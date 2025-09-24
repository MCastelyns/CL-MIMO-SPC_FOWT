%% SPC on NREL 5MW + ARX-vs-Truth check (MIMO) ---------------------------------
clear; clc; close all;


load("linearization_nrel5mw.mat", "sys");
[A,B,C,D] = ssdata(sys);     

% sizes
Nu = size(B,2);                    % [beta, tau, wind]
Ny = size(C,1);                    % [omega_g, theta_p]

% Sim settings 
dt      = 0.1;                    
p       = 2000;                    
t_past  = dt*p;
Tau_stddev      = 0.01;
Beta_stddev     = 0.01;
WindSpeed_stddev= 0.01;
debug   = true;


[y,u,x] = simulateNREL5MW(p, t_past, Tau_stddev, Beta_stddev, WindSpeed_stddev, debug);


T       = 2005;                     
t_final = dt * T;
t       = linspace(0, t_final, T+1).';
lambda  = 0.995;                   
Np      = 50;                      
m       = p*(Nu+Ny) + Nu;          

Theta   = zeros(Ny, m);
S       = 1e4*eye(m);              
sigma   = 1.0;                    


u = u.';  y = y.';

I_Np = eye(Ny*Np);

U_store = zeros(Nu, T);
Y_store = zeros(Ny, T);


xk = x(end,:).';

% Use batch LS for initial guess
batch_fit = spc_fit_predictor(u,y,p,Np,Nu,Ny);

% Set initial guesses for loop to these found values
Ckappa = batch_fit.CKappa;
D = batch_fit.D;
Theta = [Ckappa, D];

r = zeros(Ny, T+1);                



for k = (p+1):(T-1)                
    idx = (k-p):(k-1);
    z_p = reshape([u(:,idx); y(:,idx)], [], 1);     

    phi    = [z_p; u(:,k)];         
    y_next = y(:,k);                 

    % SCRLS update
    [Theta, S, sigma] = SCRLS(Theta, S, phi, y_next, lambda, sigma);

    % Build prediction matrices from Theta
    cut     = p*(Nu+Ny);
    CKappa  = Theta(:, 1:cut);              
    Dhat    = Theta(:, cut + (1:Nu));       

    [Gamma_tilde, H_tilde, G_tilde] = build_prediction_matrices(CKappa, Dhat, p, Np, Nu, Ny);
    Gamma = (I_Np - G_tilde)\Gamma_tilde;
    Hpred = (I_Np - G_tilde)\H_tilde;

    Nc  = 10;

    Qy  = diag([1;1]);
    Ru  = diag([0.1; 0.1; 0.1]);
    Rdu = diag([0.1; 0.1; 0]);

    Qy_tilde  = kron(eye(Np), Qy);
    Ru_tilde  = kron(eye(Np), Ru);
    Rdu_tilde = kron(eye(Np), Rdu);

    % Constant reference, can make it change too
    r_tilde = repmat(r(:,k), Np, 1);

    umin  = [-50; -50; -100];
    umax  = [ 50;  50;  100];
    dumin = [-10; -10;  -20];
    dumax = [ 10;  10;   20];

    [Hqp,fqp,Aeq,beq,Ain,bin,lb,ub] = build_spc_qp( ...
        Gamma, Hpred, z_p, r_tilde, Qy_tilde, Ru_tilde, Rdu_tilde, ...
        Np, Nc, Nu, umin, umax, dumin, dumax);

    % Ensure symmetric Hqp (quadprog likes this better, it sometimes throws
    % warnings without this, i assume from rounding errors)
    Hqp = (Hqp + Hqp.')/2;

    opts = optimoptions('quadprog','Display','off');
    u_tilde = quadprog(Hqp, fqp, Ain, bin, Aeq, beq, lb, ub, [], opts);

    u_apply = u_tilde(1:Nu);


    xk = A*xk + B*u_apply;
    yk = C*xk + D*u_apply;

    u(:,k+1) = u_apply;
    y(:,k+1) = yk;

    U_store(:,k) = u_apply;
    Y_store(:,k) = yk;
end


figure;
subplot(3,1,1);
plot(t(1:T), u(1,:), 'b', 'LineWidth', 1.2); hold on;
plot(t(1:T), u(2,:), 'r', 'LineWidth', 1.2);
ylabel('Control Inputs'); legend('\beta (pitch)','\tau (torque)'); grid on;

subplot(3,1,2);
plot(t(1:T), u(3,:), 'k', 'LineWidth', 1.2);
ylabel('Disturbance (wind)'); grid on;

subplot(3,1,3);
plot(t(1:T), y(1,:), 'b', 'LineWidth', 1.2); hold on;
plot(t(1:T), y(2,:), 'r', 'LineWidth', 1.2);
ylabel('Outputs'); xlabel('Time [s]');
legend('\omega_g','\theta_p'); grid on;


include_direct = true;
[A_lags, B0, B_lags] = theta_to_arx_mimo(Theta, p, Nu, Ny, include_direct);



% FRFs
nw = 1024; w = linspace(0,pi,nw);
Htrue = frf_true_ss(A,B,C,D,w);          
Hhat  = frf_from_arx(A_lags,B0,B_lags,w);


figure('Name','Bode: True vs ARX (MIMO)','Color','w');
for i = 1:Ny
  for j = 1:Nu
    k = (i-1)*Nu + j;
    % Magnitude
    subplot(2*Ny,Nu,k);
    plot(w,20*log10(abs(squeeze(Htrue(i,j,:)))),'k-','LineWidth',1.1); hold on;
    plot(w,20*log10(abs(squeeze(Hhat (i,j,:)))) ,'r--','LineWidth',1.1);
    grid on; if i==1, title(sprintf('u_%d → y_%d',j,i)); end
    if j==1, ylabel('Mag [dB]'); end
    % Phase
    subplot(2*Ny,Nu,Ny*Nu+k);
    phT = unwrap(angle(squeeze(Htrue(i,j,:))))*180/pi;
    phH = unwrap(angle(squeeze(Hhat (i,j,:))))*180/pi;
    plot(w,phT,'k-','LineWidth',1.1); hold on;
    plot(w,phH,'r--','LineWidth',1.1);
    grid on; if j==1, ylabel('Phase [deg]'); end; xlabel('\omega [rad/sample]');
  end
end
legend({'True','ARX'},'Location','bestoutside');

% Per-channel FRF error summary
fprintf('\n|Hhat - Htrue|_inf per channel:\n');
for i=1:Ny 
    for j=1:Nu
  err = max(abs(squeeze(Hhat(i,j,:)-Htrue(i,j,:))));
  fprintf('  y_%d <- u_%d : %.3e\n', i, j, err);
    end 
end


function [A_lags,B0,B_lags] = theta_to_arx_mimo(Theta,p,Nu,Ny,include_direct)
    cut = p*(Nu+Ny);
    CK  = Theta(:,1:cut);                         
    if include_direct
        B0 = Theta(:,cut+(1:Nu));                 
    else
        B0 = zeros(Ny,Nu);
    end
    A_lags = cell(1,p);  B_lags = cell(1,p);
    for i=1:p
        cols  = (i-1)*(Nu+Ny) + (1:(Nu+Ny));
        block = CK(:,cols);                       
        Bu_i  = block(:,1:Nu);                    
        Ay_i  = block(:,Nu+1:Nu+Ny);              
        ell   = p - i + 1;                        
        A_lags{ell} = Ay_i;
        B_lags{ell} = Bu_i;
    end
end

function H = frf_true_ss(A,B,C,D,w)
% H(:,:,k) = C*(zI - A)^{-1}B + D, z = e^{jω_k}
    Ny=size(C,1); 
    Nu=size(B,2); 
    nw=numel(w);
    H = zeros(Ny,Nu,nw); 
    I = eye(size(A,1));
    for k=1:nw
        z = exp(1j*w(k));
        H(:,:,k) = C*((z*I - A)\B) + D;
    end
end

function H = frf_from_arx(A_lags,B0,B_lags,w)
% H(z) = B(z^-1)/A(z^-1)
    Ny=size(B0,1); 
    Nu=size(B0,2); 
    p=numel(A_lags); 
    nw=numel(w); 
    H=zeros(Ny,Nu,nw);
    I = eye(Ny);
    for k=1:nw
        zinv = exp(-1j*w(k));
        Az = I; Bz = B0; zpow = zinv;
        for ell=1:p
            Az = Az - A_lags{ell} * zpow;
            Bz = Bz + B_lags{ell} * zpow;
            zpow = zpow * zinv;
        end
        H(:,:,k) = Az \ Bz;
    end
end
