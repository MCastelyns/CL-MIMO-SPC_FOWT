clear; clc; close all;

% Settings
Ts       = 0.1;             
p        = 120;             % past window / ARX order (?)
Tbatch   = 4000;            
Tsim     = 2000;            
lambda   = 0.9995;           % forgetting factor
Np       = 50;              % prediction horizon
Nc       = 1;               % control horizon

% Excitation stddev per input: [beta(rad); tau(Nm); wind(m/s)]
%std_u = [0.03; 100; 0.1];
ampl_u = [0.03; 100; 0.1];
% std_u = [0.03; 3e4; 0.3];

% Weights
Qy  = diag([100; 10000]);
Ru  = diag([1; 1; 0.1]);
Rdu = diag([0.1; 0.1; 0.1]);

% Input bounds
beta_bound = deg2rad(2);
umin  = [-beta_bound; -5e5; 0];
umax  = [ beta_bound;  5e5; 0];
dumin = [-10; -1e5; 0];
dumax = [ 10;  1e5; 0];

% Load continuous linearization and discretize
load linearization_nrel5mw.mat sys
sysd  = c2d(sys, Ts);
[A,B,C,D] = ssdata(sysd);
[Ny, Nu]  = size(D);


u_batch = zeros(Nu, Tbatch);

%u_batch(1,:) = std_u(1) * randn(1, Tbatch);          % beta
%u_batch(2,:) = std_u(2) * randn(1, Tbatch);          % tau
%u_batch(3,:) = std_u(3) * randn(1, Tbatch);          % V

time_batch = Tbatch*Ts;
F = 1; % Cutoff frequency of prbs
temp = idprbs(time_batch, ampl_u(1),Ts, F)';
u_batch(1,:) = temp(1:Tbatch);              % beta
temp = idprbs(time_batch, ampl_u(2),Ts, F)';
u_batch(2,:) = temp(1:Tbatch);              % tau
%temp = idprbs(time_batch, ampl_u(3),Ts, F)';
%u_batch(3,:) = temp(1:Tbatch);              % V

% Simulate true plant for batch segment
x = zeros(size(A,1),1);
y_batch = zeros(Ny, Tbatch);
for k = 1:Tbatch
    y_batch(:,k) = C*x + D*u_batch(:,k);
    x            = A*x + B*u_batch(:,k);
end
x_end = x;  % state at end of batch (for continuity into closed-loop)

% Batch LS
m   = p*(Nu + Ny) + Nu;        
K   = Tbatch - p;              
Phi = zeros(m,  K);
Y   = zeros(Ny, K);

kk = 0;
for k = (p+1):Tbatch
    kk = kk + 1;
    blocks = cell(2*p+1,1); bi = 0;
    for l = 1:p
        bi = bi+1; blocks{bi} = u_batch(:,k-l);
        bi = bi+1; blocks{bi} = y_batch(:,k-l);
    end
    blocks{end} = u_batch(:,k);
    Phi(:,kk) = vertcat(blocks{:});
    Y(:,kk)   = y_batch(:,k);
end

% Scaling
s = std(Phi, 0, 2);  s(s==0) = 1;
Phis = Phi ./ s;

rho     = 1e-5 * trace(Phis*Phis')/size(Phis,1);
ThetaS  = (Y * Phis') / (Phis*Phis' + rho*eye(m));   
Theta   = ThetaS ./ s.';                              % unscale

% ARX transfer matrix from batch fit
G_batch = Theta2ARX_MIMO(Theta, p, Ts, Nu, Ny);
G_batch.InputName  = sysd.InputName;
G_batch.OutputName = sysd.OutputName;

% Online part
T_online = p + Tsim;
u = zeros(Nu, T_online+1);
y = zeros(Ny, T_online+1);


u(:,1:p+1) = u_batch(:, Tbatch-p : Tbatch);
y(:,1:p+1) = y_batch(:, Tbatch-p : Tbatch);


P0    = inv(Phis*Phis' + rho*eye(m));    
Srls  = chol(P0,'lower');               
eLS   = Y - ThetaS*Phis;
sigma = sqrt(mean(eLS(:).^2) + eps);

% Block-diagonal weights for QP
Qy_tilde  = kron(eye(Np), Qy);
Ru_tilde  = kron(eye(Np), Ru);
Rdu_tilde = kron(eye(Np), Rdu);

% QP options
opts = optimoptions('quadprog','Display','off', ...
    'OptimalityTolerance',1e-6,'ConstraintTolerance',1e-6,'MaxIterations',80);

I_Np = eye(Ny*Np);

% Start from the true state after batch to avoid a jump
xk = x_end;

% Zero reference (linearized around operating point)
r = zeros(Ny, T_online+1);

% CL SPC 

for k = (p+1):T_online
    if mod(k,100)==0, fprintf('k=%d\n', k); end
    idx = (k-p):(k-1);

    z_p = reshape([u(:,idx); y(:,idx)], [], 1);

    phi  = [z_p; u(:,k)];

    % SCRLS update
    phis = phi ./ s;                
    y_now = y(:,k);
    [ThetaS, Srls, sigma] = SCRLS(ThetaS, Srls, phis, y_now', lambda, sigma);
    sigma = abs(sigma);
    Theta = ThetaS ./ s.';          % unscale 


    cut    = p*(Nu+Ny);
    CKappa = Theta(:, 1:cut);
    Theta(:, cut + (1:Nu)) = zeros(size(D));
    Dhat   = Theta(:, cut + (1:Nu));


    [Gamma_tilde, H_tilde, G_tilde] = build_prediction_matrices(CKappa, Dhat, p, Np, Nu, Ny);
    Gamma = (I_Np - G_tilde) \ Gamma_tilde;
    Hpred = (I_Np - G_tilde) \ H_tilde;


    % Solve QP
    r_tilde = repmat(r(:,k), Np, 1);
    [Hqp, fqp, Aeq, beq, Ain, bin, lb, ub] = build_spc_qp( ...
        Gamma, Hpred, z_p, r_tilde, Qy_tilde, Ru_tilde, Rdu_tilde, ...
        Np, Nc, Nu, umin, umax, dumin, dumax);

    Hqp = (Hqp + Hqp.')/2;
    U_tilde = quadprog(Hqp, fqp, Ain, bin, Aeq, beq, lb, ub, [], opts);

    u_apply = U_tilde(1:Nu);
    % Wind = disturbance: add noise
    %u_apply(3) = u_apply(3) + std_u(3)*randn;

    u(:,k+1) = u_apply;

    % Apply true plant
    xk = A*xk + B*u_apply;
    yk = C*xk + D*u_apply;
    y(:,k+1) = yk;
end

% Bode plots
G_final = Theta2ARX_MIMO(Theta, p, Ts, Nu, Ny);
opt = bodeoptions; 
opt.PhaseWrapping='on'; 
opt.PhaseWrappingBranch=360;

wmin = 1e-2; wmax = 100;
figure('Color','w'); 
bodeplot(G_final,{wmin,wmax},'r-.',opt); hold on;
bodeplot(sysd,   {wmin,wmax},'k',  opt);
bodeplot(G_batch,{wmin,wmax},'b--',opt);
legend('Final ARX','True','Batch ARX','Location','best');

% time domain plots
time_hist   = (0:p)*Ts;
time_online = (p:(p+Tsim))*Ts;

figure('Color','w','Name','History (from batch)');
subplot(3,1,1);
plot(time_hist, u(1,1:p+1),'LineWidth',1.2); hold on;
plot(time_hist, u(2,1:p+1),'LineWidth',1.2);
ylabel('Inputs'); legend('\beta','\tau'); grid on;

subplot(3,1,2);
plot(time_hist, u(3,1:p+1),'k','LineWidth',1.2);
ylabel('Wind'); grid on;

subplot(3,1,3);
plot(time_hist, y(1,1:p+1),'LineWidth',1.2); hold on;
plot(time_hist, y(2,1:p+1),'LineWidth',1.2);
ylabel('Outputs'); xlabel('Time [s]');
legend('\omega_g','\theta_p'); grid on;

figure('Color','w','Name','Closed-loop (online)');
subplot(3,1,1);
plot(time_online, u(1,p+1:end),'LineWidth',1.2); hold on;
plot(time_online, u(2,p+1:end),'LineWidth',1.2);
ylabel('Inputs'); legend('\beta','\tau'); grid on;

subplot(3,1,2);
plot(time_online, u(3,p+1:end),'k','LineWidth',1.2);
ylabel('Wind'); grid on;

subplot(3,1,3);
plot(time_online, y(1,p+1:end),'LineWidth',1.2); hold on;
plot(time_online, y(2,p+1:end),'LineWidth',1.2);
ylabel('Outputs'); xlabel('Time [s]');
legend('\omega_g','\theta_p'); grid on;

