clear; clc; close all;


Ts       = 0.1;
p        = 120;
Tbatch   = 4000;
Tsim     = 2000;
lambda   = 0.9995;
Np       = 50;
Nc       = 1;

% PRBS amplitudes per input [beta(rad); tau(Nm)]  
ampl_u = [0.03, 1000];

% Weights
Qy  = diag([100]);
Ru  = diag([1]);
Rdu = diag([0.1]);

% Input bounds
beta_bound = deg2rad(2);
umin  = [-beta_bound];
umax  = [ beta_bound];
dumin = [-10];
dumax = [ 10];

load linearization_nrel5mw.mat sys
sysd  = c2d(sys, Ts);
[A,B,C,D] = ssdata(sysd);
[Ny, Nu_full]  = size(D);

% Input and output dimensions and channel numbers
Nu = 1; %
Ny = 1; % 
idx_u = 1;
idx_y = 2;


% PRBS excitation
u_batch = zeros(Nu, Tbatch);
t_total = Tbatch*Ts;
F = 1; 
temp = idprbs(t_total, ampl_u(1), Ts, F).';
u_batch(1,:) = temp(:,1:Tbatch);
%temp = idprbs(t_total, ampl_u(2), Ts, F).';
%u_batch(2,:) = temp(:,1:Tbatch);

% Simulate true plant for batch
x = zeros(size(A,1),1);
y_batch = zeros(Ny, Tbatch);
for k = 1:Tbatch
    u_k_full = [u_batch(:,k); 0; 0];
    y_full = C*x + D*u_k_full;
    y_batch(:,k) = y_full(idx_y,:);
    x            = A*x + B*u_k_full;
end
x_end = x;

% Signal scaling
[us_batch, Du, ys_batch, Dy] = sigscale(u_batch, y_batch);
Du_inv = diag(1./diag(Du));
Dy_inv = diag(1./diag(Dy));

% Strictly proper batch LS
m_strict = p*(Nu + Ny);
K   = Tbatch - p;
Phis_strict = zeros(m_strict, K);
Ysf  = zeros(Ny, K);

kk = 0;
for k = (p+1):Tbatch
    kk = kk + 1;
    blocks = cell(2*p,1); bi = 0;
    for l = 1:p
        bi = bi+1; blocks{bi} = us_batch(:,k-l);
        bi = bi+1; blocks{bi} = ys_batch(:,k-l);
    end
    Phis_strict(:,kk) = vertcat(blocks{:});
    Ysf(:,kk)         = ys_batch(:,k);
end

rho    = 1e-10 * trace(Phis_strict*Phis_strict')/size(Phis_strict,1);
ThetaS_strict = (Ysf * Phis_strict') / (Phis_strict*Phis_strict' + rho*eye(m_strict));


blk = cell(2*p,1); bi = 0;
for l = 1:p
    bi = bi+1; blk{bi} = Du_inv;
    bi = bi+1; blk{bi} = Dy_inv;
end
R_strict = blkdiag(blk{:});

Theta_strict  = Dy * ThetaS_strict * R_strict;

CKappa0 = Theta_strict;
Dhat0   = zeros(Ny, Nu);
Theta_use_batch = [CKappa0, Dhat0];

G_batch_strict = Theta2ARX_MIMO(Theta_use_batch, p, Ts, Nu, Ny);


m_full = p*(Nu + Ny) + Nu;
Phis_full = zeros(m_full, K);
kk = 0;
for k = (p+1):Tbatch
    kk = kk + 1;
    blocks = cell(2*p+1,1); bi = 0;
    for l = 1:p
        bi = bi+1; blocks{bi} = us_batch(:,k-l);
        bi = bi+1; blocks{bi} = ys_batch(:,k-l);
    end
    blocks{end} = us_batch(:,k);
    Phis_full(:,kk) = vertcat(blocks{:});
end

ThetaS_full = [ThetaS_strict, zeros(Ny, Nu)];
P0    = inv(Phis_full*Phis_full' + rho*eye(m_full));
Srls  = chol(P0,'lower');
eLS   = Ysf - ThetaS_full*Phis_full;
sigma = sqrt(mean(eLS(:).^2) + eps);

% Online part
T_total = Tbatch + Tsim;

u = zeros(Nu, T_total);
y = zeros(Ny, T_total);
u(:,1:Tbatch) = u_batch;
y(:,1:Tbatch) = y_batch;

us = zeros(Nu, T_total);
ys = zeros(Ny, T_total);
us(:,1:Tbatch) = us_batch;
ys(:,1:Tbatch) = ys_batch;

Qy_tilde  = kron(eye(Np), Qy);
Ru_tilde  = kron(eye(Np), Ru);
Rdu_tilde = kron(eye(Np), Rdu);

opts = optimoptions('quadprog','Display','off', ...
    'OptimalityTolerance',1e-6,'ConstraintTolerance',1e-6,'MaxIterations',80);

I_Np = eye(Ny*Np);

xk = x_end;

r = zeros(Ny, T_total);

cut_full = p*(Nu+Ny);

% RLS (online with D term)
ThetaS = ThetaS_full;
for k = Tbatch : (T_total-1)
    if mod(k,200)==0, fprintf('k=%d/%d\n', k, T_total-1); end
    idx = (k-p):(k-1);
    z_p = reshape([u(:,idx); y(:,idx)], [], 1);
    z_p_s = reshape([us(:,idx); ys(:,idx)], [], 1);

    phi_s   = [z_p_s; us(:,k)];
    y_now_s = ys(:,k);

    [ThetaS, Srls, sigma] = SCRLS(ThetaS, Srls, phi_s, y_now_s', lambda, sigma);
    sigma = abs(sigma);

    blk = cell(2*p+1,1); bi = 0;
    for l = 1:p
        bi = bi+1; blk{bi} = Du_inv;
        bi = bi+1; blk{bi} = Dy_inv;
    end
    blk{end} = Du_inv;
    R_full = blkdiag(blk{:});

    Theta = Dy * ThetaS * R_full;

    CKappa = Theta(:, 1:cut_full);
    Dhat   = Theta(:, cut_full + (1:Nu));
    Theta_use = [CKappa, Dhat];

    [Gamma_tilde, H_tilde, G_tilde] = build_prediction_matrices(CKappa, Dhat, p, Np, Nu, Ny);
    Gamma = (I_Np - G_tilde) \ Gamma_tilde;
    Hpred = (I_Np - G_tilde) \ H_tilde;

    r_tilde = repmat(r(:,k), Np, 1);
    [Hqp, fqp, Aeq, beq, Ain, bin, lb, ub] = build_spc_qp( ...
        Gamma, Hpred, z_p, r_tilde, Qy_tilde, Ru_tilde, Rdu_tilde, ...
        Np, Nc, Nu, umin, umax, dumin, dumax);

    Hqp = (Hqp + Hqp.')/2;
    U_tilde = quadprog(Hqp, fqp, Ain, bin, Aeq, beq, lb, ub, [], opts);

    u_apply = U_tilde(1:Nu);
    u(:,k+1) = u_apply;

    u_full = [u_apply; 0; 0];
    xk = A*xk + B*u_full;
    y_full = C*xk + D*u_full;
    y(:,k+1) = y_full(idx_y,:);

    us(:,k+1) = Du_inv * u(:,k+1);
    ys(:,k+1) = Dy_inv * y(:,k+1);
end

% Bode
G_final = Theta2ARX_MIMO(Theta_use, p, Ts, Nu, Ny);

opt = bodeoptions;
opt.PhaseWrapping = 'on';
opt.PhaseWrappingBranch = 360;

wmin = 1e-4; wmax = 0.9*pi/Ts;  
figure('Color','w');
bodeplot(G_final,{wmin,wmax},'r-.',opt); hold on;
bodeplot(sysd(idx_y,idx_u),{wmin,wmax},'k',opt);     
bodeplot(G_batch_strict,{wmin,wmax},'b--',opt);
legend('Final ARX','True','Batch ARX ','Location','best');

% Time domain plots
t_batch  = (0:Tbatch-1)*Ts;
t_online = (Tbatch:T_total-1)*Ts;

figure('Color','w','Name','Batch');
subplot(2,1,1);
plot(t_batch, u_batch(1,:), 'LineWidth', 1.2);
ylabel('Inputs'); legend('\beta','\tau'); grid on;

subplot(2,1,2);
plot(t_batch, y_batch(1,:), 'LineWidth', 1.2);
ylabel('Outputs'); xlabel('Time [s]');
legend('\omega_g','\theta_p'); grid on;

figure('Color','w','Name','Online');
subplot(2,1,1);
plot(t_online, u(1,Tbatch+1:end), 'LineWidth', 1.2);
ylabel('Inputs'); legend('\beta','\tau'); grid on;

subplot(2,1,2);
plot(t_online, y(1,Tbatch+1:end), 'LineWidth', 1.2);
ylabel('Outputs'); xlabel('Time [s]');
legend('\omega_g','\theta_p'); grid on;

