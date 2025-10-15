clear; clc; close all;

Ts       = 0.1;
p        = 60;                 
Tbatch   = 12000;
Tsim     = 200;
lambda   = 0.99995;
Np       = 50;  
Nc = 1;

ampl_u = [0.3, 2000];          % [beta; tau]

Qy  = diag([100, 100]);
Ru  = diag([1, 1]);
Rdu = diag([0.1, 0.1]);

beta_bound = deg2rad(2);
umin  = [-beta_bound; -100];
umax  = [ beta_bound; 100];
dumin = [-10; -10];
dumax = [ 10; 10];

% ridge regression factor
rho_factor = 1e-2;

% True system
load linearization_nrel5mw.mat sys
sysd  = c2d(sys, Ts);
[A,B,C,D] = ssdata(sysd);
Nu = 2; Ny = 2; idx_u = 1:2; idx_y = 1:2;

% Excitation: PRBS + additional low freq. PRBS 
u_batch = zeros(Nu, Tbatch);
t_total = Tbatch*Ts;
baseF = 5;
tmp = idprbs(t_total, 1, Ts, baseF, [], [] , Nu).';
u_batch(1,:) = tmp(1,1:Tbatch)*ampl_u(1);
u_batch(2,:) = tmp(2,1:Tbatch)*ampl_u(2);

F_lf   = 0.3;
tmp_lf = idprbs(t_total, 1, Ts, F_lf, [], [], Nu).';
lf_gain = [0.25; 0.25];
u_batch = u_batch + [lf_gain(1)*ampl_u(1)*tmp_lf(1,1:Tbatch);
                     lf_gain(2)*ampl_u(2)*tmp_lf(2,1:Tbatch)];

% simulate
x = zeros(size(A,1),1);
y_batch = zeros(Ny, Tbatch);
for k = 1:Tbatch
    u_full = [u_batch(:,k); 0];
    y_full = C*x + D*u_full;
    y_batch(:,k) = y_full(idx_y,:);
    x = A*x + B*u_full;
end
x_end = x;

% scaling
[us_batch, Du, ys_batch, Dy] = sigscale(u_batch, y_batch);
Du_inv = diag(1./diag(Du));  Dy_inv = diag(1./diag(Dy));

% strictly proper batch LS
m = p*(Nu+Ny);  K = Tbatch - p;
Phis = zeros(m, K);  Ysf = zeros(Ny, K);
kk = 0;
for k = (p+1):Tbatch
    kk = kk + 1;
    blocks = cell(2*p,1); bi = 0;
    for l = 1:p
        bi=bi+1; blocks{bi} = us_batch(:,k-l);
        bi=bi+1; blocks{bi} = ys_batch(:,k-l);
    end
    Phis(:,kk) = vertcat(blocks{:});
    Ysf(:,kk)  = ys_batch(:,k);
end

rho = rho_factor * trace(Phis*Phis')/size(Phis,1);
ThetaS = (Ysf*Phis') / (Phis*Phis' + rho*eye(m));

blk = cell(2*p,1); bi=0;
for l=1:p
    bi=bi+1; blk{bi} = Du_inv;
    bi=bi+1; blk{bi} = Dy_inv;
end
R_scale = blkdiag(blk{:});
Theta_phys = Dy * ThetaS * R_scale;

CKappa0 = Theta_phys;
Dhat0   = zeros(Ny,Nu);
Theta_use_batch = [CKappa0, Dhat0];
G_batch_strict = Theta2ARX_MIMO(Theta_use_batch, p, Ts, Nu, Ny);

% online (unchanged)
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

ThetaS_full = [ThetaS, zeros(Ny,Nu)];
P0    = inv(Phis_full*Phis_full' + rho*blkdiag(eye(m), eye(Nu)));
Srls  = chol(P0,'lower');
eLS   = Ysf - ThetaS_full*Phis_full;
sigma = sqrt(mean(eLS(:).^2) + eps);

T_total = Tbatch + Tsim;
u = zeros(Nu, T_total); y = zeros(Ny, T_total);
u(:,1:Tbatch) = u_batch; y(:,1:Tbatch) = y_batch;
us = zeros(Nu, T_total); ys = zeros(Ny, T_total);
us(:,1:Tbatch)=us_batch; ys(:,1:Tbatch)=ys_batch;

Qy_tilde  = kron(eye(Np), Qy);
Ru_tilde  = kron(eye(Np), Ru);
Rdu_tilde = kron(eye(Np), Rdu);
opts = optimoptions('quadprog','Display','off', ...
    'OptimalityTolerance',1e-6,'ConstraintTolerance',1e-6,'MaxIterations',80);
I_Np = eye(Ny*Np);
xk = x_end; r = zeros(Ny, T_total);
cut_full = p*(Nu+Ny);

ThetaS_k = ThetaS_full;
for k = Tbatch : (T_total-1)
    if mod(k,200)==0, fprintf('k=%d/%d\n', k, T_total-1); end
    idx = (k-p):(k-1);
    z_p   = reshape([u(:,idx); y(:,idx)], [], 1);
    z_p_s = reshape([us(:,idx); ys(:,idx)], [], 1);
    phi_s   = [z_p_s; us(:,k)];
    y_now_s = ys(:,k);
    [ThetaS_k, Srls, sigma] = SCRLS(ThetaS_k, Srls, phi_s, y_now_s', lambda, sigma);
    blk = cell(2*p+1,1); bi=0;
    for l=1:p, bi=bi+1; blk{bi}=Du_inv; bi=bi+1; blk{bi}=Dy_inv; end
    blk{end}=Du_inv; R_full = blkdiag(blk{:});
    Theta = Dy * ThetaS_k * R_full;
    CKappa = Theta(:,1:cut_full);
    Dhat   = Theta(:,cut_full+(1:Nu));
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
    u_full = [u_apply; 0];
    xk = A*xk + B*u_full;
    y_full = C*xk + D*u_full;
    y(:,k+1) = y_full(idx_y,:);
    us(:,k+1) = Du_inv*u(:,k+1);
    ys(:,k+1) = Dy_inv*y(:,k+1);
end

G_final = Theta2ARX_MIMO([CKappa, Dhat], p, Ts, Nu, Ny);

opt = bodeoptions; opt.PhaseWrapping='on'; opt.PhaseWrappingBranch=360;
wmin = 1e-4; wmax = 0.9*pi/Ts;
figure('Color','w');
bodeplot(G_final,{wmin,wmax},'r-.',opt); hold on;
bodeplot(sysd(idx_y,idx_u),{wmin,wmax},'k',opt);
bodeplot(G_batch_strict,{wmin,wmax},'b--',opt);
legend('Final ARX','True','Batch ARX','Location','best');

t_batch  = (0:Tbatch-1)*Ts;
t_online = (Tbatch:Tbatch+Tsim-1)*Ts;
figure('Color','w','Name','Batch'); subplot(2,1,1);
plot(t_batch,u_batch(1,:),'LineWidth',1.2); hold on;
plot(t_batch,u_batch(2,:),'LineWidth',1.2);
ylabel('Inputs'); legend('\beta','\tau'); grid on;
subplot(2,1,2);
plot(t_batch,y_batch(1,:),'LineWidth',1.2); ylabel('Outputs'); xlabel('Time [s]');
legend('\omega_g','\theta_p'); grid on;

figure('Color','w','Name','Online'); subplot(2,1,1);
plot(t_online,u(1,Tbatch+1:end),'LineWidth',1.2); ylabel('Inputs'); legend('\beta'); grid on;
subplot(2,1,2);
plot(t_online,y(1,Tbatch+1:end),'LineWidth',1.2); ylabel('Outputs'); xlabel('Time [s]');
legend('\omega_g'); grid on;

Suu = (u_batch*u_batch.')/size(u_batch,2);
fprintf('cond(Suu)=%.2e, corr(u1,u2)=%.3f, p=%d\n', cond(Suu), corr(u_batch(1,:).',u_batch(2,:).'), p);
