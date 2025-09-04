%% Closed-loop test

rng(4);

% True system (ground truth)
Nu = 1; Ny = 1;
A  = 0.85;
B  = 1.00;
C  = 1.00;
D  = 0.00;

% Simple Controller design
p_cl = 0.5;                
K   = place(A, B, p_cl);  

Acl  = A - B*K;
Gr = 1 / ( C * ((eye(size(A)) - Acl) \ B) + D ); 

% Simulation settings
T  = 1400;
r  = zeros(1,T); 
r(200:end) = 1.0;     % step ref at k=200
noise_std = 0.02;     % noise stddev

x = 0;                                 
u = zeros(Nu,T);
y = zeros(Ny,T);

% Closed-loop simulation (u = -Kx*x + Gr*r + noise)
for k = 1:T-1
    y(:,k) = C*x + D*u(:,k);
    u(:,k) = -Kx*x + Gr*r(:,k) + noise_std*randn(Nu,1);
    x = A*x + B*u(:,k);
end
y(:,T) = C*x + D*u(:,T);

% solve RLS and build predictor recursively

p  = 25; 
Np = 30;

[Yf, Zp, Uf] = build_data_matrices(u, y, p);
[CKappa, Dhat] = estimate_theta_ls(Yf, Zp, Uf, p, Nu, Ny);
[Gamma_tilde, H_tilde, G_tilde] = build_prediction_matrices(CKappa, Dhat, p, Np, Nu, Ny);

I = eye(Ny*Np);
Gamma = (I - G_tilde)\Gamma_tilde;
H     = (I - G_tilde)\H_tilde;

% Predict from a time k (near the end so we have enough past samples)
k = T - 250;                             
z_p = build_zp(u(:,1:k), y(:,1:k), p);
u_future = reshape(u(:,k+1:k+Np), Nu*Np, 1);

[y_pred_vec, y_pred_mat] = spc_predict(Gamma, H, z_p, u_future, Nu, Ny, Np);
y_true = y(:, k+1:k+Np);


rmse = sqrt(mean( (y_pred_mat(:) - y_true(:)).^2 ));
fprintf('Closed-loop RMSE over horizon: %.4g\n', rmse);


% Plots 
t  = 1:T; 
tp = (k+1):(k+Np);              %Predicted timesteps
y_hat_full = nan(Ny,T); 
y_hat_full(:,tp) = y_pred_mat;

figure;
subplot(3,1,1);
plot(t, r, 'g--', 'LineWidth',1); hold on;
plot(t, y, 'w-', 'LineWidth',1.5);           
plot(t, y_hat_full, 'r--', 'LineWidth',1.5);
grid on; xlabel('k'); ylabel('y');
title('Closed-loop: true (white), predicted window (red), ref (green)');
legend('r','y true','y predicted','Location','Best');

subplot(3,1,2);
plot(t, u, 'b-','LineWidth',1);
grid on; xlabel('k'); ylabel('u'); title('Control input (with noise)');

subplot(3,1,3);
plot(tp, (y_pred_mat - y_true), 'm-','LineWidth',1);
grid on; xlabel('k'); ylabel('pred - true'); title('Prediction error over horizon');
