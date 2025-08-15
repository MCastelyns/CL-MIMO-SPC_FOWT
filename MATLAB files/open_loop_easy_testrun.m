% hyperparameters 
rng(1);

Nu=1; 
Ny=1; 
p=15; 
Np=30; 
T=1200;

% true system (ground truth)
A=0.85; B=1.0; C=1.0; D=0.0; K=0.2;
Atil = A - K*C;
Btil = B - K*D;

% excitation: random steps so system moves
u = zeros(Nu,T);
blk = 20;                   % block length
for k=1:T
    if mod(k,blk)==1
        u(:,k) = 0.8*(2*(rand>0.5)-1); % +/- amplitude
    else
        u(:,k) = u(:,k-1);
    end
end
% small extra noise
u = u + 0.15*randn(size(u));

% simulate innovation model (ground truth)
x = zeros(1,1);
y = zeros(Ny,T);
for k=1:T-1
    y(:,k) = C*x + D*u(:,k);
    x = Atil*x + Btil*u(:,k) + K*y(:,k);   % innovation recursion
end
y(:,T) = C*x + D*u(:,T);

% estimate Markov params & build predictor
[Yf,Zp,Uf] = build_data_matrices(u,y,p);
[CKappa,Dhat] = estimate_theta_ls(Yf,Zp,Uf,p,Nu,Ny);
[Gamma_tilde,H_tilde,G_tilde] = build_prediction_matrices(CKappa,Dhat,p,Np,Nu,Ny);

I = eye(Ny*Np);
Gamma = (I - G_tilde)\Gamma_tilde;
H     = (I - G_tilde)\H_tilde;

% pick a time k and predict future 
k = T-200;                       
z_p = build_zp(u(:,1:k), y(:,1:k), p);
u_future = reshape(u(:,k+1:k+Np), Nu*Np, 1);

[y_pred_vec, y_pred_mat] = spc_predict(Gamma, H, z_p, u_future, Nu, Ny, Np);

% ground truth for comparison
y_true = y(:,k+1:k+Np);

% simple rmse error check
rmse = sqrt(mean( (y_pred_mat(:) - y_true(:)).^2 ));
fprintf('Open-loop RMSE over horizon: %.4g\n', rmse);

% plotting
t  = 1:T;
tp = (k+1):(k+Np);

% have to define a same length vector to plot predictions (NaN outside the window)
y_hat_full = nan(Ny, T);
y_hat_full(:, tp) = y_pred_mat;

figure; 
subplot(2,1,1);
plot(t, y(1,:), 'w-', 'LineWidth', 1); hold on;
plot(t, y_hat_full(1,:), 'r--', 'LineWidth', 1.5);
grid on;
xlabel('k'); ylabel('y');
title('Output: true (black) and predicted window (red dashed)');
legend('y true','y predicted (window)','Location','Best');

subplot(2,1,2);
plot(t, u(1,:), 'b-','LineWidth',1);
grid on;
xlabel('k'); ylabel('u');
title('Input (excitation)');