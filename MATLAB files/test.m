rng(4);

Nu = 1; Ny = 1;
A  = 0.85;  B = 1.00;  C = 1.00;  D = 0.00;


p_cl = 0.5;
Kx   = place(A, B, p_cl);                  
Acl  = A - B*Kx;
Gr   = 1 / ( C * ((eye(size(A)) - Acl) \ B) + D );   


T  = 1400;
r  = zeros(1,T); r(200:end) = 1.0;         % step at k=200
noise_std = 0.02;

x = 0;
u = zeros(Nu,T);
y = zeros(Ny,T);

for k = 1:T-1
    y(:,k) = C*x + D*u(:,k);
    u(:,k) = -Kx*x + Gr*r(:,k) + noise_std*randn(Nu,1);
    x = A*x + B*u(:,k);
end
y(:,T) = C*x + D*u(:,T);


p   = 25;                  
Np  = 1;                   
m   = p*(Nu+Ny) + Nu;      
lam = 0.995;               


Theta = zeros(Ny, m);      
S     = eye(m);            
sigma = 1.0;


y_pred1 = nan(Ny, T);
err1    = nan(Ny, T);

I = eye(Ny*Np);

for k = p:(T-1)
    idx  = (k-p+1):k;
    z_p  = reshape([u(:,idx); y(:,idx)], [], 1);  
    phi  = [z_p; u(:,k+1)];                      


    y_next = y(:,k+1);

    [Theta, S, sigma] = SCRLS(Theta, S, phi, y_next, lam, sigma);

   


    cut     = p*(Nu+Ny);
    CKappa  = Theta(:, 1:cut);            
    Dhat    = Theta(:, cut + (1:Nu));     


    [Gamma_tilde, H_tilde, G_tilde] = build_prediction_matrices(CKappa, Dhat, p, Np, Nu, Ny);
    Gamma = (I - G_tilde)\Gamma_tilde;
    H     = (I - G_tilde)\H_tilde;


    u_future = u(:,k+1);  
    [y_pred_vec, ~] = spc_predict(Gamma, H, z_p, u_future, Nu, Ny, Np);

    y_pred1(:,k+1) = y_pred_vec;
    err1(:,k+1)    = y(:,k+1) - y_pred_vec;
end


t = 1:T;

figure;
subplot(2,1,1);
plot(t, y, 'k-', 'LineWidth', 1.2); hold on;
plot(t, y_pred1, 'r--', 'LineWidth', 1.2);
grid on; xlabel('k'); ylabel('y');
title('Online one-step prediction SCLS');
legend('True y','Predicted y','Location','Best');

subplot(2,1,2);
plot(t, err1, 'm-', 'LineWidth', 1.0);
grid on; xlabel('k'); ylabel('y - yhat');
title('Instantaneous one-step prediction error');




