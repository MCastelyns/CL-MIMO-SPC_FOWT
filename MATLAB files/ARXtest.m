clear; clc; close all;

A  = 0.85;  B = 1.00;  C = 1.00;  D = 0.00;   %
Atrue = [1, -A];
Btrue = [D, B*C];   

rng(4);
Nu = 1; Ny = 1;

p_cl = 0.5;
Kx   = place(A, B, p_cl);
Acl  = A - B*Kx;
Gr   = 1 / ( C * ((eye(1) - Acl) \ B) + D );

T  = 1400;
r  = zeros(1,T); r(200:end) = 1.0;
noise_std = 0.02;

x = 0;
u = zeros(1,T);
y = zeros(1,T);
for k = 1:T-1
    y(:,k) = C*x + D*u(:,k);
    u(:,k) = -Kx*x + Gr*r(:,k) + noise_std*randn(Nu,1);
    x = A*x + B*u(:,k);
end
y(:,T) = C*x + D*u(:,T);


p   = 10;                   
m   = p*(Nu+Ny) + Nu;          
lam = 0.995;                   

Theta = zeros(Ny, m);          
S     = eye(m);
sigma = 1.0;

for k = p:(T-1)
    idx = (k-p+1):k;
    z_p = reshape([u(:,idx); y(:,idx)], [], 1);
    phi = [z_p; u(:,k+1)];
    y_next = y(:,k+1);

    [Theta, S, sigma] = SCRLS(Theta, S, phi, y_next, lam, sigma);
end


cut     = p*(Nu+Ny);
CKappa  = Theta(:, 1:cut);        
D   = Theta(:, cut + (1:Nu)); 

Kmat  = reshape(CKappa, (Nu+Ny), p); 
beta  = fliplr(Kmat(1,:));           % multiplies u(k-1), u(k-2), ...
alpha = fliplr(Kmat(2,:));           % multiplies y(k-1), y(k-2), ...

Ahat = [1, -alpha];
Bhat = [D, beta];


npts = 2048;
[Htrue, w] = freqz(Btrue, Atrue, npts);
Hhat        = freqz(Bhat,  Ahat,  npts);

figure;
subplot(2,1,1);
plot(w, 20*log10(abs(Htrue)), 'k-', 'LineWidth', 1.3); hold on;
plot(w, 20*log10(abs(Hhat)),  'r--', 'LineWidth', 1.3);
grid on; ylabel('Magnitude [dB]');
title('Bode: True ARX vs Estimated ARX');
legend('True G(z)', 'Estimated \hat{G}(z)', 'Location','Best');

subplot(2,1,2);
plot(w, unwrap(angle(Htrue))*180/pi, 'k-', 'LineWidth', 1.3); hold on;
plot(w, unwrap(angle(Hhat))*180/pi,  'r--', 'LineWidth', 1.3);
grid on; xlabel('\omega [rad/sample]'); ylabel('Phase [deg]');

nA = max(numel(Atrue), numel(Ahat));
nB = max(numel(Btrue), numel(Bhat));
Atrue_pad = [Atrue, zeros(1, nA - numel(Atrue))];
Ahat_pad  = [Ahat,  zeros(1, nA - numel(Ahat))];
Btrue_pad = [Btrue, zeros(1, nB - numel(Btrue))];
Bhat_pad  = [Bhat,  zeros(1, nB - numel(Bhat))];

fprintf('||Atrue - Ahat||_inf = %.3g\n', max(abs(Atrue_pad - Ahat_pad)));
fprintf('||Btrue - Bhat||_inf = %.3g\n', max(abs(Btrue_pad - Bhat_pad)));
disp('A polynomials (padded):'); disp([Atrue_pad; Ahat_pad]);
disp('B polynomials (padded):'); disp([Btrue_pad; Bhat_pad]);

