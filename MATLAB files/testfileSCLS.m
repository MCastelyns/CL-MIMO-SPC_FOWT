%% SCLS (square-root covariance RLS) vs plain RLS -> ARX Bode (no FIR)
clear; clc; close all; rng(4);

%% --- True plant (discrete-time, Ts=1) ---
A = 0.85; B = 1.00; C = 1.00; D = 0.00;
Atrue = [1, -A];
Btrue = [D, B*C];     % [0, 1]

%% --- Closed-loop excitation (your controller) ---
p_cl = 0.5;
Kx   = place(A, B, p_cl);
Acl  = A - B*Kx;
Gr   = 1 / ( C * ((eye(1) - Acl) \ B) + D );

T  = 1400;
r  = zeros(1,T); r(200:end) = 1;
noise_std = 0.02;

x=0; u=zeros(1,T); y=zeros(1,T);
for k=1:T-1
    y(:,k) = C*x + D*u(:,k);
    u(:,k) = -Kx*x + Gr*r(:,k) + noise_std*randn(1,1);
    x = A*x + B*u(:,k);
end
y(:,T) = C*x + D*u(:,T);


p     = 10;               
lam   = 0.995;            
m     = 2*p + 1;          
S0    = eye(m);           
sigma0= 1.0;


Theta_rls = zeros(1,m); S_rls = S0; sigma_rls = sigma0;
for k=p:(T-1)
    idx = (k-p+1):k;
    z_p = reshape([u(:,idx); y(:,idx)], [], 1);
    phi = [z_p; u(:,k+1)];
    y_next = y(:,k+1);
    [Theta_rls, S_rls, sigma_rls] = RLS_cov(Theta_rls, S_rls, phi, y_next, lam, sigma_rls);
end
[Ahat_rls, Bhat_rls] = theta_to_arx(Theta_rls, p);  


Theta_scls = zeros(1,m); S_scls = S0; sigma_scls = sigma0;
for k=p:(T-1)
    idx = (k-p+1):k;
    z_p = reshape([u(:,idx); y(:,idx)], [], 1);
    phi = [z_p; u(:,k+1)];
    y_next = y(:,k+1);
    [Theta_scls, S_scls, sigma_scls] = SCRLS(Theta_scls, S_scls, phi, y_next, lam, sigma_scls);
end
[Ahat, Bhat] = theta_to_arx(Theta_scls, p);         



fprintf('||Theta_scls - Theta_rls||_inf = %.3g\n', max(abs(Theta_scls - Theta_rls)));
fprintf('||Ahat(SCLS)-Ahat(RLS)||_inf  = %.3g\n', max(abs(pad(Ahat, numel(Ahat_rls)) - pad(Ahat_rls, numel(Ahat)))));
fprintf('||Bhat(SCLS)-Bhat(RLS)||_inf  = %.3g\n', max(abs(pad(Bhat, numel(Bhat_rls)) - pad(Bhat_rls, numel(Bhat)))));


npts = 2048;
[Htrue,w] = freqz(Btrue, Atrue, npts);
Hhat       = freqz(Bhat,  Ahat,  npts);

figure;
subplot(2,1,1);
plot(w, 20*log10(abs(Htrue)),'k-','LineWidth',1.3); hold on;
plot(w, 20*log10(abs(Hhat)), 'r--','LineWidth',1.3);
grid on; ylabel('Magnitude [dB]');
title('Bode: True ARX vs Estimated ARX (SCLS)');
legend('True G(z)','Estimated \hat{G}(z)','Location','Best');

subplot(2,1,2);
plot(w, unwrap(angle(Htrue))*180/pi,'k-','LineWidth',1.3); hold on;
plot(w, unwrap(angle(Hhat))*180/pi, 'r--','LineWidth',1.3);
grid on; xlabel('\omega [rad/sample]'); ylabel('Phase [deg]');


function [Theta,S,sigma] = RLS_cov(Theta,S,phi,y,lambda,sigma)
% Standard covariance-form RLS
    K = (S*phi) / (lambda + phi' * S * phi);
    e = y - Theta * phi;
    Theta = Theta + e * K';
    S = (S - K * phi' * S) / lambda;
    sigma = lambda * sigma + (1-lambda) * (e' * e); 
end


function [Ahat, Bhat] = theta_to_arx(Theta, p)
    cut   = 2*p;

    CK = Theta(:,1:cut); beta0 = Theta(:,cut+1);

    Kmat  = reshape(CK, 2, p);     % row1=u-weights, row2=y-weights
    beta  = fliplr(Kmat(1,:));     % multiplies u(k-1), u(k-2), ...
    alpha = fliplr(Kmat(2,:));     % multiplies y(k-1), y(k-2), ...
    Ahat  = [1, -alpha];
    Bhat  = [beta0, beta];
end

function y = pad(x, n)
    if numel(x) >= n, y = x(1:n);
    else, y = [x, zeros(1, n-numel(x))];
    end
end
