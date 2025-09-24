clear; clc; close all; 


k = 1;
m = 2;
b = 3;

Ts = 0.1;
A  = [0 1; -k/m -b/m];
B  = [0; 1/m];
C  = [1 0];
D  = [0];

% Get the ground truth ARX from the SS description
[num, den] = ss2tf(A,B,C,D,1);     % SISO: input #1
num = num./den(1);  den = den./den(1);
Btrue = num;                        % B(q)
Atrue = den;                        % A(q)


Tbatch = 400;      
Tadapt = 4000;     
T      = Tbatch + Tadapt;

Nu = size(B,2);
Ny = size(D,1);

u = randn(T,1);                     
x = zeros(size(A,1),1);
y = zeros(T,1);
for k = 1:T
    %idx_y = ((k-1)*Ny)+(1:Ny);
    %idx_u = ((k-1)*Nu)+(1:Nu);
    y(k) = C*x + D*u(k);
    x    = A*x + B*u(k);
end

% Batch LS on first Tbatch samples
p = 40;                             
K = Tbatch - p;

Phi = zeros(1 + p + p, K);          % [u(k); u(k-1..k-p); y(k-1..k-p)]
Y   = zeros(1, K);
kk = 0;
for k = (p+1):Tbatch
    kk = kk + 1;
    Phi(:,kk) = [ u(k); ...
                  u(k-1:-1:k-p); ...
                  y(k-1:-1:k-p)  ];
    Y(kk) = y(k); 
end

% normalization to improve batch estimation performance
rho   = 1e-6 * trace(Phi*Phi')/size(Phi,1);
Theta = (Y * Phi') / (Phi*Phi' + rho*eye(size(Phi,1)));   

% Translate batch predictions into ARX model
b0    = Theta(1);
beta  = Theta(1+(1:p));
alpha = Theta(1+p+(1:p));
Ahat_batch = [1, -alpha];
Bhat_batch = [b0, beta];

% Get initial guesses for recursive LS
P0    = inv(Phi*Phi' + rho*eye(size(Phi,1)));
S     = chol(P0,'lower');                          
e_ls  = Y - Theta*Phi;                              
sigma = sqrt(mean(e_ls.^2));                        


lambda = 0.995;                                    

for k = (Tbatch+1):T
    zU = u(k-1:-1:k-p);
    zY = y(k-1:-1:k-p);
    phi = [ u(k); zU; zY ];                         
    [Theta,S,sigma] = SCRLS(Theta, S, phi, y(k), lambda, sigma);
    %[Theta,S,sigma] = scls_step(Theta, S, phi, y(k), lambda, sigma);
end


b0    = Theta(1);
beta  = Theta(1+(1:p));
alpha = Theta(1+p+(1:p));
Ahat_scls = [1, -alpha];
Bhat_scls = [b0, beta];


npts = 2048;
[Htrue,w]  = freqz(Btrue,     Atrue,     npts);
Hbatch     = freqz(Bhat_batch,Ahat_batch,npts);
Hscls      = freqz(Bhat_scls, Ahat_scls, npts);

figure('Color','w','Name','Bode comparison');
subplot(2,1,1);
plot(w, 20*log10(abs(Htrue)), 'k-', 'LineWidth', 1.6); hold on;
plot(w, 20*log10(abs(Hbatch)),'b--','LineWidth', 1.2);
plot(w, 20*log10(abs(Hscls)), 'r-.','LineWidth', 1.2);
grid on; ylabel('Magnitude [dB]');
title(sprintf('SISO 2nd-order, Ts=%.2g s, p=%d, Ttrain=%d, Tadapt=%d',Ts,p,Tbatch,Tadapt));
legend('True','Batch ARX','SCRLS','Location','best');

subplot(2,1,2);
plot(w, unwrap(angle(Htrue))*180/pi,  'k-', 'LineWidth', 1.6); hold on;
plot(w, unwrap(angle(Hbatch))*180/pi, 'b--','LineWidth', 1.2);
plot(w, unwrap(angle(Hscls))*180/pi,  'r-.','LineWidth', 1.2);
grid on; xlabel('\omega [rad/sample]'); ylabel('Phase [deg]');

nA = max([numel(Atrue), numel(Ahat_batch), numel(Ahat_scls)]);
nB = max([numel(Btrue), numel(Bhat_batch), numel(Bhat_scls)]);

pad = @(pvec,n) [pvec, zeros(1, n-numel(pvec))];

Atrue_p  = pad(Atrue,     nA);
Abatch_p = pad(Ahat_batch,nA);
Ascls_p  = pad(Ahat_scls, nA);

Btrue_p  = pad(Btrue,     nB);
Bbatch_p = pad(Bhat_batch,nB);
Bscls_p  = pad(Bhat_scls, nB);

figure('Color','w','Name','ARX coefficients');
tiledlayout(2,1);

nexttile; 
stem(0:nA-1, Atrue_p,'k','filled'); hold on;
stem(0:nA-1, Abatch_p,'b--','filled');
stem(0:nA-1, Ascls_p, 'r-.','filled');
grid on; xlabel('lag (coeff index)'); ylabel('A coeff');
title('Denominator A(q) coefficients');
legend('True','Batch','SCRLS','Location','best');

nexttile; 
stem(0:nB-1, Btrue_p,'k','filled'); hold on;
stem(0:nB-1, Bbatch_p,'b--','filled');
stem(0:nB-1, Bscls_p, 'r-.','filled');
grid on; xlabel('lag (coeff index)'); ylabel('B coeff');
title('Numerator B(q) coefficients');

fprintf('\nSup-norm coefficient errors (padded):\n');
fprintf('  ||Atrue - Abatch||_inf = %.3g\n', max(abs(Atrue_p - Abatch_p)));
fprintf('  ||Atrue - Ascls ||_inf = %.3g\n', max(abs(Atrue_p - Ascls_p)));
fprintf('  ||Btrue - Bbatch||_inf = %.3g\n', max(abs(Btrue_p - Bbatch_p)));
fprintf('  ||Btrue - Bscls ||_inf = %.3g\n', max(abs(Btrue_p - Bscls_p)));


