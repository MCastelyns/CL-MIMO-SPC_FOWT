clear; clc; close all;

% Main simulation settings
Ts = 0.01;                  
p  = 40;                    % past window
Tbatch = 400;               % batch window
Tadapt = 4000;              % recursive window
stddev_u = 0.05;            

% I/O channels to plot
inputnr  = 1;              
outputnr = 1;               

% Define simple system
A= [-0.80  1.00  0.00  0.00;
    -2.00 -0.60  0.30  0.00;
     0.00  0.00 -0.90  1.00;
     0.10  0.00 -2.50 -0.70];

B = [1.0 0.5;
     0.3 0.8;
     0.2 1.0;
     0.6 0.4];

C = [1 0 0 0;
     0 0 1 0];

D = zeros(2,2);

[Ny, Nu] = size(D);

% Discretize
sysc = ss(A,B,C,D);
sysd = c2d(sysc, Ts);
[Ad,Bd,Cd,Dd] = ssdata(sysd);

% True system model
[num, den] = ss2tf(Ad,Bd,Cd,Dd, inputnr);  % rows = outputs, cols = num coeffs
Btrue = num(outputnr,:)./den(1);
Atrue = den./den(1);

% Simulate system
T = Tbatch + Tadapt;

u = stddev_u * randn(T, Nu);
x = zeros(size(Ad,1),1);
y = zeros(T, Ny);
for k = 1:T
    y(k,:) = (Cd*x + Dd*u(k,:).').';
    x      = Ad*x + Bd*u(k,:).';
end

% Batch LS problem
rowsPhi = p*(Nu+Ny) + Nu;   
K       = Tbatch - p;       % Length of recursive estimation part

Phi = zeros(rowsPhi, K);
Y   = zeros(Ny, K);

kk = 0;
for k = (p+1):Tbatch
    kk = kk + 1;

    blocks = cell(2*p+1, 1);
    bi = 0;
    % Stacking the input and output vertically for each lag (z_p)
    for l = 1:p
        bi = bi + 1; 
        blocks{bi} = u(k-l,:).'; 
        bi = bi + 1; 
        blocks{bi} = y(k-l,:).'; 
    end
    blocks{end} = u(k,:).';      % Last block is u(k)

    Phi(:,kk) = vertcat(blocks{:}); 
    Y(:,kk)   = y(k,:).';
end

% Batch Least Squares
rho   = 1e-6 * trace(Phi*Phi')/size(Phi,1);
Theta = (Y * Phi') / (Phi*Phi' + rho*eye(rowsPhi));   

[Ahat_batch, Bhat_batch, ARX_batch] = Theta2ARX(Theta, inputnr, outputnr, p, Ts, Nu, Ny);
G_batch = Theta2ARX_MIMO(Theta, p, Ts, Nu, Ny);

% Set initial values for SCRLS 
% (Reasonable guesses starting from batch prediction)
P0    = inv(Phi*Phi' + rho*eye(rowsPhi));
S     = chol(P0,'lower');          % square-root covariance
e_ls  = Y - Theta*Phi;
sigma = sqrt(mean(e_ls(:).^2) + eps);
lambda = 0.995;

% Recursive least squares loop
for k = (Tbatch+1):T
    blocks = cell(2*p+1, 1);    
    bi = 0;                     % Block index
    for l = 1:p
        bi = bi + 1; 
        blocks{bi} = u(k-l,:).';
        bi = bi + 1; 
        blocks{bi} = y(k-l,:).';
    end
    blocks{end} = u(k,:).';
    phi = vertcat(blocks{:});      % Stack all the vectors stored in the blocks vertically to build phi

    [Theta,S,sigma] = SCRLS(Theta, S, phi, y(k,:), lambda, sigma);
end

[Ahat_SCRLS, Bhat_SCRLS, ARX_SCRLS] = Theta2ARX(Theta, inputnr, outputnr, p, Ts, Nu, Ny);
G_SCRLS = Theta2ARX_MIMO(Theta, p, Ts, Nu, Ny);

% Bode plot to compare ARX transfer functions
% Phase wrapping on or off. Can make checking phase plots easier in some
% cases, (its mod 360 degrees with this on)
opts = bodeoptions;
opts.PhaseWrapping      = 'off';         


figure('Color','w','Name','Bode: True vs Batch vs RLS');
bode(G_batch, 'b--', opts)
hold on;
bode(G_SCRLS, 'r-.', opts)
bode(sysd, 'k', opts)
grid on;
legend('Batch ARX','SCRLS ARX','True','Location','best');
hold off;

% ARX coefficient comparison (not sure if this is actually even relevant,
% even when they are very different the bode plots match quite nicely)
nA = max([numel(Atrue), numel(Ahat_batch), numel(Ahat_SCRLS)]);
nB = max([numel(Btrue), numel(Bhat_batch), numel(Bhat_SCRLS)]);
pad = @(v,n) [v, zeros(1, n-numel(v))];     % Function to make plotting different length vectors possible (padding with zeros)

Atrue_p  = pad(Atrue,      nA);
Abatch_p = pad(Ahat_batch, nA);
Ascrls_p = pad(Ahat_SCRLS, nA);

Btrue_p  = pad(Btrue,      nB);
Bbatch_p = pad(Bhat_batch, nB);
Bscrls_p = pad(Bhat_SCRLS, nB);

figure('Color','w','Name','ARX coefficients');
tiledlayout(2,1);

nexttile;
stem(0:nA-1, Atrue_p,'k','filled'); hold on;
stem(0:nA-1, Abatch_p,'b--','filled');
stem(0:nA-1, Ascrls_p,'r-.','filled');
grid on; xlabel('lag index'); ylabel('A coeff');
title('Denominator A(q)'); legend('True','Batch','SCRLS','Location','best');

nexttile;
stem(0:nB-1, Btrue_p,'k','filled'); hold on;
stem(0:nB-1, Bbatch_p,'b--','filled');
stem(0:nB-1, Bscrls_p,'r-.','filled');
grid on; xlabel('lag index'); ylabel('B coeff');
title(sprintf('Numerator B(q) for u_%d \\rightarrow y_%d',inputnr,outputnr));



