% MIMO simple SS system test
clear; clc; close all;
Ts = 0.01;

% Define continuous time true system
A= [-0.80  1.00  0.00  0.00;
       -2.00 -0.60  0.30  0.00;
        0.00  0.00 -0.90  1.00;
        0.10  0.00 -2.50 -0.70];
B = [ 1.0  0.5;
        0.3  0.8;
        0.2  1.0;
        0.6  0.4];
C = [1 0 0 0;
       0 0 1 0];
D = zeros(2,2);


[Ny, Nu] = size(D);

% Discretize true system
sysc = ss(A,B,C,D);
sysd = c2d(sysc,Ts);

[Ad, Bd, Cd, Dd] = ssdata(sysd);

% Extract true system ARX coefficients
inputnr = 1;
outputnr = 1;
[num, den] = ss2tf(Ad,Bd,Cd,Dd,inputnr);     
num = num(outputnr,:);

num = num/den(1);  den = den/den(1);
Btrue = num;                        
Atrue = den;                        


% Define simulation sizes
Tbatch = 400;      
Tadapt = 4000;     
T      = Tbatch + Tadapt;

stddev_u = 0.05;
u = stddev_u * randn(T,Nu);                     
x = zeros(size(A,1),1);
y = zeros(T,Ny);
for k = 1:T
    %idx_y = ((k-1)*Ny)+(1:Ny);
    %idx_u = ((k-1)*Nu)+(1:Nu);
    y(k,:) = (Cd*x + Dd*u(k,:)')';
    x    = Ad*x + Bd*u(k,:)';
end

% Batch LS on first Tbatch samples
p = 40;                             
K = Tbatch - p;

Phi = zeros(Nu + p*(Nu + Ny), K);   
Y   = zeros(Ny, K);

kk = 0;
for k = (p+1):Tbatch
    kk = kk + 1;

    blocks = cell(2*p + 1, 1);
    for i = 1:p
        blocks{2*i-1} = u(k-i,:).';   
        blocks{2*i}   = y(k-i,:).';   
    end
    blocks{2*p+1} = u(k,:).';         
    Phi(:,kk) = vertcat(blocks{:});

    Y(:,kk) = y(k,:).';
end

% normalization to improve batch estimation performance
rho   = 1e-6 * trace(Phi*Phi')/size(Phi,1);
Theta = (Y * Phi') / (Phi*Phi' + rho*eye(size(Phi,1)));   

% Translate to ARX
m = Nu; r = Ny;
uk_cols = p*(m + r) + (1:m);     % last m columns
getU = @(i) (i-1)*(m + r) + (1:m);           % i = 1..p  (uk-i)
getY = @(i) (i-1)*(m + r) + m + (1:r);       % i = 1..p  (yk-i)

b0    = Theta(:, uk_cols);                   % Ny x Nu  (D)
beta  = Theta(:, cell2mat(arrayfun(@(i) getU(i), 1:p, 'UniformOutput', false)));
alpha = Theta(:, cell2mat(arrayfun(@(i) getY(i), 1:p, 'UniformOutput', false)));
beta = beta(outputnr,:);        % Ny x ...
b0 = b0(outputnr,inputnr);      % Ny x Nu (D)
alpha = alpha(inputnr,:);
Ahat_batch = [1, -alpha];
Bhat_batch = [b0, beta];

% Get initial guesses for recursive LS
P0    = inv(Phi*Phi' + rho*eye(size(Phi,1)));
S     = chol(P0,'lower');                          
e_ls  = Y - Theta*Phi;
sigma = sqrt(mean(e_ls(:).^2));
sigma = 1;



lambda = 0.995;                                    

for k = (Tbatch+1):T
    blocks = cell(2*p + 1, 1);
    for i = 1:p
        blocks{2*i-1} = u(k-i,:).';
        blocks{2*i}   = y(k-i,:).';
    end
    blocks{2*p+1} = u(k,:).';
    phi = vertcat(blocks{:});   

    [Theta,S,sigma] = SCRLS(Theta, S, phi, y(k,:), lambda, sigma);
end



m = Nu; r = Ny;
uk_cols = p*(m + r) + (1:m);     % last m columns
getU = @(i) (i-1)*(m + r) + (1:m);           % i = 1..p  (uk-i)
getY = @(i) (i-1)*(m + r) + m + (1:r);       % i = 1..p  (yk-i)

b0    = Theta(:, uk_cols);                   % Ny x Nu  (D)
beta  = Theta(:, cell2mat(arrayfun(@(i) getU(i), 1:p, 'UniformOutput', false)));
alpha = Theta(:, cell2mat(arrayfun(@(i) getY(i), 1:p, 'UniformOutput', false)));
beta = beta(outputnr,:);        % Ny x ...
b0 = b0(outputnr,inputnr);      % Ny x Nu (D)
alpha = alpha(inputnr,:);
Ahat_scrls = [1, -alpha];
Bhat_scrls = [b0, beta];

% Compare bode plots of true and estimated systems

bode(tf(Bhat_batch, Ahat_batch, Ts), 'b--');
hold on;
bode(tf(Bhat_scrls, Ahat_scrls, Ts), 'r-.');
bode(sysd, 'k')

nA = max([numel(Atrue), numel(Ahat_batch), numel(Ahat_scrls)]);
nB = max([numel(Btrue), numel(Bhat_batch), numel(Bhat_scrls)]);

pad = @(pvec,n) [pvec, zeros(1, n-numel(pvec))];

Atrue_p  = pad(Atrue,     nA);
Abatch_p = pad(Ahat_batch,nA);
Ascrls_p  = pad(Ahat_scrls, nA);

Btrue_p  = pad(Btrue,     nB);
Bbatch_p = pad(Bhat_batch,nB);
Bscrls_p  = pad(Bhat_scrls, nB);

figure('Color','w','Name','ARX coefficients');
tiledlayout(2,1);

nexttile; 
stem(0:nA-1, Atrue_p,'k','filled'); hold on;
stem(0:nA-1, Abatch_p,'b--','filled');
stem(0:nA-1, Ascrls_p, 'r-.','filled');
grid on; xlabel('lag (coeff index)'); ylabel('A coeff');
title('Denominator A(q) coefficients');
legend('True','Batch','SCRLS','Location','best');

nexttile; 
stem(0:nB-1, Btrue_p,'k','filled'); hold on;
stem(0:nB-1, Bbatch_p,'b--','filled');
stem(0:nB-1, Bscrls_p, 'r-.','filled');
grid on; xlabel('lag (coeff index)'); ylabel('B coeff');
title('Numerator B(q) coefficients');

fprintf('\nSup-norm coefficient errors (padded):\n');
fprintf('  ||Atrue - Abatch||_inf = %.3g\n', max(abs(Atrue_p - Abatch_p)));
fprintf('  ||Atrue - Ascls ||_inf = %.3g\n', max(abs(Atrue_p - Ascrls_p)));
fprintf('  ||Btrue - Bbatch||_inf = %.3g\n', max(abs(Btrue_p - Bbatch_p)));
fprintf('  ||Btrue - Bscls ||_inf = %.3g\n', max(abs(Btrue_p - Bscrls_p)));


