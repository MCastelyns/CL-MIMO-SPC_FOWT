clear; clc; close all;

% Simulation settings
dt       = 0.1;    
p        = 100;              
Tbatch   = 4000;             
Tadapt   = 12000;
lambda   = 0.995;

% excitation std per input [beta(rad); tau(Nm); wind(m/s)]
std_u = [0.03; 3e4; 0.3];

% True system
load linearization_nrel5mw.mat sys
sysd = c2d(sys, dt);         
[Ad,Bd,Cd,Dd] = ssdata(sysd);
[Ny, Nu] = size(Dd);


T = Tbatch + Tadapt;


u = zeros(Nu,T);
u(1,:) = std_u(1) * randn(1,T);     
u(2,:) = std_u(2) * randn(1,T);     
u(3,:) = std_u(3) * randn(1,T);     



x = zeros(size(Ad,1),1);
y = zeros(Ny, T);
for k = 1:T
    y(:,k) = Cd*x + Dd*u(:,k);
    x      = Ad*x + Bd*u(:,k);
end

% Batch LS
m     = p*(Nu + Ny) + Nu;   
K     = Tbatch - p;          
Phi   = zeros(m,  K);
Y     = zeros(Ny, K);

kk = 0;
for k = (p+1):Tbatch
    kk = kk + 1;
    blocks = cell(2*p+1,1);
    bi = 0;
    for l = 1:p
        bi = bi+1; blocks{bi} = u(:,k-l);
        bi = bi+1; blocks{bi} = y(:,k-l);
    end
    blocks{end} = u(:,k);
    Phi(:,kk) = vertcat(blocks{:});
    Y(:,kk)   = y(:,k);
end

% Scaling
s = std(Phi,0,2);           
s(s==0) = 1;                % avoid divide-by-zero
Phis = Phi ./ s;            % scaled regressor

% Batch LS
rho    = 1e-6 * trace(Phis*Phis')/size(Phis,1);
ThetaS = (Y * Phis') / (Phis*Phis' + rho*eye(m));    % Ny x m, scaled
Theta  = ThetaS ./ s.';                              % unscale → Ny x m

% Convert to transfer matrix
G_batch = Theta2ARX_MIMO(Theta, p, dt, Nu, Ny);
G_batch.InputName  = sysd.InputName;  
G_batch.OutputName = sysd.OutputName;

% SCRLS
P0  = inv(Phis*Phis' + rho*eye(m));
S   = chol(P0,'lower');                      
eLS = Y - ThetaS*Phis;                       % residual in scaled space
sigma = sqrt(mean(eLS(:).^2) + eps);

for k = (Tbatch+1):T
    blocks = cell(2*p+1,1);
    bi = 0;
    for l = 1:p
        bi = bi+1; blocks{bi} = u(:,k-l);
        bi = bi+1; blocks{bi} = y(:,k-l);
    end
    blocks{end} = u(:,k);
    phi  = vertcat(blocks{:});     % m x 1 (unscaled)
    phis = phi ./ s;               
    [ThetaS,S,sigma] = SCRLS(ThetaS, S, phis, y(:,k), lambda, sigma);
end

Theta  = ThetaS ./ s.';            % unscale once at the end
G_scrls = Theta2ARX_MIMO(Theta, p, dt, Nu, Ny);
G_scrls.InputName  = sysd.InputName;  
G_scrls.OutputName = sysd.OutputName;


opt = bodeoptions; opt.Grid='on'; opt.MagUnits='dB';
opt.PhaseWrapping='on'; opt.PhaseWrappingBranch=360;

% Focus on interesting freq range
wmin = 1e-3; wmax = 1000;     % rad/s

figure('Color','w');
bodeplot(G_batch, {wmin,wmax}, 'b--', opt); hold on;
bodeplot(G_scrls, {wmin,wmax}, 'r-.',  opt);
bodeplot(sysd,    {wmin,wmax}, 'k',    opt);
legend('Batch ARX','SCRLS ARX','True','Location','best');


