%% NON RECURSIVE FORMULATION FIRST. LATER LOOP RECURSIVE ESTIMATION OF THETA USING NEW DATA %%

% Define data sizes
p = 25;         % Past Window Dimension (Should be larger than expected system order but not too high for recursive formulation)
Np = 50;        % Prediction Horizon (Chosen such that prediction interval contains at least one period of each of the modes to be damped)
Nu = 2;         % Input Numbers 
Ny = 2;         % Output Numbers


% Build Yf, Zp, Uf from past data
[Yf, Zp, Uf] = build_data_matrices(u, y, p);

% Solve least squares problem and extract [CKappa, D]
[Ckappa, D] = estimate_theta_ls(Yf,Zp,Uf,p,Nu,Ny);


% Build prediction matrices using estimated markov parameters
[Gamma_tilde, H_tilde, G_tilde] = build_prediction_matrices(Ckappa, D, p, Np, Nu, Ny);


% Reformulate to open loop predictor
I = eye(Ny*Np); 
Gamma = (I - G_tilde)\Gamma_tilde; 
H = (I - G_tilde)\H_tilde; 

% Define stacked vectors and use subspace predictor to predict outputs up
% to time step k + Np