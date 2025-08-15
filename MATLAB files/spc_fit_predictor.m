function pred = spc_fit_predictor(u, y, p, Np, Nu, Ny)
% spc_fit_predictor  Batch-fit SPC predictor from data and build (Gamma, H).
% Automatically calls the right functions in the correct order to set up
% the LS problem and solve it and extract the matrices
%
% Inputs:
%   u  : Nu x T
%   y  : Ny x T
%   p  : past window size
%   Np : prediction horizon
%   Nu : #inputs
%   Ny : #outputs
%
% Output:
%   pred : struct with fields
%          .CKappa, .D
%          .Gamma_tilde, .H_tilde, .G_tilde
%          .Gamma, .H
%          .p, .Np, .Nu, .Ny

    % stack data
    [Yf, Zp, Uf] = build_data_matrices(u, y, p);

    % LS estimate of predictor Markov parameters
    [CKappa, D] = estimate_theta_ls(Yf, Zp, Uf, p, Nu, Ny);

    % build predictor matrices
    [Gamma_tilde, H_tilde, G_tilde] = build_prediction_matrices(CKappa, D, p, Np, Nu, Ny);

    % rewrite to predictor form
    I = eye(Ny * Np);
    Gamma = (I - G_tilde) \ Gamma_tilde;
    H     = (I - G_tilde) \ H_tilde;

    % pack results (so we have access to intermediate results too for later
    % computations/tests)
    pred = struct( ...
        'CKappa',       CKappa, ...
        'D',            D, ...
        'Gamma_tilde',  Gamma_tilde, ...
        'H_tilde',      H_tilde, ...
        'G_tilde',      G_tilde, ...
        'Gamma',        Gamma, ...
        'H',            H, ...
        'p',            p, ...
        'Np',           Np, ...
        'Nu',           Nu, ...
        'Ny',           Ny );
end
