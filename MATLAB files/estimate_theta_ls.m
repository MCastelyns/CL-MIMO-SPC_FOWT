function [CKappa, D] = estimate_theta_ls(Yf, Zp, Uf, p, Nu, Ny)
% estimate_theta_ls  Estimates the predictor Markov parameters [C Kappa, D] using LS
%
%   [CKappa, D] = estimate_theta_ls(Yf, Zp, Uf, p, Nu, Ny) returns the estimated
%   predictor Markov parameter matrix [CKappa, D], where:
%
%     CKappa - [C A^{p-1}[B, K], ..., C[B, K]] (Ny x p*(Nu + Ny))
%     D - (Ny x Nu)
%
%   Inputs:
%     Yf - future output matrix (Ny*Nf x N)
%     Zp - stacked past vector matrix (p*(Nu+Ny) x N)
%     Uf - future input matrix (Nu*Nf x N)
%     p  - past window size
%     Nu - number of inputs
%     Ny - number of outputs
%
%   Output:
%     CKappa - (Ny x p*(Nu+Ny))
%     D      - (Ny x Nu)

    Phi = [Zp; Uf];  

    % Solve the least squares using psuedoinverse
    Theta = Yf * pinv(Phi);  

    % Extract [CKappa, D] from Theta
    CKappa = Theta(:, 1 : p * (Nu + Ny));       
    D = Theta(:, p * (Nu + Ny) + 1 : end);      
end
