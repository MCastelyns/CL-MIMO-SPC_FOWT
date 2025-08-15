function [y_pred_vec, y_pred_mat] = spc_predict(Gamma, H, z_p, u_future, Nu, Ny, Np)
% spc_predict  Predict future outputs using SPC predictor matrices.
%
% Inputs:
%   Gamma     : (Ny*Np) x (p*(Nu+Ny))
%   H         : (Ny*Np) x (Nu*Np)
%   z_p       : (p*(Nu+Ny)) x 1
%   u_future  : (Nu*Np) x 1  if we give [] it assumes zero inputs
%   Nu, Ny    : scalars
%   Np        : scalar
%
% Outputs:
%   y_pred_vec : (Ny*Np) x 1
%   y_pred_mat : Ny x Np

    if isempty(u_future)
        u_vec = zeros(Nu*Np, 1);
    else
        u_vec = u_future(:);  
    end

    y_pred_vec = Gamma * z_p + H * u_vec;
    y_pred_mat = reshape(y_pred_vec, Ny, Np);
end
