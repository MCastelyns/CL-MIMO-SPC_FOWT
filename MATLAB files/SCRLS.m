function [Theta, S, sigma, G] = SCRLS(Theta, S, phi, y, lambda, sigma)
% SCRLS Computes a time update of the Markov parameters 
%
%   Using the Square-root Covariance RLS algorithm as described in
%   "Round-off Error Propagation in Four Generally-Applicable, Recursive,
%   Least-Squares Estimation Schemes" by Verhaegen et al.
%   Inputs:
%       Theta   - Previous Markov Parameters
%       S       - Previous Cholesky factor of Covariance
%       phi     - New data vector 
%       y       - New output measurement
%       lambda  - Forgetting factor
%       sigma   - Previous estimate of standard deviation
%
%   Outputs:
%       Theta   - Updated Markov Parameters
%       S       - Updated Cholesky factor of Covariance
%       sigma   - Updated estimate of standard deviation
%       G       - Matrix used in computation (not sure if we even need to return this atm)



    % Build the pre-matrix as described in the paper

    Pre = [sqrt(lambda)*sigma , phi'*S;
           zeros(size(S,1),1) , S/sqrt(lambda)];
    
    % Do a QR decomposition on the transpose of Pre. 
    [~, R] = qr(Pre.', "econ");     % Use econ version for possible faster computation
    Post  = R';                     % We want lower triangular, not upper triangular. So we transpose it


    % Extract the values from the blocks in Post
    sigma = Post(1, 1); 
    G     = Post(2:end, 1);
    S     = Post(2:end, 2:end);   

    % Parameter update
    r = y - (Theta * phi);                   % Ny x 1 residual
    
    %size(G')
    %size(r')
    %size(Theta)
    Theta = Theta + (sqrt(lambda) / sigma)* r * G';

    % Fix sigma to 1, this is the only way it works at the moment. I see
    % that the piezoelectric paper also isn't working with an updated sigma
    %sigma = 1;
end