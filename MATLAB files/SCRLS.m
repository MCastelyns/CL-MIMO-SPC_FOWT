function [Theta, S, sigma, G] = SCRLS(Theta, S, phi, y, lambda, sigma)
% SCRLS Computes a time update of the Markov parameters 
%
%   Using the Square-root Covariance RLS algorithm as described in
%   "Round-off Error Propagation in Four Generally-Applicable, Recursive,
%   Least-Squares Estimation Schemes" by Verhaegen et al.

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
    
    %size(G)
    %size(r')
    Theta = Theta + (G' / sigma) * (r.');       
end