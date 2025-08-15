function [Yf, Zp, Uf] = build_data_matrices(u, y, p)
% build_data_matrices - Constructs data matrices Yf, Zp, and Uf for LS estimation
%
% Inputs:
%   u - input data matrix of size (Nu x N_total)
%   y - output data matrix of size (Ny x N_total)
%   p - past window size
%
% Outputs:
%   Yf - (Ny x N) matrix of future outputs starting at time k+p
%   Zp - (p*(Nu+Ny) x N) matrix of stacked past input/output vectors
%   Uf - (Nu x N) matrix of future inputs

    % Dimensions
    [Nu, N_total_u] = size(u);
    [Ny, N_total_y] = size(y);
    assert(N_total_u == N_total_y, 'Input and output must have same length');

    % Number of samples with past window
    N = N_total_u - p;


    Yf = zeros(Ny, N);
    Zp = zeros((Nu + Ny) * p, N);
    Uf = zeros(Nu, N);

    for i = 1:N
        k = i - 1;  % zero-based indexing, used to python... (this is really ugly, i should work out how to fix this)

        % y_{k+p}
        Yf(:, i) = y(:, k + p + 1);

        % z_k^(p) = [z_k; z_{k+1}; ...; z_{k+p-1}]
        zp = [];
        for j = 0:(p-1)
            zj = [u(:, k + j + 1); y(:, k + j + 1)];  % z_{k+j} = [u_{k+j}; y_{k+j}]
            zp = [zp; zj];
        end
        Zp(:, i) = zp;

        % u_{k+p}
        Uf(:, i) = u(:, k + p + 1);
    end
end
