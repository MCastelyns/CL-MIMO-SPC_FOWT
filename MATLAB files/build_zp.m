function z_p = build_zp(u_hist, y_hist, p)
% build_zp  Stack the last p samples into z_k^(p).
% u_hist: Nu x T (latest sample at column T, T has to be larger than p)
% y_hist: Ny x T
% z_p   : (p*(Nu+Ny)) x 1  = [u_{T-p+1}; y_{T-p+1}; ... ; u_T; y_T]

    T = size(u_hist, 2);
    if T < p
        error('Not enough past samples: got %d, need at least %d.', T, p);
    end

    index = (T - p + 1) : T;          % last p columns
    U   = u_hist(:, index);           % Nu x p
    Y   = y_hist(:, index);           % Ny x p
    z_p = reshape([U; Y], [], 1);     % (Nu+Ny)*p x 1
end
