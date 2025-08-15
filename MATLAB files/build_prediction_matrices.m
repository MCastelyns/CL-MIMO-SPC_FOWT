function [Gamma_tilde, H_tilde, G_tilde] = build_prediction_matrices(CKappa, D, p, Np, Nu, Ny)
% build_prediction_matrices
% Constructs the predictor matrices GammaTilde, HTilde, GTilde from [CKappa, D].

    % Extract C A^i Btilde and C A^i K from CKappa and store them for
    % easier use
    CAB_terms = cell(1, p);   % should be (Ny x Nu) but this way it auto creates right size 
    CAK_terms = cell(1, p);   % should be (Ny x Ny) but this way it auto creates right size
    for j = 1:p
        i = p - j;  % exponent for this CKappa block
        cols  = (j-1)*(Nu+Ny) + (1:(Nu+Ny));
        CABK  = CKappa(:, cols);          % Ny x (Nu+Ny)
        CAB_terms{i+1} = CABK(:, 1:Nu);   % C A^i BTilde
        CAK_terms{i+1} = CABK(:, Nu+1:end); % C A^i K
    end

    Gamma_tilde = zeros(Ny*Np, (Nu+Ny)*p);
    H_tilde     = zeros(Ny*Np, Nu*Np);
    G_tilde     = zeros(Ny*Np, Ny*Np);

    % Build GammaTilde
    for i = 1:Np        % Block row
        r = (i-1)*Ny + (1:Ny); % row indexes in GammaTilde corresponding to block
        for j = 1:p     % Block column
            if i <= j   % Nonzero ones, if i is larger than j it is zero by definition
                e = p + i - j - 1;                  % exponent for A
                c = (j-1)*(Nu+Ny) + (1:(Nu+Ny));    % Column indices
                Gamma_tilde(r, c) = [CAB_terms{e+1}, CAK_terms{e+1}];
            end
        end
    end

    % Build HTilde and GTilde 
    % Diagonal of HTilde is D; GTilde diagonal is zero.
    for i = 1:Np
        r = (i-1)*Ny + (1:Ny);

        % diagonal D on H_tilde (is 0 in paper? did they just set D=0?)
        cH_diag = (i-1)*Nu + (1:Nu);    %Column indices of diagonal block columns for H
        H_tilde(r, cH_diag) = D;

        % blocks below diagonal
        for j = 1:i-1
            e = i - j - 1;                  % exponent for A
            if e <= p-1
                cH = (j-1)*Nu + (1:Nu);     % Column indices in H
                cG = (j-1)*Ny + (1:Ny);     % Column indices in G
                H_tilde(r, cH) = CAB_terms{e+1};   % C A^{e} BTilde
                G_tilde(r, cG) = CAK_terms{e+1};   % C A^{e} K
            end
        end
    end
end
