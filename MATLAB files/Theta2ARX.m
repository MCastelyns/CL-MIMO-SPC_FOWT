function [Ahat, Bhat, delay] = Theta2ARX(Theta, iu, iy, p, Ts, Nu, Ny)
% Map one (iy,iu) element from stacked Theta -> discrete SISO ARX polynomials.
% Assumes Theta = [CKappa , Dhat] and CKappa packs blocks as [u(l); y(l)], l=1..p.
% Strictly proper: leading zero in Bhat; monic A with MINUS on y-lags.

    % split
    CK = Theta(:, 1:p*(Nu+Ny));         % Ny x (p*(Nu+Ny))
    % gather the p input and output lag coefficients for this row
    bu = zeros(1,p);     % Kyu(iy,iu,l)
    ay = zeros(1,p);     % self-output Kyy(iy,iy,l)
    for l = 1:p
        off = (l-1)*(Nu+Ny);
        Kyu_l = CK(:, off + (1:Nu));          % Ny x Nu
        Kyy_l = CK(:, off + Nu + (1:Ny));     % Ny x Ny
        bu(l) = Kyu_l(iy, iu);                % row iy, col iu
        ay(l) = Kyy_l(iy, iy);                % ONLY self-output lag
    end

    % Discrete tf expects coefficients in descending powers of z^-1 starting at z^0.
    % Strictly proper -> one-sample delay: leading zero in numerator.
    Bhat = [0, bu];                 % [b0=0, b1, ..., bp]
    Ahat = [1, -ay];                % [1, -a1, -a2, ..., -ap]

    delay = 0;                      % explicit delay already encoded via leading zero
    % Note: You return Ahat,Bhat to your Theta2ARX_MIMO which does tf(Num,Den,Ts).
end
