function [Ahat, Bhat, ARX] = Theta2ARX(Theta, inputnr, outputnr, p, Ts, Nu, Ny)
%Theta2ARX Translates estimated Theta into an ARX model
%   Detailed explanation goes here

anum = zeros(1,p);
bnum = zeros(1,p);
for l = 1:p
    anum(l) = Theta(outputnr, ((l-1)*(Nu+Ny) + Nu + outputnr));   
    bnum(l) = Theta(outputnr, ((l-1)*(Nu+Ny) + inputnr));     
end

b0 = Theta(outputnr, (p*(Nu+Ny) + inputnr));

Ahat = [1, -anum];
Bhat = [b0, bnum];

ARX = tf(Bhat, Ahat, Ts);
end