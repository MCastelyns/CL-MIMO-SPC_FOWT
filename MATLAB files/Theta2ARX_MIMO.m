function G = Theta2ARX_MIMO(Theta, p, Ts, Nu, Ny)
% Build full Ny x Nu transfer matrix of ARX models from Theta
    Num = repmat({[]}, Ny, Nu);
    Den = repmat({[]}, Ny, Nu);
    for iy = 1:Ny
        for iu = 1:Nu
            [Ahat, Bhat, ~] = Theta2ARX(Theta, iu, iy, p, Ts, Nu, Ny);
            Num{iy,iu} = Bhat;   
            Den{iy,iu} = Ahat;
        end
    end
    G = tf(Num, Den, Ts);
end