function [Hqp,fqp,Aeq,beq,Ain,bin,lb,ub] = build_spc_qp(Gamma,H,z_p,r_tilde,...
        Qy_tilde,Ru_tilde,Rdu_tilde, Np,Nc, m, umin,umax, dumin,dumax)

Im = eye(m);
SDelta = kron(tril(ones(Np)), Im) - kron(tril(ones(Np),-1), Im); 


S0 = zeros(m*Np, size(z_p,1));   

v0 = zeros(m*Np,1);

% Cost matrices
%size(H')
%size(Qy_tilde)

Hqp = H.'*Qy_tilde*H + Ru_tilde + SDelta.'*Rdu_tilde*SDelta;
fqp = H.'*Qy_tilde*(Gamma*z_p - r_tilde) + SDelta.'*Rdu_tilde*(S0*z_p + v0);


dumin_h = repmat(dumin, Np, 1);   
dumax_h = repmat(dumax, Np, 1);   

Ain = [ SDelta; -SDelta ];
bin = [ dumax_h - (S0*z_p + v0);  -(dumin_h - (S0*z_p + v0)) ];

% Force u to be constant after Nc
E = kron([zeros(Nc,1); ones(Np-Nc,1)], Im);                
Aeq = E' * SDelta;                                         
beq = - E' * (S0*z_p + v0);

% Disturbance trajectory (for now fixed at 0, might want to implement
% forecasted wind speeds here later)
d_traj = zeros(Np,1);

% Equality constraints: enforce disturbance input = d_traj
Aeq_dist = kron(eye(Np), [0 0 1]);   
beq_dist = d_traj;

Aeq = [Aeq; Aeq_dist];
beq = [beq; beq_dist];


lb  = repmat(umin, Np, 1);
ub  = repmat(umax, Np, 1);
end
