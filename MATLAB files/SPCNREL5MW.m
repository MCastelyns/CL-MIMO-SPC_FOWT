% Use linearized model of NREL 5 MW reference turbine at 16 m/s to obtain
% data for SPC and run SPC
load("linearization_nrel5mw.mat", "sys");
[A,B,C,D] = ssdata(sys);

% Define I/O dimensions
Nu = 3;     % [Beta, Tau, Wind speed]
Ny = 2;     % [GeneratorSpeed, PlatformPitch]


% Define simulation and past data window variables
dt = 0.1;       % Time step [s]
p = 200;        % Past data window
t_past = dt*p;  % Past simulated time [s]

% Define noise to achieve persistently exciting inputs
Tau_stddev = 0.05;  
Beta_stddev = 0.05; 
WindSpeed_stddev = 0.01;

% If debug is true we show plots of the simulated trajectory to check if we stay within reasonable bounds for the linearization
debug = true;

% First we simulate the system to get some data
[y, u, x] = simulateNREL5MW(p, t_past, Tau_stddev, Beta_stddev, WindSpeed_stddev, debug);

% Define SPC simulation
T = 205;           % Total simulated timesteps
t_final = dt * T;   % Total simulated time


t = linspace(0, t_final, T+1)';


% Define SPC variables
lambda = 0.995;
Np = 50;
m   = p*(Nu+Ny) + Nu;



% Define initial guess to kickstart recursive SPC (replace with better
% guess)
Theta = zeros(Ny, m);      
S     = eye(m);            
sigma = 1.0;

% Was working with transposes in previous version. I do this so I dont have
% to rewrite functions
u = u';
y = y';

I = eye(Ny*Np);


U_store = zeros(Nu, T);    
Y_store = zeros(Ny, T);   

xk = x(end,:)';

% Loop from end of past window to the end of the simulation
for k = p:T-1
    idx = (k-p+1):k;
    z_p = reshape([u(:,idx); y(:,idx)], [], 1);

    phi = [z_p; u(:,k)];         
    y_next = y(:,k);

    [Theta, S, sigma] = SCRLS(Theta, S, phi, y_next, lambda, sigma);

   
    cut     = p*(Nu+Ny);
    CKappa  = Theta(:, 1:cut);            
    Dhat    = Theta(:, cut + (1:Nu));     


    [Gamma_tilde, H_tilde, G_tilde] = build_prediction_matrices(CKappa, Dhat, p, Np, Nu, Ny);
    Gamma = (I - G_tilde)\Gamma_tilde;
    H     = (I - G_tilde)\H_tilde;

    
    % Set up quadprog problem
    Nc = 10;
    
    % Weights
    Qy  = diag([1;1]);
    Ru  = diag([0.1; 0.1; 0.1]);
    Rdu = diag([0.1; 0.1; 0]);
    
    Qy_tilde  = kron(eye(Np), Qy);          
    Ru_tilde  = kron(eye(Np), Ru);          
    Rdu_tilde = kron(eye(Np), Rdu);        
    
    % Reference (for now 0, but can be implemented later)
    r = zeros(Ny, T);
    r_tilde = repmat(r(:,k+1), Np, 1);     
    
    umin = [-50; -50; -100];
    umax = [50; 50; 100];

    dumin = [-10; -10; -20];
    dumax = [10; 10; 20];

    [Hqp,fqp,Aeq,beq,Ain,bin,lb,ub] = build_spc_qp( ...
        Gamma, H, z_p, r_tilde, Qy_tilde, Ru_tilde, Rdu_tilde, ...
        Np, Nc, Nu, umin, umax, dumin, dumax);
    
    % Fix Hqp to be symmetric, to prevent MATLAB throwing warnings
    Hqp = (Hqp + Hqp.')/2;

    
    opts = optimoptions('quadprog','Display','off');
    u_tilde = quadprog(Hqp, fqp, Ain, bin, Aeq, beq, lb, ub, [], opts);
    
    u_apply = u_tilde(1:Nu);   

    % Instead we add noise on the disturbance input
    %u_apply(3) = 0.1*randn;


    xk = A*xk + B*u_apply;        
    yk = C*xk + D*u_apply;

    u(:,k+1) = u_apply;
    y(:,k+1) = yk;

    U_store(:,k) = u_apply;
    Y_store(:,k) = yk;
    
    k
end

figure;
subplot(3,1,1);
plot(t(1:T), u(1,:), 'b', 'LineWidth', 1.2); hold on;
plot(t(1:T), u(2,:), 'r', 'LineWidth', 1.2);
ylabel('Control Inputs');
legend('\beta (pitch)','\tau (torque)');
grid on;

subplot(3,1,2);
plot(t(1:T), u(3,:), 'k', 'LineWidth', 1.2);
ylabel('Disturbance (wind noise)');
grid on;

subplot(3,1,3);
plot(t(1:T), y(1,:), 'b', 'LineWidth', 1.2); hold on;
plot(t(1:T), y(2,:), 'r', 'LineWidth', 1.2);
ylabel('Outputs');
xlabel('Time [s]');
legend('\omega_g (gen speed)','\theta_p (platform pitch)');
grid on;