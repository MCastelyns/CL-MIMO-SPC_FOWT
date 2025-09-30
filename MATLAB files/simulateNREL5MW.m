function [y, u, x] = simulateNREL5MW(steps, t, Tau_stddev, Beta_stddev, WindSpeed_stddev, debug, dt)
% SIMULATENREL5MW  Discrete-time simulation of the linearized NREL 5MW SS model
%   [y,u,x] = simulateNREL5MW(steps, t, Tau_stddev, Beta_stddev, WindSpeed_stddev, debug, dt)
%   - steps:       number of simulation steps (the function returns steps+1 samples incl. k=0)
%   - t:           total simulation time [s]
%   - *_stddev:    standard deviations for input white noise (Tau, Beta, WindSpeed)
%   - debug:       true/false for quick plots (optional; default = false)
%   - dt:          sample time [s] (optional; default = t/steps)
%
%   Outputs:
%     y  : (steps+1) x Ny  outputs   [GeneratorSpeed(deviation), PlatformPitch]
%     u  : (steps+1) x Nu  inputs    [Beta, Tau, WindSpeed]
%     x  : (steps+1) x Nx  states (x(1,:) is x_0)

    if nargin < 6 || isempty(debug), debug = false; end
    if nargin < 7 || isempty(dt),    dt    = t/steps; end

    % Load continuous-time linearization
    load("linearization_nrel5mw.mat", "sys");

    % Discretize to the requested sample time
    if sys.Ts == 0
        sysd = c2d(sys, dt, 'zoh');
    else
        % If already discrete, convert to this dt just in case
        sysd = d2d(sys, dt, 'zoh');
    end
    [Ad,Bd,Cd,Dd] = ssdata(sysd);

    % Sizes
    Nx = size(Ad,1);
    Nu = size(Bd,2);
    Ny = size(Cd,1);

    % Generate inputs (steps+1 samples: k = 0..steps)
    stepsp1   = steps + 1;
    Tau       = Tau_stddev       * randn(stepsp1, 1);
    Beta      = Beta_stddev      * randn(stepsp1, 1);
    WindSpeed = WindSpeed_stddev * randn(stepsp1, 1);

    % Arrange inputs as [Beta, Tau, WindSpeed] to match your convention
    u = [Beta, Tau, WindSpeed];           % (steps+1) x Nu

    % Allocate and simulate the discrete-time SS model
    x = zeros(stepsp1, Nx);               % store x_k (before update) each step
    y = zeros(stepsp1, Ny);
    xk = zeros(Nx,1);                     % initial state x_0

    for k = 1:stepsp1
        uk      = u(k,:).';               % Nu x 1
        y(k,:)  = (Cd*xk + Dd*uk).';      % output at step k-1 (time k-1)*dt
        x(k,:)  = xk.';                   % store current state
        xk      = Ad*xk + Bd*uk;          % state update
    end

    % Optional quick-look plots (relative deviations)
    if debug
        time = linspace(0, t, stepsp1).';
        RatedSpeed = 1173.7;  % rpm, for reference

        figure('Name','Generator speed (deviation)'); 
        plot(time, y(:,1)); grid on;
        ylabel('\omega_g deviation [rpm]');
        xlabel('Time [s]'); hold on; yline(0,'r--'); hold off;

        figure('Name','Platform pitch angle (deviation)');
        plot(time, y(:,2)); grid on;
        ylabel('\theta_p deviation [deg]');
        xlabel('Time [s]'); hold on; yline(0,'r--'); hold off;

        % If you want absolute generator speed for context:
        figure('Name','Generator speed (absolute)');
        plot(time, y(:,1) + RatedSpeed); grid on;
        ylabel('\omega_g [rpm]');
        xlabel('Time [s]'); hold on; yline(RatedSpeed,'r--'); hold off;
    end
end
