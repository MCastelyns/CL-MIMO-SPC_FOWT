function [y, u, x] = simulateNREL5MW(steps, t, Tau_stddev, Beta_stddev, WindSpeed_stddev, debug)
% SIMULATENREL5MW Simulates the Linearized SS model of the NREL 5MW
% reference wind turbine at 16 m/s with a noise on the input channels
%   Inputs:
%       steps       - Number of simulation steps
%       t           - Total simulation time [s]
%       Tau_stddev  - Standard deviation for generator torque noise [Nm]
%       Beta_stddev - Standard deviation for blade pitch noise [rad]
%       debug       - Boolean to enable/disable debugging output (default is false)
%
%   Outputs:
%       y               - Output    [GeneratorSpeed, PlatformPitch]
%       u               - Input     [Beta, Tau, Wind speed]
%       x               - State     [37 states]     (needed to continue simulating from current state when running)

    % rng(13); % Can add this to make it reproducible
    stepsp1 = steps+1;      % Needed a lot when simulating a steps amount, we actually need a vector thats 1 longer (include beginning state)
    % Import linearization of NREL 5MW (still water, 16 m/s wind speed)
    load("linearization_nrel5mw.mat", "sys");
    
    % We use a white noise input on both input channels, with low magnitude
    Tau = Tau_stddev * randn(stepsp1, 1);  % Generator torque
    Beta = Beta_stddev * randn(stepsp1, 1); 
    WindSpeed = WindSpeed_stddev * randn(stepsp1, 1); 

    SS = sys; % I like working with SS instead of sys

    % Simulate the system response to the generated inputs
    time = linspace(0, t, stepsp1)'; 
    u = [Beta, Tau, WindSpeed]; 

    % Simulate the system response using the state-space model
    [y, ~, x] = lsim(SS, u, time);

    % We are in above rated, the rated generator speed is 1173.7 rpm
    RatedSpeed = 1173.7;

    GeneratorSpeed = y(:, 1) + RatedSpeed;
    PlatformPitch = y(:, 2);

    % Set default value for debug if not provided
    if nargin < 5
        debug = false;
    end

    % Plot results, mainly for checking if we stay within reasonable bounds
    % of the linearization point
    if debug
        figure;
        plot(time, GeneratorSpeed);
        % Plot a horizontal line for the Rated Speed
        hold on; 
        yline(RatedSpeed, 'r--'); 
        xlabel('Time (s)');
        ylabel('Rotational Speed [rpm]');
        grid on;
        legend(["Generator Speed", "Rated Speed"])
        hold off;

        figure;
        plot(time, PlatformPitch);
        hold on;
        yline(0, 'r--');
        xlabel('Time (s)');
        ylabel('Platform Pitch Angle [deg]');
        grid on;
    end

    % We should add values to other results too, since we are using a
    % linearization model.

    % To keep rated power at 16 m/s, blade pitch should be 12.06 deg 
    % (table 7.1 NREL 5MW ref turbine paper)
    % BladePitch = Beta + deg2rad(12.06); % [rad]
end