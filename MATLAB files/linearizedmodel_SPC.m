clear; clc;

%rng(13); % Can add this to make it reproducable

% Import linearization of NREL 5MW (still water, 16 m/s wind speed)
load("linearization_nrel5mw.mat");

% Inputs: 
% - Collective blade pitch [rad]
% - Generator torque [Nm]
% - Horizontal wind speed (disturbance?) [m/s]

% Outputs: 
% - Generator speed [rpm]
% - Platform pitch angle [deg]

% Don't really care about state definitions. We only work with input/output
% 37 states

% Define baseline controller, for data collection and testing of
% identification

% %% For now, (GS)PI attempt as this seems to be a common benchmark controller
% % LQR is also possible, perhaps easier to get a simple implementation
% 
% G = tf(sys);
% 
% % We want transfer function from blade pitch to omega_g
% TF = G(1,1);
% 
% 
% % Set bandwidth and PM
% wb   = 0.2;       
% PM   = 50;        
% 
% % PI on blade pitch for now. GSPI is better though as a benchmark
% % Standard form
% 
% s = tf('s');
% Cpi = @(Kp, Ti) Kp*(Ti*s+1)/(Ti*s);
% 
% 
% [mag,phase] = bode(TF,wb);
% phase = phase*pi/180; 
% 
% phi = phase;
% 
% phi_needed = -(pi - (PM*pi/180) + phi);
% 
% Ti = 1/tan(-phi_needed)/wb;   
% 
% [magC,~] = bode((Ti*s+1)/(Ti*s),wb);
% magC = squeeze(magC);
% Kp = 1/(mag*magC);
% 
% 
% C = Cpi(Kp, Ti);
% 
% fprintf('Tuned PI gains: Kp = %.4f, Ti = %.4f\n', Kp, Ti);
% 
% L = C*TF;
% figure;
% margin(L); grid on;
% title('Loop transfer function with tuned PI');



%% INSTEAD USE OPEN LOOP TO GATHER DATA
% We use a white noise input on both input channels, with low magnitude
T = 500; % Timesteps
Tau = 0.01*randn(T,1);  % Generator torque
Beta = 0.01*randn(T,1); % Blade pitch

SS = sys; % I like working with SS instead of sys

% Simulate the system response to the generated inputs
time = linspace(0,10,T)'; % Time vector
u = [Beta, Tau, zeros(T,1)]; % Input matrix

% Simulate the system response using the state-space model
[y, t, x] = lsim(SS, u, time);

% Analyze the system response

% We are in above rated, the rated generator speed is 1173.7 rpm
% Since we use a linearization around 16 m/s (above rated). We should add
% this value to the value we obtain in the linearization model to obtain
% the true value
RatedSpeed = 1173.7;

GeneratorSpeed = y(:,1) + 1173.7;
PlatformPitch = y(:,2);

figure;
plot(t, GeneratorSpeed);
% Plot a horizontal line for the Rated Speed
hold on; 
yline(RatedSpeed, 'r--'); 
xlabel('Time (s)');
ylabel('Rotational Speed [rpm]');
grid on;
legend(["Generator Speed", "Rated Speed"])
hold off;

figure;
plot(t, PlatformPitch);
xlabel('Time (s)');
ylabel('Platform Pitch Angle [deg]');
grid on;

% Get metrics on deviation from rated speed
% Calculate the deviation from the rated speed
deviation = GeneratorSpeed - RatedSpeed;

% Calculate RMSE for generator speed
rmse = sqrt(mean(deviation.^2));
fprintf('Root Mean Square Error (RMSE) from rated speed: %.4f rpm\n', rmse);

% Calculate RMSE for platform pitch
rmsePitch = sqrt(mean((PlatformPitch).^2));
fprintf('Root Mean Square Error (RMSE) for platform pitch: %.4f deg\n', rmsePitch);