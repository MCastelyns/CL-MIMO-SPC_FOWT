clear; clc;

rng(13);

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

%% For now, (GS)PI attempt as this seems to be a common benchmark controller
% LQR is also possible, perhaps easier to get a simple implementation

G = tf(sys);

% We want transfer function from blade pitch to omega_g
TF = G(1,1);


% Set bandwidth and PM
wb   = 0.2;       
PM   = 50;        

% PI on blade pitch for now. GSPI is better though as a benchmark
% Standard form

s = tf('s');
Cpi = @(Kp, Ti) Kp*(Ti*s+1)/(Ti*s);


[mag,phase] = bode(TF,wb);
phase = phase*pi/180; 

phi = phase;

phi_needed = -(pi - (PM*pi/180) + phi);

Ti = 1/tan(-phi_needed)/wb;   

[magC,~] = bode((Ti*s+1)/(Ti*s),wb);
magC = squeeze(magC);
Kp = 1/(mag*magC);


C = Cpi(Kp, Ti);

fprintf('Tuned PI gains: Kp = %.4f, Ti = %.4f\n', Kp, Ti);

L = C*TF;
figure;
margin(L); grid on;
title('Loop transfer function with tuned PI');



