Cf = 60000;
Cr = 60000;
lf = 1.22;
lr = 1.62;
Iz = 2920;
m = 1590;
L = 5;

% Sampling period
Ts = 1e-2;

v_x = 20 * rand() + 15;
R = 100 * (14 * rand() + 1);
phi_dot_des = v_x / R;

A = [0, 1, 0, 0;
     0, -(Cf+Cr)/(m*v_x), (Cf + Cr)/m, -(Cf*lf - Cr*lr)/(m*v_x);
     0, 0, 0, 1;
     0, -(Cf*lf - Cr*lr)/(Iz*v_x), (Cf*lf - Cr*lr)/Iz, -(Cf*lf^2 + Cr*lr^2)/(Iz*v_x)];
B = [0;
     Cf/m;
     0;
     Cf*lf/Iz];
B_dist = [0;
          -(Cf*lf - Cr*lr)/(m*v_x) - v_x
          0;
          -(Cf*lf.^2 + Cr*lr.^2) / (Iz*v_x)];
B_overall = [B, B_dist];
C = [1, 0, 0, 0;
     0, 0, 1, 0];

errorModel = ss(A, B_overall, C, 0);
errorModel = setmpcsignals(errorModel, MV=1, MD=2);
curvModel = ss(zeros(4),B_dist, zeros(1,4), 0);

numSteps = 1000;
maxSteer = pi / 6;
maxSteerRate = pi / 3 * Ts;

% Lateral error for which to fail the episode
latErrorThreshold = 1.5;
% Heading error for which to fail the episode
headingErrorThreshold = pi / 2;

manipVars = struct('Min', -maxSteer, ...
                   'Max', maxSteer, ...
                   'RateMin', -maxSteerRate, ...
                   'RateMax', maxSteerRate, ...
                   'Name', 'Steering Angle', ...
                   'Units', 'Radians', ...
                   'ScaleFactor', 2 * maxSteer);
weights = struct('ManipulatedVariables', 0.1, ...
                 'ManipulatedVariablesRate', 10, ...
                 'OutputVariables', [20, 0.05], ...
                 'ECR', 0.0001);
outVars = [struct('Min', -latErrorThreshold/10, ...
                  'Max', latErrorThreshold/10, ...
                  'Name', 'Lateral Error', ...
                  'Units', 'meters', ...
                  'ScaleFactor', 2 * latErrorThreshold), ...
           struct('Min', -headingErrorThreshold/10, ...
                  'Max', headingErrorThreshold/10, ...
                  'Name', 'Heading Error', ...
                  'Units', 'radians', ...
                  'ScaleFactor', 2 * headingErrorThreshold)];
distVars = struct('Name', 'Road Curvature Disturbance');

mpcObj = mpc(errorModel, Ts, 100, 100, weights, manipVars, outVars, distVars)
% setindist(mpcObj, 'model', curvModel);

controllerState = mpcstate(mpcObj);
outputReference = [0, 0];

e_y0 = 1.5 * rand() - 0.75;
% e_y0 = 0.2;
de_y0 = rand() - 0.5;
e_theta0 = pi / 6 * rand() - pi / 12;
% e_theta0 = -0.1;
de_theta0 = pi / 10 * rand() - pi / 20;

initialState = [e_y0; de_y0; e_theta0; de_theta0];

t = 0:Ts:numSteps * Ts;
N = length(t);
output = zeros(N, 2);
input = zeros(N, 1);

simOpts = mpcsimopt;
simOpts.PlantInitialState = initialState;
sim(mpcObj, numSteps, outputReference, phi_dot_des, simOpts);

% for i = 1:N
%     output(i, :) = C * controllerState.Plant;
%     input(i, :) = mpcmove(mpcObj, controllerState, output(i, :), outputReference);
% end
