plotTitle = "Agent Performance Example";
plotDSteer = 0;

Cf = 60000;
Cr = 60000;
lf = 1.22;
lr = 1.62;
Iz = 2920;
m = 1590;
L = 5;

% Sampling period
Ts = 1e-2;

R0 = 1000;

% m/s
v_x = 25;
phi_dot_des = v_x / R0;

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


% e_y0 = 0.5 * rand() - 0.25;
e_y0 = 0.2;
% de_y0 = rand() - 0.5;
de_y0 = 0.5;
% e_theta0 = pi / 6 * rand() - pi / 12;
e_theta0 = -0.1;
% de_theta0 = pi / 10 * rand() - pi / 20;
de_theta0 = -pi / 20;

u_prev = 0;

numSteps = 500;
t = 1:numSteps;

states = zeros(4, numSteps);
actions = zeros(1, numSteps);
actionsCom = zeros(1, numSteps);

%% Create Environment
obsInfo = rlNumericSpec([7,1]);
obsInfo.Name = "Bicycle Model States";
obsInfo.Description = 'e_y, de_y, e_phi, de_phi, u_prev, v_x, R';

maxSteer = pi / 6;

actInfo = rlNumericSpec(1);
actInfo.Name = "Steering Action";
actInfo.Description = 'delta';
actInfo.LowerLimit = -maxSteer;
actInfo.UpperLimit = maxSteer;

addpath("../UtilFunctions")
env = rlFunctionEnv(obsInfo, actInfo, "environmentStepFunction", "simEnvironmentResetFunction");

simOptions = rlSimulationOptions(MaxSteps=numSteps);
experience = sim(env,saved_agent,simOptions);

states(:, 1) = experience.Observation.BicycleModelStates.Data(1:4,:,1); 

agentPerfLength = size(experience.Observation.BicycleModelStates.Data, 3) - 1;

for i = t
    % action = experience.Action.SteeringAction.Data(:,:,i);
    % actions(1, i) = action;
    if i < agentPerfLength
        states(:, i+1) = experience.Observation.BicycleModelStates.Data(1:4,:,i+1);
        actions(1, i) = experience.Observation.BicycleModelStates.Data(5,:,i+1);
        actionsCom(1, i) = experience.Action.SteeringAction.Data(:,:,i);
        % states(:, i+1) = states(:, i) + Ts *  (A * states(:, i) + B * action + B_dist * phi_dot_des);
    end
end

if plotDSteer
    subplot(4,1,1)
    plot(t(1:agentPerfLength), states(1, 1:agentPerfLength))
    title(plotTitle)
    ylabel("Lateral Error (m)")
    subplot(4,1,2)
    plot(t(1:agentPerfLength), states(3, 1:agentPerfLength))
    ylabel("Heading Error (rad)")
    subplot(4,1,3)
    plot(t(1:agentPerfLength), actions(1:agentPerfLength))
    ylabel("Steering Angle (rad)")
    subplot(4,1,4)
    plot(t(1:agentPerfLength), [0, diff(actions(1:agentPerfLength))])
    ylabel("dSteering Angle (rad/s)")
    xlabel("Time Step")
else
    subplot(3,1,1)
    plot(t(1:agentPerfLength), states(1, 1:agentPerfLength))
    title(plotTitle)
    ylabel("Lateral Error (m)")
    subplot(3,1,2)
    plot(t(1:agentPerfLength), states(3, 1:agentPerfLength))
    ylabel("Heading Error (rad)")
    subplot(3,1,3)
    plot(t(1:agentPerfLength), actions(1:agentPerfLength))
    ylabel("Steering Angle (rad)")
    xlabel("Time Step")
end