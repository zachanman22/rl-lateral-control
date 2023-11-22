function [NextObservation,Reward,IsDone,NextState] = environmentStepFunction(Action,State)
%ENVIRONMENTSTEPFUNCTION Summary of this function goes here
%   Detailed explanation goes here
Cf = 60000;
Cr = 60000;
lf = 1.22;
lr = 1.62;
Iz = 2920;
m = 1590;
L = 5;

% Sampling period
Ts = 1e-2;

v_x = State(6);
phi_dot_des = v_x / State(7);

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
maxSteer = pi / 6;
maxSteerRate = pi / 100;

% if ~(Action <= maxSteer && Action >= -maxSteer)
%     error('Action must be between %g and %g.',...
%         -maxSteer,maxSteer);
% end

% Won't consider max steer rate

% Lateral error for which to fail the episode
latErrorThreshold = 1.5;
% Heading error for which to fail the episode
headingErrorThreshold = pi / 2;

NextState = zeros(7,1);

% Compute integral of lateral error
% NextState(5) = State(5) + State(1);

% Compute integral of heading error
% NextState(6) = State(6) + State(2);

NextState(5) = Action;

% Maintain same longitudinal speed
NextState(6) = v_x;
% Constant radius
NextState(7) = State(7);

prevAction = State(5);

if Action - prevAction > maxSteerRate
    Action = prevAction + maxSteerRate;
elseif Action - prevAction < -maxSteerRate
    Action = prevAction - maxSteerRate;
end

NextState(1:4) = State(1:4) + Ts *  (A * State(1:4) + B * Action + B_dist * phi_dot_des);

% Full visible/observable state assumption
NextObservation = NextState;

nextLatError = NextObservation(1);
nextHeadingError = NextObservation(3);

IsDone = abs(nextLatError) > latErrorThreshold || ...
    abs(nextHeadingError) > headingErrorThreshold; % || ...
    % abs(Action - prevAction) > maxSteerRate;

% IsDone = abs(nextLatError) > latErrorThreshold;
% IsDone = false;

if IsDone
    Reward = -100;
else
    if abs(nextLatError) <= latErrorThreshold / 10
        lateralErrorReward = 3 * (1 - abs(nextLatError) / latErrorThreshold) .^ 2;
    else
        lateralErrorReward = 0;
    end

    if abs(nextHeadingError) <= headingErrorThreshold / 10
        headingErrorReward = 1 * (1 - abs(nextHeadingError) / headingErrorThreshold).^2;
    else
        headingErrorReward = 0;
    end
    
    if abs(Action - prevAction) > maxSteerRate
        maxSteerReward = -100;
    else 
        maxSteerReward = 0;
    end

    if NextState(1) > 0 && State(1) < 0
        overshootReward = -10;
    elseif NextState(1) < 0 && State(1) > 0
        overshootReward = -10;
    else
        overshootReward = 1;
    end

    if abs(NextState(1)) > abs(State(1))
        smoothReward = -10;
    else
        smoothReward = 1;
    end

    % Reward = (lateralErrorReward + headingErrorReward) * Ts;
    Reward = (lateralErrorReward + headingErrorReward - (Action / maxSteer) ^ 2) + overshootReward + smoothReward + maxSteerReward;
end

% if IsDone
%     Reward = -100;
% else
%     if nextLatError ^ 2 < 0.001
%         lateralErrorReward = 2;
%     else
%         lateralErrorReward = 0;
%     end
% 
%     if nextHeadingError ^ 2 < 0.001
%         headingErrorReward = 1;
%     else
%         headingErrorReward = 0;
%     end
%     Reward = (0.1 + lateralErrorReward + headingErrorReward);
% end

end

