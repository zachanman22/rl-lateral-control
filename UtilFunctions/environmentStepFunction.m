function [NextObservation,Reward,IsDone,NextState] = environmentStepFunction(Action,State)
%ENVIRONMENTSTEPFUNCTION Summary of this function goes here
%   Detailed explanation goes here

% Set to 0 for base reward and 1 for modified reward
modifiedReward = 1;

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
maxSteerRate = pi / 3 * Ts;

% Lateral error for which to fail the episode
latErrorThreshold = 1.5;
% Heading error for which to fail the episode
headingErrorThreshold = pi / 2;

NextState = zeros(7,1);

% Maintain same longitudinal speed
NextState(6) = v_x;
% Constant radius
NextState(7) = State(7);

prevAction = State(5);

% Rate limiting the action
if Action - prevAction > maxSteerRate
    updatedAction = prevAction + maxSteerRate;
elseif Action - prevAction < -maxSteerRate
    updatedAction = prevAction - maxSteerRate;
else
    updatedAction = Action;
end

% True action after rate limiting
NextState(5) = updatedAction;

NextState(1:4) = State(1:4) + Ts *  (A * State(1:4) + B * updatedAction + B_dist * phi_dot_des);

% Full visible/observable state assumption
NextObservation = NextState;

nextLatError = NextObservation(1);
nextHeadingError = NextObservation(3);

IsDone = abs(nextLatError) > latErrorThreshold || ...
    abs(nextHeadingError) > headingErrorThreshold;

if IsDone
    Reward = -1000;
else
    % If within 1/10 of the lateral threshold, provide reward as fn of
    % error
    if abs(nextLatError) <= latErrorThreshold / 10
        % Care more about lateral error than heading error, so factor of 3
        lateralErrorReward = 3 * (1 - abs(nextLatError) / latErrorThreshold) .^ 2;
    else
        lateralErrorReward = 0;
    end
    
    % If within 1/10 of the heading threshold, provide reward as fn of
    % error
    if abs(nextHeadingError) <= headingErrorThreshold / 10
        headingErrorReward = 1 * (1 - abs(nextHeadingError) / headingErrorThreshold).^2;
    else
        headingErrorReward = 0;
    end
    
    % Penalize sinusoidal lateral error by penalizing the lateral error
    % crossing the zero line
    if modifiedReward
        if NextState(1) > 0 && State(1) < 0
        overshootReward = -100;
        elseif NextState(1) < 0 && State(1) > 0
            overshootReward = -100;
        else
            overshootReward = 0;
        end
    else
        overshootReward = 0;
    end
    
    
    % Penalize the lateral error increasing, want it to constantly decrease
    % smoothly
    if modifiedReward
        if abs(NextState(1)) > abs(State(1))
            smoothReward = -10;
        else
            smoothReward = 1;
        end
    else
        smoothReward = 0;
    end
    
    
    % Want to perform task with minimal action, so penalize larger actions
    actionReward = -(Action / maxSteer) ^ 2;

    % Reward = (lateralErrorReward + headingErrorReward) * Ts;
    Reward = lateralErrorReward + headingErrorReward + actionReward + overshootReward + smoothReward;
end

end

