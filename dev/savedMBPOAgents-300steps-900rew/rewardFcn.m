function Reward = rewardFcn(State, Action, NextState)
if iscell(State)
    State = State{1};
end

if iscell(Action)
    Action = Action{1};
end

if iscell(NextState)
    NextState = NextState{1};
end

maxSteer = pi / 6;

% Lateral error for which to fail the episode
latErrorThreshold = 1.5;
% Heading error for which to fail the episode
headingErrorThreshold = pi / 2;

% Full visible/observable state assumption
NextObservation = NextState;

Action = Action(1, :);

nextLatError = NextObservation(1, :);
nextHeadingError = NextObservation(3, :);

IsDone = isDone(State, Action, NextState);

% IsDone = abs(nextLatError) > latErrorThreshold;
% IsDone = false;

% If within 1/10 of the lateral threshold, provide reward as fn of
% error
lateralErrorMask = abs(nextLatError) <= latErrorThreshold / 10;
% Care more about lateral error than heading error, so factor of 3
lateralErrorReward = 3 * (1 - abs(nextLatError) / latErrorThreshold) .^ 2;
lateralErrorReward = lateralErrorReward .* lateralErrorMask;

% If within 1/10 of the heading threshold, provide reward as fn of
% error
headingErrorMask = abs(nextHeadingError) <= headingErrorThreshold / 10;
headingErrorReward = 1 * (1 - abs(nextHeadingError) / headingErrorThreshold).^2;
headingErrorReward = headingErrorReward .* headingErrorMask;

% % If difference between the requested action and the prev action has a
% % magnitude greater than the max steering rate, penalize
% if abs(Action - prevAction) > maxSteerRate
%     maxSteerReward = -100;
% else 
%     maxSteerReward = 0;
% end
maxSteerReward = 0;

% Penalize sinusoidal lateral error by penalizing the lateral error
% crossing the zero line
% if NextState(1) > 0 && State(1) < 0
%     overshootReward = -100;
% elseif NextState(1) < 0 && State(1) > 0
%     overshootReward = -100;
% else
%     overshootReward = 0;
% end

overshootReward = 0;

% Penalize the lateral error increasing, want it to constantly decrease
% smoothly
% if abs(NextState(1)) > abs(State(1))
%     smoothReward = -10;
% else
%     smoothReward = 1;
% end

smoothReward = 0;

% Want to perform task with minimal action, so penalize larger actions
actionReward = -(Action / maxSteer) .^ 2;

Reward = zeros(size(IsDone));

Reward(logical(IsDone)) = -1000;
Reward(~logical(IsDone)) = lateralErrorReward(~logical(IsDone)) + headingErrorReward(~logical(IsDone)) + actionReward(~logical(IsDone)) + overshootReward + smoothReward + maxSteerReward;


end