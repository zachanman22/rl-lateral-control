function Reward = rewardFcn(State, Action, NextState)

modifiedReward = 0;

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

Action = Action(1, :);

nextLatError = NextState(1, :);
nextHeadingError = NextState(3, :);

IsDone = isDone(State, Action, NextState);

% If within 1/10 of the lateral threshold, provide reward as fn of
% error
lateralErrorMask = abs(nextLatError) <= latErrorThreshold / 10;
% Care more about lateral error than heading error, so factor of 3
lateralErrorReward = 3 * (1 - abs(nextLatError) / latErrorThreshold) .^ 2;
lateralErrorReward = lateralErrorReward .* lateralErrorMask;

prevAction = State(5, :);

% If within 1/10 of the heading threshold, provide reward as fn of
% error
headingErrorMask = abs(nextHeadingError) <= headingErrorThreshold / 10;
headingErrorReward = 1 * (1 - abs(nextHeadingError) / headingErrorThreshold).^2;
headingErrorReward = headingErrorReward .* headingErrorMask;

% Penalize sinusoidal lateral error by penalizing the lateral error
% crossing the zero line

if modifiedReward
    overshootMask = (NextState(1, :) > 0 & State(1, :) < 0) | (NextState(1, :) < 0 & State(1, :) > 0);
    overshootReward = -100 * ones(size(State(1, :))) .* overshootMask;
else
    overshootReward = zeros(size(State(1, :)));
end

% Penalize the lateral error increasing, want it to constantly decrease
% smoothly
if modifiedReward
    smoothMask = abs(NextState(1, :)) > abs(State(1, :));
    smoothReward = -11 * ones(size(State(1, :))) .* smoothMask + 1;
else
    smoothReward = zeros(size(State(1, :)));
end 

% Want to perform task with minimal action, so penalize larger actions
actionReward = -(Action / maxSteer) .^ 2;

Reward = zeros(size(IsDone));

Reward(logical(IsDone)) = -1000;
Reward(~logical(IsDone)) = lateralErrorReward(~logical(IsDone)) + headingErrorReward(~logical(IsDone)) + actionReward(~logical(IsDone)) + overshootReward(~logical(IsDone)) + smoothReward(~logical(IsDone));


end