function IsDone = isDone(State, Action, NextState)
if iscell(State)
    State = State{1};
end

if iscell(Action)
    Action = Action{1};
end

if iscell(NextState)
    NextState = NextState{1};
end

% Lateral error for which to fail the episode
latErrorThreshold = 1.5;
% Heading error for which to fail the episode
headingErrorThreshold = pi / 2;

% Full visible/observable state assumption
NextObservation = NextState;

nextLatError = NextObservation(1, :);
nextHeadingError = NextObservation(3, :);

IsDone = abs(nextLatError) > latErrorThreshold | ...
    abs(nextHeadingError) > headingErrorThreshold; % || ...
    % abs(Action - prevAction) > maxSteerRate;

end