function [InitialObservation,InitialState] = environmentResetFunction()
%ENVIRONMENTRESETFUNCTION Summary of this function goes here
%   Detailed explanation goes here

% Randomize initial state
e_y0 = 1.5 * rand() - 0.75;
de_y0 = rand() - 0.5;
e_theta0 = pi / 6 * rand() - pi / 12;
de_theta0 = pi / 10 * rand() - pi / 20;

% Zero for initial previous action
u_prev = 0;

% Randomize longitudinal velocity and radius of curvature
v_x0 = 20 * rand() + 15;
R0 = 100 * (14 * rand() + 1);

InitialState = [e_y0; de_y0; e_theta0; de_theta0; u_prev; v_x0; R0];
InitialObservation = InitialState;
end

