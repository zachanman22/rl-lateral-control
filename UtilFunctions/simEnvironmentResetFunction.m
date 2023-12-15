function [InitialObservation,InitialState] = simEnvironmentResetFunction()
%ENVIRONMENTRESETFUNCTION Summary of this function goes here
%   Detailed explanation goes here
% e_y0 = 1.5 * rand() - 0.75;
e_y0 = 0.5;
% de_y0 = rand() - 0.5;
de_y0 = 0;
% e_theta0 = pi / 6 * rand() - pi / 12;
e_theta0 = -pi / 12;
% de_theta0 = pi / 10 * rand() - pi / 20;
de_theta0 = 0;

% inte_y0 = 0;
% inte_phi0 = 0;

u_prev = 0;

% m/s
% v_x0 = 25;
v_x0 = 25;
R0 = 1000;

InitialState = [e_y0; de_y0; e_theta0; de_theta0; u_prev; v_x0; R0];
InitialObservation = InitialState;
end

