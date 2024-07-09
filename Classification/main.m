clc,clear all, close;
% RLSE Based Binary Classification
% Written By: Rasit
% 07-Jul-2024
%% Create Data
NoD = 200; % Number of Data
X = [randn(NoD,2)+1.5; -randn(NoD,2)-1.5];  % Class Data
Y = [ones(NoD,1); -ones(NoD,1)];           % Class Label


% Create LS and RLSE Model
model_oder = 3;
x_rlse = zeros(model_oder,1);  % Random start RLSE state vector
P = 1e-6*eye(model_oder,model_oder);

figure('units','normalized','outerposition',[0 0 1 1],'color','w')
gif('RLSEClassification.gif')
for k=1:NoD
    A = [X(k,:) 1];  % Create Regressor Matrix
    b = Y(k);        % Get Measurement
    [x_rlse,K,P] = rlse_online(A,b,x_rlse,P);

    clf
    % Plot Result
    [i, j] = meshgrid(min(X(:,1)):1e-1:max(X(:,1)), min(X(:,2)):1e-1:max(X(:,2)));
    decision_boundary = x_rlse(1) * i + x_rlse(3) + j * x_rlse(2);

    contourf(i, j, sign(decision_boundary), 'LineColor', 'none'),hold on
    scatter(X(Y == 1, 1), X(Y == 1, 2), 'k',"filled")
    scatter(X(Y == -1, 1), X(Y == -1, 2), 'g',"filled");
    x_line = linspace(min(X(:,1)), max(X(:,1)), 100);
    y_line = -(x_rlse(1) * x_line + x_rlse(3)) / x_rlse(2);
    plot(x_line, y_line, 'r',LineWidth=4);
    xlabel('X1'),ylabel('X2'),title('RLSE Based Binary Classification');
    axis([min(min(X)), max(max(X)), min(min(X)), max(max(X))]),axis equal
    drawnow
    gif
end


function [x,K,P] = rlse_online(a_k,b_k,x,P)
% One step of RLSE (Recursive Least Squares Estimation) algorithm
a_k = a_k(:);
b_k = b_k(:);
K = (P*a_k)/(a_k'*P*a_k+1); % Compute Gain K (Like Kalman Gain!)
x = x + K*(b_k-a_k'*x);     % State Update
P = P - K*a_k'*P;           % Covariance Update
end

