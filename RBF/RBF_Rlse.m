clc,clear all,close all;
%% RLSE RBF Model Fitting
% Written By: Rasit Evduzen
% 06-Apr-2024
%% Create Data
x_tr = (-3:5e-2:3)';
num_of_data = size(x_tr,1);
a1 = radbas(x_tr);
a2 = radbas(x_tr-1.5);
a3 = radbas(x_tr+2);
y = 1*a1 + 2*a2 + 3*a3; % Original Data
y_tr = y + .25*randn(num_of_data,1);  % Noisy Data

% Confidence Plot
xconf = [x_tr' x_tr(end:-1:1)'];
yconf = [y'+0.15 y(end:-1:1)'-0.15];


% Create LS and RLSE Model
model_oder = 3;
x_rlse = randn(model_oder,1);  % Random start RLSE state vector
P = 1e2 * eye(model_oder,model_oder);

figure('units','normalized','outerposition',[0 0 1 1],'color','w')
for k=1:num_of_data
    A = [radbas(x_tr(k)) radbas(x_tr(k)-1.5) radbas(x_tr(k)+2)];  % Create Regressor Matrix
    b = y_tr(k);  % Get Measurement

    [x_rlse,K,P] = rlse_online(A,b,x_rlse,P);
    clf
    plot(x_tr,y_tr,'ro-',LineWidth=2.5) % Plot Original Data
    hold on, grid minor
    y_rlse = model_evaluate(x_rlse,a1,a2,a3);
    plot(x_tr,y_rlse,'k',LineWidth=2) % Plot Original Data
    title("RBF Model Fittin via Recursive Least Squares")
    axis([-3 3 -.5 4])
    fill(xconf,yconf,'red',FaceColor=[0 0 1],EdgeColor="none",FaceAlpha=.3);
    legend("Noisy Data","RLSE","Confidence Bounds")
    drawnow
end

function [y] = model_evaluate(model_param,a1,a2,a3)
y = model_param(1)*a1 + model_param(2)*a2 + model_param(3)*a3;
end


function [x,K,P] = rlse_online(a_k,b_k,x,P)
% One step of RLSE (Recursive Least Squares Estimation) algorithm
a_k = a_k(:);
b_k = b_k(:);
K = (P*a_k)/(a_k'*P*a_k+1); % Compute Gain K (Like Kalman Gain!)
x = x + K*(b_k-a_k'*x);     % State Update
P = P - K*a_k'*P;           % Covariance Update
end