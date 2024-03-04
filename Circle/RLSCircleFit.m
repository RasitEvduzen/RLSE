clc,clear all,close all;
%% LS and RLSE Compare
% Written By: Rasit Evduzen
% 04-Mar-2024
%% Create Data
C = [5.5 , 7.8]; 
R = 4.5;
N = 2e2;
k = 5e-2;
model_oder = 3;
% Create noisy circle
alpha = 2*pi*rand(1,N);
noise = k*2*randn(1,N)-1;
Points = C + [ R*noise.*cos(alpha) ; R*noise.*sin(alpha) ]';

A = [Points(:,1) Points(:,2) ones(N,1)];
b = [Points(:,1).*Points(:,1) + Points(:,2).*Points(:,2)];

% Create LS and RLSE Model
x_rlse = rand(model_oder,1);  % Random start RLSE state vector
P = 1e2 * eye(model_oder,model_oder);

figure('units','normalized','outerposition',[0 0 1 1],'color','w')
for k=1:N
    [x_rlse,K,P] = rlse_online(A(k,:),b(k,:),x_rlse,P);
    xc = x_rlse(1)/2;
    yc = x_rlse(2)/2;
    r = sqrt(4*x_rlse(3) + x_rlse(1)*x_rlse(1) + x_rlse(2)*x_rlse(2) )/2;
    % Plot Result
    clf
    plot (Points(:,1) , Points(:,2),'ko', 'LineWidth', 2);
    axis([-10 10 -5 15]),axis equal,grid on,hold on
    xline(0),yline(0)
    title({"RLSE - Circle Fitting"; "Center x: "+num2str(xc); ...
        "Center y: "+num2str(yc);"Radius r: "+num2str(r)})

    circle = [xc + r*cos([0:pi/50:2*pi]) ; yc + r*sin([0:pi/50:2*pi])]';
    plot (circle(:,1), circle(:,2), 'r', 'LineWidth', 3);
    xlabel('X'),ylabel('Y')
    drawnow
end


function [x,K,P] = rlse_online(a_k,b_k,x,P)
    % One step of RLSE (Recursive Least Squares Estimation) algorithm
    a_k = a_k(:); 
    b_k = b_k(:); 
    K = (P*a_k)/(a_k'*P*a_k+1); % Compute Gain K (Like Kalman Gain!)
    x = x + K*(b_k-a_k'*x);     % State Update
    P = P - K*a_k'*P;           % Covariance Update
end