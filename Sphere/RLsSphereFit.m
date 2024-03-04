clc,clear all,close all;
%% LS and RLSE Compare
% Written By: Rasit Evduzen
% 04-Mar-2024
%% Create Data
C = [30 , 20, 15];
R = 12;
model_oder = 4;
N = 3e2; % Number of points
k = 0.02; % Noise
% Create noisy sphere
alpha = 2*pi*rand(1,N);
beta  = 2*pi*rand(1,N);
noise = k*2*randn(1,N)-1;
Points = C + [ R*noise.*cos(alpha) ; R*noise.*sin(alpha).*cos(beta) ; R*noise.*sin(alpha).*sin(beta)  ]';

% Prepare matrices
A = [Points(:,1) Points(:,2) Points(:,3) ones(N,1)];
b = [Points(:,1).*Points(:,1) + Points(:,2).*Points(:,2) + Points(:,3).*Points(:,3)];

% Create LS and RLSE Model
x_rlse = rand(model_oder,1);  % Random start RLSE state vector
P = 1e2 * eye(model_oder,model_oder);

figure('units','normalized','outerposition',[0 0 1 1],'color','w')
for k=1:N
    [x_rlse,K,P] = rlse_online(A(k,:),b(k,:),x_rlse,P);
    xc = x_rlse(1)/2;
    yc = x_rlse(2)/2;
    zc = x_rlse(3)/2;
    r = sqrt(4*x_rlse(4) + x_rlse(1)*x_rlse(1) + x_rlse(2)*x_rlse(2) + x_rlse(3)*x_rlse(3))/2;
    % Plot Result
    clf
    plot3 (Points(:,1) , Points(:,2), Points(:,3), 'ko','LineWidth',2);
    axis square equal, grid on,hold on
    xlabel ('X'),ylabel ('Y'),zlabel ('Z');
    title({"RLSE - Sphere Fitting"; "Center x: "+num2str(xc); ...
        "Center y: "+num2str(yc);"Center z: "+num2str(zc);"Radius r: "+num2str(r)})
    [X, Y, Z] = sphere;
    X = xc + X*r;
    Y = yc + Y*r;
    Z = zc + Z*r;
    surf (X, Y, Z, 'FaceAlpha', 0.2, 'EdgeAlpha', 0.1,'FaceColor','r');
    axis square equal;
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