clc,clear,close all,warning off;
% RLSE Based 3D RBF Model Fitting
% Written By: Rasit
% 14-Apr-2024
%% Create Data
rbfFunc = @(x,y,cx,cy,gamma,weight) weight*exp(-((x-cx).^2+(y-cy).^2)/(2*gamma^2));

NoD = 2e3;
noise = 2*rand(NoD,1);
CenterX = [1 0 1 0];
CenterY = [0 1 1 0];
Gamma = [.25 .25 .25 .25];
Weight = [2 2 1.5 1.5];
ModelOrder = 4;
x = 3*rand(NoD,1)-1;
y = 3*rand(NoD,1)-1;
z = 0;
for i=1:ModelOrder
    z = z + rbfFunc(x,y,CenterX(i),CenterY(i),Gamma(i),Weight(i));
end
z = z + .25*rand(NoD,1);


% Generate A matrix
A = [];
b = z;
for i=1:ModelOrder
    A = [A rbfFunc(x,y,CenterX(i),CenterY(i),Gamma(i),1)]; % Standard basis
end

% Model Fitting Phase
xrlse = rand(ModelOrder,1);  % Random start RLSE state vector
P = 1e2*eye(ModelOrder,ModelOrder);

figure('units','normalized','outerposition',[0 0 1 1],'color','w')
for k=1:NoD
    [xrlse,K,P] = rlse_online(A(k,:),b(k,:),xrlse,P);

    clf
    scatter3(x,y,z,"k","filled"),hold on
%     view(k/10,25)
    % Plot Fitted Function!
    Range = [-1 2];
    [X,Y]= meshgrid(linspace(Range(1),Range(2),1e2));
    Z = 0;

    for i=1:ModelOrder
        Z = Z + rbfFunc(X,Y,CenterX(i),CenterY(i),Gamma(i),xrlse(i));
    end
    surf(X,Y,Z,EdgeColor="none",FaceAlpha=.75),axis([Range(1) Range(2) Range(1) Range(2) -.5 3])
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