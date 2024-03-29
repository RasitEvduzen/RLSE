clc,clear,close all;
% Plane Fitting via Recursive Least Squares
% Written By: Rasit
% 29-Mar-2024
%% Create Data
NoD = 5e1;
xslope = 7;
yslope = 9;
zoffset = 5;
noise = 2*rand(NoD,1);
x = rand(NoD,1);
y = rand(NoD,1);
z = (xslope*x) + (yslope*y) + (zoffset) + (noise);


% Create LS and RLSE Model
A = [x y ones(NoD,1)];
b = z;
% xlse = inv(A'*A)*A'*b  % lse Solution
% residue = b - A*xlse;  % lse Residue
model_oder = size(A,2);
x_rlse = rand(model_oder,1);  % Random start RLSE state vector
P = 1e2 * eye(model_oder,model_oder);

tspan = linspace(0,1,10)';
[xx,yy] = meshgrid(tspan);
figure('units','normalized','outerposition',[0 0 1 1],'color','w')
for k=1:NoD
    [x_rlse,K,P] = rlse_online(A(k,:),b(k,:),x_rlse,P);
    
    % Plot Result
    clf
    scatter3(x,y,z,'b','filled',LineWidth=5),hold on,grid on
    xlabel("X"),ylabel("Y"),zlabel("Z"),title("RLSE Plane Fitting")
    zz = x_rlse(1)*xx + x_rlse(2)*yy + x_rlse(3);  % Fitted Plane
    surf(xx,yy,zz,FaceColor="r",FaceAlpha=0.25,EdgeColor="k")
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
