clc,clear all,close all;
%% LS and RLSE Compare
% Written By: Rasit Evduzen
% 26-Feb-2024
%%

% Create synthetic data 
model_oder = 3;
model_coeff = randn(model_oder+1,1);
num_of_data = 50;
x_tr = linspace(-10,10,num_of_data)';
y_tr = 0;
for i=0:model_oder
    y_tr = y_tr + [model_coeff(i+1)*x_tr.^(i)];
end

% Create System matrix
b = y_tr; 
A = [];
for i=0:model_oder
    A = [A x_tr.^(i)];
end


% Create LS and RLSE Model
x_lse = A\b;                     % inv(A'*A)*A'*b
x_rlse = randn(model_oder+1,1);  % Random start RLSE state vector
P = 1e2 * eye(model_oder+1,model_oder+1);

figure('units','normalized','outerposition',[0 0 1 1],'color','w')
for k=1:num_of_data
    [x_rlse,K,P] = rlse_online(A(k,:),b(k,:),x_rlse,P);
    
    clf
    subplot(121)
    plot(x_tr,y_tr,'r*') % Plot Original Data
    hold on, grid minor
    y_lse = model_evaluate(x_tr,x_lse);
    plot(x_tr,y_lse,'b') % Plot LSE Solution Data
    xline(0),yline(0)
    title("Linear Model Fitting via Batch Least Squares")

    subplot(122)
    plot(x_tr,y_tr,'r*') % Plot Original Data
    hold on, grid minor
    y_lse = model_evaluate(x_tr,x_rlse);
    plot(x_tr,y_lse,'b') % Plot RLSE Solution Data
    xline(0),yline(0);
    title("Linear Model Fitting via Recursive Least Squares");
    display(["Number of iter: "+num2str(k); "Real Model Coeff: " + num2str(model_coeff'); "Batch LSE Coeff: " + num2str(x_lse'); "Recursive LSE Coeff: " + num2str(x_rlse')]);
    drawnow

end

function [y] = model_evaluate(t_span,model_param)
    y = 0;
    for i=1:size(model_param)
        y = y + [model_param(i)*t_span.^(i-1)];
    end
end


function [x,K,P] = rlse_online(a_k,b_k,x,P)
    % One step of RLSE (Recursive Least Squares Estimation) algorithm
    a_k = a_k(:); 
    b_k = b_k(:); 
    K = (P*a_k)/(a_k'*P*a_k+1); % Compute Gain K (Like Kalman Gain!)
    x = x + K*(b_k-a_k'*x);     % State Update
    P = P - K*a_k'*P;           % Covariance Update
end
