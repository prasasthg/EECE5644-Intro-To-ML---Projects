%ML HW 2
%Q2
clc; clear all; close all;

Ntrain = 100;
Nvalidate = 10000;

[trainx,trainy,valx,valy] = hw2q2(Ntrain,Nvalidate);

%ML ESTIMATOR
train_z = [trainx.^3;trainx.^2;trainx;ones(2,Ntrain)];
train_z = reshape(train_z,[2,4,Ntrain]);

w1 = zeros(2,2);
w2 = zeros(2,4);

for i = 1:Ntrain
    w1 = w1 + (squeeze(train_z(:,:,i))*squeeze(train_z(:,:,i))');
    w2 = w2 + (squeeze(train_z(:,:,i))*trainy(i));
end

theta_ML = inv(w1)*w2;

for i = 1:Nvalidate
    y(i) = estimate(valx(:,i),theta_ML);
end

square_error = sum((valy-y).^2,'all');
mean_square_error = square_error/Nvalidate;

disp('ML ESTIMATOR');
disp('Mean Squared Error = ');
disp(mean_square_error);

%MAP ESTIMATOR
gamma = linspace(0.01, 100000, 100);
variance = 0.001;
noise = normrnd(0,variance);

for i = 1:length(gamma)
    gamma_val = gamma(i);
    w1 = zeros(2);
    w2 = zeros(2, 1);
    for j = 1:Ntrain
        w1 = w1 + (squeeze(train_z(:,:,j))*squeeze(train_z(:,:,j))') + ((variance/gamma_val)*eye(2));
        w2 = w2 + (squeeze(train_z(:,:,j))*trainy(j)); 
    end
    theta_MAP = inv(w1)*w2;
    for j = 1:Nvalidate
        y(j) = estimate(valx(:,j),theta_MAP);
    end
    square_error = sum((valy-y).^2,'all');
    mean_square_error = square_error/Nvalidate;
    MSE(i) = mean_square_error;
end
Min_MSE = min(MSE);
Min_MSE_ind = find(MSE==Min_MSE);
Max_MSE = max(MSE);
Max_MSE_ind = find(MSE==Max_MSE);

disp('MAP ESTIMATOR');
disp('Minimum Gamma = ');
disp(gamma(Min_MSE_ind));
disp('Mean Squared Error = ');
disp(Min_MSE);
disp('Maximum Gamma = ');
disp(gamma(Max_MSE_ind));
disp('Mean Squared Error = ');
disp(Max_MSE);

figure;
plot(gamma, MSE);
xlabel('Gamma');
ylabel('Mean Squared Error');
title('Variation in MSE with respect to Gamma');

function y = estimate(x, w)
    y = (x.^3)' * w(:,1) + (x.^2)' * w(:,2) + (x') * w(:,3) + (ones(1,2) * w(:,4));
end

function [trainx,trainy,valx,valy] = hw2q2(Ntrain,Nvalidate)
    data = generateData(Ntrain);
    figure(1), plot3(data(1,:),data(2,:),data(3,:),'.'), axis equal,
    xlabel('x1'),ylabel('x2'), zlabel('y'), title('Training Dataset'),
    trainx = data(1:2,:); 
    trainy = data(3,:);
    data = generateData(Nvalidate);
    figure(2), plot3(data(1,:),data(2,:),data(3,:),'.'), axis equal,
    xlabel('x1'),ylabel('x2'), zlabel('y'), title('Validation Dataset'),
    valx = data(1:2,:); 
    valy = data(3,:);
end

function x = generateData(N)
    gmmParameters.priors = [.3,.4,.3];
    gmmParameters.meanVectors = [-10 0 10;0 0 0;10 0 -10];
    gmmParameters.covMatrices(:,:,1) = [1 0 -3;0 1 0;-3 0 15];
    gmmParameters.covMatrices(:,:,2) = [8 0 0;0 .5 0;0 0 .5];
    gmmParameters.covMatrices(:,:,3) = [1 0 -3;0 1 0;-3 0 15];
    [x,labels] = generateDataFromGMM(N,gmmParameters);
end

function [x,labels] = generateDataFromGMM(N,gmmParameters)
    priors = gmmParameters.priors;
    meanVectors = gmmParameters.meanVectors;
    covMatrices = gmmParameters.covMatrices;
    n = size(gmmParameters.meanVectors,1);
    C = length(priors);
    x = zeros(n,N); labels = zeros(1,N);
    u = rand(1,N); thresholds = [cumsum(priors),1];
    for l = 1:C
    indl = find(u <= thresholds(l)); Nl = length(indl);
    labels(1,indl) = l*ones(1,Nl);
    u(1,indl) = 1.1*ones(1,Nl);
    x(:,indl) = mvnrnd(meanVectors(:,l),covMatrices(:,:,l),Nl)';
    end
end

