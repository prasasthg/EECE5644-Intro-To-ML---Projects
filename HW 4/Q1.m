
%Q1
clc; clear; close all;

n_train = 1000;
n_test = 10000;
r_minus = 2;
r_plus = 4;

n_classes = 2;
n_features = 2;

%Generate Samples
[Xtrain,Ytrain] = generate_samples(n_train,r_minus,r_plus,n_features);
[Xtest,Ytest] = generate_samples(n_test,r_minus,r_plus,n_features);

%Plot Input Data
plot_data(Xtrain,Ytrain)
plot_data(Xtest,Ytest)

%MLP
n_perceptrons = 10;
num_folds = 10;

%Create output matrix
outputMatrix = zeros(n_classes, n_train);
for i = 1:n_classes
    outputMatrix(i,:) = (Ytrain == i-1);
end

%Create folds
f_size = n_train/num_folds;
fold_i = [1:f_size:n_train,n_train];

%Find optimal number of perceptrons
min_ce = 1;
for i = 1:n_perceptrons

    %10 Fold cross validation
    for j = 1:num_folds
        val_indices = fold_i(j):fold_i(j+1);
        train_indices = setdiff(1:n_train,val_indices); 

        %Train MLP
        net = feedforwardnet(i);
        net.trainParam.epochs = 100;
        net.trainParam.lr = 0.01;
        net.layers{1}.transferFcn = 'logsig';
        net.layers{end}.transferFcn = 'softmax';
        net.trainParam.showWindow = false;
        [net,~] = train(net, Xtrain(:,train_indices), outputMatrix(:,train_indices));

        %Validate MLP
        yVal = net(Xtrain(:,val_indices));
        [~, pred] = max(yVal);
        pred = pred - 1;
        targ = Ytrain(val_indices);
        corr = 0;
        for n = 1:numel(pred)
            if pred(n) == targ(n)
                corr = corr + 1;
            end
        end
        fold_ce(j) = 1 - (corr/numel(pred));
    end
    avg_fold_ce(i) = mean(fold_ce);

    % Update the minimum error and the corresponding neural network parameters
    if avg_fold_ce(i) <= min_ce
        min_ce = avg_fold_ce(i);
        best_n_hidden = i;
    end 
end
disp([avg_fold_ce;1:10]);
fprintf('Optimum number of perceptrons = %d\n',best_n_hidden);
fprintf('Min Classification Error = %f\n',min_ce);

%Train MLP using optimum number of perceptrons
net = feedforwardnet(i);
net.trainParam.epochs = 100;
net.trainParam.lr = 0.01;
net.layers{1}.transferFcn = 'logsig';
net.layers{end}.transferFcn = 'softmax';
net.trainParam.showWindow = false;
[net,~] = train(net, Xtrain, outputMatrix);

%Validate MLP 
yVal = net(Xtest);
[~, pred] = max(yVal);
pred = pred - 1;
targ = Ytest;
plot_output(Xtest,Ytest,pred)
conf_M = zeros(n_classes);
for n = 1:numel(pred)
    p = pred(n) + 1;
    t = targ(n) + 1;
    conf_M(t,p) = conf_M(t,p) + 1;
end
error = 1 - (trace(conf_M)/n_test);
fprintf('Confusion Matrix =  \n');
disp(conf_M);
fprintf('MLP Error = %f\n',error);

%SVM
Xtrain = Xtrain';
Ytrain = Ytrain';
Xtest = Xtest';
Ytest = Ytest';
% Set hyperparameters to try
C_vals = [0.01,0.1,1,10,100];
sigma_vals = [0.001,0.01,0.1,1];

% Split data into 10 folds
%Create folds
num_folds = 10;
f_size = n_train/num_folds;
fold_i = [1:f_size:n_train,n_train];

% Perform grid search using 10-fold cross-validation
best_err = 1;
for C = C_vals
    for sigma = sigma_vals
        err = zeros(num_folds,1);
        for i = 1:num_folds
            % Split data into training and validation sets
            val_indices = fold_i(i):fold_i(i+1);
            train_indices = setdiff(1:n_train,val_indices); 
            X_train_fold = Xtrain(train_indices,:);
            Y_train_fold = Ytrain(train_indices);
            X_val_fold = Xtrain(val_indices,:);
            Y_val_fold = Ytrain(val_indices);

            % Train SVM classifier using training set
            svm_model = fitcsvm(X_train_fold, Y_train_fold, 'BoxConstraint', C, 'KernelFunction', 'gaussian', 'KernelScale', sigma);

            % Predict labels for validation set and compute error
            pred = predict(svm_model, X_val_fold);
            corr = 0;
            for n = 1:numel(pred)
                if pred(n) == Y_val_fold(n)
                    corr = corr + 1;
                end
            end
            err(i) = 1 - (corr/numel(val_indices));
        end

        % Compute average error over all folds
        avg_err = mean(err);

        % Update best hyperparameters if average error is lower
        if avg_err < best_err
            best_C = C;
            best_sigma = sigma;
            best_err = avg_err;
        end
    end
end

fprintf('Optimum C value = %d\n',best_C);
fprintf('Optimum sigma value = %d\n',best_sigma);
fprintf('Min Classification Error = %f\n',best_err);

% Train SVM classifier using entire training set with best hyperparameters
svm_model = fitcsvm(Xtrain, Ytrain, 'BoxConstraint', best_C, 'KernelFunction', 'gaussian', 'KernelScale', best_sigma);

% Test SVM classifier on testing set and compute probability of error
pred = predict(svm_model, Xtest);
plot_output(Xtest',Ytest',pred')
targ = Ytest + 1;
conf_M = zeros(n_classes);
for n = 1:numel(pred)
    p = pred(n) + 1;
    t = targ(n);
    conf_M(t,p) = conf_M(t,p) + 1;
end
error = 1 - (trace(conf_M)/n_test);
fprintf('Confusion Matrix =  \n');
disp(conf_M);
fprintf('SVM Error = %f\n',error);


function [X,Y] = generate_samples(n_samples,r_minus,r_plus,n_features)
    % Generate data
    X = zeros(n_samples, n_features);
    priori = [0.5,0.5];
    Y = zeros(1, n_samples);
    for i = 1:n_samples
        if rand >= priori(2)
            Y(i) = 0;
        else
            Y(i) = 1;
        end
    end
    sd = 1;
    ind0 = find(Y == 0);
    ind1 = find(Y == 1);
    N0 = numel(ind0);
    N1 = numel(ind1);
    theta0 = 2*pi*randn(N0, 1);
    theta1 = 2*pi*randn(N1, 1);
    x0 = sd^2*randn(N0,n_features) + r_minus.*[cos(theta0), sin(theta0)];
    x1 = sd^2*randn(N1,n_features) + r_plus.*[cos(theta1), sin(theta1)];
    X(ind0,:) = x0;
    X(ind1,:) = x1;
    mu_n = mean(X);
    sd_n = std(X);
    X = (X - repmat(mu_n,n_samples,1)) ./ repmat(sd_n,n_samples,1);
    X = X';
end

function plot_data(X,Y)
    figure
    scatter(X(1,find(Y==0)),X(2,find(Y==0)),'o', 'g')
    hold on
    scatter(X(1,find(Y==1)),X(2,find(Y==1)),'X', 'r')
    xlabel('Feature 1');
    ylabel('Feature 2');
    grid on;
    legend show;
end

function plot_output(X,Y,pred)
    X1 = X(1,:);
    X2 = X(2,:);
    figure
    scatter(X1(find(Y==0 & pred==0)), X2(find(Y==0 & pred==0)),'*','g');
    hold on;
    scatter(X1(find(Y==0 & pred==1)), X2(find(Y==0 & pred==1)),'*','r');
    scatter(X1(find(Y==1 & pred==0)), X2(find(Y==1 & pred==0)),'+','r');
    scatter(X1(find(Y==1 & pred==1)), X2(find(Y==1 & pred==1)),'+','g');
    xlabel('Feature 1');
    ylabel('Feature 2');
    grid on;
    legend show;
end