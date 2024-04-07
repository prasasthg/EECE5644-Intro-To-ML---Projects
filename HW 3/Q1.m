clc; clear all; close all;


n = 3;
CL = 4;


priori = [0.25,0.25,0.25,0.25];
mean_ij(1,:) = [0,0,0];
mean_ij(2,:) = [0,0,5];
mean_ij(3,:) = [0,5,0];
mean_ij(4,:) = [5,0,0];
cov_m(:,:,1) = [1,0,-2;0,1,0;-2,0,12];
cov_m(:,:,2) = [6,0,0;0,0.3,0;0,0,0.3];
cov_m(:,:,3) = [1,0,-2;0,0.3,0;-2,0,12];
cov_m(:,:,4) = [6,0,2;0,1,0;2,0,12];


DS_train = [100,200,500,1000,2000,5000];
DS_val = 10000;

%Perceptron Range
PVal = 1:10;

%Learning Rate and Epochs for finding best number of Perceptrons
alpha1 = 0.01;
epochs1 = 100;

%Learning Rate and Epochs for training MLP
alpha2 = 0.01;
epochs2 = 100;

%Generate Test Data
DSval = generate_samples(mean_ij,cov_m,priori,DS_val,n);
Xval = DSval(:,1:n);
mu_n = mean(Xval);
sd_n = std(Xval);
Xval = (Xval - repmat(mu_n, DS_val, 1)) ./ repmat(sd_n, DS_val, 1);
Yval = DSval(:,CL) + 1;

for t = 1:numel(DS_train) 
    %Find Optimum Number of Perceptrons
    %Generate Training Data
    DS_size = DS_train(t);
    Dtrain = generate_samples(mean_ij,cov_m,priori,DS_size,n);
    fprintf('MLP MODEL %d (%d training samples)\n',t,DS_size);
    fprintf('-------------------------------------\n');
    % Shuffle the rows of the data randomly
    shf_idx = randperm(size(Dtrain, 1));
    Dtrain_shf = Dtrain(shf_idx, :);
    
    % Separate the input data from the class labels
    X = Dtrain_shf(:,1:end-1);
   
    mu_n = mean(X);
    sd_n = std(X);
    X = (X - repmat(mu_n, DS_size, 1)) ./ repmat(sd_n, DS_size, 1);
    Y = Dtrain_shf(:,end);
    Y = Y + 1;
    
    % Convert the class labels to binary vectors
    CL = max(Y);
    T = zeros(length(Y),CL);
    for i = 1:length(Y)
        T(i,Y(i)) = 1;
    end
    
    %Splitting the data into 10 folds
    k = 10;
    f_size = floor(DS_size/k);
    Xfolds = zeros(f_size,n,k);
    for fold = 1:k
        fold_start = (fold-1)*f_size + 1;
        fold_end = fold*f_size;
        Xfolds(:,:,fold) = X(fold_start:fold_end,:);
    end

    Tfolds = zeros(f_size,n+1,k);
    for fold = 1:k
        fold_start = (fold-1)*f_size + 1;
        fold_end = fold*f_size;
        Tfolds(:,:,fold) = T(fold_start:fold_end,:);
    end
    
    % Set up the neural network with ELU activation function and SoftMax output layer
    min_ce = 1;
    for i = 1:length(PVal)
        inp = size(X,2);
        numH = PVal(i);
        outp = CL;
        
        % Evaluate the performance of the neural network using 10-fold cross-validation
        fold_ce = zeros(k,1);
        for fold = 1:k
            % Divide the data into training and validation sets
            Xval_fold = squeeze(Xfolds(:,:,fold));
            Tval_fold = squeeze(Tfolds(:,:,fold));
            
            t_idx = [1:(fold-1)*f_size,fold*f_size+1:DS_size];
            Xtrain_fold = X(t_idx,:);
            Ttrain_fold = T(t_idx,:);
            r_ce = zeros(10,1);
            best_ll = -Inf;
            for j = 1:10
                % Train the neural network on the training set
                wt1 = randn(inp,numH);
                bs1 = randn(numH,1);
                wt2 = randn(numH,outp);
                bs2 = randn(outp,1);
                for epoch = 1:epochs1
                    % Forward propagation
                    Z1 = Xtrain_fold*wt1 + repmat(bs1',size(Xtrain_fold,1),1);
                    A1 = elu(Z1,alpha1);
                    Z2 = A1*wt2 + repmat(bs2',size(Xtrain_fold,1),1);
                    Y = s_max(Z2);
                    
                    % Backward propagation
                    dZ2 = Y - Ttrain_fold;
                    dwt2 = A1' * dZ2;
                    dbs2 = sum(dZ2,1)';
                    dA1 = dZ2 * wt2';
                    dZ1 = eluG(Z1) .* dA1;
                    dwt1 = Xtrain_fold' * dZ1;
                    dbs1 = sum(dZ1,1)';
                    
                    % Update the weights and biases of the neural network
                    wt1 = wt1 - alpha1*dwt1;
                    bs1 = bs1 - alpha1*dbs1;
                    wt2 = wt2 - alpha1*dwt2;
                    bs2 = bs2 - alpha1*dbs2;
                    ytrain_fold = zeros(size(Ttrain_fold,1),1);
                    for s = 1:size(Ttrain_fold,1)
                        for c = 1:CL
                            if Ttrain_fold(s,c) == 1
                                ytrain_fold(s) = c;
                            end
                        end
                    end
                    train_ll(epoch) = sum(log(Y(ytrain_fold',1)));
                end
                if (train_ll(end) > best_ll)
                    best_ll = train_ll(end);
                    rwt1 = wt1;
                    rwt2 = wt2;
                    rbs1 = bs1;
                    rbs2 = bs2;
                end
            end     
            % Evaluate the performance of the trained neural network on the validation set
            Z1 = Xval_fold*rwt1 + repmat(rbs1',size(Xval_fold,1),1);
            A1 = elu(Z1,alpha1);
            Z2 = A1*rwt2 + repmat(rbs2',size(Xval_fold,1),1);
            Y = s_max(Z2);
            [~,pred] = max(Y,[],2);
            disp(pred);
            corr = 0;
            for a = 1:numel(pred)
                p = pred(a);
                if (Tval_fold(a,p) == 1)
                    corr = corr + 1;
                end
            end        
            fold_ce(fold) = 1 - (corr/numel(pred));
        end
        
        % Calculate the average error over all folds
        avg_fold_ce = mean(fold_ce);

        % Update the minimum error and the corresponding neural network parameters
        if avg_fold_ce <= min_ce
            min_ce = avg_fold_ce;
            bestwt1 = rwt1;
            bestbs1 = rbs1;
            bestwt2 = rwt2;
            bestbs2 = rbs2;
            bestnumH = numH;
        end        
    end
    best_P(t) = bestnumH;
    fprintf('Optimum number of perceptrons = %d\n',bestnumH);
    fprintf('Min Classification Error = %f\n',min_ce);

    %Train MLP models using optimum number of perceptrons for each set
    Xtrain = Dtrain(:,1:n);
    mu_n = mean(Xtrain);
    sd_n = std(Xtrain);
    Xtrain = (Xtrain - repmat(mu_n, DS_size, 1)) ./ repmat(sd_n, DS_size, 1);
    Ytrain = Dtrain(:,n+1) + 1;
    Ttrain = zeros(length(Ytrain),CL);
    for i = 1:length(Ytrain)
        Ttrain(i,Ytrain(i)) = 1;
    end
    inp = size(Xtrain,2);
    outp = CL;
    numH = best_P(t);
    best_ll = -Inf;
    for j = 1:10
        wt1 = randn(inp,numH);
        bs1 = zeros(numH,1);
        wt2 = randn(numH,outp);
        bs2 = zeros(outp,1);
        train_ll = zeros(epochs2, 1);
        for epoch = 1:epochs2
            % Forward pass
            Z1 = Xtrain*wt1 + repmat(bs1',DS_size,1);
            A1 = sig(Z1);
            Z2 = A1*wt2 + repmat(bs2',DS_size,1);
            Y = s_max(Z2);
    
            % Compute cross-entropy loss
            loss = -sum(log(Y(Ytrain',1)));
    
            % Backward pass
            dZ2 = Y - Ttrain;
            dwt2 = A1' * dZ2;
            dbs2 = sum(dZ2,1)';
            dA1 = dZ2 * wt2';
            dZ1 = sigG(Z1) .* dA1;
            dwt1 = Xtrain' * dZ1;
            dbs1 = sum(dZ1,1)';
            wt1 = wt1 - alpha2 * dwt1;
            bs1 = bs1 - alpha2 * dbs1;
            wt2 = wt2 - alpha2 * dwt2;
            bs2 = bs2 - alpha2 * dbs2; 
            train_ll(epoch) = sum(log(Y(Ytrain',1)));
        end
        if (train_ll(end) > best_ll)
            best_ll = train_ll(end);
            bestwt1 = wt1;
            bestwt2 = wt2;
            bestbs1 = bs1;
            bestbs2 = bs2;
        end
    end

    %Test  MLP with the test data
    Z1 = Xval*bestwt1 + repmat(bestbs1',DS_val,1);
    A1 = sig(Z1);
    Z2 = A1*bestwt2 + repmat(bestbs2',DS_val,1);
    Y = s_max(Z2);

    %Calculate Error and Confusion matrix
    [~,pred] = max(Y,[],2);
    conf_m = zeros(CL);
    for i = 1:numel(pred)
        a = Yval(i);
        p = pred(i);
        conf_m(a,p) = conf_m(a,p) + 1;  
    end
    error = 1 - (trace(conf_m)/DS_val);
    fprintf('Probability of Error = %f\n',error);
    fprintf('Confusion Matrix =  \n');
    disp(conf_m);
    errors(t) = error;
end

%Theoretical Classifier
%Generate Test Data
DSval = generate_samples(mean_ij,cov_m,priori,DS_val,n);
Xval = DSval(:,1:n);
mu_n = mean(Xval);
sd_n = std(Xval);
Xval = (Xval - repmat(mu_n, DS_val, 1)) ./ repmat(sd_n, DS_val, 1);
Yval = DSval(:,CL) + 1;

%Loss matrix
loss_m = ones(CL) - eye(CL);

%Class-Conditional Probabilities
probX_L = zeros(CL,DS_val);
for i = 1:CL
    probX_L(i,:) = mvnpdf(Xval,mean_ij(i,:),squeeze(cov_m(:,:,i)));
end

%Class Posteriors
probX = priori * probX_L;
Class_Pos = (probX_L .* repmat(priori', 1, DS_val)) ./ repmat(probX, CL, 1);
exp_risk = loss_m * Class_Pos;
[~,dec] = min(exp_risk,[],1);
dec = dec';

%Calculate Risk and Confusion Matrix
avg_exp_risk = sum(min(exp_risk,[],1))/DS_val;
opt_conf_m = zeros(CL);
for i = 1:CL
    for j = 1:CL
        opt_conf_m(i,j) = numel(find((i == Yval) & (j == dec)));
    end
end

% Plot graph to compare theoretical classifier and various MLP models
plot(DS_train, errors, '-', 'MarkerSize', 15);
hold on;
yline(avg_exp_risk, 'r-', 'LineWidth', 1);
xlabel('Number of Samples');
ylabel('Error');
title('Plot to Compare Errors of the trained models');
hold off;

function DS = generate_samples(mean_ij,cov_m,priori,DS_size,n)
    Priori_Cum = cumsum(priori);
    rand = randn(DS_size, 1);
    CL = zeros(size(rand));
    for i = 1:DS_size
        if rand(i) <= Priori_Cum(1)
            CL(i) = 0;
        elseif rand(i) <= Priori_Cum(2)
            CL(i) = 1;
        elseif rand(i) <= Priori_Cum(3)
            CL(i) = 2;
        else 
            CL(i) = 3;
        end
    end
    DS = zeros(DS_size,n);
    for i = 1:DS_size
        if CL(i) == 0           
            DS(i,:) = mvnrnd(mean_ij(1,:),squeeze(cov_m(:,:,1)));
        elseif CL(i) == 1
            DS(i,:) = mvnrnd(mean_ij(2,:),squeeze(cov_m(:,:,2)));       
        elseif CL(i) == 2      
            DS(i,:) = mvnrnd(mean_ij(3,:),squeeze(cov_m(:,:,3)));
        elseif CL(i) == 3
            DS(i,:) = mvnrnd(mean_ij(4,:),squeeze(cov_m(:,:,4)));                   
        end
    end
    DS = [DS,CL];
end

function y = s_max(x)
    m = max(x,[],2);
    y = exp(x - m) ./ sum(exp(x - m),2);
end

function y = elu(x,alpha)
    y = x .* (x > 0) + alpha * (exp(x) - 1) .* (x <= 0);
end

function y = eluG(x)
    y = ones(size(x)); 
    y(x < 0) = exp(x(x < 0));
end

function y = sig(x)
    % Sigmoid activation function
    y = 1 ./ (1 + exp(-x));
end

function y = sigG(x)
    % Computes the gradient of the sigmoid function at x
    y = sig(x).*(1-sig(x));
end