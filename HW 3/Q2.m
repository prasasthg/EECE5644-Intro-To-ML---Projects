clc; clear all; close all;


n_features = 2;
n_comp = 4;
priori = [0.2,0.3,0.4,0.1];
comp_mu(1,:) = [0,0];
comp_mu(2,:) = [0,5];
comp_mu(3,:) = [5,0];
comp_mu(4,:) = [5,5];
comp_sd(:,:,1) = [10,4;4,10];
comp_sd(:,:,2) = [8,6;6,8];
comp_sd(:,:,3) = [5,0;0,5];
comp_sd(:,:,4) = [4,1;1,5];
set_size = [10,100,1000,10000];
folds = 10;
g_comp = 6;


select = zeros(numel(set_size),g_comp);
for s = 1:numel(set_size)
    %Generate Data
    n_samples = set_size(s);
    DS = generate_samples(comp_mu,comp_sd,priori,n_samples,n_features,n_comp);
    plot_data(DS);
    n_repeat = 30;
    
    
    for rep = 1:n_repeat
        %Shuffle Data
        shf_idx = randperm(n_samples);
        DS = DS(shf_idx, :);
        
        %Split data into X and Y
        X = DS(:,1:n_features);
        Y = DS(:,n_features+1);
        
        %Split data into 10 folds
        n_folds = 10;
        f_size = floor(n_samples/n_folds);
        Xfolds = zeros(f_size,n_features,n_folds);
        for f = 1:n_folds
            fold_start = (f-1)*f_size + 1;
            fold_end = f*f_size;
            Xfolds(:,:,f) = X(fold_start:fold_end,:);
        end
           
        %Perform 10-fold cross validation for each number of components
        max_ll = 0;
        comp_ll = zeros(g_comp,1);
        for g = 1:g_comp
            fold_ll = zeros(n_folds,1);
            for f = 1:n_folds
                %Assign Training and Validation Folds
                Xval_fold = squeeze(Xfolds(:,:,f));
                t_idx = [1:(f-1)*f_size,f*f_size+1:n_samples];
                Xtrain_fold = X(t_idx,:);
                
                % Estimate GMM parameters using EM algorithm
                gmFit = fitgmdist(Xtrain_fold, g, 'RegularizationValue', 0.01, 'ProbabilityTolerance',1e-6, 'Options', statset('MaxIter', 1000));
                
                % Evaluate log-likelihood of validation set
                fold_ll(f) = sum(log(pdf(gmFit,Xval_fold)));
            end
            comp_ll(g) = mean(fold_ll);
        end
        [max_ll,g_select] = max(comp_ll);
        select(s,g_select) = select(s,g_select) + 1;
    end
    fprintf('Set with %d samples completed.\n',n_samples);
    figure
    bar(select(s,:))
end

function DS = generate_samples(comp_mu,comp_sd,priori,n_samples,n,c)
    
    x = zeros(n, n_samples);
    labels = zeros(1, n_samples);
   
    u = rand(1, n_samples);
    th = zeros(1, c+1);
    th(1:c) = cumsum(priori);
    th(c+1) = 1;
    
    for l = 1:c
        indl = find(u <= th(l));
        Nl = length(indl);
        labels(indl) = (l-1)*ones(1, Nl);
        u(indl) = 1.1;
        x(:, indl) = mvnrnd(comp_mu(l, :), squeeze(comp_sd(:,:,l)), Nl)';
    end
    x = x';
    labels = labels' + 1;
    DS = [x,labels];
end

function plot_data(DS)
    figure
    scatter(DS(find(DS(:,3)==1),1),DS(find(DS(:,3)==1),2),'o', 'g')
    hold on
    scatter(DS(find(DS(:,3)==2),1),DS(find(DS(:,3)==2),2),'X', 'b')
    scatter(DS(find(DS(:,3)==3),1),DS(find(DS(:,3)==3),2),'*', 'r')
    scatter(DS(find(DS(:,3)==4),1),DS(find(DS(:,3)==4),2),'+', 'k')

end