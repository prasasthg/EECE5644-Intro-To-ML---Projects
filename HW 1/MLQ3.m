%Read Data Sets
D_S_R = readmatrix('/Users/prasasth/Downloads/winequality-red.csv');
D_S_W = readmatrix('/Users/prasasth/Downloads/winequality-white.csv');
D_S = [D_S_R;D_S_W];

labels = 11;
features = 11;

%Input variables (based on physicochemical tests):
%    1 - fixed acidity
%    2 - volatile acidity
%    3 - citric acid
%    4 - residual sugar
%    5 - chlorides
%    6 - free sulfur dioxide
%    7 - total sulfur dioxide
%    8 - density
%    9 - pH
%    10 - sulphates
%    11 - alcohol
%    Output variable (based on sensory data): 
%    12 - quality (score between 0 and 10)
D_Var = 1:12;
D_Lab = 1:11;

figure
for i = 1:labels
    scatter3(D_S((D_Lab==i),1),D_S((D_Lab==i),2),D_S((D_Lab==i),3));
end
xlabel('X')
ylabel('Y')
zlabel('Z')
legend show;
title('INPUT')

D_S = normalize(D_S,'zscore');

[coeff,D_S,latent] = pca(D_S);

mean_M = zeros(labels,features);
cv_M = zeros(labels,features,features);

for i = 1:labels
    mean_M(i, :) = mean(D_S(find(D_Lab==i),:));
    cv_M(i,:,:) = cov(D_S(find(D_Lab==i),:));
    cv_M(i,:,:) = squeeze(cv_M(i,:,:)) + (0.00001)*(trace(squeeze(cv_M(i,:,:)))/rank(squeeze(cv_M(i,:,:))))*eye(11);
end

loss_matrix = ones(labels,labels) - eye(labels);

Prob_X_given_L = zeros(labels,size(D_S,1));

for i = 1:labels
    Prob_X_given_L(i, :) = mvnpdf(D_S(:,1:11),mean_M(i, :),squeeze(cv_M(i, :,:)));
end

priori = zeros(labels,1);
for i = 1:labels
    priori(i,1) = (size(D_Lab(find(D_Lab==i)),1)) / size(D_S,1);
end

P_X = priori'*Prob_X_given_L;
CP_rep1 = repmat(priori,1,size(D_S,1));
CP_rep2 = repmat(P_X,labels,1);
posteriori = (Prob_X_given_L' * CP_rep1)/CP_rep2;

exp_risk = loss_matrix*posteriori';
dec = min(exp_risk);
avg_exp_risk = sum(min(exp_risk', [], 1))/6497;


D_Lab = D_S(:,12);
conf_M = zeros(labels,labels);
for i = 1:labels
    for j = 1:labels
        conf_M(i,j) = numel(find((i-1==dec) & (j-1==D_Lab))) / numel(find(D_Lab==j-1));
    end
end