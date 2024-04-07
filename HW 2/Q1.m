%ML HW 2
%Q1
clc; clear all; close all;
rng(10);


%PART 1
n = 2;
priori = [0.6,0.4];
mean_ij(1,:) = [-1,-1];
mean_ij(2,:) = [1,1];
mean_ij(3,:) = [-1,1];
mean_ij(4,:) = [1,-1];
cov_m = eye(2);
w = [0.5,0.5];

%Generating Samples with true class labels
DV_10K = generate_samples (mean_ij,cov_m,priori,n,w,10000);

figure
scatter(DV_10K(find(DV_10K(:,3)==0),1),DV_10K(find(DV_10K(:,3)==0),2),'o', 'g')
hold on
scatter(DV_10K(find(DV_10K(:,3)==1),1),DV_10K(find(DV_10K(:,3)==1),2),'X', 'b')
xlabel('X_1')
ylabel('X_2')
legend('0','1')
title('INPUT')


[disc,sort_disc,tau] = calc_threshold(DV_10K,mean_ij,cov_m,w);

[dec,true_pos,false_pos,error,min_error,min_error_ind] = classify_data(DV_10K,tau,disc,priori);

[tau_T,dec_T,true_pos_T,false_pos_T,error_T] = calc_theoretical_tau(priori,DV_10K,disc);

disp('Ideal Threshold = ');
disp(exp(tau_T));
disp('Ideal Minimum Error = ');
disp(error_T);

disp('Practical Threshold = ');
disp(exp(tau(min_error_ind)));
disp('Practical Minimum Error = ');
disp(min_error);

figure
plot(false_pos, true_pos, 'r')
hold on
plot(false_pos(min_error_ind), true_pos(min_error_ind), 'square', 'color', 'k')
xlabel('False Positive')
ylabel('True Positive')
title('ROC Curve')

%PART 2A
%Logistic Linear
DT_20 = generate_samples(mean_ij,cov_m,priori,n,w,20);
DT_200 = generate_samples(mean_ij,cov_m,priori,n,w,200);
DT_2K = generate_samples(mean_ij,cov_m,priori,n,w,2000);

[theta_ML,z] = cost_optimization_linear(DT_20,n);
plot_data(DT_20);
DV_10K = generate_samples(mean_ij,cov_m,priori,n,w,10000);
[theta_test,z_test] = cost_optimization_linear(DV_10K,n);
[error_20,dec_test_20] = logistic(theta_ML,z_test,DV_10K);
plot_class_data(DV_10K,dec_test_20');

[theta_ML,z] = cost_optimization_linear(DT_200,n);
plot_data(DT_200);
DV_10K = generate_samples(mean_ij,cov_m,priori,n,w,10000);
[theta_test,z_test] = cost_optimization_linear(DV_10K,n);
[error_200,dec_test_200] = logistic(theta_ML,z_test,DV_10K);
plot_class_data(DV_10K,dec_test_200');

[theta_ML,z] = cost_optimization_linear(DT_2K,n);
plot_data(DT_2K);
DV_10K = generate_samples(mean_ij,cov_m,priori,n,w,10000);
[theta_test,z_test] = cost_optimization_linear(DV_10K,n);
[error_2K,dec_test_2K] = logistic(theta_ML,z_test,DV_10K);
plot_class_data(DV_10K,dec_test_2K');

disp("LOGISTIC LINEAR REGRESSION");
disp('Error for 20 training samples = ');
disp(error_20);
disp('Error for 200 training samples = ');
disp(error_200);
disp('Error for 2000 training samples = ');
disp(error_2K);


%PART 2B
%Logistic Quadratic
DT_20 = generate_samples(mean_ij,cov_m,priori,n,w,20);
DT_200 = generate_samples(mean_ij,cov_m,priori,n,w,200);
DT_2K = generate_samples(mean_ij,cov_m,priori,n,w,2000);

[theta_MQ,z] = cost_optimization_quadratic(DT_20,n);
plot_data(DT_20);
DV_10K = generate_samples(mean_ij,cov_m,priori,n,w,10000);
[theta_test,z_test] = cost_optimization_quadratic(DV_10K,n);
[error_20,dec_test_20] = logistic(theta_MQ,z_test,DV_10K);
plot_class_data(DV_10K,dec_test_20');

[theta_MQ,z] = cost_optimization_quadratic(DT_200,n);
plot_data(DT_200);
DV_10K = generate_samples(mean_ij,cov_m,priori,n,w,10000);
[theta_test,z_test] = cost_optimization_quadratic(DV_10K,n);
[error_200,dec_test_200] = logistic(theta_MQ,z_test,DV_10K);
plot_class_data(DV_10K,dec_test_200');

[theta_MQ,z] = cost_optimization_quadratic(DT_2K,n);
plot_data(DT_2K);
DV_10K = generate_samples(mean_ij,cov_m,priori,n,w,10000);
[theta_test,z_test] = cost_optimization_quadratic(DV_10K,n);
[error_2K,dec_test_2K] = logistic(theta_MQ,z_test,DV_10K);
plot_class_data(DV_10K,dec_test_2K');

disp("LOGISTIC QUADRATIC REGRESSION");
disp('Error for 20 training samples = ');
disp(error_20);
disp('Error for 200 training samples = ');
disp(error_200);
disp('Error for 2000 training samples = ');
disp(error_2K);


function DS = generate_samples(mean_ij,cov_m,priori,n,w,set_size)
    DS = zeros(set_size,n+1);
    CL = (rand(set_size, 1) >= priori(1));
    CL = double(CL);
    for i = 1:set_size
        if CL(i) == 0
            if rand(1,1) >= w(1)
                sample(i,:) = mvnrnd(mean_ij(1,:),cov_m);
            else
                sample(i,:) = mvnrnd(mean_ij(2,:),cov_m);
            end
        elseif CL(i) == 1
            if rand(1,1) >= w(1)
                sample(i,:) = mvnrnd(mean_ij(3,:),cov_m);
            else
                sample(i,:) = mvnrnd(mean_ij(4,:),cov_m);        
            end
        end
    end
    DS = [sample,CL];
end

function plot_data(DS)
    figure
    scatter(DS(find(DS(:,3)==0),1),DS(find(DS(:,3)==0),2),'o', 'g')
    hold on
    scatter(DS(find(DS(:,3)==1),1),DS(find(DS(:,3)==1),2),'X', 'b')
end

function plot_class_data(DS,dec)
    sample1 = DS(:,1);
    sample2 = DS(:,2);
    label = DS(:,3);
    figure;
    scatter(sample1(label==0 & dec == 0),sample2(label==0 & dec == 0),'o','g');
    hold on;
    scatter(sample1(label==0 & dec == 1),sample2(label==0 & dec == 1),'X','r');
    scatter(sample1(label==1 & dec == 0),sample2(label==1 & dec == 0),'X','r');    
    scatter(sample1(label==1 & dec == 1),sample2(label==1 & dec == 1),'o','g');
    xlabel("Feature x1");
    ylabel("Feature x2");
    title("Classified Data");
end


function [disc,sort_disc,tau] = calc_threshold(DS,mean_ij,cov_m,w)
    PXL0 = w(1)*mvnpdf(DS(:,1:2),mean_ij(1,:),cov_m) + w(2)*mvnpdf(DS(:,1:2),mean_ij(2,:),cov_m);
    PXL1 = w(1)*mvnpdf(DS(:,1:2),mean_ij(3,:),cov_m) + w(2)*mvnpdf(DS(:,1:2),mean_ij(4,:),cov_m);    
    disc = log(PXL1) - log(PXL0);
    sort_disc = sort(disc);
    tau = (sort_disc(1:end-1) + sort_disc(2:end)) / 2;
end

function [dec,true_pos,false_pos,error,min_error,min_error_ind] = classify_data(DS,tau,disc,priori)
    for i = 1:length(tau)
        dec = disc >= tau(i);
        true_pos(i) = numel(find((dec==1) & (DS(:,3)==1))) / numel(find(DS(:,3)==1));
        false_pos(i) = numel(find((dec==1) & (DS(:,3)==0))) / numel(find(DS(:,3)==0));
        error(i) = priori(2)*false_pos(i) + priori(1)*(1-true_pos(i));
    end
    min_error = min(error);
    min_error_ind = find(error==min(error));
end

function [tau_T,dec_T,true_pos_T,false_pos_T,error_T] = calc_theoretical_tau(priori,DS,disc)
    tau_T = log(priori(1) / priori(2));
    dec_T = disc >= tau_T;
    true_pos_T = numel(find((dec_T==1) & (DS(:,3)==1))) / numel(find(DS(:,3)==1));
    false_pos_T = numel(find((dec_T==1) & (DS(:,3)==0))) / numel(find(DS(:,3)==0));
    error_T = priori(2)*false_pos_T + priori(1)*(1-true_pos_T);
end


function cost = cost_function(theta,z,m,Y)
    h = 1 ./ (1 + exp(-(theta' * z)));
    cost = (-1 / m) * (sum(Y' * log(h'),'all') + sum((1 - Y)' * log(1 - h'),'all'));
end

function [theta_ML,z] = cost_optimization_linear(DS,n)
    m = size(DS,1);
    z = [ones(m, 1),DS(:,1:2)]';
    theta_j = zeros(n+1,1);
    Y = DS(:,3);
    theta_ML = fminsearch(@(theta) cost_function(theta, z, m, Y),theta_j);
end

function [theta_MQ,z] = cost_optimization_quadratic(DS,n)
    m = size(DS,1);
    x1 = DS(:,1);
    x2 = DS(:,2);
    z = [ones(m, 1),x1,x2,x1.^2,x1.*x2,x2.^2]';
    theta_j = zeros(6,1);
    Y = DS(:,3);
    theta_MQ = fminsearch(@(theta) cost_function(theta, z, m, Y),theta_j);
end


function [error,dec_test] = logistic(theta_ML,z_test,DS)
    Y = DS(:,3);
    dec_test = (1 ./ (1+ exp(-(theta_ML' * z_test)))) >= 0.5;
    dec_test_T = dec_test';
    true_pos = numel(find((dec_test_T==1) & (Y==1)));
    true_neg = numel(find((dec_test_T==0) & (Y==0)));
    error = (10000 - (true_pos + true_neg))/100;
end