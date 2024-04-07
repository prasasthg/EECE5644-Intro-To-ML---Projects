clear all; close all; clc 


%
rng(10);

% INPUT PARAMETERS
mean1 = [-0.5 -0.5 -0.5 -0.5]; 
SD1 = (1/4)*[2 -0.5 0.3 0;-0.5 1 -0.5 0;0.3 -0.5 1 0;0 0 0 2]; 
mean2 = [1 1 1 1];
SD2 = [1 0.3 -0.2 0;0.3 2 0.3 0;-0.2 0.3 1 0;0 0 0 3];
priori = [0.35, 0.65];
n = 4; %4D Gaussian
sample_size = 10000;



%PART A

%Class Label Generation
[C_L,smple] = input_and_class_labels(mean1,mean2,SD1,SD2,sample_size, priori,n);

%Input Data Graphical Plot
plot_input_data(C_L,smple);

%Discriminant Score, Threshold Calculation
[DS,sorted_DS,threshold] = calc_threshold(mean1,mean2,SD1,SD2,smple);

%Theoretical Threshold Calculation
theoretical_threshold = log(priori(2) / priori(1));
theoretical_decision = DS >= theoretical_threshold;
theoretical_true_p = numel(find((theoretical_decision==1) & (C_L==1))) / numel(find(C_L==1));
theoretical_false_p = numel(find((theoretical_decision==1) & (C_L==0))) / numel(find(C_L==0));
theoretical_error = priori(2)*theoretical_false_p + priori(1)*(1-theoretical_true_p);

%Data Classification
for i = 1:length(threshold)
    decision = DS >= threshold(i);
    true_p(i) = numel(find((decision==1) & (C_L==1))) / numel(find(C_L==1));
    false_p(i) = numel(find((decision==1) & (C_L==0))) / numel(find(C_L==0));
    error(i) = priori(2)*false_p(i) + priori(1)*(1-true_p(i));
end

%ROC Curve Graphical Plot
plot_ROC_Curve(true_p,false_p,error);

%OUTPUT DATA
fprintf('Threshold (Empirical) = %.4f\n', exp(threshold(find(error==min(error)))))
fprintf('Minimum Error (Empirical) = %.4f\n', min(error))
fprintf('Threshold (theoretical) = %.4f\n', exp(theoretical_threshold))
fprintf('Minimum Error (theoretical) = %.4f\n', theoretical_error)


%PART B

%Class Label Generation
[C_L,smple] = input_and_class_labels(mean1,mean2,SD1,SD2,sample_size, priori,n);

%Input Data Graphical Plot
plot_input_data(C_L,smple);

%Discriminant Score, Threshold Calculation
[DS,sorted_DS,threshold] = calc_threshold(mean1,mean2,eye(n,n),eye(n,n),smple);

%Theoretical Threshold Calculation
theoretical_threshold = log(priori(2) / priori(1));
theoretical_decision = DS >= theoretical_threshold;
theoretical_true_p = numel(find((theoretical_decision==1) & (C_L==1))) / numel(find(C_L==1));
theoretical_false_p = numel(find((theoretical_decision==1) & (C_L==0))) / numel(find(C_L==0));
theoretical_error = priori(2)*theoretical_false_p + priori(1)*(1-theoretical_true_p);

%Data Classification
for i = 1:length(threshold)
    decision = DS >= threshold(i);
    true_p(i) = numel(find((decision==1) & (C_L==1))) / numel(find(C_L==1));
    false_p(i) = numel(find((decision==1) & (C_L==0))) / numel(find(C_L==0));
    error(i) = priori(2)*false_p(i) + priori(1)*(1-true_p(i));
end

%ROC Curve Graphical Plot
plot_ROC_Curve(true_p,false_p,error);

%OUTPUT DATA
fprintf('Threshold (Empirical) = %.4f\n', exp(threshold(find(error==min(error)))))
fprintf('Minimum Error (Empirical) = %.4f\n', min(error))
fprintf('Threshold (Theoretical) = %.4f\n', exp(theoretical_threshold))
fprintf('Minimum Error (Theoretical) = %.4f\n', theoretical_error)

%PART C

%Scatter Matrices Calculation
SB_4 = transpose(mean1-mean2)*(mean1-mean2);
SB = SB_4(1,:);
SW = SD1 + SD2;
% disp(SW);
% disp(SB);
% disp(inv(SW));

%Class Label Generation
[C_L,smple] = input_and_class_labels(mean1,mean2,SD1,SD2,sample_size, priori,n);

%Input Data Graphical Plot
plot_input_data(C_L,smple);

S_E = inv(SW)*SB_4;
% disp(S_E);
[weight,e_v_M] = eig(S_E);
e_v = eig(S_E);
% disp(weight);
% disp(e_v);
w_max = weight(find(e_v==max(e_v)));
%disp (w_max);


C_L_0 = smple(find(C_L == 0));
C_L_1 = smple(find(C_L == 1));

w_max_T = w_max.';
C_L_0_T = C_L_0.';
C_L_1_T = C_L_1.';
Y_0 = w_max_T*C_L_0_T;
Y_1 = w_max_T*C_L_1_T;
Y = [Y_0 Y_1];
Y_S = sort(Y.');


threshold = (Y_S(1:end-1) + Y_S(2:end)) / 2;

%Theoretical Threshold Calculation
theoretical_threshold = log(priori(2) / priori(1));
theoretical_decision = Y.' >= theoretical_threshold;
theoretical_true_p = numel(find((theoretical_decision==1) & (C_L==1))) / numel(find(C_L==1));
theoretical_false_p = numel(find((theoretical_decision==1) & (C_L==0))) / numel(find(C_L==0));
theoretical_error = priori(2)*theoretical_false_p + priori(1)*(1-theoretical_true_p);

%Data Classification
for i = 1:length(threshold)
    decision = Y.' >= threshold(i);
    true_p(i) = numel(find((decision==1) & (C_L==1))) / numel(find(C_L==1));
    false_p(i) = numel(find((decision==1) & (C_L==0))) / numel(find(C_L==0));
    error(i) = priori(2)*false_p(i) + priori(1)*(1-true_p(i));
end

%ROC Curve Graphical Plot
plot_ROC_Curve(true_p,false_p,error);

%OUTPUT DATA
fprintf('Threshold (Empirical) = %.4f\n', threshold(find(error==min(error))))
fprintf('Minimum Error (Empirical) = %.4f\n', min(error))
fprintf('Threshold (Theoretical) = %.4f\n', exp(theoretical_threshold))
fprintf('Minimum Error (Theoretical) = %.4f\n', theoretical_error)

%FUNCTIONS
function [C_L,smple] = input_and_class_labels (mean1,mean2,SD1,SD2,sample_size, priori,n)
    C_L = (rand(sample_size, 1) >= priori(1));
    C_L = double(C_L);
    smple = zeros(sample_size, n);
    for i = 1:sample_size
        if C_L(i) == 0
            smple(i,:) = mvnrnd(mean1, SD1);
        elseif C_L(i) == 1
            smple(i,:) = mvnrnd(mean2, SD2);
        end
    end
end

function plot_input_data(C_L,smple)
    figure
    scatter3(smple(C_L==0,4), smple(C_L==0,2), smple(C_L==0,3), 'o', 'g')
    hold on
    scatter3(smple(C_L==1,4), smple(C_L==1,2), smple(C_L==1,3), 'X', 'b')
    xlabel('X')
    ylabel('Y')
    zlabel('Z')
    legend('0','1')
    title('INPUT')
end

function [DS,sorted_DS,threshold] = calc_threshold(mean1,mean2,SD1,SD2,smple)
    DS = log(mvnpdf(smple, mean2, SD2)) - log(mvnpdf(smple, mean1, SD1));
    sorted_DS = sort(DS);
    threshold = (sorted_DS(1:end-1) + sorted_DS(2:end)) / 2;
end

function plot_ROC_Curve(true_p,false_p,error)
    figure
    plot(false_p, true_p, 'r')
    hold on
    plot(false_p(find(error==min(error))), true_p(find(error==min(error))), 'square', 'color', 'k')
    xlabel('False Positive')
    ylabel('True Positive')
    title('ROC Curve')
end
