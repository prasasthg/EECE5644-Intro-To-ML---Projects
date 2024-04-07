%Q2
clc; clear; close all;

% Read the image from the specified path
path = '/Users/prasasth/Desktop/';
file_name = '119082.jpeg';
image = imread(fullfile(path,file_name));

disp(size(image));

% Downsample the image with specified scale
p_scale = 60; % percent of original size
w = int32(size(image,2) * (p_scale / 100));
h = int32(size(image,1) * (p_scale / 100));
dim = [h,w];

% Resizing the input image
img = imresize(image,dim,'method','bilinear');

% Intialize parameters of the image
rows = size(img,1);
columns = size(img,2);
n_channels = size(img,3);
n_pixels = rows * columns;
n_features = 5;

% Create raw feature vector
raw_FV = zeros(n_pixels,n_features);
p = 1;
for r = 1:rows
    for c = 1:columns
        raw_FV(p,1) = r;
        raw_FV(p,2) = c;
        for ch = 1:n_channels
            raw_FV(p,ch + 2) = img(r,c,ch);
        end
        p = p + 1;
    end
end

% Normalize the feature values
raw_FV = normalize(raw_FV);

%Split data into 10 folds
n_folds = 10;
f_size = floor(n_pixels/n_folds);
Xfolds = zeros(f_size,n_features,n_folds);
for f = 1:n_folds
    fold_start = (f-1)*f_size + 1;
    fold_end = f*f_size;
    Xfolds(:,:,f) = raw_FV(fold_start:fold_end,:);
end
avg_LL = [];
n_comp = 1:6;

% Test model orders ranging from 1 to 6
for n = n_comp
    % Initialize a list to store the log-likelihoods for each fold
    comp_LL = zeros(n_folds,1);
    for fold = 1:n_folds
        % Get training and validation sets
        Xval_fold = squeeze(Xfolds(:,:,f));
        t_idx = [1:(f-1)*f_size,f*f_size+1:n_pixels];
        Xtrain_fold = raw_FV(t_idx,:);

        % Fit the GMM using the EM algorithm
        gmdist = fitgmdist(Xtrain_fold,n,'RegularizationValue',0.01,'ProbabilityTolerance',1e-6,'Options',statset('MaxIter',3000));

        % Calculate the log-likelihood of the validation set
        LL = sum(log(pdf(gmdist,Xval_fold)),'all');

        % Append the log-likelihood to the list for this fold
        comp_LL(fold) = LL;
    end

    % Calculate the average log-likelihood across all K folds
    avg_comp_LL = mean(comp_LL);
    avg_LL(n) = avg_comp_LL;
end


% Plot Log Likelihood Score with respect to number of components in each model
plot(n_comp,avg_LL);
xlabel('Number of Components');
ylabel('Average Log Likelihood');

% Final Model Fitting
[~,max_ll] = max(avg_LL);
best_n_comp = n_comp(max_ll);
best_gmdist_model = fitgmdist(raw_FV,best_n_comp,'RegularizationValue',0.01,'ProbabilityTolerance',1e-6,'Options',statset('MaxIter',3000));

% Compute class post using model weights and conditional probabilties
post = zeros(best_n_comp,n_pixels);
for i = 1:best_n_comp
    PDF = mvnpdf(raw_FV,best_gmdist_model.mu(i,:),best_gmdist_model.Sigma(:,:,i));
    post(i,:) = (best_gmdist_model.PComponents(i) * PDF)';
end

% Decide label for each pixel with maximum posterior value
[~,seg_image] = max(post);

% Reshape seg_image to the shape of the input image
seg_image = reshape(seg_image,[columns,rows]);
seg_image = seg_image';

% Display the segmented image
figure;
imshow(seg_image,[]);
seg_image = label2rgb(seg_image);
figure; 
imshow(seg_image,[]);