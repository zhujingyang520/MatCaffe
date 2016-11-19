function transformed_image = preprocess_image(image)
% Preprocess the input image, including mean subtraction, channel reshape,
% and cropping. 
% 
% Input:
% - image: original image read by Matlab, of shape [Width*Height*Channel]
% 
% Return: 
% - transformed_image: transformed image after preprocess, of shape
% [Crop*Crop*Channel], where Channel is organized in BGR, and spatial
% dimension is ordered as width x height

CROPPED_DIM = 227;  % cropped dimension for CaffeNet

% Read mean from ImageNet image (as distributed with Caffe) for mean
% subtraction
caffe_root = '/home/jingyang/ProgramFiles/deep_learning/caffe';
mu = load([caffe_root '/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat']);
mu = mu.mean_data;
% average over pixels to obtain the mean (BGR) pixel values
mu = mean(mean(mu, 1), 2);
fprintf('mean-subtracted values: ');
fprintf('B: %f; G: %f; R: %f\n', mu(1), mu(2), mu(3));

% preprocess raw image 
transformed_image = image(:, :, [3 2 1]);   % RGB -> BGR
transformed_image = permute(transformed_image, [2 1 3]);    % flip width & height dimension
transformed_image = single(transformed_image);  % convert uint8 to single
transformed_image = imresize(transformed_image, [CROPPED_DIM CROPPED_DIM], 'bilinear'); % resize image 
transformed_image = bsxfun(@minus, transformed_image, mu);  % mean subtraction

