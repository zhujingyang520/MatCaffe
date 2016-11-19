%% Main script for Matlab CaffeNet demonstration
% It demonstrates the usage of MatCaffe for classification, reproduce the
% python implementations: 
% https://github.com/BVLC/caffe/blob/master/examples/00-classification.ipynb

close all; clear; clc;

% Caffe installation file path. Add MatCaffe to the search path
caffe_root = '/home/jingyang/ProgramFiles/deep_learning/caffe';
matcaffe = [caffe_root '/matlab'];
addpath(matcaffe);

%% Load CaffeNet pre-trained model file 
% Make sure the trained weights & deploy protobuffer have been downloaded 
% the file path can NOT include '~', otherwise it can not been interpretted
model = './deploy.prototxt';
weights = './bvlc_reference_caffenet.caffemodel';

% Use CPU
caffe.set_mode_cpu();
% % For GPU, uncomment the following program
% caffe.set_mode_gpu();
% caffe.set_device(gpu_id); % gpu_id should be replaced by the actual value

%% Load network from pre-trained file 
fprintf('Load pre-trained CaffeNet ... \n');
net = caffe.Net(model, weights, 'test'); % create net and load weights

%% Load & preprocess the image for classification
image = imread('cat.jpg');
figure; imshow(image);
title('Image to be classified');
transformed_image = preprocess_image(image);    % preprocess image

%% Network feedforward (classification)
batch_size = 10;
% reshape input image of network into width, height, channel, batch_size
net.blobs('data').reshape([227 227 3 batch_size]);   
% replicate image along the batch dimension
input_data = repmat(transformed_image, [1 1 1 batch_size]);
% feedforward
fprintf('Conducting image classfication of batch size: %d ...\n', batch_size);
tic;
output = net.forward({input_data});
toc;
output_prob = output{1}(:, 1);  % extract the output probability
[~, class_idx] = max(output_prob);
fprintf('Predicted class is %d\n', class_idx);

%% Load ImageNet labels for interpretable
labels_file = [caffe_root '/data/ilsvrc12/synset_words.txt'];
fid = fopen(labels_file, 'r');
labels = textscan(fid, '%s', 'delimiter', '\t');
fclose(fid);
fprintf('Predicted label: %s\n', string(labels{1}(class_idx)));
% Top-5 predictions
[~, sort_idx] = sort(output_prob, 'descend');   % sort the probability
top5_idx = sort_idx(1:5);
fprintf('Top-5 predicted labels: %s\n', string(labels{1}(top5_idx)));

%% Examining intermediate output 
% For each layer, let's examine the activation shapes, of shape (width,
% height, channel_depth, batch_size). Note: it is reversed with Python, C++
% since Matlab adopts the column-major to store the matrix.
fprintf('shape of intermediate activations\n');
for i = 1 : length(net.blob_names)
    fprintf('%s\t', net.blob_names{i});
    print_1Darray(net.blobs(net.blob_names{i}).shape);
    fprintf('\n');
end

% Show CONV1 feature
conv1_feat = net.blobs('conv1').get_data();
vis_square(conv1_feat(:, :, 1:36, 1));
title('First 36 activations after CONV1');

% Show POOL5 feature
pool5_feat = net.blobs('pool5').get_data();
vis_square(pool5_feat(:, :, :, 1));
title('Activations after POOL5');


%% Examining parameters shape
% layers parameter is a 2D array, where [1] for weights, and [0] for biases
% Note: Matlab's start index is 1 instead of 0 in Python, C++
% The shape is of (filter_width, filter_height, input_channels,
% output_chaneels), which is reversed from Python, C++
fprintf('shape of layers parameters\n');
for i = 1 : length(net.layer_names)
    if strcmp(net.layers(net.layer_names{i}).type, 'Convolution') || ...
        strcmp(net.layers(net.layer_names{i}).type, 'InnerProduct')
        % Only CONV & FC contain parameters
        fprintf('%s\t', net.layer_names{i});
        fprintf('W: ');
        print_1Darray(net.layers(net.layer_names{i}).params(1).shape);
        fprintf('b: ');
        print_1Darray(net.layers(net.layer_names{i}).params(2).shape);
        fprintf('\n');
    end
end

% Show the CONV1 Kernel
vis_square(net.layers('conv1').params(1).get_data());
title('Kernels of CONV1 in CaffeNet');

%% Time evaluation on CPU
fprintf('Evaulate execution time on CPU ...\n');
% vary the batch sizes
batch_sizes = [1 2 4 6 8 10 20];
exe_times = zeros(size(batch_sizes));
for i = 1 : length(batch_sizes)
    fprintf('@ batch size %d: ', batch_sizes(i));
    net.blobs('data').reshape([227 227 3 batch_sizes(i)]); 
    input_data = repmat(transformed_image, [1 1 1 batch_sizes(i)]);
    func_handler = @() net.forward({input_data});
    exe_times(i) = timeit(func_handler);
    fprintf('%fs\n', exe_times(i));
end

figure; plot(batch_sizes, exe_times, 'o-', 'LineWidth', 2);
xlabel('Batch size'); ylabel('Execution time (s)');
title('Execution time of CaffeNet on CPU');
throughput = exe_times ./ batch_sizes;
fprintf('Execution time: Avg: %fs; Best: %fs; Worst: %fs\n', ...
    mean(throughput), min(throughput), max(throughput));

%% Clear nets and solvers
caffe.reset_all();