function vis_square(data)
% Helper function to visualize n features or output features in a sqaure
% grid of size approx. sqrt(n) by sqrt(n). 
%
% Input:
% - data: of shape (width, height, 3, n) [RGB] or (width, height, n) [gray]

% sanity check
assert(ndims(data) == 3 || (ndims(data) == 4 && size(data, 3) == 3));

% normalize data to [0, 1]
data = (data - min(data(:))) / (max(data(:)) - min(data(:)));

% force the number of filters to be square
n = ceil(sqrt(size(data, ndims(data))));
if ndims(data) == 4
    % color images, pad along width, height, and feature number dimension
    padsize = [1 1 0 n^2-size(data, ndims(data))];
else
    % gray images 
    padsize = [1 1 n^2-size(data, ndims(data))];
end
data = padarray(data, padsize, 1, 'post');  % pad with 1 (white)

% tile the filters into a 2D image
if ndims(data) == 4
    % color images 
    data_reshape = reshape(data, size(data, 1), size(data, 2), 3, n, n);
    data_reshape = permute(data_reshape, [2 5 1 4 3]);
    data_reshape = reshape(data_reshape, n*size(data, 1), n*size(data, 2), 3);
else
    % gray images 
    data_reshape = reshape(data, size(data, 1), size(data,2), n, n);
    data_reshape = permute(data_reshape, [2 4 1 3]);
    data_reshape = reshape(data_reshape, n*size(data, 1), n*size(data, 2));
end

figure; imshow(data_reshape);


