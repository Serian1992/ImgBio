function mask = seperateBack(img, display)
%seperateBack seperates the white background from the tissue on the 16
%downsampled .jpg images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This method is a color-based segmentation using K-Means Clustering in
% Matlab Image Processing Toolbox
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Convert image from RGB color space to L*a*b color space
tic;
lab_he = rgb2lab(img);

% Classify the colors in 'a*b*' space using K-means Clustering
ab = lab_he(:,:,2:3);
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);

nColors = 3;
% repeat the clustering 3 times to avoid local minima
[cluster_idx, ~] = kmeans(ab,nColors,'distance','sqEuclidean','Replicates',3);

% Label Every pixel in the image using the results from KMEANS
mask = reshape(cluster_idx,nrows,ncols);

if display>0
    figure(299),clf
    subplot(2,1,1),imshow(img);
    subplot(2,1,2),imshow(mask,[]);
end

toc;

end

