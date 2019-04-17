%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file computes the ratio of tumor and minimum convexArea, ratio of
% stroma and minimum convexArea
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;clc;

bigPath = 'C:\Users\luzix\Desktop\Zixiao_TissueSeg\cropped_32\';
load('C:\Users\luzix\Desktop\Zixiao_TissueSeg\code\imgList.mat');
savePath = 'C:\Users\luzix\Desktop\Zixiao_TissueSeg\';


for i=1:numel(imgList)
    tic;
    list = dir(strcat(bigPath, imgList{i}, '\*_rgb_post.png'));
    
    tumorCount = 0;
    stromaCount = 0;
    
    for j=1:numel(list)
        rgbMask = imread(strcat(list(j).folder, '\', list(j).name));
        
        % Find the tumor area
        t1 = (rgbMask(:,:,1)==190) .* (rgbMask(:,:,2)==0) .* (rgbMask(:,:,3)==1);
        tumorCount = tumorCount + numel(find(t1==1));
        % Find the stroma area
        t2 = (rgbMask(:,:,1)==0) .* (rgbMask(:,:,2)==255) .* (rgbMask(:,:,3)==0);
        stromaCount = stromaCount + numel(find(t2==1));
        
    end
    
    
    totalCount = tumorCount + stromaCount;
    ratio(i).id = imgList{i};
    ratio(i).tumorRatio = tumorCount/totalCount;
    ratio(i).stromaRatio = stromaCount/totalCount;
    
    toc;
    fprintf('Finish %i/%i\n', i, numel(imgList));
end

save([savePath, 'ratio.mat'], 'ratio');

    