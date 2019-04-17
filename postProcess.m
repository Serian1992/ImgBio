%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file post-process the predicted mask by CNN to smooth the boundaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;clc;

bigFolderPath = 'C:\Users\luzix\Desktop\Zixiao_TissueSeg\cropped_32\';
load('C:\Users\luzix\Desktop\Zixiao_TissueSeg\code\imgList.mat');

for i=1:numel(imgList)
    tic;
    subPath = strcat(bigFolderPath, imgList{i}, '\');
    list = dir(strcat(subPath, '*.jpg'));
    for j=1:numel(list)
        maskName = strcat(subPath, replace(list(j).name, 'jpg', 'png'));
        if exist(maskName, 'file')
            mask = double(imread(maskName));
            temp = zeros(size(mask));
            temp(mask==2)=1;
            
            se = strel('disk', 10);
            temp = imdilate(temp, se);
            mask(temp==1) = 2;
            rgbMask = uint8(getRGBMask(mask - 1));
            
            imwrite(uint8(mask), strcat(subPath, replace(list(j).name, '.jpg', '_post.png')));
            imwrite(rgbMask, strcat(subPath, replace(list(j).name, '.jpg', '_rgb_post.png')));
        end
    end
    fprintf('Finish %i/%i\n', i, numel(imgList));
    toc;
end

%delete(pool);




