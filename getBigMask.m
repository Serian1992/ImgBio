%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file generates the predicted map and rgbMask for all the big patches
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;clc;

bigFolderPath = 'C:\Users\luzix\Desktop\Zixiao_TissueSeg\cropped_32\';
smallFolderPath = 'C:\Users\luzix\Desktop\Zixiao_TissueSeg\CNNTestSet\';
load('C:\Users\luzix\Desktop\Zixiao_TissueSeg\code\imgList.mat');

for i=1:numel(imgList)
    bigPath = strcat(bigFolderPath, imgList{i}, '\');
    
    predtPath = strcat(smallFolderPath, imgList{i}, '\score\');
    
    tic;
    integratePredts(bigPath,predtPath);
    fprintf('Finish %i/%i\n',i, numel(imgList));
    toc;
end

