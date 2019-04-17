clear;clc;

smallPath = 'C:\Users\luzix\Desktop\Zixiao_TissueSeg\TIL\CNNTestSet\';
bigPath = 'C:\Users\luzix\Desktop\Zixiao_TissueSeg\TIL\cropped_32\';
load('C:\Users\luzix\Desktop\Zixiao_TissueSeg\TIL\imgList.mat');

for i=1:numel(imgList)
    tic;
    bigPatchPath = strcat(bigPath, '\', imgList{i}, '\');
    savePath = strcat(smallPath, imgList{i}, '\');
    if ~exist(savePath, 'dir')
        mkdir(savePath);
    end
    getSmallPatches_Cell(bigPatchPath, savePath);
    fprintf('Finish %i/%i\n', i, numel(imgList));
    toc;
end