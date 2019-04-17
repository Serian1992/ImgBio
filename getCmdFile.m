%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file writes the system command into the cmd file, which is for the
% test of all the folders
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;clc;

fileName = 'TCGAGroupTest.cmd';

if exist(fileName, 'file')
    delete(fileName);
end
fid = fopen(fileName, 'a+');

folder = 'C:\Users\luzix\Desktop\Zixiao_TissueSeg\CNNTestSet\';
load('C:\Users\luzix\Desktop\Zixiao_TissueSeg\code\imgList.mat');

for i=1:numel(imgList)
    result_dir = strcat(folder, imgList{i}, '/');
    result_dir = replace(result_dir, '\', '/');
    
    image_list = strcat(result_dir, 'val_FullImage_list.lst');
    test_img_dir = replace(strcat(folder, imgList{i}), '\', '/');
    
    fprintf(fid, 'python CrossGroupTest_Main.py --result_dir %s --image_list %s --test_img_dir %s\n\n',result_dir, image_list, test_img_dir);
end
fprintf(fid, 'pause\n');


fclose(fid);




