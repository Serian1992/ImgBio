clear;clc;

folder = 'C:\Users\luzix\Desktop\Zixiao_TissueSeg\CNNTestSet\';
load('C:\Users\luzix\Desktop\Zixiao_TissueSeg\code\imgList.mat');

for i=1:numel(imgList)
    txtSave = strcat(folder, imgList{i}, '\', 'val_FullImage_list.lst');
    if exist(txtSave, 'file')
        delete(txtSave);
    end
    contentList = dir(strcat(folder, imgList{i}, '\*.jpg')); 
    getTXTFile(txtSave, contentList);
    fprintf('Finish %i/%i\n',i,numel(imgList));
end
