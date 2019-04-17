clear;clc;

list = dir('C:\Users\luzix\Desktop\Zixiao_TissueSeg\TIL\16_down\img\*.jpg');
savePath = 'C:\Users\luzix\Desktop\Zixiao_TissueSeg\TIL\16_down\backMask\';

for i=1:numel(list) 
    imi = imread([list(i).folder, '\', list(i).name]);
    mask = seperateBack(imi, 0);
    
    mask(mask==mask(1,1)) = 0;
    mask(mask~=0) = 1;
    
    imwrite(mask, [savePath, list(i).name(1:end-4), '_backMask.png']);
    fprintf('Finish %i/%i\n', i, numel(list));
end

imgList = struct2cell(list);
imgList = imgList(1,:);
imgList = imgList';
imgList = cellfun(@(x) x(1:end-4), imgList, 'UniformOutput', false);
save('C:\Users\luzix\Desktop\Zixiao_TissueSeg\TIL\imgList.mat', 'imgList');


    
