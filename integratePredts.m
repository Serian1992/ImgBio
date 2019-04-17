function integratePredts(bigPath,predtPath)
%INTEGRATEPREDTS integrates the small predicted pieces into a mask of the
% same size as big patch
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

imgList = dir(strcat(bigPath, '*.jpg'));
%maskList = dir(strcat(predtPath, '*.png'));
patchSize = [2880,4512];

for i=1:numel(imgList)
    info = imfinfo(strcat(bigPath, imgList(i).name));
    
    % The width and height stored in the info are opposite
    width = info.Height;
    height = info.Width;
    mask = zeros(width, height, 'uint8');
    
    if width>=patchSize(1) && height>=patchSize(2)
        w_num = ceil((width-patchSize(1)) / (patchSize(1)-720) + 1);
        h_num = ceil((height-patchSize(2)) / (patchSize(2)-1128) + 1);
        count = 1;
        
        for ww = 1:w_num
            for hh = 1:h_num
                patch = imread(strcat(predtPath, imgList(i).name(end-7:end-4), '_', int2str(count), '.png'));
                
                x_start = min(width-patchSize(1)+1, 1+(ww-1)*(patchSize(1)-720));
                y_start = min(height-patchSize(2)+1, 1+(hh-1)*(patchSize(2)-1128));
                
                mask(x_start:x_start+patchSize(1)-1, y_start:y_start+patchSize(2)-1) = patch;
                count = count+1;
            end
        end
        
        imwrite(uint8(mask), strcat(bigPath, replace(imgList(i).name, '.jpg', '.png')));
        rgbMask = uint8(getRGBMask(mask - 1));
        imwrite(rgbMask, strcat(bigPath, replace(imgList(i).name, '.jpg', '_rgb.png')));
    end
    
end
end

