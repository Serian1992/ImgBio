function getSmallPatches(bigPatchPath, savePath, varargin)
%getCNNTestPatches generate small patches for the CNN segmentation.
%Crop patches of 40X resolution at patchSize and save them on the disk

% patchSize is default to be [720*4, 1128*4] = [2880,4512]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

list = dir(strcat(bigPatchPath, '*.jpg'));
patchSize = [2880,4512];
if nargin==3
    patchSize = varargin{1};
elseif nargin>3
    error('Too many inputs!\n');
end

% Make sure the patchSize contains only two values
if numel(patchSize)~=2
    error('Unknown patchSize!\n');
end


for i=1:numel(list)
    imi = imread(strcat(bigPatchPath, list(i).name));
    width = size(imi,1);
    height = size(imi,2);
    
    if width>=patchSize(1) && height>=patchSize(2)
        
        % The stride along x and y axis is 720 and 1128, respectively.
        % This is to ensure the receptive field of CNN when test (not sure if this is correct)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        w_num = ceil((width-patchSize(1)) / (patchSize(1)-720) + 1);
        h_num = ceil((height-patchSize(2)) / (patchSize(2)-1128) + 1);             
        count = 1;
        
        for ww = 1:w_num
            for hh = 1:h_num
                x_start = min(width-patchSize(1)+1, 1+(ww-1)*(patchSize(1)-720));
                y_start = min(height-patchSize(2)+1, 1+(hh-1)*(patchSize(2)-1128));
                patch = imi(x_start:x_start+patchSize(1)-1, y_start:y_start+patchSize(2)-1, :);
                
                imwrite(patch, strcat(savePath, list(i).name(end-7:end-4), '_', int2str(count), '.jpg'));
                
                count = count+1;
            end
        end
    end
end





