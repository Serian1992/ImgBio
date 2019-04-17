function preProcessWSI(wsiPath , wsiName, backPath, savePath, varargin)
%PREPROCESSWSI pre-processes the WSI file based on the backMask.
%Crop patches of 40X resolution at patchSize and save them on the disk

% patchSize is default to be [720*4, 1128*4] = [2880,4512]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%patchSize = [2880,4512];
magnification = 40;

if nargin==5
    magnification = varargin{1};
elseif nargin>5
    error('Too many inputs!\n');
end

% Prepare the openslide library
openslide_load_library();
WSI = strcat(wsiPath, wsiName);
slidePtr = openslide_open(WSI);

% Load backMask
backMask = imread(strcat(backPath, replace(wsiName, '.svs', '_backMask.png')));
validInd = getValidBbox(backMask);
mag = 16 * magnification / 40;

% Transfer the valid index to magnification index
validInd(1,:) = round((max(validInd(1,:) - 1, 0)) * mag + 1);
validInd(2,:) = round((max(validInd(2,:) - 1, 0)) * mag + 1);
validInd(3,:) = validInd(3,:) * mag;
validInd(4,:) = validInd(4,:) * mag;
index = cell(size(validInd,2), 1);


for i=1:size(validInd,2)
    % Make sure the patches are not over the memory limit (10GB)
    chek = ceil(validInd(3,i)*validInd(4,i)*4*8 / (10*1024.^3));
    if validInd(3,i) < validInd(4,i)
        w_num = round(sqrt(chek));
        h_num = ceil(sqrt(chek));
    else
        w_num = ceil(sqrt(chek));
        h_num = round(sqrt(chek));
    end
    
    subind = zeros(4,w_num*h_num);
    count = 1;
    for ww = 1:w_num
        for hh = 1:h_num
            subind(1,count) = (ww-1)*round(validInd(3,i)/w_num) + validInd(1,i);
            subind(2,count) = (hh-1)*round(validInd(4,i)/h_num) + validInd(2,i);
            subind(3,count) = min(round(validInd(3,i)/w_num), validInd(1,i)+validInd(3,i)-subind(1,count));
            subind(4,count) = min(round(validInd(4,i)/h_num), validInd(2,i)+validInd(4,i)-subind(2,count));
            
            count = count+1;
        end
    end
    
    for cc = 1:size(subind,2)
        if ~exist([savePath, wsiName(1:end-4), '_', int2str(i), '_', int2str(cc), '.jpg'],'file')
            imargb = openslide_read_region(slidePtr, subind(1,cc), subind(2,cc), subind(3,cc), subind(4,cc), 0);
            imwrite(imargb, strcat(savePath, wsiName(1:end-4), '_', int2str(i), '_', int2str(cc), '.jpg'));
        end
    end
    
    index{i} = subind;
    
    
end
disp(size(validInd,2));
save(strcat(savePath, 'index.mat'), 'index');

% Unload library
openslide_close(slidePtr)
clear slidePtr
openslide_unload_library

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function validInd = getValidBbox(backMask)
mask = imfill(backMask, 'holes');
cc = bwconncomp(mask);
bbx = regionprops(cc, 'BoundingBox');

% Obtain the width of bbox, sort and pick out the valid bboxes
width = zeros(numel(bbx),1);
for kk=1:numel(bbx)
    width(kk) = bbx(kk).BoundingBox(3);
end
[~, ind] = sort(width, 'descend');
%tt = value > (value(1)/5);
ind = ind(1);
validInd = zeros(4,numel(ind));

% Assign the value of valid bboxes
for j=1:numel(ind)
    validInd(:,j) = bbx(ind(j)).BoundingBox;
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
