function rgbMask = getRGBMask(label)
%GETRGBMASK2 Summary of this function goes here

rgbMask = zeros(numel(label),3);
temp = label(:);
rgbMask(temp==1,1)=190;
rgbMask(temp==1,3)=2;
rgbMask(temp==2,2)=255;
rgbMask(temp==1,3)=1;
rgbMask = reshape(rgbMask,size(label,1),size(label,2),3);
rgbMask = uint8(rgbMask);


end

