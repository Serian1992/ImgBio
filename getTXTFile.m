function getTXTFile(txtSave, contentList)
%GETTXTFILE generates a txt file as the CNN test list 
% txtSave is the path and name to save the txt file
% contentList must be a struct containing folder and name  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~isstruct(contentList)
    error('The contentList must be a struct!\n');
end

fid = fopen(txtSave, 'a+');

for i=1:numel(contentList)
    fileName = strcat(contentList(i).folder,'\',contentList(i).name);
    fprintf(fid, '%i	%s\n',i,fileName);
end

fclose(fid);   
end

