function predictLOCKLOSS(mag,vel,distance,depth,azimuth,ifo,outfile)

%predictLOCKLOSS Classify using SVM Model 
%  predictLOCKLOSS classifies the measurements in data 
%  using the SVM model in the file trainedClassifier.mat, and then 
%  returns class labels in label and corresponding score 
% Author: Nikhil Mukund Menon (26-Feb-2017)
% EMail: nikhil@iucaa.in
% Uses  : Matlab 2016b

if isstr(mag)
   mag = str2num(mag);
end

if isstr(vel)
   vel = str2num(vel);
end

if isstr(distance)
   distance = str2num(distance);
end

if isstr(depth)
   depth = str2num(depth);
end

if isstr(azimuth)
   azimuth = str2num(azimuth);
end

if strcmp(ifo,'H1')
    trainedClassifier = 'optim_mdlSVM_lho.mat';
elseif strcmp(ifo,'L1')
    trainedClassifier = 'optim_mdlSVM_llo.mat';
else
    trainedClassifier = 'optim_mdlSVM_lho.mat';
end
    
Mdl = load(trainedClassifier);
mdlSVM = Mdl.mdlSVM;

data = [mag,vel,distance,depth,azimuth];

[label,score] = predict(mdlSVM,data); 

fid = fopen(outfile,'w+');
fprintf(fid,'%d %.5f\n',label,score(2));
fclose(fid);

end
