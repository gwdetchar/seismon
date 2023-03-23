function [label,score] = predictLOCKLOSS(site,data)
%predictLOCKLOSS Classify using SVM Model 
%  predictLOCKLOSS classifies the measurements in data 
%  using the SVM model in the file trainedClassifier.mat, and then 
%  returns class labels in label and corresponding score 
% Author: Nikhil Mukund Menon (26-Feb-2017)
% EMail: nikhil@iucaa.in
% Uses  : Matlab 2016b

if strcmp(site,'lho')
    trainedClassifier = 'optim_mdlSVM_lho.mat';
elseif strcmp(site,'llo')
    trainedClassifier = 'optim_mdlSVM_llo.mat';
end
    
Mdl = load(trainedClassifier);
mdlSVM = Mdl.mdlSVM;


[label,score] = predict(mdlSVM,data); 
end