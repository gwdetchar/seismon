%% Run SVM for Lockloss Prediction from EQ Parameters
% Workflow
%     1) Select EQ Features
%     2) Artificially increase sample size
%     3) Equalize In-Lock/Out-of-Lock Cases
%     4) Normalize Features
%     5) Create Training and testing data sets
%     6) Use 10 fold cross validation
%     7) Select optimal Hyperparameters selected using Bayesian Optimization
%     8) Calculate SVM Model
%     9) Test on unseen data
% Author: Nikhil Mukund Menon (26-Feb-2017)
% EMail: nikhil@iucaa.in
% Uses  : Matlab 2016b

site  = 'llo';

SAVE_DATA = 0;
SAVE_MODEL = 1;  

Artificial_Samples = 1;
Make_Equal_Distribution = 1; % Makes samples so that In-Lock/Lock-Loss samples are similar in size
Multiply_factor = 10;        % Artificially increase samples [LLO:10, LHO:2]
Noise_level     = 0.01;      % Add noise to normalized parameters 
KFold            = 30; % KFold Cross Validatipon Factor [LLO:30, LHO:5]
Normalize       = 1;  % Normalize features


if strcmp(site,'lho')
    filename = './CLASSIFIER_ROC/H1_O1_O2.txt';
elseif strcmp(site,'llo')
    filename = './CLASSIFIER_ROC/L1_O1_O2.txt';
end

DATA =   load(filename);


% Select ID when IFO was locked
ID = find(logical(DATA(:,23)==1) + logical(DATA(:,23)==2));

% Select Predictor Variables
% 1: earthquake gps time
% 2: earthquake mag
% 3: p gps time
% 4: s gps time
% 5: r (2 km/s)
% 6: r (3.5 km/s)
% 7: r (5 km/s)
% 8: predicted ground motion (m/s)
% 9: lower bounding time
% 10: upper bounding time
% 11: latitude
% 12: longitude
% 13: distance
% 14: depth (m)
% 15: azimuth (deg) 
% 16: peak ground velocity gps time
% 17: peak ground velocity (m/s)
% 18: peak ground acceleration gps time
% 19: peak ground acceleration (m/s^2)
% 20: peak ground displacement gps time
% 21: peak ground displacement (m/s)
% 22: lockloss time (if available)
% 23: Detector Status

VAR = [2 8 13 14 15  ];
Nfeatures = numel(VAR);

pred = DATA(ID,VAR);

resp = logical(DATA(ID,23)-1);
pred_orig = pred;
resp_orig = resp;


%% Artificially increase sample size
if Artificial_Samples
pred = repmat(pred_orig,Multiply_factor,1);
resp = repmat(resp_orig,Multiply_factor,1);
pred_orig_boosted_size = pred;
resp_orig_boosted_size = resp;
pred = pred + Noise_level*repmat(std(pred_orig,1),size(pred,1),1).*randn(size(pred));
end



%% Adjust for unequal distribution 
if Make_Equal_Distribution
IDX = resp ==1;
data1 = pred(IDX,:);
data1_orig = data1;
data2 = resp(IDX,:);
Equal_Factor = round((length(pred) - 2*sum(IDX))/sum(IDX));
data1 = repmat(data1,Equal_Factor,1);
data2 = repmat(data2,Equal_Factor,1);
data1 = data1 + Noise_level*repmat(std(data1_orig,1),size(data1,1),1).*randn(size(data1));
pred  = [pred; data1 ];
resp  = [resp; data2 ];
end

%% Normalize Features
if Normalize
data = pred;
data = (data - repmat(min(data,[],1),size(data,1),1))*spdiags(1./(max(data,[],1)-min(data,[],1))',0,size(data,2),size(data,2)) ;
pred = data;
end

DATA = [pred resp];
RandID = randperm(length(DATA));
DATA = DATA(RandID,:);


% Divide to TRAIN data (70%) and TEST DATA(30%)
[trainInd,valInd,testInd] = dividerand(length(pred),0.7,0.005,0.295);
TRAIN_DATA = DATA(trainInd,:);
TEST_DATA  = DATA(testInd,:);



if SAVE_DATA ==1    
csvwrite('TRAIN_DATA.dat',TRAIN_DATA);
csvwrite('TEST_DATA.dat',TEST_DATA);
end

%load TRAIN_DATA.dat;
%load TEST_DATA.dat;

%% Start SVM Training
cdata = TRAIN_DATA(:,1:Nfeatures);
grp   = TRAIN_DATA(:,end);

cdata = cdata(:,1:Nfeatures);
cdata = bsxfun(@rdivide,...
    bsxfun(@minus,cdata,min(cdata,[],1)),...
    range(cdata,1));

% Cross Validation Partioning
c = cvpartition(length(grp),'KFold',KFold); 

% SVM options
opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
    'AcquisitionFunctionName','expected-improvement-plus');
svmmod = fitcsvm(cdata,grp,'KernelFunction','rbf',...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts,'Standardize',true);

lossnew = kfoldLoss(fitcsvm(cdata,grp,'CVPartition',c,'KernelFunction','rbf',...
    'BoxConstraint',svmmod.HyperparameterOptimizationResults.XAtMinObjective.BoxConstraint,...
    'KernelScale',svmmod.HyperparameterOptimizationResults.XAtMinObjective.KernelScale));


mdlSVM = fitPosterior(svmmod);
[~,score_svm] = resubPredict(mdlSVM);
resp = logical(grp);      
[Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(resp,score_svm(:,logical(mdlSVM.ClassNames)),'true');  

% Area under cruve
disp(['Areas under Curve : ', num2str(AUCsvm)]);

if SAVE_MODEL == 1
 save(sprintf('optim_mdlSVM_%s.mat',char(site)),'mdlSVM');
end

% Plot ROC Curve
figure(111)
plot(Xsvm,Ysvm,'linewidth',2);
xlabel('False positive rate');
ylabel('True positive rate'); 
title(sprintf('%s : ROC Curve for SVM Classification',char(upper(site))))
print( 'SVM_ROC','-dpdf','-r0','-bestfit');

% TEST on Unseen Data
data = TEST_DATA(:,1:Nfeatures);
data = (data - repmat(min(data,[],1),size(data,1),1))*spdiags(1./(max(data,[],1)-min(data,[],1))',0,size(data,2),size(data,2)) ;
[label,score] = predict(mdlSVM,data);
C = confusionmat(TEST_DATA(:,end),label); 
disp('Confusion Matrix for analysis on Unseen TEST_DATA ....');
disp(C)