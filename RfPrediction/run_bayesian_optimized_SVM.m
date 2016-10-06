%% Run SVM for Lockloss Prediction from EQ Parameters
% Workflow
%     1) Normalize features
%     2) Artificially increase sample size
%     3) Equalize In-Lock/Out-of-Lock Cases
%     4) Select Relevant features via Neighbourhood component analysis (run_NCA.m)
%     5) Create Training and testing data sets
%     6) Use 10 fold cross validation
%     7) Select optimal Hyperparameters selected using Bayesian Optimization
%     8) Calculate SVM Model
%     9) Test on unseen data
% Author: Nikhil Mukund Menon (5-Oct-2016)
% Uses  : Matlab 2016b

site  = 'llo';

SAVE_DATA = 0;
SAVE_MODEL = 0; 
SAVE_FIG  = 1; 
Normalize = 1;
Artificial_Samples = 1;
Make_Equal_Distribution = 1; % Makes samples so that In-Lock/Lock-Loss samples are similar in size
Multiply_factor = 50;        % Artificially increase samples
Noise_level     = 0.01;      % Add noise to normalized parameters

cd ./data
if strcmp(site,'llo')
  load('llo_pred.mat'); % predictors
  load('llo_resp.mat'); % response
elseif strcmp(site,'lho')
  load('lho_pred.mat');
  load('lho_resp.mat');
end
cd ../

pred = PRED;

resp = logical(RESP);
pred_orig = pred;
resp_orig = resp;

%% Normalize Features
if Normalize
data = pred;
data = (data - repmat(min(data,[],1),size(data,1),1))*spdiags(1./(max(data,[],1)-min(data,[],1))',0,size(data,2),size(data,2)) ;
pred = data;
end


%% Artificially increase sample size
if Artificial_Samples
pred = repmat(pred_orig,Multiply_factor,1);
resp = repmat(resp_orig,Multiply_factor,1);
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



DATA = [pred resp];

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


cdata = TRAIN_DATA(:,1:3);
grp   = TRAIN_DATA(:,5);

cdata = cdata(:,1:3);
cdata = bsxfun(@rdivide,...
    bsxfun(@minus,cdata,min(cdata,[],1)),...
    range(cdata,1));

% Cross Validation Partioning
c = cvpartition(length(grp),'KFold',10); 

% SVM options
opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
    'AcquisitionFunctionName','expected-improvement-plus');
svmmod = fitcsvm(cdata,grp,'KernelFunction','rbf',...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);

lossnew = kfoldLoss(fitcsvm(cdata,grp,'CVPartition',c,'KernelFunction','rbf',...
    'BoxConstraint',svmmod.HyperparameterOptimizationResults.XAtMinObjective.BoxConstraint,...
    'KernelScale',svmmod.HyperparameterOptimizationResults.XAtMinObjective.KernelScale));


mdlSVM = fitPosterior(svmmod);
[~,score_svm] = resubPredict(mdlSVM);
resp = logical(grp);      
[Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(resp,score_svm(:,logical(mdlSVM.ClassNames)),'true');  

% Area under cruve
AUCsvm

if SAVE_MODEL == 1
 save('mdlSVM.mat','mdlSVM');
end

% Plot ROC Curve
if SAVE_FIG == 1
figure(111)
plot(Xsvm,Ysvm,'linewidth',2);
xlabel('False positive rate');
ylabel('True positive rate'); 
title('ROC Curve for SVM Classification')
print( 'SVM_ROC','-dpdf','-r0','-bestfit');
end


% TEST on Unseen Data
data = TEST_DATA(:,1:3);
data = (data - repmat(min(data,[],1),size(data,1),1))*spdiags(1./(max(data,[],1)-min(data,[],1))',0,size(data,2),size(data,2)) ;
[label,score] = predict(mdlSVM,data);
C = confusionmat(TEST_DATA(:,5),label)     
