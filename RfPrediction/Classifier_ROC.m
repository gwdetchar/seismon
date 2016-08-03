%  Different Classifier Performance @  Lockloss Probability Predcition
% Nikhil (3/August/2016)
% Uses scores (resp.mat ) and labels(pred.mat)

site  = 'lho';

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

%% ROC Without Confidence Intervals
mdl = fitglm(pred,resp,'Distribution','binomial','Link','logit');
score_log = mdl.Fitted.Probability; % Probability estimates
[Xlog,Ylog,Tlog,AUClog] = perfcurve(resp,score_log,'true');
mdlSVM = fitcsvm(pred,resp,'Standardize',true);
mdlSVM = fitPosterior(mdlSVM);
[~,score_svm] = resubPredict(mdlSVM);
[Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(resp,score_svm(:,mdlSVM.ClassNames),'true');
mdlNB = fitcnb(pred,resp);
[~,score_nb] = resubPredict(mdlNB);
[Xnb,Ynb,Tnb,AUCnb] = perfcurve(resp,score_nb(:,mdlNB.ClassNames),'true');
AUClog
AUCsvm
AUCnb

plot(Xlog,Ylog)
hold on
plot(Xsvm,Ysvm)
plot(Xnb,Ynb)
xlim([-0.02,1.02]); ylim([-0.02,1.02]);
legend('Logistic Regression','Support Vector Machines','Naive Bayes','Location','Best')
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curves for Logistic Regression, SVM, and Naive Bayes Classification')
hold off

if exist('ROC_plots') ~= 7
mkdir ROC_plots
end

if strcmp(site,'llo')
saveas(gcf,['./plots/llo_lockloss_ROC_usgs.pdf']);
elseif strcmp(site,'lho')
saveas(gcf,['./plots/lho_lockloss_ROC_usgs.pdf']);
end

%% ROC with Pointwise Confidence Interval
mdl = fitglm(pred,resp,'Distribution','binomial','Link','logit');
score_log = mdl.Fitted.Probability; % Probability estimates
[Xlog,Ylog,Tlog,AUClog] = perfcurve(resp,score_log,'true','NBoot',1000,'TVals',0:0.05:1);
mdlSVM = fitcsvm(pred,resp,'Standardize',true);
mdlSVM = fitPosterior(mdlSVM);
[~,score_svm] = resubPredict(mdlSVM);
[Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(resp,score_svm(:,mdlSVM.ClassNames),'true','NBoot',1000,'TVals',0:0.05:1);
mdlNB = fitcnb(pred,resp);
[~,score_nb] = resubPredict(mdlNB);
[Xnb,Ynb,Tnb,AUCnb] = perfcurve(resp,score_nb(:,mdlNB.ClassNames),'true','NBoot',1000,'TVals',0:0.05:1);

errorbar(Xlog(:,1),Ylog(:,1),Ylog(:,1)-Ylog(:,2),Ylog(:,3)-Ylog(:,1));
hold on
errorbar(Xsvm(:,1),Ysvm(:,1),Ysvm(:,1)-Ysvm(:,2),Ysvm(:,3)-Ysvm(:,1));
errorbar(Xnb(:,1),Ynb(:,1),Ynb(:,1)-Ynb(:,2),Ynb(:,3)-Ynb(:,1));
xlim([-0.02,1.02]); ylim([-0.02,1.02]);
title('ROC Curve with Pointwise Confidence Bounds')
legend('Logistic Regression','Support Vector Machines','Naive Bayes','Location','Best')
xlabel('False positive rate'); ylabel('True positive rate');
hold off
if strcmp(site,'llo')
saveas(gcf,['./plots/llo_lockloss_ROC_usgs_cb.pdf']);
elseif strcmp(site,'lho')
saveas(gcf,['./plots/lho_lockloss_ROC_usgs_cb.pdf']);
end

AUClog
AUCsvm
AUCnb