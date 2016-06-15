cla;
set(0,'DefaultAxesFontSize',20);
set(0,'DefaultTextFontSize',20);
LLO_file = load('data/LLO_O1_binary_Z.txt');
Xunique = [LLO_file(:,1) LLO_file(:,2) LLO_file(:,3) LLO_file(:,4)];
Xunique = unique(Xunique,'rows','stable');
Xtrain = [Xunique(1:end/2,1) Xunique(1:end/2,2) Xunique(1:end/2,3)];
Y = [Xunique(:,4)];
Ytrain = [Y(1:end/2)];
Xtest = [Xunique(end/2:end,1) Xunique(end/2:end,2) Xunique(end/2:end,3)];
Mdl = classregtree(Xtrain,Ytrain);
yfit = eval(Mdl, Xtest);
yfit == Y(end/2:end);
figure
plot([1:length(Xtest)], yfit, 'or', [1:length(Xtest)], Y(end/2:end), 'xb');
title('LLO Test Data Vs. Actual Data')
xlabel('Xtest(second half)')
ylabel('y(predicted and actual)')
legend('predicted','actual')
saveas(gcf,'/home/eric.coughlin/gitrepo/seismon/RfPrediction/plots/LLO_comparision_test.png')
saveas(gcf,'/home/eric.coughlin/public_html/LLO_comparision_test.png')

LHO_file = load('data/LHO_O1_binary_Z.txt');
Xunique = [LHO_file(:,1) LHO_file(:,2) LHO_file(:,3) LHO_file(:,4)];
Xunique = unique(Xunique,'rows','stable');
Xtrain = [Xunique(1:end/2,1) Xunique(1:end/2,2) Xunique(1:end/2,3)];
Y = [Xunique(:,4)];
Ytrain = [Y(1:end/2)];
Xtest = [Xunique(end/2:end,1) Xunique(end/2:end,2) Xunique(end/2:end,3)];
Mdl = classregtree(Xtrain,Ytrain);
yfit = eval(Mdl, Xtest);
yfit == Y(end/2:end);
figure
plot([1:length(Xtest)], yfit, 'or', [1:length(Xtest)], Y(end/2:end), 'xb');
title('LHO Test Data Vs. Actual Data')
xlabel('Xtest(second half)')
ylabel('y(predicted and actual)')
legend('predicted','actual')
saveas(gcf,'/home/eric.coughlin/gitrepo/seismon/RfPrediction/plots/LHO_comparision_test.png')
saveas(gcf,'/home/eric.coughlin/public_html/LHO_comparision_test.png')

