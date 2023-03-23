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


file1 = load('data/LLO_analysis_locks.txt');
index = find(file1(:,16) == 1 | file1(:,16) == 2);
indexcut = file1(index,16);
indexcut(indexcut == 1) = 0;
indexcut(indexcut == 2) = 1;
file1Unique = unique([file1(index,1) file1(index,3) file1(index,15) indexcut],'rows','stable');
index2 = find(Xunique(1:length(file1Unique),:) == file1Unique(:,:));
disp(length(Xunique))
disp(length(file1Unique))



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

