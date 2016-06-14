LLO_file = load('data/LLO_O1_binary_Z.txt');
Xtrain = [LLO_file(1:end/2,1) LLO_file(1:end/2,2) LLO_file(1:end/2,3)];
Y = [LLO_file(:,4)];
Ytrain = [LLO_file(1:end/2,4)];
Xtest = [LLO_file(end/2:end,1) LLO_file(end/2:end,2) LLO_file(end/2:end,3)];
Mdl = classregtree(Xtrain,Ytrain,'Categorical',3,'MinParent',20,'Names',{'EQ','PW','PV'})
yfit = eval(Mdl, Xtest);
yfit == LLO_file(end/2:end,4);
figure
plot(Xtest, yfit, '--r', Xtest, LLO_file(end/2:end,4), 'xb');
title('LLO Test Data Vs. Actual Data')
xlabel('Xtest(second half)')
ylabel('y(predicted and actual)')
legend('predicted','actual')
saveas(gcf,'/home/eric.coughlin/public_html/LLO_comparision_test.png')

LHO_file = load('data/LHO_O1_binary_Z.txt');
Xtrain = [LHO_file(1:end/2,1) LHO_file(1:end/2,2) LHO_file(1:end/2,3)];
Y = [LHO_file(:,4)];
Ytrain = [LHO_file(1:end/2,4)];
Xtest = [LHO_file(end/2:end,1) LHO_file(end/2:end,2) LHO_file(end/2:end,3)];
Mdl = classregtree(Xtrain,Ytrain,'Categorical',3,'MinParent',20,'Names',{'EQ','PW','PV'})
yfit = eval(Mdl, Xtest);
yfit == LHO_file(end/2:end,4);
figure
plot(Xtest, yfit, 'r', Xtest, LHO_file(end/2:end,4), 'b');
title('LHO Test Data Vs. Actual Data')
xlabel('Xtest(second half)')
ylabel('y(predicted and actual)')
saveas(gcf,'/home/eric.coughlin/public_html/LHO_comparision_test.png')
