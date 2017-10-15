
set(0,'DefaultAxesFontSize',20);
set(0,'DefaultTextFontSize',20);

site = 'LHO';
site = 'LLO';

data_out = load(['./plots/lockloss_' site '.mat'])
eqs = data_out.eqs;
flags = data_out.flags;

indexes = find(flags == 1 | flags == 2);
eqs = eqs(indexes,:);
flags = flags(indexes);
flags(flags == 1) = 0;
flags(flags == 2) = 1;

train = 1:2:length(eqs);
test = 2:2:length(eqs);

eqs_train = eqs(train,:); flags_train = flags(train);
eqs_test = eqs(test,:); flags_test = flags(test);

params = [2 13 15];
%params = 15;

X_train = eqs_train(:,params);
Y_train = flags_train;

X_test = eqs_test(:,params);
Y_test = flags_test;

Mdl = classregtree(X_train,Y_train);
Y_test_pred = eval(Mdl, X_test);

figure;
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
clf
plot(Y_test,'ro')
hold on
plot(Y_test_pred, 'bx');
hold off
grid
%caxis([-6 -3])
%set(gca,'xticklabel',{'0','5000','10000','15000','20000'})
xlabel('Earthquake Number')
ylabel('Lockloss')
%cb = colorbar;
%set(get(cb,'ylabel'),'String','Peak ground motion, log10 [m/s]')
saveas(gcf,['./plots/lockloss_ml_' site '.pdf'])