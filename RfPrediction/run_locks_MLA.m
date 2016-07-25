
set(0,'DefaultAxesFontSize',20);
set(0,'DefaultTextFontSize',20);

site = 'LHO';
lho = load(['./plots/lockloss_' site '.mat']);
%eqs_lho = load(sprintf('data/%s_analysis_locks.txt',site));
eqs_lho = load(sprintf('data/%s_analysis_locks.txt',site));
site = 'LLO';
llo = load(['./plots/lockloss_' site '.mat']);
eqs_llo = load(sprintf('data/%s_analysis_locks.txt',site));

indexes_lho = find(eqs_lho(:,21) == 1 | eqs_lho(:,21) == 2);
eqs_lho = eqs_lho(indexes_lho,:);
indexes_llo = find(eqs_llo(:,21) == 1 | eqs_llo(:,21) == 2);
eqs_llo = eqs_llo(indexes_llo,:);

flags_lho = eqs_lho(:,21);
flags_lho(flags_lho == 1) = 0;
flags_lho(flags_lho == 2) = 1;

flags_llo = eqs_llo(:,21);
flags_llo(flags_llo == 1) = 0;
flags_llo(flags_llo == 2) = 1;

indexes_lho = find(eqs_lho(:,21) == 2);
indexes_llo = find(eqs_llo(:,21) == 2);
fprintf('LHO: %d, LLO: %d\n',length(indexes_lho),length(indexes_llo));

[nlho,nsize] = size(eqs_lho); [nllo,~] = size(eqs_llo);
nlho_train = floor(nlho/2);
nllo_train = floor(nllo/2);

vars_usgs = [2 8 13 14];
%vars_all = [2 8 13 14 16];
%vars_usgs = vars_all;

idx_lho = randperm(nlho);
idx_train_lho = idx_lho(1:nlho_train);
idx_test_lho = idx_lho(nlho_train+1:end);
%idx_test_lho = idx_lho(1:nlho_train);
thetas_lho = glmfit(eqs_lho(idx_train_lho,vars_usgs),[flags_lho(idx_train_lho) ones(size(idx_train_lho))'],'binomial','link','logit');

idx_llo = randperm(nllo);
idx_train_llo = idx_llo(1:nllo_train);
idx_test_llo = idx_llo(nllo_train+1:end);
idx_test_llo = idx_llo(1:nllo_train);
thetas_llo = glmfit(eqs_llo(idx_train_llo,vars_usgs),[flags_llo(idx_train_llo) ones(size(idx_train_llo))'],'binomial','link','logit');

pvals = 0:0.01:1;
z_lho = [];
for ii = 1:length(idx_test_lho)
   z_lho = [z_lho thetas_lho(1)+sum(thetas_lho(2:end).*eqs_lho(idx_test_lho(ii),vars_usgs)')];
end
hh_lho=1./(1+exp(-z_lho));
fap_lho = []; esp_lho = [];
for ii = 1:length(pvals)
   indexes1 = intersect(find(hh_lho<=pvals(ii)),find(flags_lho(idx_test_lho) == 1));
   indexes0 = intersect(find(hh_lho<=pvals(ii)),find(flags_lho(idx_test_lho) == 0));
   fap_lho(ii) = length(indexes1) / length(find(flags_lho(idx_test_lho) == 1));
   esp_lho(ii) = length(indexes0) / length(find(flags_lho(idx_test_lho) == 0));
end

z_llo = []; 
for ii = 1:length(idx_test_llo)
   z_llo = [z_llo thetas_llo(1)+sum(thetas_llo(2:end).*eqs_llo(idx_test_llo(ii),vars_usgs)')];
end
hh_llo=1./(1+exp(-z_llo));
fap_llo = []; esp_llo = [];
for ii = 1:length(pvals)
   indexes1 = intersect(find(hh_llo<=pvals(ii)),find(flags_llo(idx_test_llo) == 1));
   indexes0 = intersect(find(hh_llo<=pvals(ii)),find(flags_llo(idx_test_llo) == 0));
   fap_llo(ii) = length(indexes1) / length(find(flags_llo(idx_test_llo) == 1));
   esp_llo(ii) = length(indexes0) / length(find(flags_llo(idx_test_llo) == 0));
end

figure;
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
clf
plot(fap_lho,esp_lho,'kx')
hold on
plot(fap_llo,esp_llo,'go')
plot(linspace(0,1),linspace(0,1),'k--');
hold off
grid
%caxis([-6 -3])
xlabel('FAP')
ylabel('ESP');
leg1 = legend({'LHO','LLO'},'Location','SouthEast');
%cb = colorbar;
%set(get(cb,'ylabel'),'String','Peak ground motion, log10 [m/s]')
saveas(gcf,['./plots/lockloss_fap.pdf'])
close;


