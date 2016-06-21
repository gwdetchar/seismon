
set(0,'DefaultAxesFontSize',20);
set(0,'DefaultTextFontSize',20);

site = 'LHO';
lho = load(['./plots/lockloss_' site '.mat']);
eqs_lho = load(sprintf('data/%s_analysis_locks.txt',site));
site = 'LLO';
llo = load(['./plots/lockloss_' site '.mat']);
eqs_llo = load(sprintf('data/%s_analysis_locks.txt',site));

figure;
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
clf
semilogx(10.^lho.peakampcut,lho.flagscutsum,'kx')
hold on
semilogx(10.^llo.peakampcut,llo.flagscutsum,'go')
hold off
grid
%caxis([-6 -3])
xlim([min(10.^lho.peakampcut) 1e-4])
xlabel('Peak ground motion, log10 [m/s]')
ylabel('Lockloss Probability');
leg1 = legend({'LHO','LLO'},'Location','SouthEast');
%cb = colorbar;
%set(get(cb,'ylabel'),'String','Peak ground motion, log10 [m/s]')
saveas(gcf,['./plots/lockloss_vel.pdf'])

indexes = find(eqs_lho(:,18)~=-1);
eqs_lho_cut = eqs_lho(indexes,:);
indexes = find(eqs_llo(:,18)~=-1);
eqs_llo_cut = eqs_llo(indexes,:);

eqs_lho_timediff = eqs_lho_cut(:,18) - eqs_lho_cut(:,15); 
eqs_llo_timediff = eqs_llo_cut(:,18) - eqs_llo_cut(:,15);

bins = -3600:600:3600;
N_lho = histcounts(eqs_lho_timediff,bins)/length(eqs_lho_timediff);
N_llo = histcounts(eqs_llo_timediff,bins)/length(eqs_llo_timediff);
bins = (bins(1:end-1) + bins(2:end))/2;

figure;
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
clf
plot(bins,N_lho,'k')
hold on
plot(bins,N_llo,'g')
hold off
grid
%caxis([-6 -3])
%xlim([min(10.^lho.peakampcut) 1e-4])
xlabel('Difference in lockloss and peak ground velocity time [s]')
ylabel('Probability Density Function');
leg1 = legend({'LHO','LLO'},'Location','SouthEast');
%cb = colorbar;
%set(get(cb,'ylabel'),'String','Peak ground motion, log10 [m/s]')
saveas(gcf,['./plots/lockloss_timediff.pdf'])
