
set(0,'DefaultAxesFontSize',20);
set(0,'DefaultTextFontSize',20);

site = 'LHO';
lho = load(['./plots/lockloss_' site '.mat']);
%eqs_lho = load(sprintf('data/%s_analysis_locks.txt',site));
eqs_lho = load(sprintf('WithPublishTime/%s_analysis_locks_with_pub_time.txt',site));
site = 'LLO';
llo = load(['./plots/lockloss_' site '.mat']);
eqs_llo = load(sprintf('WithPublishTime/%s_analysis_locks_with_pub_time.txt',site));

figure;
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
clf
semilogx(10.^lho.peakampcut,lho.flagscutsumvel,'kx')
hold on
semilogx(10.^llo.peakampcut,llo.flagscutsumvel,'go')
hold off
grid
%caxis([-6 -3])
xlim([min([min(10.^lho.peakampcut) min(10.^llo.peakampcut)]) 1e-4])
xlabel('Peak ground motion, log10 [m/s]')
ylabel('Lockloss Probability');
leg1 = legend({'LHO','LLO'},'Location','SouthEast');
%cb = colorbar;
%set(get(cb,'ylabel'),'String','Peak ground motion, log10 [m/s]')
saveas(gcf,['./plots/lockloss_vel.pdf'])

figure;
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
clf
semilogx(10.^lho.peakacccut,lho.flagscutsumacc,'kx')
hold on
semilogx(10.^llo.peakacccut,llo.flagscutsumacc,'go')
hold off
grid
%caxis([-6 -3])
xlim([min([min(10.^lho.peakacccut) min(10.^llo.peakacccut)]) 1e-4])
xlabel('Peak ground acceleration, log10 [m/s^2]')
ylabel('Lockloss Probability');
leg1 = legend({'LHO','LLO'},'Location','SouthEast');
%cb = colorbar;
%set(get(cb,'ylabel'),'String','Peak ground motion, log10 [m/s]')
saveas(gcf,['./plots/lockloss_acc.pdf'])

figure;
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
clf
semilogx(10.^lho.peakdispcut,lho.flagscutsumdisp,'kx')
hold on
semilogx(10.^llo.peakdispcut,llo.flagscutsumdisp,'go')
hold off
grid
%caxis([-6 -3])
xlim([1e-6 1e-4])
xlabel('Peak ground displacement, log10 [m]')
ylabel('Lockloss Probability');
leg1 = legend({'LHO','LLO'},'Location','SouthEast');
%cb = colorbar;
%set(get(cb,'ylabel'),'String','Peak ground motion, log10 [m/s]')
saveas(gcf,['./plots/lockloss_disp.pdf'])

indexes = find(eqs_lho(:,22)~=-1);
eqs_lho_cut = eqs_lho(indexes,:);
indexes = find(eqs_llo(:,22)~=-1);
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
close;

eqs_lho_notice = eqs_lho(:,end);
eqs_llo_notice = eqs_llo(:,end);

bins = 0:50:1000;
N_lho = histcounts(eqs_lho_notice,bins)/length(eqs_lho_notice);
N_llo = histcounts(eqs_llo_notice,bins)/length(eqs_llo_notice);
bins = (bins(1:end-1) + bins(2:end))/2;
N_lho = cumsum(N_lho);

figure;
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
clf
plot(bins/60,N_lho,'k')
%hold on
%plot(bins,N_llo,'g')
%hold off
grid
%caxis([-6 -3])
%xlim([min(10.^lho.peakampcut) 1e-4])
xlabel('Notice latency [min]')
ylabel('Cumulative Density Function');
xlim([0 900/60]);
%leg1 = legend({'LHO','LLO'},'Location','SouthEast');
%cb = colorbar;
%set(get(cb,'ylabel'),'String','Peak ground motion, log10 [m/s]')
saveas(gcf,['./plots/earthquake_notice.pdf'])
close;

eqs_lho_notice = (eqs_lho(:,6)-eqs_lho(:,1)) - eqs_lho_notice;
eqs_llo_notice = (eqs_llo(:,6)-eqs_llo(:,1)) - eqs_llo_notice;

bins = 0:200:4800;
N_lho = histcounts(eqs_lho_notice,bins)/length(eqs_lho_notice);
N_llo = histcounts(eqs_llo_notice,bins)/length(eqs_llo_notice);
bins = (bins(1:end-1) + bins(2:end))/2;
N_lho = cumsum(N_lho); N_llo = cumsum(N_llo);

figure;
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
clf
plot(bins/60,N_lho,'k')
hold on
plot(bins/60,N_llo,'g')
hold off
grid
%caxis([-6 -3])
%xlim([min(10.^lho.peakampcut) 1e-4])
xlabel('Earthquake Notice [min]')
ylabel('Cumulative Density Function');
leg1 = legend({'LHO','LLO'},'Location','SouthEast');
xlim([0 4800/60]);
%cb = colorbar;
%set(get(cb,'ylabel'),'String','Peak ground motion, log10 [m/s]')
saveas(gcf,['./plots/lockloss_notice.pdf'])
close;

