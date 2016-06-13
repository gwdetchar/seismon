
set(0,'DefaultAxesFontSize',20);
set(0,'DefaultTextFontSize',20);

site = 'LHO';
lho = load(['./plots/lockloss_' site '.mat']);
site = 'LLO';
llo = load(['./plots/lockloss_' site '.mat']);

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

