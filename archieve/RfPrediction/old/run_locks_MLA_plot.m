
set(0,'DefaultAxesFontSize',20);
set(0,'DefaultTextFontSize',20);

data_usgs = load('./data/lockloss_fap_usgs.mat');
data_all = load('./data/lockloss_fap_all.mat');

figure;
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
clf
plot(data_usgs.fap_lho,data_usgs.esp_lho,'kx')
hold on
plot(data_usgs.fap_llo,data_usgs.esp_llo,'gx')
plot(data_all.fap_lho,data_all.esp_lho,'ko')
plot(data_all.fap_llo,data_all.esp_llo,'go')
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

figure;
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
clf
errorbar(data_usgs.fap_lho_unique,data_usgs.esp_lho_mean,data_usgs.esp_lho_mean-data_usgs.esp_lho_min,data_usgs.esp_lho_max-data_usgs.esp_lho_mean,'k-')
hold on
errorbar(data_usgs.fap_llo_unique,data_usgs.esp_llo_mean,data_usgs.esp_llo_mean-data_usgs.esp_llo_min,data_usgs.esp_llo_max-data_usgs.esp_llo_mean,'g-.')
%errorbar(data_all.fap_lho_unique,data_all.esp_lho_mean,data_all.esp_lho_mean-data_all.esp_lho_min,data_all.esp_lho_max-data_all.esp_lho_mean,'k-')
%errorbar(data_all.fap_llo_unique,data_all.esp_llo_mean,data_all.esp_llo_mean-data_all.esp_llo_min,data_all.esp_llo_max-data_all.esp_llo_mean,'g-')
plot(linspace(0,1),linspace(0,1),'k--');
hold off
grid
xlim([-.05 1])
xlabel('False Alarm Probability')
ylabel('Efficiency Standard Probability');
leg1 = legend({'LHO','LLO'},'Location','SouthEast');
%cb = colorbar;
%set(get(cb,'ylabel'),'String','Peak ground motion, log10 [m/s]')
saveas(gcf,['./plots/lockloss_fap_errorbars.pdf'])
close;

fprintf('LHO (USGS only) & %.1f & %.1e & %.1e & %.1e & %.1e \\\\ \\hline \n',data_usgs.thetas_lho(1),data_usgs.thetas_lho(2),data_usgs.thetas_lho(3),data_usgs.thetas_lho(4),data_usgs.thetas_lho(5));
fprintf('LHO & %.1f & %.1e & %.1e & %.1e & %.1e \\\\ \\hline \n',data_all.thetas_lho(1),data_all.thetas_lho(2),data_all.thetas_lho(3),data_all.thetas_lho(4),data_all.thetas_lho(5));
fprintf('LLO (USGS only) & %.1f & %.1e & %.1e & %.1e & %.1e \\\\ \\hline \n',data_usgs.thetas_llo(1),data_usgs.thetas_llo(2),data_usgs.thetas_llo(3),data_usgs.thetas_llo(4),data_usgs.thetas_llo(5));
fprintf('LLO & %.1f & %.1e & %.1e & %.1e & %.1e \\\\ \\hline \n',data_all.thetas_llo(1),data_all.thetas_llo(2),data_all.thetas_llo(3),data_all.thetas_llo(4),data_all.thetas_llo(5));


