
set(0,'DefaultAxesFontSize',20);
set(0,'DefaultTextFontSize',20);

site = 'LHO';
lho = load(['./plots/lockloss_' site '.mat']);
eqs_lho_orig = load(sprintf('data/%s_analysis_locks.txt',site));
eqs_lho = load(sprintf('WithPublishTime/%s_analysis_locks_with_pub_time.txt',site));
eqs_lho(:,1:22) = eqs_lho_orig(:,1:22);
site = 'LLO';
llo = load(['./plots/lockloss_' site '.mat']);
eqs_llo_orig = load(sprintf('data/%s_analysis_locks.txt',site));
eqs_llo = load(sprintf('WithPublishTime/%s_analysis_locks_with_pub_time.txt',site));
eqs_llo(:,1:22) = eqs_llo_orig(:,1:22);

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

eqs_lho_notice = eqs_lho(:,23);
eqs_llo_notice = eqs_llo(:,23);

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

eqs_lho_time_pde = eqs_lho(:,24);
eqs_lho_mag_pde = eqs_lho(:,25); 
eqs_lho_lat_pde = eqs_lho(:,26); eqs_lho_lon_pde = eqs_lho(:,27);
eqs_lho_depth_pde = eqs_lho(:,28);
eqs_lho_time_initial = eqs_lho(:,29);
eqs_lho_mag_initial = eqs_lho(:,30);
eqs_lho_lat_initial = eqs_lho(:,31); eqs_lho_lon_initial = eqs_lho(:,32);
eqs_lho_depth_initial = eqs_lho(:,33);
eqs_lho_est_timediff = abs(eqs_lho_time_initial - eqs_lho_time_pde);

eqs_llo_time_pde = eqs_llo(:,24);
eqs_llo_mag_pde = eqs_llo(:,25); 
eqs_llo_lat_pde = eqs_llo(:,26); eqs_llo_lon_pde = eqs_llo(:,27);
eqs_llo_depth_pde = eqs_llo(:,28);
eqs_llo_time_initial = eqs_llo(:,29);
eqs_llo_mag_initial = eqs_llo(:,30);
eqs_llo_lat_initial = eqs_llo(:,31); eqs_llo_lon_initial = eqs_llo(:,32);
eqs_llo_depth_initial = eqs_llo(:,33);
eqs_llo_est_timediff = abs(eqs_llo_time_initial - eqs_llo_time_pde);

filename = 'SA_ampRf/SiteUsed_LHO_DataUsed_S5S6_SitePred_LHO_DataPred_O1_Z_Trial_1/SA_Rfest_LHO_S5_S6_to_O1Z.mat';
lho_o1 = load(filename);
lho_o1 = lho_o1.SA_Rfest_LHO_S5_S6_to_O1Z;
filename = 'SA_ampRf/SiteUsed_LLO_DataUsed_S5S6_SitePred_LLO_DataPred_O1_Z_Trial_1/SA_Rfest_LLO_S5_S6_to_O1Z.mat';
llo_o1 = load(filename);
llo_o1 = llo_o1.SA_Rfest_LLO_S5_S6_to_O1Z;
%lho_o1 = llo_o1;

filename = 'data/TABLE.mat';
sites = load(filename);
lho_o1 = sites.TABLE(1,:);
llo_o1 = sites.TABLE(6,:);
%geo_o1 = sites.TABLE(1,:);
%virgo_o1 = sites.TABLE(end,:);

%keyboard

M = eqs_lho_mag_pde; h = eqs_lho_depth_pde; 
r = greatCircleDistance(deg2rad(46.6475), deg2rad(-119.5986), deg2rad(eqs_lho_lat_pde),deg2rad(eqs_lho_lon_pde));
%Rf0 = lho_o1.Rf0_out; Rfs = lho_o1.Rfs_out;
%cd = lho_o1.cd_out; rs = lho_o1.rs_out;
Rf0 = lho_o1.Rf0; Rfs = lho_o1.Rfs;
cd = lho_o1.cd; rs = lho_o1.rs;

Rf0
Rfs
cd
rs

%Rf0 = 1.263;
%Rfs = 0.821;
%cd  = 2610.348;
%rs = 0.996;

%Rf0 = 1.1345;
%Rfs = 1.1544;
%cd = -4754.6;
%rs = 1.0432;

%Rf0 = 0.37436;
%Rfs = 1.1617;
%cd = -4935.7;
%rs = 0.86596;

%Rf0 = 22.679;
%Rfs = 1.7628;
%cd = 507.89;
%rs = 1.4866;

%Rf0 = 0.3693;
%Rfs = 1.5193;
%cd = 1247.3;
%rs = 0.93732;

lho_Rf_pde = ampRf(M,r,h,Rf0,Rfs,cd,rs);

%indexes = find(M > 5.5 & M < 8.0);
%indexes = find(M > 5.5);
indexes = find(M > 5.5);

M = eqs_lho_mag_initial; h = eqs_lho_depth_initial;
r = greatCircleDistance(deg2rad(46.6475), deg2rad(-119.5986), deg2rad(eqs_lho_lat_initial),deg2rad(eqs_lho_lon_initial));
lho_Rf_initial = ampRf(M,r,h,Rf0,Rfs,cd,rs);

indexes = find(M > 5.5);
%indexes = find(M > 5.5 & M < 7.0);

lho_Rf = eqs_lho(indexes,20);
lho_Rf_pde = lho_Rf_pde(indexes);
lho_Rf_initial = lho_Rf_initial(indexes);

diff_lho = max([(lho_Rf_pde./lho_Rf_initial)'; (lho_Rf_initial./lho_Rf_pde)']);
diff_lho_pde = max([(lho_Rf_pde./lho_Rf)'; (lho_Rf./lho_Rf_pde)']);
diff_lho_initial = max([(lho_Rf_initial./lho_Rf)'; (lho_Rf./lho_Rf_initial)']);

[junk,index] = max(diff_lho_pde);
M = eqs_llo_mag_pde; h = eqs_llo_depth_pde;
r = greatCircleDistance(deg2rad(30.4986), deg2rad(-90.7483), deg2rad(eqs_llo_lat_pde),deg2rad(eqs_llo_lon_pde));
%Rf0 = llo_o1.Rf0_out; Rfs = llo_o1.Rfs_out;
%cd = llo_o1.cd_out; rs = llo_o1.rs_out;
Rf0 = llo_o1.Rf0; Rfs = llo_o1.Rfs;
cd = llo_o1.cd; rs = llo_o1.rs;

%Rf0 = 0.37436;
%Rfs = 1.1617;
%cd = -4935.7;
%rs = 0.86596;

%Rf0 = 0.3693;
%Rfs = 1.5193;
%cd = 1247.3;
%rs = 0.93732;



llo_Rf_pde = ampRf(M,r,h,Rf0,Rfs,cd,rs);

%indexes = find(M > 5.0 & M < 9.0);
indexes = find(M > 5.5);

M = eqs_llo_mag_initial; h = eqs_llo_depth_initial;
r = greatCircleDistance(deg2rad(30.4986), deg2rad(-90.7483), deg2rad(eqs_llo_lat_initial),deg2rad(eqs_llo_lon_initial));
llo_Rf_initial = ampRf(M,r,h,Rf0,Rfs,cd,rs);

indexes = find(M > 5.5);
%indexes = find(M > 5.5 & M < 7.0);

llo_Rf = eqs_llo(indexes,20);
llo_Rf_pde = llo_Rf_pde(indexes);
llo_Rf_initial = llo_Rf_initial(indexes);

diff_llo = max([(llo_Rf_pde./llo_Rf_initial)'; (llo_Rf_initial./llo_Rf_pde)']);
diff_llo_pde = max([(llo_Rf_pde./llo_Rf)'; (llo_Rf./llo_Rf_pde)']);
diff_llo_initial = max([(llo_Rf_initial./llo_Rf)'; (llo_Rf./llo_Rf_initial)']);

bins = 0.90:0.01:10;
N_lho = histcounts(diff_lho,bins)/length(diff_lho);
N_llo = histcounts(diff_llo,bins)/length(diff_llo);
N_lho = cumsum(N_lho); N_llo = cumsum(N_llo);

N_lho_pde = histcounts(diff_lho_pde,bins)/length(diff_lho_pde);
N_llo_pde = histcounts(diff_llo_pde,bins)/length(diff_llo_pde);
N_lho_pde = cumsum(N_lho_pde); N_llo_pde = cumsum(N_llo_pde);

N_lho_initial = histcounts(diff_lho_initial,bins)/length(diff_lho_initial);
N_llo_initial = histcounts(diff_llo_initial,bins)/length(diff_llo_initial);
N_lho_initial = cumsum(N_lho_initial); N_llo_initial = cumsum(N_llo_initial);

bins = (bins(1:end-1) + bins(2:end))/2;
[junk,index]=min(abs(bins-5));
fprintf('LHO Initial: %.5f\n',N_lho_initial(index));
fprintf('LHO Final: %.5f\n',N_lho_pde(index));
fprintf('LLO Initial: %.5f\n',N_llo_initial(index));
fprintf('LLO Final: %.5f\n',N_llo_pde(index));

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
xlabel('max(Rf / <Rf>, <Rf> / Rf)')
ylabel('Cumulative Density Function');
leg1 = legend({'LHO','LLO'},'Location','SouthEast');
xlim([1 5]);
%cb = colorbar;
%set(get(cb,'ylabel'),'String','Peak ground motion, log10 [m/s]')
saveas(gcf,['./plots/initial_vs_final.pdf'])
close;

figure;
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
clf
plot(bins,N_lho_pde,'k')
hold on
plot(bins,N_llo_pde,'g')
plot(bins,N_lho_initial,'k--')
plot(bins,N_llo_initial,'g--')
hold off
grid
%caxis([-6 -3])
%xlim([min(10.^lho.peakampcut) 1e-4])
xlabel('max(Rf / <Rf>, <Rf> / Rf)')
ylabel('Cumulative Density Function');
leg1 = legend({'LHO','LLO'},'Location','SouthEast');
xlim([1 10]);
%cb = colorbar;
%set(get(cb,'ylabel'),'String','Peak ground motion, log10 [m/s]')
saveas(gcf,['./plots/initial_final_vs_real.pdf'])
close;

figure;
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
clf
semilogy(llo_Rf,'kx')
hold on
semilogy(llo_Rf_pde,'k*')
semilogy(llo_Rf_initial,'ko')
semilogy(lho_Rf,'bx')
semilogy(lho_Rf_pde,'b*')
semilogy(lho_Rf_initial,'bo')
hold off
grid
%caxis([-6 -3])
%xlim([min(10.^lho.peakampcut) 1e-4])
%xlabel('max(Rf / <Rf>, <Rf> / Rf)')
%ylabel('Cumulative Density Function');
leg1 = legend({'LHO','LLO'},'Location','SouthEast');
%xlim([1 5]);
%cb = colorbar;
%set(get(cb,'ylabel'),'String','Peak ground motion, log10 [m/s]')
saveas(gcf,['./plots/initial_vs_final_vs_real.pdf'])
close;

figure;
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
clf
plot(eqs_llo_mag_initial,eqs_llo_mag_pde,'kx')
hold on
plot([4 9],[4 9],'k--');
hold off
grid
%caxis([-6 -3])
%xlim([min(10.^lho.peakampcut) 1e-4])
xlabel('Initial Magnitude')
ylabel('Final Magnitude');
%leg1 = legend({'LHO','LLO'},'Location','SouthEast');
xlim([4 9]);
ylim([4 9]);
%cb = colorbar;
%set(get(cb,'ylabel'),'String','Peak ground motion, log10 [m/s]')
saveas(gcf,['./plots/initial_vs_final_mag.pdf'])
close;

bins = 0:0.1:15;
N_lho = histcounts(eqs_lho_est_timediff,bins)/length(eqs_lho_est_timediff);
N_llo = histcounts(eqs_llo_est_timediff,bins)/length(eqs_llo_est_timediff);
bins = (bins(1:end-1) + bins(2:end))/2;
N_lho = cumsum(N_lho); N_llo = cumsum(N_llo);

figure;
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
clf
plot(bins,N_lho,'k')
%hold on
%plot(bins,N_llo,'g')
%hold off
grid
%caxis([-6 -3])
%xlim([min(10.^lho.peakampcut) 1e-4])
xlabel('Difference in initial and final earthquake time [s]')
ylabel('Cumulative Density Function');
%leg1 = legend({'LHO','LLO'},'Location','SouthEast');
%cb = colorbar;
%set(get(cb,'ylabel'),'String','Peak ground motion, log10 [m/s]')
saveas(gcf,['./plots/lockloss_est_timediff.pdf'])
close;


