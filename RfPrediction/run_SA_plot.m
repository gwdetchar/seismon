
set(0,'DefaultAxesFontSize',20);
set(0,'DefaultTextFontSize',20);

plotLocation = 'plots';
if ~exist(plotLocation)
   system(['mkdir -p ' plotLocation]);
end

filename = 'SA_ampRf/Site_LHO_Data_S5S6_Trial_1/Optim.mat';
lho = load(filename);
filename = 'SA_ampRf/Site_LLO_Data_S5S6_Trial_1/Optim.mat';
llo = load(filename);
filename = 'SA_ampRf/Site_Virgo_Data_VSR4_Trial_1/Optim.mat';
virgo = load(filename);
filename = 'SA_ampRf/Site_GEO_Data_HF_Trial_1/Optim.mat';
geo = load(filename);

filename = 'data/TABLE.mat';
sites = load(filename);
lho = sites.TABLE(2,:);
llo = sites.TABLE(7,:);
geo = sites.TABLE(1,:);
virgo = sites.TABLE(end,:);

fprintf('Detector & $a$ & $b$ & $c$ & $d$ \\\\ \\hline\n');
fprintf('%s & %.2f & %.2f & %.2f & %.2f \\\\ \\hline\n',upper('lho'),lho.Rf0,lho.Rfs,lho.cd,lho.rs);
fprintf('%s & %.2f & %.2f & %.2f & %.2f \\\\ \\hline\n',upper('llo'),llo.Rf0,llo.Rfs,llo.cd,llo.rs);
fprintf('%s & %.2f & %.2f & %.2f & %.2f \\\\ \\hline\n',upper('virgo'),virgo.Rf0,virgo.Rfs,virgo.cd,virgo.rs);
fprintf('%s & %.2f & %.2f & %.2f & %.2f \\\\ \\hline\n',upper('geo'),geo.Rf0,geo.Rfs,geo.cd,geo.rs);

%fprintf('%s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\ \\hline\n',upper('lho'),lho.Rf0,lho.Rfs,lho.Q0,lho.Qs,lho.cd,lho.ch,lho.rs);
%fprintf('%s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\ \\hline\n',upper('llo'),llo.Rf0,llo.Rfs,llo.Q0,llo.Qs,llo.cd,llo.ch,llo.rs);
%fprintf('%s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\ \\hline\n',upper('virgo'),virgo.Rf0,virgo.Rfs,virgo.Q0,virgo.Qs,virgo.cd,virgo.ch,virgo.rs);
%fprintf('%s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\ \\hline\n',upper('geo'),geo.Rf0,geo.Rfs,geo.Q0,geo.Qs,geo.cd,geo.ch,geo.rs);

filename = 'SA_ampRf/SiteUsed_LHO_DataUsed_S5S6_SitePred_LHO_DataPred_O1_Z_Trial_1/SA_Rfest_LHO_S5_S6_to_O1Z.mat';
lho_o1 = load(filename);
lho_o1 = lho_o1.SA_Rfest_LHO_S5_S6_to_O1Z;
filename = 'SA_ampRf/SiteUsed_LLO_DataUsed_S5S6_SitePred_LLO_DataPred_O1_Z_Trial_1/SA_Rfest_LLO_S5_S6_to_O1Z.mat';
llo_o1 = load(filename);
llo_o1 = llo_o1.SA_Rfest_LLO_S5_S6_to_O1Z;

M = lho_o1.magnitudes; r = lho_o1.r; h = lho_o1.depths;
Rf0 = lho_o1.Rf0_out; Rfs = lho_o1.Rfs_out;
cd = lho_o1.cd_out; rs = lho_o1.rs_out;
Rf0 = lho.Rf0; Rfs = lho.Rfs;
cd = lho.cd; rs = lho.rs;
peakamp = lho_o1.peakamp;
lho_Rf = ampRf(M,r,h,Rf0,Rfs,cd,rs);
diff_lho = max([(lho_Rf./peakamp)'; (peakamp./lho_Rf)']);

M = llo_o1.magnitudes; r = llo_o1.r; h = llo_o1.depths;
Rf0 = llo_o1.Rf0_out; Rfs = llo_o1.Rfs_out;
cd = llo_o1.cd_out; rs = llo_o1.rs_out;
Rf0 = llo.Rf0; Rfs = llo.Rfs;
cd = llo.cd; rs = llo.rs;
peakamp = llo_o1.peakamp;
llo_Rf = ampRf(M,r,h,Rf0,Rfs,cd,rs);
diff_llo = max([(llo_Rf./peakamp)'; (peakamp./llo_Rf)']);

bins = 1:1:10;
N_lho = histcounts(diff_lho,bins);
N_llo = histcounts(diff_llo,bins);
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
xlim([1 8])
xlabel('max(Rf / <Rf>, <Rf> / Rf)')
ylabel('Counts');
leg1 = legend({'LHO','LLO'},'Location','NorthEast');
saveas(gcf,[plotLocation '/pred_diff.pdf']);
close;

ms = linspace(4,8,11); dists = logspace(2,5,11);
[M,r] = meshgrid(ms,dists);

%M = lho_o1.magnitudes; r = lho_o1.r; h = lho_o1.depths;
h = 10.0;
Rf0 = lho_o1.Rf0_out; Rfs = lho_o1.Rfs_out;
cd = lho_o1.cd_out; rs = lho_o1.rs_out;
Rf0 = lho.Rf0; Rfs = lho.Rfs;
cd = lho.cd; rs = lho.rs;
lho_Rf = ampRf(M,r,h,Rf0,Rfs,cd,rs);

Rf0 = llo_o1.Rf0_out; Rfs = llo_o1.Rfs_out;
cd = llo_o1.cd_out; rs = llo_o1.rs_out;
Rf0 = llo.Rf0; Rfs = llo.Rfs;
cd = llo.cd; rs = llo.rs;
llo_Rf = ampRf(M,r,h,Rf0,Rfs,cd,rs);

% Frequency-time plot
figure;
set(gcf, 'PaperSize',[10 8])
set(gcf, 'PaperPosition', [0 0 10 8])
clf

hC = pcolor(M,r,log10(lho_Rf * 1e6));
set(gcf,'Renderer','zbuffer');
colormap(jet)
%shading interp

t = colorbar('peer',gca);
set(get(t,'ylabel'),'String','log10(Peak seismic velocity [\mum/s])');
set(gca,'xscale','lin','XLim',[4,8])
set(gca,'yscale','log','YLim',[100,1e5])
set(hC,'LineStyle','none');
grid

set(gca,'clim',[-3 6])
xlabel('Magnitude')
ylabel('Distance [km]');
print('-dpng',[plotLocation '/LHO_M_r.png']);
print('-depsc2',[plotLocation '/LHO_M_r.eps']);
print('-dpdf',[plotLocation '/LHO_M_r.pdf']);
close;

% Frequency-time plot
figure;
set(gcf, 'PaperSize',[10 8])
set(gcf, 'PaperPosition', [0 0 10 8])
clf

hC = pcolor(M,r,log10(llo_Rf * 1e6));
set(gcf,'Renderer','zbuffer');
colormap(jet)
%shading interp

t = colorbar('peer',gca);
set(get(t,'ylabel'),'String','log10(Peak seismic velocity [\mum/s])');
set(gca,'xscale','lin','XLim',[4,8])
set(gca,'yscale','log','YLim',[100,1e5])
set(hC,'LineStyle','none');
grid

set(gca,'clim',[-3 6])
xlabel('Magnitude')
ylabel('Distance [km]');
print('-dpng',[plotLocation '/LLO_M_r.png']);
print('-depsc2',[plotLocation '/LLO_M_r.eps']);
print('-dpdf',[plotLocation '/LLO_M_r.pdf']);
close;
