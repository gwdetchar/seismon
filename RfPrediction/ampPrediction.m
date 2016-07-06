
function LocklossPrediction(site)
%% Function to estmate lockloss probability from EQ Parameters
% First loads previously estimated (via Simulated Annealing) best fit parametres for the Rf amplitude
% prediction model. Then eatimates a best fit cutve to the observed ground
% motion to lockloss probability curve (via Smoothing Spline Fit). Finally
% converts the predicted ground motion induced by EQ to lockloss
% probability.

%% Load Data

if strcmp(site,'LHO')
lho = load(['./plots/lockloss_' site '.mat']);
% eqs_lho = load(sprintf('data/%s_analysis_locks.txt',site));
load('./data/SA_Rfest_LHO_O1_Z.mat');
Rfest_param = SA_Rfest_LHO_O1_Z;
pkdispcut_orig = 10.^lho.peakdispcut ; 
flagscutsumdisp_orig = lho.flagscutsumdisp;

elseif strcmp(site,'LLO')
llo = load(['./plots/lockloss_' site '.mat']);
% eqs_llo = load(sprintf('data/%s_analysis_locks.txt',site));
load('./data/SA_Rfest_LLO_O1_Z.mat');
Rfest_param = SA_Rfest_LLO_O1_Z;
pkdispcut_orig = 10.^llo.peakdispcut ; 
flagscutsumdisp_orig = llo.flagscutsumdisp;
end

pkdispcut = sort(pkdispcut_orig);
flagscutsumdisp = sort(flagscutsumdisp_orig);

% Symptote Lockloss probability to 1 for large ground motion
if strcmp(site,'LHO')
pkdispcut = [pkdispcut(1:end-1)' linspace(pkdispcut(end),1e-4,20) ];
flagscutsumdisp = [flagscutsumdisp(1:end-1)' linspace(flagscutsumdisp(end),1,20)];
end


%% Fit Curve

[xData, yData] = prepareCurveData( pkdispcut, flagscutsumdisp );

% Set up fittype and options.
ft = fittype( 'smoothingspline' );
opts = fitoptions( ft );
opts.SmoothingParam = 0.9999;
opts.Normalize = 'on';

% Fit model to data.
[curvefit, ~] = fit( xData, yData, ft, opts );


%% Parameters for Rfest
M     = Rfest_param.magnitudes;
r     = Rfest_param.distances;
h     = Rfest_param.depths;
Rf0   = Rfest_param.Rf0_out; 
Rfs   = Rfest_param.Rfs_out; 
Q0    = Rfest_param.Q0_out; 
Qs    = Rfest_param.Qs_out; 
cd    = Rfest_param.cd_out; 
ch    = Rfest_param.ch_out; 
rs    = Rfest_param.rs_out; 


%% Original ampRf code

% function Rf = ampRf(M,r,h,Rf0,Rfs,Q0,Qs,cd,ch)
% M = magnitude
% r = distance [km]
% h = depth [km]
%
% Rf0 = Rf amplitude parameter
% Rfs = exponent of power law for f-dependent Rf amplitude
% Q0 = Q-value of Earth for Rf waves at 1Hz
% Qs = exponent of power law for f-dependent Q
% cd = speed parameter for surface coupling  [km/s]
% ch = speed parameter for horizontal propagation  [km/s]
%
% exp(-2*pi*h.*fc./cd), coupling of source to surface waves
% exp(-2*pi*r.*fc./ch./Q), dissipation

fc = 10.^(2.3-M/2);
Q = Q0./fc.^Qs;
Af = Rf0./fc.^Rfs;

% Rf = Rf0.*(M./fc).*exp(-2*pi*h.*fc/c).*exp(-2*pi*r.*fc/c./Q)./r*1e-3;
Rfpred = M.*Af.*exp(-2*pi*h.*fc./cd).*exp(-2*pi*r.*fc./ch./Q)./r.^rs*1e-3;

%% EQ-Rfs to Lockloss Probabliity 

MaxRflevel = 1;
lockloss_Rfpred = curvefit(Rfpred);
Rfpred = Rfpred(Rfpred < MaxRflevel);
lockloss_Rfpred = lockloss_Rfpred(Rfpred < MaxRflevel);

%% Plotting

figure(111); clf; hold on
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
set(gcf,'color','w')
set(gca,'FontSize',25,'FontWeight','bold')
set(gca,'xscale','log')
set(gca,'yscale','log')

plot(curvefit,pkdispcut,flagscutsumdisp);
plot(Rfpred,lockloss_Rfpred,'ko','markersize',10,'linewidth',2)
xlabel('Peak ground displacement,  [m]','FontSize',25,'FontWeight','bold')
ylabel('Lockloss Probability','FontSize',25,'FontWeight','bold');
title( sprintf([site, ' O1Z']),'FontSize',25,'FontWeight','bold')
lgnd = legend('Data','Fitted Curve','Lockloss Predictions','Location','SouthEast');
set(lgnd,'FontSize',13)
axis tight
grid on; 


