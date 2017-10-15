

set(0,'DefaultAxesFontSize',22);
set(0,'DefaultTextFontSize',22);

folder = './data/';

site = 'LLO'; %LLO,LHO
data = 'O1_Z'; %S5,S6,O1_Z

datafile = load([folder site '_' data '.txt']);

% thresh = 1.3e-6;
% cut1 = find(datafile(:,15) > thresh);
cut1 = find(datafile(:,2) > 6);
% cut1 = find((datafile(:,2) < 6).*(datafile(:,15) > 3e-6));
datafile = datafile(cut1,:);
    
[magnitudes,iis] = sort(datafile(:,2));
% [peakamp,iis] = sort(datafile(:,16));
% magnitudes = datafile(iis,2);
peakamp = datafile(iis,16);
latitudes = datafile(iis,11);
longitudes = datafile(iis,12);
distances = datafile(iis,13) / 1000;
% distances = ones(size(magnitudes));
depths = datafile(iis,14);
% depths = zeros(size(magnitudes));
N = length(magnitudes);


%%

figure(1)
set(gcf, 'PaperSize',[13 9])
set(gcf, 'PaperPosition', [0 0 13 9])
clf
subplot(2,3,1)
hist(latitudes,30)
grid
xlabel('Latitude')
axis tight
subplot(2,3,2)
hist(longitudes,30)
grid
xlabel('Longitude')
axis tight
subplot(2,3,3)
hist(distances,30)
grid
xlabel('Distance [km]')
axis tight
subplot(2,3,4)
hist(magnitudes,30)
grid
xlabel('Magnitude')
axis tight
subplot(2,3,5)
hist(log10(depths),30)
grid
xlabel('Depth, log10  [km]')
axis tight
subplot(2,3,6)
hist(log10(peakamp),30)
grid
xlabel('Rf peak velocity, log10 [m/s]')
axis tight

% mkdir(['home/eric.coughlin/public_html/plots/' site '/' data '/']);
% saveas(gcf,['/home/eric.coughlin/public_html/plots/' site '/' data '/Histograms.png'])

%%
figure(2)
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
clf
scatter(distances,magnitudes,40,log10(peakamp),'filled','Marker','o','MarkerEdgeColor','k')
grid
caxis([-6 -3])
set(gca,'xticklabel',{'0','5000','10000','15000','20000'})
xlabel('Distance [km]')
ylabel('Magnitude')
cb = colorbar;
set(get(cb,'ylabel'),'String','Peak ground motion, log10 [m/s]')
% saveas(gcf,['/home/eric.coughlin/public_html/plots/' site '/' data '/Histo_ground_' site '.pdf'])

%%
optset =    optimset(        'TolFun', 1e-16, ...   % Termination tolerance on the function value
					              'TolX', 1e-16, ...     % Termination tolerance on x
							   'MaxIter',  5e6, ...     % Show the iteration steps ['iter' -> 'off']
                                                                'MaxFunEval', inf);

cost = @(x)norm(log10(ampRf(magnitudes,distances,depths,x(1),x(2),x(3),x(4),x(5),x(6),x(7))./peakamp));

% function Rf = ampRf(M,r,h, Rf0,Rfs,Q0,Qs,cd,ch,rs)
% [ov, res] = fminsearch(cost, 100*rand(1,7), optset);

% ov = [280.6703 1.113023 41.342 26.3735 404.9048 -8.4239e-017 1.652343]; %S5, LHO
ov = [183.61632 1.662635 7.1031912 -0.48252104 823.56256 -2246.8295 1.8035421]; %S5, LLO

disp('Best-fit parameters (Rf0,Rfs,Q0,Qs,cd,ch,rs):')
disp('   fc = 10.^(2.3-M/2)');
disp('   Q = Q0./fc.^Qs');
disp('   Af = Rf0./fc.^Rfs');
disp('   Rf = M.*Af.*exp(-2*pi*h.*fc./cd).*exp(-2*pi*r.*fc./ch./Q)./r.^rs*1e-3');
disp(num2str(ov'))

ii = 1:N;

%%
% res_rel = abs(peakamp-ampRf(magnitudes,distances,depths,ov(1),ov(2),ov(3),ov(4),ov(5),ov(6),ov(7))) ...
%     ./ampRf(magnitudes,distances,depths,ov(1),ov(2),ov(3),ov(4),ov(5),ov(6),ov(7));

acc_rel = max(peakamp./ampRf(magnitudes,distances,depths,ov(1),ov(2),ov(3),ov(4),ov(5),ov(6),ov(7)), ...
    ampRf(magnitudes,distances,depths,ov(1),ov(2),ov(3),ov(4),ov(5),ov(6),ov(7))./peakamp); %Ratio of predicted to actual ground motion amplitude

fac = 5;
ii2 = find(acc_rel<fac);
ii3 = find(acc_rel>=fac);

dotcolors = zeros(N,1);
dotcolors(ii2) = 1;
dotcolors(ii3) = 3;

figure(3)
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
clf
semilogy(ii,ampRf(magnitudes,distances,depths,ov(1),ov(2),ov(3),ov(4),ov(5),ov(6),ov(7))/1e-6,'rx','LineWidth',2)
hold on
scatter(ii,peakamp(ii)/1e-6,[],dotcolors(ii),'filled','LineWidth',2)
colormap('winter')
hold off
grid
ylabel('Peak Rf Motion [\mum/s]')
legend('Prediction','Data','Location','SouthEast')

% saveas(gcf,['./plots/Regression_' site '_' data '.pdf'])


figure(4)
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
clf
scatter(ii,acc_rel',[],dotcolors,'filled')
colormap('winter')
grid
ylabel('max(Rf/<Rf>,<Rf>/Rf)')
axis tight
set(gca,'yscale','log')
% set(gca,'xscale','log','yscale','log')
title([num2str(length(ii2)/N*100) '% within factor ' num2str(fac)])

%  saveas(gcf,['/home/eric.coughlin/public_html/plots/' site '/' data '/Accuracy_rel_F' num2str(fac) '_' site '.pdf'])

%%
facfac = linspace(1,10,100);
for k = 1:length(facfac) 
    ii = find(acc_rel<facfac(k));
    ratio(k) = length(ii)/N*100;
end

figure(5)
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
clf
plot(facfac,ratio,'LineWidth',3)
grid
set(gca,'xlim',[1 facfac(end)])
xlabel('Factor Interval')
ylabel('Percent of Captured EQs')

% saveas(gcf,['./plots/Captured_' site '_' data '.pdf'])
