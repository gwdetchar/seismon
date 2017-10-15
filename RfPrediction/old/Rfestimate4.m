function Rfestimate4(threshS5input, threshS6input, threshO1input, siteinput)

set(0,'DefaultAxesFontSize',22);
set(0,'DefaultTextFontSize',22);

folder = '';

site = siteinput;

S5 = load([folder site '_S5.txt']);
S6 = load([folder site '_S6.txt']);
O1 = load([folder site '_O1.txt']);

threshS5 = threshS5input;
threshS6 = threshS6input;
threshO1 = threshO1input;
cut1 = find(S5(:,15) > threshS5);
S5 = S5(cut1,:);
cut2 = find(S6(:,15) > threshS6);
S6 = S6(cut2,:);
cut3 = find(O1(:,15) > threshO1);
O1 = O1(cut3,:);
    
[peakamp,iis] = sort([S5(:,15); S6(:,15); O1(:,15)]);
latitudes = [S5(:,11); S6(:,11); O1(:,11)];
latitudes = latitudes(iis);
longitudes = [S5(:,12); S6(:,12); O1(:,12)];
longitudes = longitudes(iis);
distances = [S5(:,13); S6(:,13); O1(:,13)] / 1000;
distances = distances(iis);
magnitudes = [S5(:,2); S6(:,2); O1(:,2)];
magnitudes = magnitudes(iis);
depths = zeros(size(magnitudes));
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

foldir = 'home/eric.coughlin/public_html/secondplots/';
mkdir(['/home/eric.coughlin/public_html/secondplots/' site]);
saveas(gcf,['/home/eric.coughlin/public_html/secondplots/' site '/Histograms.png'])

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
saveas(gcf,['/home/eric.coughlin/public_html/secondplots/' site '/Histo_ground_' site '.pdf'])

%%
optset =    optimset(        'TolFun', 1e-5, ...   % Termination tolerance on the function value
					              'TolX', 1e-5, ...     % Termination tolerance on x
							   'MaxIter',  5e4, ...     % Show the iteration steps ['iter' -> 'off']
                                                                'MaxFunEval',inf);

cost = @(x)norm(log10(ampRf(magnitudes,distances,depths,x(1),x(2),x(3),x(4),x(5),x(6),x(7))./peakamp));

% function Rf = ampRf(M,r,h, Rf0,Rfs,Q0,Qs,cd,ch,rs)
[ov, res] = fminsearch(cost, [10 1 1000 0 100 10 1], optset);

disp(num2str(ov'))

ii = 1:N;

%%
% res_rel = abs(peakamp-ampRf(magnitudes,distances,depths,ov(1),ov(2),ov(3),ov(4),ov(5),ov(6),ov(7))) ...
%     ./ampRf(magnitudes,distances,depths,ov(1),ov(2),ov(3),ov(4),ov(5),ov(6),ov(7));

acc_rel = max(peakamp./ampRf(magnitudes,distances,depths,ov(1),ov(2),ov(3),ov(4),ov(5),ov(6),ov(7)), ...
    ampRf(magnitudes,distances,depths,ov(1),ov(2),ov(3),ov(4),ov(5),ov(6),ov(7))./peakamp);

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
semilogy(ii,ampRf(magnitudes,distances,depths,ov(1),ov(2),ov(3),ov(4),ov(5),ov(6),ov(7))/1e-6,'gx','LineWidth',2)
hold on
scatter(ii,peakamp(ii)/1e-6,[],dotcolors(ii),'LineWidth',2)
colormap('jet')
hold off
grid
ylabel('Peak Rf Motion [\mum/s]')
legend('Prediction','Data','Location','SouthEast')

 saveas(gcf,['/home/eric.coughlin/public_html/secondplots/' site '/Regression_' site '.pdf'])


figure(4)
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
clf
scatter(ii,acc_rel',[],dotcolors,'filled')
colormap('jet')
grid
ylabel('max(Rf/<Rf>,<Rf>/Rf)')
axis tight
set(gca,'yscale','log')
% set(gca,'xscale','log','yscale','log')
title([num2str(length(ii2)/N*100) '% within factor ' num2str(fac)])

 saveas(gcf,['/home/eric.coughlin/public_html/secondplots/' site '/Accuracy_rel_F' num2str(fac) '_' site '.pdf'])

