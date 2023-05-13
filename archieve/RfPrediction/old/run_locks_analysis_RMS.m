
set(0,'DefaultAxesFontSize',20);
set(0,'DefaultTextFontSize',20);

site = 'LHO';
%site = 'LLO';

if strcmp(site,'LHO')
   eqfilename = 'data/LHO_O1_RMS_Z.txt';
   segfilename = 'data/segs_Locked_H_1126569617_1136649617.txt';
elseif strcmp(site,'LLO')
   eqfilename = 'data/LLO_O1_RMS_Z.txt';
   segfilename = 'data/segs_Locked_L_1126569617_1136649617.txt';
end

eqs = load(eqfilename);
segments = load(segfilename);
locklosses = segments(:,2);

thresh = 1.3e-6;
cut1 = find(eqs(:,16) > thresh);
%eqs = eqs(cut1,:);
[~,indexes] = sort(eqs(:,16),'descend');
eqs = eqs(indexes,:);

peakamp = log10(abs(eqs(:,16)));
peakacc = log10(abs(eqs(:,18)));
peakdisp = log10(abs(eqs(:,20)));
latitudes = eqs(:,11); longitudes = eqs(:,12); 
distances = eqs(:,13); magnitudes = eqs(:,2);

filename = sprintf('data/%s_analysis_locks_RMS.txt',site)
fid = fopen(filename,'w+')

total_locks = 0;
total_time = 0;
flags = [];

indexes = [];
for ii = 1:length(eqs)

   eq = eqs(ii,:);
   eqStart = eq(3); eqEnd = eq(7);
   eqPeakAmp = eq(end);

   over = 0;
   for jj = 1:length(indexes)
      eq2 = eqs(indexes(jj),:);
      eqStart2 = eq2(3); eqEnd2 = eq2(7);
      if sum(intersect(floor(eqStart):ceil(eqEnd),floor(eqStart2):ceil(eqEnd2))) > 0
         over = 1;
      end
   end
  
   if over == 0
      indexes = [indexes ii];
   end
end
eqs = eqs(indexes,:);
peakamp = log10(abs(eqs(:,16)));
peakacc = log10(abs(eqs(:,18)));
peakdisp = log10(abs(eqs(:,20)));
latitudes = eqs(:,11); longitudes = eqs(:,12); 
distances = eqs(:,13); magnitudes = eqs(:,2);

for ii = 1:length(eqs)

   eq = eqs(ii,:);
   eqStart = eq(3); eqEnd = eq(7);
   eqPeakAmp = eq(end);

   indexes = [];
   for jj = 1:length(segments)
      segStart = segments(jj,1); segEnd = segments(jj,2);
      if sum(intersect(floor(eqStart):ceil(eqEnd),floor(segStart):ceil(segEnd))) > 0
         indexes = [indexes jj];
      end
   end

   locklosstime = -1;
   if length(indexes) == 0
      flag = 0;
   else
      segs = segments(indexes,:);
      checkloss = find(segs(:,2) <= eqEnd);
      if length(checkloss) == 0
         flag = 1;
      else
         flag = 2;
         locklosstime = segs(checkloss(1),2);
      end
   end

   if flag == 2
      total_time = total_time + segments(indexes(checkloss(1))+1,1) - segments(indexes(checkloss(1)),2);
      total_locks = total_locks + 1;
   end

   fprintf(fid,'%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.5e %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.5e %.1f %.5e %.1f %.5e %d %d\n',eqs(ii,1),eqs(ii,2),eqs(ii,3),eqs(ii,4),eqs(ii,5),eqs(ii,6),eqs(ii,7),eqs(ii,8),eqs(ii,9),eqs(ii,10),eqs(ii,11),eqs(ii,12),eqs(ii,13),eqs(ii,14),eqs(ii,15),eqs(ii,16),eqs(ii,17),eqs(ii,18),eqs(ii,19),eqs(ii,20),flag,locklosstime);
   flags = [flags flag];
end
fclose(fid);
flags = flags';

fprintf('%d %.5f %d %.5f\n',total_locks,total_time/86400,length(segments),sum(segments(:,2)-segments(:,1))/86400);

indexes = find(flags == 1 | flags == 2);
peakampcut = peakamp(indexes);
peakacccut = peakacc(indexes);
peakdispcut = peakdisp(indexes);
flagscut = flags(indexes);
flagscut(flagscut == 1) = 0;
flagscut(flagscut == 2) = 1;

[peakampcut,ii] = sort(peakampcut,'descend');
flagscutvel = flagscut(ii);
flagsallvel = ones(size(flagscutvel));
flagscutsumvel = cumsum(flagscutvel) ./ cumsum(flagsallvel);
peakampcut = fliplr(peakampcut);
flagscutsumvel = fliplr(flagscutsumvel);

[peakacccut,ii] = sort(peakacccut,'descend');
flagscutacc = flagscut(ii);
flagsallacc = ones(size(flagscutacc));
flagscutsumacc = cumsum(flagscutacc) ./ cumsum(flagsallacc);
peakacccut = fliplr(peakacccut);
flagscutsumacc = fliplr(flagscutsumacc);

[peakdispcut,ii] = sort(peakdispcut,'descend');
flagscutdisp = flagscut(ii);
flagsalldisp = ones(size(flagscutdisp));
flagscutsumdisp = cumsum(flagscutdisp) ./ cumsum(flagsalldisp);
peakdispcut = fliplr(peakdispcut);
flagscutsumdisp = fliplr(flagscutsumdisp);


probs = [0.5 0.75 0.9 0.95];
[~,ii] = unique(flagscutsumvel);
flagscutsumvel_sort = flagscutsumvel(ii);
peakampcut_sort = peakampcut(ii);

thresholds = interp1(flagscutsumvel_sort,peakampcut_sort,probs);
for ii = 1:length(probs)
   fprintf('vel: %.2f %.5e\n',probs(ii),10.^thresholds(ii));
end

[~,ii] = unique(flagscutsumacc);
flagscutsumacc_sort = flagscutsumacc(ii);
peakacccut_sort = peakacccut(ii);

thresholds = interp1(flagscutsumacc_sort,peakacccut_sort,probs);
for ii = 1:length(probs)
   fprintf('acc: %.2f %.5e\n',probs(ii),10.^thresholds(ii));
end

[~,ii] = unique(flagscutsumdisp);
flagscutsumdisp_sort = flagscutsumdisp(ii);
peakdispcut_sort = peakdispcut(ii);

thresholds = interp1(flagscutsumdisp_sort,peakdispcut_sort,probs);
for ii = 1:length(probs)
   fprintf('disp: %.2f %.5e\n',probs(ii),10.^thresholds(ii));
end

figure;
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
clf
plot(peakampcut,flagscutsumvel,'kx')
grid
%caxis([-6 -3])
xlabel('Peak ground motion, log10 [m/s]')
ylabel('Lockloss Probability');
%cb = colorbar;
%set(get(cb,'ylabel'),'String','Peak ground motion, log10 [m/s]')
saveas(gcf,['./plots/lockloss_vel_RMS_' site '.pdf'])

figure;
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
clf
plot(peakacccut,flagscutsumacc,'kx')
grid
%caxis([-6 -3])
xlabel('Peak ground acceleration, log10 [m/s^2]')
ylabel('Lockloss Probability');
%cb = colorbar;
%set(get(cb,'ylabel'),'String','Peak ground motion, log10 [m/s]')
saveas(gcf,['./plots/lockloss_acc_RMS_' site '.pdf'])

figure;
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
clf
plot(peakdispcut,flagscutsumdisp,'kx')
grid
%caxis([-6 -3])
xlabel('Peak ground displacement, log10 [m]')
ylabel('Lockloss Probability');
%cb = colorbar;
%set(get(cb,'ylabel'),'String','Peak ground motion, log10 [m/s]')
saveas(gcf,['./plots/lockloss_disp_RMS_' site '.pdf'])

figure;
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
clf
indexes = find(flags == 0);
scatter(distances(indexes),peakamp(indexes),40,'b','filled','Marker','o','MarkerEdgeColor','k')
hold on
indexes = find(flags == 1);
scatter(distances(indexes),peakamp(indexes),40,'g','filled','Marker','o','MarkerEdgeColor','k')
indexes = find(flags == 2);
scatter(distances(indexes),peakamp(indexes),40,'r','filled','Marker','o','MarkerEdgeColor','k')
hold off
grid
%caxis([-6 -3])
set(gca,'xticklabel',{'0','5000','10000','15000','20000'})
xlabel('Distance [km]')
ylabel('Peak ground motion, log10 [m/s]')
%cb = colorbar;
%set(get(cb,'ylabel'),'String','Peak ground motion, log10 [m/s]')
saveas(gcf,['./plots/lockloss_vel_distance_RMS_' site '.pdf'])

figure;
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
clf
indexes = find(flags == 0);
scatter(distances(indexes),peakacc(indexes),40,'b','filled','Marker','o','MarkerEdgeColor','k')
hold on
indexes = find(flags == 1);
scatter(distances(indexes),peakacc(indexes),40,'g','filled','Marker','o','MarkerEdgeColor','k')
indexes = find(flags == 2);
scatter(distances(indexes),peakacc(indexes),40,'r','filled','Marker','o','MarkerEdgeColor','k')
hold off
grid
%caxis([-6 -3])
set(gca,'xticklabel',{'0','5000','10000','15000','20000'})
xlabel('Distance [km]')
ylabel('Peak ground acceleration, log10 [m/s^2]')
%cb = colorbar;
%set(get(cb,'ylabel'),'String','Peak ground motion, log10 [m/s]')
saveas(gcf,['./plots/lockloss_acc_distance_RMS_' site '.pdf'])

figure;
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
clf
indexes = find(flags == 0);
scatter(distances(indexes),peakdisp(indexes),40,'b','filled','Marker','o','MarkerEdgeColor','k')
hold on
indexes = find(flags == 1);
scatter(distances(indexes),peakdisp(indexes),40,'g','filled','Marker','o','MarkerEdgeColor','k')
indexes = find(flags == 2);
scatter(distances(indexes),peakdisp(indexes),40,'r','filled','Marker','o','MarkerEdgeColor','k')
hold off
grid
%caxis([-6 -3])
set(gca,'xticklabel',{'0','5000','10000','15000','20000'})
xlabel('Distance [km]')
ylabel('Peak ground displacement, log10 [m]')
%cb = colorbar;
%set(get(cb,'ylabel'),'String','Peak ground motion, log10 [m/s]')
saveas(gcf,['./plots/lockloss_disp_distance_RMS_' site '.pdf'])

figure;
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
clf
indexes = find(flags == 0);
scatter(distances(indexes),magnitudes(indexes),40,'b','filled','Marker','o','MarkerEdgeColor','k')
hold on
indexes = find(flags == 1);
scatter(distances(indexes),magnitudes(indexes),40,'g','filled','Marker','o','MarkerEdgeColor','k')
indexes = find(flags == 2);
scatter(distances(indexes),magnitudes(indexes),40,'r','filled','Marker','o','MarkerEdgeColor','k')
hold off
grid
%caxis([-6 -3])
set(gca,'xticklabel',{'0','5000','10000','15000','20000'})
xlabel('Distance [km]')
ylabel('Magnitude')
%cb = colorbar;
%set(get(cb,'ylabel'),'String','Peak ground motion, log10 [m/s]')
saveas(gcf,['./plots/lockloss_mag_distance_RMS_' site '.pdf'])

save(['./plots/lockloss_RMS_' site '.mat'])
