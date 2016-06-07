
set(0,'DefaultAxesFontSize',20);
set(0,'DefaultTextFontSize',20);

site = 'LHO';
site = 'LLO';

if strcmp(site,'LHO')
   eqfilename = 'LHO_O1.txt';
   segfilename = 'segs_Locked_H_1126569617_1136649617.txt';
elseif strcmp(site,'LLO')
   eqfilename = 'LLO_O1.txt';
   segfilename = 'segs_Locked_L_1126569617_1136649617.txt';
end

eqs = load(eqfilename);
segments = load(segfilename);
locklosses = segments(:,2);

thresh = 1.3e-6;
cut1 = find(eqs(:,15) > thresh);
eqs = eqs(cut1,:);

peakamp = log10(eqs(:,15));
latitudes = eqs(:,11); longitudes = eqs(:,12); 
distances = eqs(:,13); magnitudes = eqs(:,2);

filename = sprintf('%s_analysis_locks.txt',site)
fid = fopen(filename,'w+')

total_locks = 0;
total_time = 0;
flags = [];

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

   if length(indexes) == 0
      flag = 0;
   else
      segs = segments(indexes,:);
      checkloss = find(segs(:,2) <= eqEnd);
      if length(checkloss) == 0
         flag = 1;
      else
         flag = 2;
      end
   end

   if flag == 2
      total_time = total_time + segments(indexes(checkloss(1))+1,1) - segments(indexes(checkloss(1)),2);
      total_locks = total_locks + 1;
   end

   fprintf(fid,'%.1f %.1f %.1f %.5e %d\n',eqs(ii,1),eqs(ii,2),eqs(ii,3),eqs(ii,15),flag);
   flags = [flags flag];
end
fclose(fid);
flags = flags';

fprintf('%d %.5f %d %.5f\n',total_locks,total_time/86400,length(segments),sum(segments(:,2)-segments(:,1))/86400);

indexes = find(flags == 1 | flags == 2);
peakampcut = peakamp(indexes);
flagscut = flags(indexes);
flagscut(flagscut == 1) = 0;
flagscut(flagscut == 2) = 1;
[peakampcut,ii] = sort(peakampcut,'descend');
flagscut = flagscut(ii);
flagsall = ones(size(flagscut));
flagscutsum = cumsum(flagscut) ./ cumsum(flagsall);
peakampcut = fliplr(peakampcut);
flagscutsum = fliplr(flagscutsum);

figure;
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
clf
plot(peakampcut,flagscutsum,'kx')
grid
%caxis([-6 -3])
xlabel('Peak ground motion, log10 [m/s]')
ylabel('Lockloss Probability');
%cb = colorbar;
%set(get(cb,'ylabel'),'String','Peak ground motion, log10 [m/s]')
saveas(gcf,['./plots/lockloss_vel_' site '.pdf'])

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
saveas(gcf,['./plots/lockloss_vel_distance_' site '.pdf'])

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
saveas(gcf,['./plots/lockloss_mag_distance_' site '.pdf'])

