
set(0,'DefaultAxesFontSize',20);
set(0,'DefaultTextFontSize',20);

site = 'LHO';
site = 'LLO';

if strcmp(site,'LHO')
   channelx = 'H1_ISI-GND_STS_HAM5_X_BLRMS_30M_100M';
   channely = 'H1_ISI-GND_STS_HAM5_Y_BLRMS_30M_100M';
   channelz = 'H1_ISI-GND_STS_HAM5_Z_BLRMS_30M_100M';

   segfilename = 'data/segs_Locked_H_1126569617_1136649617.txt';
elseif strcmp(site,'LLO')
   channelx = 'L1_ISI-GND_STS_HAM5_X_BLRMS_30M_100M';
   channely = 'L1_ISI-GND_STS_HAM5_Y_BLRMS_30M_100M';
   channelz = 'L1_ISI-GND_STS_HAM5_Z_BLRMS_30M_100M';

   segfilename = 'data/segs_Locked_L_1126569617_1136649617.txt';
end

segments = load(segfilename);
locklosses = segments(:,2);

filename = sprintf('data/%s_groundmotion_locks_RMS.txt',site)
fid = fopen(filename,'w+')

for jj = 1:length(segments)
   segStart = segments(jj,2) - 60; segEnd = segments(jj,2) + 60;
 
   xvelgps = -1; xvel = -1;
   yvelgps = -1; yvel = -1;
   zvelgps = -1; zvel = -1;
   xaccgps = -1; xacc = -1;
   yaccgps = -1; yacc = -1;
   zaccgps = -1; zacc = -1;
   xdispgps = -1; xdisp = -1;
   ydispgps = -1; ydisp = -1;
   zdispgps = -1; zdisp = -1;

   filename = sprintf('/home/mcoughlin/Seismon/Text_Files/Timeseries/%s/64/%d-%d.txt',channelx,segStart,segEnd);
   if exist(filename)
      data_out = load(filename);
      xvelgps = data_out(2,1); xvel = data_out(2,2);
   end
   filename = sprintf('/home/mcoughlin/Seismon/Text_Files/Timeseries/%s/64/%d-%d.txt',channely,segStart,segEnd);
   if exist(filename)
      data_out = load(filename);
      yvelgps = data_out(2,1); yvel = data_out(2,2);
   end
   filename = sprintf('/home/mcoughlin/Seismon/Text_Files/Timeseries/%s/64/%d-%d.txt',channelz,segStart,segEnd);
   if exist(filename)
      data_out = load(filename);
      zvelgps = data_out(2,1); zvel = data_out(2,2);
   end

   filename = sprintf('/home/mcoughlin/Seismon/Text_Files/Acceleration/%s/64/%d-%d.txt',channelx,segStart,segEnd);
   if exist(filename)
      data_out = load(filename);
      xaccgps = data_out(2,1); xacc = data_out(2,2);
   end

   filename = sprintf('/home/mcoughlin/Seismon/Text_Files/Acceleration/%s/64/%d-%d.txt',channely,segStart,segEnd);
   if exist(filename)
      data_out = load(filename);
      yaccgps = data_out(2,1); yacc = data_out(2,2);
   end

   filename = sprintf('/home/mcoughlin/Seismon/Text_Files/Acceleration/%s/64/%d-%d.txt',channelz,segStart,segEnd);
   if exist(filename)
      data_out = load(filename);
      zaccgps = data_out(2,1); zacc = data_out(2,2);
   end

  filename = sprintf('/home/mcoughlin/Seismon/Text_Files/Displacement/%s/64/%d-%d.txt',channelx,segStart,segEnd);
   if exist(filename)
      data_out = load(filename);
      xdispgps = data_out(2,1); xdisp = data_out(2,2);
   end

   filename = sprintf('/home/mcoughlin/Seismon/Text_Files/Displacement/%s/64/%d-%d.txt',channely,segStart,segEnd);
   if exist(filename)
      data_out = load(filename);
      ydispgps = data_out(2,1); ydisp = data_out(2,2);
   end

   filename = sprintf('/home/mcoughlin/Seismon/Text_Files/Displacement/%s/64/%d-%d.txt',channelz,segStart,segEnd);
   if exist(filename)
      data_out = load(filename);
      zdispgps = data_out(2,1); zdisp = data_out(2,2);
   end


   fprintf(fid,'%d %d %.1f %.5e %.1f %.5e %.1f %.5e %.1f %.5e %.1f %.5e %.1f %.5e %.1f %.5e %.1f %.5e %.1f %.5e\n',segStart,segEnd,xvelgps,xvel,yvelgps,yvel,zvelgps,zvel,xaccgps,xacc,yaccgps,yacc,zaccgps,zacc,xdispgps,xdisp,ydispgps,ydisp,zdispgps,zdisp);
end
fclose(fid);

filename = sprintf('data/%s_groundmotion_locks_RMS.txt',site)
data_out = load(filename);

tt = (data_out(:,1) - data_out(1,1) + 60)/86400;

figure;
set(gcf, 'PaperSize',[8 6])
set(gcf, 'PaperPosition', [0 0 8 6])
clf
semilogy(tt,data_out(:,4),'kx')
hold on
semilogy(tt,data_out(:,6),'gx')
semilogy(tt,data_out(:,8),'bx')
semilogy(tt,data_out(:,10),'ko')
semilogy(tt,data_out(:,12),'go')
semilogy(tt,data_out(:,14),'bo')
semilogy(tt,data_out(:,16),'k*')
semilogy(tt,data_out(:,18),'g*')
semilogy(tt,data_out(:,20),'b*')
hold off
grid
%caxis([-6 -3])
xlabel('Lockloss time [days]')
ylabel('Peak ground motion, acceleration, and displacement, [m/s, m/s^2, m]');
%cb = colorbar;
%set(get(cb,'ylabel'),'String','Peak ground motion, log10 [m/s]')
saveas(gcf,['./plots/lockloss_groundmotion_RMS_' site '.pdf'])

save(['./plots/lockloss_groundmotion_RMS_' site '.mat'])
