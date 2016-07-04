
set(0,'DefaultAxesFontSize',20);
set(0,'DefaultTextFontSize',20);

files = {'data/Virgo_VSR2.txt','data/Virgo_VSR4.txt','data/GEO_HF.txt','data/LHO_S5.txt','data/LHO_S6.txt','data/LHO_O1_Z.txt','data/LLO_S5.txt','data/LLO_S6.txt','data/LLO_O1_Z.txt'};

for ff = 1:length(files)
   eqfilename = files{ff};

   eqs = load(eqfilename);
   [~,indexes] = sort(eqs(:,16),'descend');
   eqs = eqs(indexes,:);

   filename = strrep(eqfilename,'.txt','_noverlap.txt');

   fid = fopen(filename,'w+')

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

   for ii = 1:length(eqs)
      fprintf(fid,'%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.5e %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.5e %.1f %.5e %.1f %.5e\n',eqs(ii,1),eqs(ii,2),eqs(ii,3),eqs(ii,4),eqs(ii,5),eqs(ii,6),eqs(ii,7),eqs(ii,8),eqs(ii,9),eqs(ii,10),eqs(ii,11),eqs(ii,12),eqs(ii,13),eqs(ii,14),eqs(ii,15),eqs(ii,16),eqs(ii,17),eqs(ii,18),eqs(ii,19),eqs(ii,20));
   end
   fclose(fid);
end

