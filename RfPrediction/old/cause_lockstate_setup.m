LHOfile = load('data/LHO_O1_Z.txt');
thresh = 1.3e-6;
cut1 = find(LHOfile(:,15) > thresh);
LHOfile = LHOfile(cut1,:);
[~,indexes] = sort(LHOfile(:,15),'descend');
LHOfile = LHOfile(indexes,:);
fid = fopen('data/cut_LHO_O1_Z.txt','w')
fprintf(fid, '%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.5e %.1f %.1f %.1f %.1f %.1f %.1f %.5e\n',LHOfile(indexes,1),LHOfile(indexes,2),LHOfile(indexes,3),LHOfile(indexes,4),LHOfile(indexes,5),LHOfile(indexes,6),LHOfile(indexes,7),LHOfile(indexes,8),LHOfile(indexes,9),LHOfile(indexes,10),LHOfile(indexes,11),LHOfile(indexes,12),LHOfile(indexes,13),LHOfile(indexes,14),LHOfile(indexes,15));
fclose(fid);
