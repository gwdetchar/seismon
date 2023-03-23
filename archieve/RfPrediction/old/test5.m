set(0,'DefaultAxesFontSize',20)
set(0,'DefaultTextFontSize',20)

ud = getenv('userprofile');
fs = [ud '\Documents\MATLAB\'];
sd = [ud '\Documents\gitrepo\seismon\RfPrediction\data\'];

file = load([sd  'LHO_O1_predicted_binary_Z.txt']);
sof = size(file);
vartest = {'pvelocity','magnitude','distance','depth'};
        
% _vartest_ is where the names of the parameters are stored.
        
%testvar = calibrate_data(file(1:fix(end/1.5),3:3+length(vartest)-1),length(vartest));
testvar = calibrate_data(file(1:fix(end/1.5),[3 4 5 6]),length(vartest));
for ind4 = 1:size(testvar,2)
    file(1:fix(end/1.5),2+ind4) = testvar(:,ind4);
end
loc = find(file(1:fix(end/1.5),sof(2))==0);
locloss = find(file(1:fix(end/1.5),sof(2))==1);
p_training_set = [];
p_training_set = [p_training_set file(1:fix(end/1.5),3:6)];
y_training_set = [file(1:fix(end/1.5),sof(2))];
thetas = glmfit(p_training_set(:,:),[y_training_set ones(length(y_training_set),1)],'binomial','link','logit');
ec_lock = [thetas(1)];
ec_lockloss = [thetas(1)];
ec_com = [];
ec_FAP = [];
ec_ESP = [];
ec_lock = [ec_lock + thetas(2) * p_training_set(loc,1) + thetas(3) * p_training_set(loc,2) + thetas(4) * p_training_set(loc,3) + thetas(5) * p_training_set(loc,4)];
ec_lockloss = [ec_lockloss + thetas(2) * p_training_set(locloss,1) + thetas(3) * p_training_set(locloss,2) + thetas(4) * p_training_set(locloss,3) + thetas(5) * p_training_set(locloss,4)];
ec_com = [ec_com ec_lock];
ec_com = [ec_com; ec_lockloss];
ec_cm = min(ec_com);
ec_cma = max(ec_com);
p_star = ec_cm:0.005:ec_cma;
for p = p_star
    count3 = length(find(ec_lockloss < p));
    count3 = count3/length(ec_lockloss);
    ec_FAP = [ec_FAP count3];
    count4 = length(find(ec_lock < p));
    count4 = count4/length(ec_lock);
    ec_ESP = [ec_ESP count4];
end
file2 = load([sd  'LHO_O1_binary_Z.txt']);
sof = size(file2);
vartest = {'velocity','accleration','displacement','magnitude','distance','depth'};
        
% _vartest_ is where the names of the parameters are stored.
        
%testvar = calibrate_data(file(1:fix(end/1.5),3:3+length(vartest)-1),length(vartest));
testvar = calibrate_data(file2(1:fix(end/1.5),(3:8)),length(vartest));
for ind4 = 1:size(testvar,2)
    file2(1:fix(end/1.5),2+ind4) = testvar(:,ind4);
end
loc = find(file2(1:fix(end/1.5),sof(2))==0);
locloss = find(file2(1:fix(end/1.5),sof(2))==1);
p_training_set = [];
p_training_set = [p_training_set file2(1:fix(end/1.5),3:8)];
y_training_set = [file2(1:fix(end/1.5),sof(2))];
thetas2 = glmfit(p_training_set(:,:),[y_training_set ones(length(y_training_set),1)],'binomial','link','logit');
ec2_lock = [thetas2(1)];
ec2_lockloss = [thetas2(1)];
ec2_com = [];
ec2_FAP = [];
ec2_ESP = [];
ec2_lock = [ec2_lock + thetas2(2) * p_training_set(loc,1) + thetas2(3) * p_training_set(loc,2) + thetas2(4) * p_training_set(loc,3) + thetas2(5) * p_training_set(loc,4) + thetas2(6) * p_training_set(loc,5) + thetas2(7) * p_training_set(loc,6)];
ec2_lockloss = [ec2_lockloss + thetas2(2) * p_training_set(locloss,1) + thetas2(3) * p_training_set(locloss,2) + thetas2(4) * p_training_set(locloss,3) + thetas2(5) * p_training_set(locloss,4) + thetas2(6) * p_training_set(locloss,5) + thetas2(7) * p_training_set(locloss,6)];
ec2_com = [ec2_com ec2_lock];
ec2_com = [ec2_com; ec2_lockloss];
ec2_cm = min(ec2_com);
ec2_cma = max(ec2_com);
p_star2 = ec2_cm:0.05:ec2_cma;

for p = p_star2
    count3 = length(find(ec2_lockloss < p));
    count3 = count3/length(ec2_lockloss);
    ec2_FAP = [ec2_FAP count3];
    count4 = length(find(ec2_lock < p));
    count4 = count4/length(ec2_lock);
    ec2_ESP = [ec2_ESP count4];
end

name = ['USGS Data with Predicted Velocity'];
name2 = ['All of the Real Parameters'];
figure('visible','off')
set(gcf,'PaperSize',[10 6])
set(gcf,'PaperPosition',[0 0 10 6])
linear = plot(linspace(0,1),linspace(0,1),'--');
set(linear,{'DisplayName'},{'x = y'})
hold on
ec = plot(ec_FAP,ec_ESP);
set(ec,{'DisplayName'},{[name]})
hold on
ec2 = plot(ec2_FAP,ec2_ESP,'-.');
set(ec2,{'DisplayName'},{[name2]})
legend('show','Location','southeast')
xlabel('False Alarm Probability')
ylabel('Efficiency Standard Probability')
saveas(gcf,[fs 'LHO_mat_Z_all_vs_predicted_USGS_FAP_ESP.pdf'])

