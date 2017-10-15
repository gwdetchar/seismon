set(0,'DefaultAxesFontSize',20)
set(0,'DefaultTextFontSize',20)

ud = getenv('userprofile');
fs = [ud '\Documents\MATLAB\'];
sd = [ud '\Documents\gitrepo\seismon\RfPrediction\data\'];
load('seismon_logistic_regression.mat','s')
file = load([sd 'LHO_O1_predicted_binary_Z.txt']);
file2 = load([sd 'LHO_O1_predicted_binary_Y.txt']);
file3 = load([sd 'LHO_O1_predicted_binary_X.txt']);
p_ts = [];
for p = {'pvelocity'}
    for dir = ['Z']
        p_ts = [p_ts s.LHO.(dir).(char(p)).combined(:,1)];
    end
end
for p = {'magnitude' 'distance' 'depth'}
    for dir = ['Z']
        p_ts = [p_ts s.LHO.(dir).(char(p)).combined(:,1)];
    end
end
lockstatus_ts = file(1:fix(end/1.5),7);
y = lockstatus_ts;
alpha_step_size = 3;
thetas_z = glmfit(p_ts,[y ones(length(y),1)],'binomial','link','logit');
%[thetas_z, J] = logistic_regression_dx(p_ts,y,18,alpha_step_size);
z_velocity = file(fix(end/1.5):end,3:6);
z_velocity = calibrate_data(z_velocity,4);
y_velocity = file2(fix(end/1.5):end,3:6);
y_velocity = calibrate_data(y_velocity,4);
x_velocity = file3(fix(end/1.5):end,3:6);
x_velocity = calibrate_data(x_velocity,4);
p_ts2 = [z_velocity];
y = file(fix(end/1.5):end,7);
z = thetas_z(1) + thetas_z(2) * p_ts2(:,1) + thetas_z(3) * p_ts2(:,2) + thetas_z(4) * p_ts2(:,3) + ...
    thetas_z(5) * p_ts2(:,4);
hh=1./(1+exp(-z));

check_good_above = 0;
check_bad_above = 0;
check_good_below = 0;
check_bad_below = 0;

for i = 1:size(y_velocity,1)
    if y(i) == 1
        if hh(i) >= 0.5
            check_good_above = check_good_above + 1;
        elseif hh(i) < 0.5
            check_bad_above = check_bad_above + 1;
        end
    elseif y(i) == 0
        if hh(i) < 0.5
            check_good_below = check_good_below + 1;
        elseif hh(i) >= 0.5
            check_bad_below = check_bad_below + 1;
        end
    end
end
figure('visible','off')
set(gcf,'PaperSize',[10 6])
set(gcf,'PaperPosition',[0 0 10 6])
signomoid_curve = plot(z,hh,'o');
hold on
set(signomoid_curve,{'DisplayName'},{'z vs. h'})
prediction_line = plot(z,y,'*');
set(prediction_line,{'DisplayName'},{'z vs. y'})
str = ['good above: ' num2str(check_good_above) '/' num2str(length(y))];
title('LHO Z USGS with Predicted Velocity Lockloss Prediction')
text(.5,.80,str)
text(.5,.75,['bad above: ' num2str(check_bad_above) '/' num2str(length(y))])
text(.5,.70,['good below: ' num2str(check_good_below) '/' num2str(length(y))])
text(.5,.65,['bad below: ' num2str(check_bad_below) '/' num2str(length(y))])
legend('show')
saveas(gcf,[fs 'LHO_Z_USGS_predicted_velocity_Lockloss_Prediction.pdf'])