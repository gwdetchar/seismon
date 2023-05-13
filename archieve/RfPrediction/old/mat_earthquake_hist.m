cla;
n1 = normrnd(1.0e-06,1.0e-02,400,1);
n2 = normrnd(1.0e-05,1.0e-02,400,1);
histfit(n1);
hold on
histfit(n2);
title('Velocity Histogram(generated data)')
saveas(gcf,'/home/eric.coughlin/public_html/mat_test_hist.png')
figure()
acceleration_n1 = diff(n1);
acceleration_n2 = diff(n2);
histfit(acceleration_n1)
hold on
histfit(acceleration_n2)
title('Acceleration Histogram(generated data)')
saveas(gcf,'/home/eric.coughlin/public_html/mat_test_acc_hist.png')
figure()
velocity_data = [n1, n2];
acceleration_data = [acceleration_n1, acceleration_n2]
hist3(velocity_data)
hold on 
hist3(acceleration_data)
saveas(gcf,'/home/eric.coughlin/public_html/mat_test_vel_acc_hist.epsc')

cla;
n1 = normrnd(1.0e-06,1.0e-02,400,1);
n2 = normrnd(1.0e-05,1.0e-02,400,1);
hn1 = histfit(n1);
nob = hn1(1)
v_star = 0:.01:max(n1);
set(hn1,{'DisplayName'},{'Locked'})
hold on
hn2 = histfit(n2);
set(hn2,{'DisplayName'},{'Lockloss'})
set(hn2(1),'facecolor','g'); set(hn2(2),'color','k')
hold on
hn3 = histfit(ratio_velocity);
set(hn3,{'DisplayName'},{'Ratio'})
delete(hn3(1))
set(hn3(2),'color','m')
title('Velocity Histogram(generated data)')
legend('show')
saveas(gcf,'C:\Users\ericcoug\Documents\MATLAB\mat_test_hist.png')
figure()
acceleration_n1 = diff(n1);
acceleration_n2 = diff(n2);
ratio_acceleration = acceleration_n1 ./ acceleration_n2;
han1 = histfit(acceleration_n1);
set(han1,{'DisplayName'},{'Locked'})
hold on
han2 = histfit(acceleration_n2);
set(han2,{'DisplayName'},{'Lockloss'})
set(han2(1),'facecolor','g'); set(han2(2),'color','k')
title('Acceleration Histogram(generated data)')
legend('show')
saveas(gcf,'C:\Users\ericcoug\Documents\MATLAB\mat_test_acc_hist.png')
figure()
lock_data = [n1(1:399), acceleration_n1];
lockloss_data = [n2(1:399), acceleration_n2];
lav = surf(lock_data,'EdgeColor','None');
colorbar
view(2);
title('Lock Velocity and Acceleration')
saveas(gcf,'C:\Users\ericcoug\Documents\MATLAB\mat_test_lock_vel_acc_hist.png')

figure()
lock_data = [n1(1:399), acceleration_n1];
lockloss_data = [n2(1:399), acceleration_n2];
llav = surf(lockloss_data,'EdgeColor','None');
colorbar
view(2);
title('Lockloss Velocity and Acceleration')
saveas(gcf,'C:\Users\ericcoug\Documents\MATLAB\mat_test_lockloss_vel_acc_hist.png')

