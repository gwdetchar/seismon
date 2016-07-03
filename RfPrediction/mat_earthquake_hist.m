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

