n1 = normrnd(1.0e-06,1.0e-02,400,1);
h_plot = histfit(n1);
saveas(gcf,'/home/eric.coughlin/public_html/mat_test_hist.png')



