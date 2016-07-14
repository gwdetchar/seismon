function [thetas] = logistic_regression_dx(p_training_set,y_training_set,num_of_parameters, ... 
    alpha_step_size,convergence_iterations)

%% Load Data
load('test3.mat','dir','site','fs')
%{
load /home/sebastien/Documents/Earthquake/Eric/data/LHO_O1_binary_X.txt

data=LHO_O1_binary_X;
acc=data(1:200,4);
vel=data(1:200,3);
di=data(1:200,5);
status=data(1:200,6);
%}

%% Sort vel from min to max

%{
sort_vel=[vel,status];
sort_vel=sortrows(sort_vel);

sort_acc=[acc,status];
sort_acc=sortrows(sort_acc);

sort_di=[di,status];
sort_di=sortrows(sort_di);
%}

%% Calibrate data
for ind = 1:num_of_parameters
    calibrat_p_training_set(:,ind) = (p_training_set(:,ind) ...
        - mean(p_training_set(:,ind))) ./ ...
        (std(p_training_set(:,ind)));
end
%aa=(p_training_set(:,2)-mean(p_training_set(:,2)))./(std(p_training_set(:,2)));
%bb=(acc(:,1)-mean(acc(:,1)))./(std(acc(:,1)));
%cc=(di(:,1)-mean(di(:,1)))./(std(di(:,1)));
%% Define matrices
x=[ones(length(calibrat_p_training_set),1) calibrat_p_training_set];
y=y_training_set;

thetas = zeros(num_of_parameters + 1,1);

%% Define hypothesis function
z = transpose(thetas)*transpose(x);

hh=1./(1+exp(-z));

%{
figure
plot(x(:,2),y,'o','Linewidth',5)
figure
semilogx(sort_vel(:,1),sort_vel(:,2),'xr','LineWidth',5)
%}

%% Gradient descent
alpha=alpha_step_size/length(y);
j = 1;
j
for i=1:num_of_parameters + 1
    ss(:,i)=(hh(:)-y(:)).*x(:,i);
end
for k=1:num_of_parameters + 1
    thetas(k)=thetas(k)-alpha*sum(ss(:,k));
end
    
z = x*thetas;
    
hh=1./(1+exp(-z));
for m = 1:num_of_parameters + 1
    temp(j,m)=thetas(m);
end
j = j + 1;
j
for i=1:num_of_parameters + 1
    ss(:,i)=(hh(:)-y(:)).*x(:,i);
end
for k=1:num_of_parameters + 1
    thetas(k)=thetas(k)-alpha*sum(ss(:,k));
end
    
z = x*thetas;
    
hh=1./(1+exp(-z));
for m = 1:num_of_parameters + 1
    temp(j,m)=thetas(m);
end
while abs(diff(temp(j,:)) - diff(temp(j-1,:))) > 1.0e-07
    j = j + 1;
    j
    for i=1:num_of_parameters + 1
        ss(:,i)=(hh(:)-y(:)).*x(:,i);
    end
    for k=1:num_of_parameters + 1
        thetas(k)=thetas(k)-alpha*sum(ss(:,k));
    end
    
    z = x*thetas;
    
    hh=1./(1+exp(-z));
    for m = 1:num_of_parameters + 1
        temp(j,m)=thetas(m);
    end
end

thetas=[temp(end,:)];

%% Convergence plot

figure('visible','off')
for indx2 = 1:size(temp,2)
convergence_plot =  plot(temp(:,indx2));
set(convergence_plot,{'DisplayName'},{['theta' num2str(indx2)]})
hold on
end
xlabel('j')
ylabel('temp')
title([char(site) ' ' dir ' Convergence Plots'])
legend('show')
if ispc
    saveas(gcf,[fs char(site) '\' char(site) '_mat_' dir '_convergence_plots.png'])
else
    saveas(gcf,[fs char(site) '_mat_' dir '_convergence_plots.png'])
end
% plot(temppp,'k')
% hold on
% plot(tempppp,'g')
% hold on

%{
legend('theta 0','theta 1','theta 2','theta3')

figure
plot(x(:,2),y,'x',...
    x(:,2),hh)
figure
plot(x(:,2),hh)

%% Test
vel=data(201:end,3);
acc=data(201:end,4);
di=data(201:end,5);
status=data(201:end,6);

sort_vel=[vel,status];
sort_vel=sortrows(sort_vel);

sort_acc=[acc,status];
sort_acc=sortrows(sort_acc);

sort_di=[di,status];
sort_di=sortrows(sort_di);


aa=(sort_vel(:,1)-mean(sort_vel(:,1)))./(std(sort_vel(:,1)));
bb=(sort_acc(:,1)-mean(sort_acc(:,1)))./(std(sort_acc(:,1)));
cc=(sort_di(:,1)-mean(sort_di(:,1)))./(std(sort_di(:,1)));

z=theta0+theta1.*aa;
hh=1./(1+exp(-z));
figure
plot(sort_vel(:,1),hh,'LineWidth',2)
hold on
plot(sort_vel(:,1),sort_vel(:,2),'xr','LineWidth',5)
grid on
%}
end