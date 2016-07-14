cla;
alpha_step_size = 8;
convergence_iterations = 75000;
%% Checking operating system
if ispc
    ud = getenv('userprofile');
    fs = [ud '\Documents\MATLAB\'];
    sd = [ud '\Documents\gitrepo\seismon\RfPrediction\data\'];
elseif isunix
    ud = getenv('HOME');
    fs = [ud '/gitrepo/seismon/RfPrediction/plots/'];
    sd = [ud '/gitrepo/seismon/RfPrediction/data/'];
elseif ismac
    ud = getenv('HOME');
end
%% Actual script
for site = {'LHO' 'LLO'}
    %% checking existance for certain directories
    if ispc && ~exist([fs,char(site)],'dir')
        mkdir(fs,char(site))
        mkdir([fs '\' char(site)],'figures')
    end
    for dir = ['Z' 'Y' 'X']
        %% initialize variables
        s.(dir) = [];
        file = load([sd char(site) '_O1_binary_' dir '.txt']);
        sof = size(file);
        vartest = {'velocity','accleration','displacement','magnitude','distance','depth'};
        nop = length(vartest);
        vartest2 = {'combined','lockloss','lock'};
        si = 3;
        loc = find(file(:,sof(2))==0);
        locloss = find(file(:,sof(2))==1);
        %% creating structures for parameters
        for ind = 1:length(vartest)
            s.(dir).(vartest{ind}) = [];
            for ind2 = 1:length(vartest2)
                if strcmp(vartest2{ind2},'lockloss')
                    s.(dir).(vartest{ind}).(vartest2{ind2}) = [file(locloss,si) file(file==1)];
                elseif strcmp(vartest2{ind2},'lock')
                    s.(dir).(vartest{ind}).(vartest2{ind2}) = [file(loc,si) file(file==0)];
                else
                    s.(dir).(vartest{ind}).(vartest2{ind2}) = [file(:,si) file(:,sof(2))];
                end
            end
            si = si + 1;
        end
        %% calculating FAP and ESP numbers
        for ind = 1:length(vartest)
            s.(dir).(vartest{ind}).FAP = [];
            s.(dir).(vartest{ind}).ESP = [];
            p_lockloss = [];
            p_lockloss = s.(dir).(vartest{ind}).lockloss;
            p_lock = [];
            p_lock = s.(dir).(vartest{ind}).lock;
            if strcmp(vartest{ind},'velocity') || strcmp(vartest{ind},'accleration')
                p_star = 0:0.000001:max(s.(dir).(vartest{ind}).combined(:,1));
                s.(dir).(vartest{ind}).p_star = p_star;
            elseif strcmp(vartest{ind},'displacement')
                p_star = 0:0.000001:max(s.(dir).(vartest{ind}).combined(:,1));
                s.(dir).(vartest{ind}).p_star = p_star;
            elseif strcmp(vartest{ind},'distance')
                p_star = 0:10000:max(s.(dir).(vartest{ind}).combined(:,1));
                s.(dir).(vartest{ind}).p_star = p_star;
            else
                p_star = 0:0.1:max(s.(dir).(vartest{ind}).combined(:,1));
                s.(dir).(vartest{ind}).p_star = p_star;
            end
            %% plotting histograms
            figure('visible','off')
            if strcmp(vartest{ind},'velocity') || strcmp(vartest{ind},'acceleration') || strcmp(vartest{ind},'displacement')
                pledge = 0:0.5e-06:max(s.(dir).(vartest{ind}).combined(1,:));
            elseif strcmp(vartest{ind},'distance')
                pledge = min(s.(dir).(vartest{ind}).combined(1,:)):1.0e+05:max(s.(dir).(vartest{ind}).combined(1,:));
            else
                pledge = 0:0.1:max(s.(dir).(vartest{ind}).combined(1,:));
            end
            vllhf = histogram(p_lockloss(:,1),pledge);
            set(vllhf,{'DisplayName'},{'Lockloss'})
            hold on
            vlhf = histogram(p_lock(:,1),pledge);
            set(vlhf,{'DisplayName'},{'Locked'})
            title([char(site) ' ' dir ' ' vartest{ind} ' Locked and Lockloss histograms'])
            xlabel(vartest{ind})
            ylabel(['Count at ' vartest{ind} ' Bin'])
            legend('show')
            if ispc
                saveas(gcf, [fs char(site) '\' char(site) '_mat_' ...
                    dir '_' vartest{ind}(1:4) '.png'])
                savefig(gcf, [fs char(site) '\figures\' char(site) '_mat_' ...
                    dir '_' vartest{ind}(1:4) '.fig'])
            else
                saveas(gcf, [fs char(site) '_mat_' dir '_' vartest{ind}(1:4) '.png'])
            end
            %% calculating p
            for p = p_star
                count = length(find(p_lock(:,1) < p));
                count = count/length(p_lock(:,1));
                s.(dir).(vartest{ind}).FAP = [s.(dir).(vartest{ind}).FAP count];
                count2 = length(find(p_lockloss(:,1) < p));
                count2 = count2/length(p_lockloss(:,1));
                s.(dir).(vartest{ind}).ESP = [s.(dir).(vartest{ind}).ESP count2];
            end
        end
        %% plotting FAP vs ESP curves 
        figure('visible','off')
        x = .25;
        y = .75;
        for ind = 1:length(vartest)
            hold on
            curve = plot(s.(dir).(vartest{ind}).FAP,s.(dir).(vartest{ind}).ESP,'x');
            set(curve,{'DisplayName'},{vartest{ind}})
            str = [vartest{ind}(1:4) 'Star: ' num2str(length(s.(dir).(vartest{ind}).p_star))];
            text(x,y,str)
            x = x -.05;
            y = y - .1;
        end
        title([char(site) ' ' dir ' FAP vs. ESP'])
        xlabel('FAP')
        ylabel('ESP')
        legend('show')
        if ispc
            saveas(gcf,[fs char(site) '\' char(site) '_mat_' dir '_FAP_ESP.png'])
            savefig(gcf,[fs char(site) '\figures\' char(site) '_mat_' dir '_FAP_ESP.fig'])
        else
            saveas(gcf,[fs char(site) '_mat_' dir '_FAP_ESP.png'])
        end
        %% calculating logistic regression of data and plotting the curves
        p_training_set = [];
        for indx = 1:length(vartest)
            p_training_set = [p_training_set s.(dir).(vartest{indx}).combined(:,1)];
        end
        y_training_set = [s.(dir).(vartest{1}).combined(:,2)];
        save('test3.mat','dir','site','fs')
        thetas = logistic_regression_dx(p_training_set,y_training_set,nop,alpha_step_size, ...
            convergence_iterations);
        name = '';
        name2 = '';
        figure('visible','off')
        for indx2 = 1:length(vartest)
            name = [name vartest{indx2}(1:4) ','];
            name2 = [name2  vartest{indx2} '_'];
            s.(dir).(name2).effiency_curve_1 = thetas(1);
            s.(dir).(name2).effiency_curve_2 = thetas(1);
            s.(dir).(name2).effiency_curve_combined = [];
            s.(dir).(name2).ESP = [];
            s.(dir).(name2).FAP = [];
            s.(dir).(name2).effiency_curve_1 = s.(dir).(name2).effiency_curve_1 + (thetas(indx2+1)* ...
                s.(dir).(vartest{indx2}).lockloss(:,1));
            s.(dir).(name2).effiency_curve_combined = [s.(dir).(name2).effiency_curve_combined ...
                s.(dir).(name2).effiency_curve_1];
            s.(dir).(name2).effiency_curve_2 = s.(dir).(name2).effiency_curve_2 + (thetas(indx2+1)* ...
                s.(dir).(vartest{indx2}).lock(:,1));
            s.(dir).(name2).effiency_curve_combined = [s.(dir).(name2).effiency_curve_combined; ...
                s.(dir).(name2).effiency_curve_2];
            p_star = min(s.(dir).(name2).effiency_curve_combined):0.000001:max(s.(dir).(name2).effiency_curve_combined);
            for p = p_star
                count3 = length(find(s.(dir).(name2).effiency_curve_1 < p));
                count3 = count3/length(s.(dir).(name2).effiency_curve_1);
                s.(dir).(name2).ESP = [s.(dir).(name2).ESP count3];
                count4 = length(find(s.(dir).(name2).effiency_curve_2 < p));
                count4 = count4/length(s.(dir).(name2).effiency_curve_2);
                s.(dir).(name2).FAP = [s.(dir).(name2).FAP count4];
            end
            ec = plot(s.(dir).(name2).FAP,s.(dir).(name2).ESP);
            hold on
            set(ec,{'DisplayName'},{[name]})
        end
        xlabel('FAP')
        ylabel('ESP')
        title([char(site) ' ' dir ' Efficiency Curves'])
        legend('show')
        if ispc
            saveas(gcf,[fs char(site) '\' char(site) '_mat_' dir '_efficiency_curves.png'])
        else
            saveas(gcf,[fs char(site) '_mat_' dir '_efficiency_curves.png'])
        end
    end
    for ind = 1:length(vartest)
        figure('visible','off')
        for dir = ['Z' 'Y' 'X']
            curve2 = plot(s.(dir).(vartest{ind}).FAP, s.(dir).(vartest{ind}).ESP,'o');
            set(curve2,{'DisplayName'},{[dir ' ' vartest{ind}]})
            hold on
        end
        title([char(site) ' ' vartest{ind} ' FAP vs. ESP'])
        xlabel('FAP')
        ylabel('ESP')
        legend('show')
        if ispc
            saveas(gcf,[fs char(site) '\' char(site) '_mat_' vartest{ind}(1:4) ...
                '_FAP_ESP.png'])
            savefig([fs char(site) '\' 'figures\' char(site) '_mat_' vartest{ind}(1:4) ...
                '_FAP_ESP.fig'])
        else
            saveas(gcf,[fs char(site) '_mat_' dir ' ' vartest{ind}(1:3) '_FAP_ESP.png'])
        end
    end
end