set(0,'DefaultAxesFontSize',20)
set(0,'DefaultTextFontSize',20)

cla;
alpha_step_size = 9;
thetas_matrix = {};
%% Checking operating system
% This section checks the user's operating system in order to find the
% correct directories to save and read in. 
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
for site = {'LHO'  'LLO'}
    s.(char(site)) = [];
    %% checking existance for directories to save in
    if ispc && ~exist([fs,char(site)],'dir')
        mkdir(fs,char(site))
        mkdir([fs '\' char(site)],'figures')
    end
    for dir = ['Z']
        
        %% initialize variables
        s.(char(site)).(dir) = [];
        file = load([sd char(site) '_O1_predicted_binary_' dir '.txt']);
        sof = size(file);
        vartest = {'pvelocity','magnitude','distance','depth'};
        
        % _vartest_ is where the names of the parameters are stored.
        
        %testvar = calibrate_data(file(1:fix(end/1.5),3:3+length(vartest)-1),length(vartest));
        testvar = calibrate_data(file(1:fix(end/1.5),[3 4 5 6]),length(vartest));
        for ind4 = 1:size(testvar,2)
            file(1:fix(end/1.5),2+ind4) = testvar(:,ind4);
        end
        nop = length(vartest);
        vartest2 = {'combined','lockloss','lock'};
        si = 3;
        loc = find(file(1:fix(end/1.5),sof(2))==0);
        locloss = find(file(1:fix(end/1.5),sof(2))==1);
        %% creating structures for parameters
        % We create structures so that the parameters information can be
        % easily stored and access for use later in the program. Each
        % channel is divided into each parameter which is further divided
        % into lockloss, lock and combined. i.e. s.Z.velocity.combined
        % which is where all of the values for velocity are stored.
        for ind = 1:length(vartest)
            s.(char(site)).(dir).(vartest{ind}) = [];
            for ind2 = 1:length(vartest2)
                if strcmp(vartest2{ind2},'lockloss')
                    s.(char(site)).(dir).(vartest{ind}).(vartest2{ind2}) = [file(locloss,si) file(locloss,sof(2))];
                elseif strcmp(vartest2{ind2},'lock')
                    s.(char(site)).(dir).(vartest{ind}).(vartest2{ind2}) = [file(loc,si) file(loc,sof(2))];
                else
                    s.(char(site)).(dir).(vartest{ind}).(vartest2{ind2}) = [file(1:fix(end/1.5),si) file(1:fix(end/1.5),sof(2))];
                end
            end
            si = si + 1;
        end
        %% calculating FAP and ESP numbers
        for ind = 1:length(vartest)
            s.(char(site)).(dir).(vartest{ind}).FAP = [];
            s.(char(site)).(dir).(vartest{ind}).ESP = [];
            p_lockloss = [];
            p_lockloss = s.(char(site)).(dir).(vartest{ind}).lockloss;
            p_lock = [];
            p_lock = s.(char(site)).(dir).(vartest{ind}).lock;
            p_star = min(s.(char(site)).(dir).(vartest{ind}).combined(:,1)):0.01:max(s.(char(site)).(dir).(vartest{ind}).combined(:,1));
            s.(char(site)).(dir).(vartest{ind}).p_star = p_star;
            
            %{
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
            %}
            
            %% calculating p
            % This loop describes the process of calculating the False
            % Alarm Probablity(FAP) and Efficiency Standard
            % Probablity(ESP) for each individual parameter.
            % This is to see how each parameter affects the Efficiency
            % Curves on their own.
            for p = p_star
                count = length(find(p_lock(:,1) < p));
                count = count/length(p_lock(:,1));
                s.(char(site)).(dir).(vartest{ind}).ESP = [s.(char(site)).(dir).(vartest{ind}).ESP count];
                count2 = length(find(p_lockloss(:,1) < p));
                count2 = count2/length(p_lockloss(:,1));
                s.(char(site)).(dir).(vartest{ind}).FAP = [s.(char(site)).(dir).(vartest{ind}).FAP count2];
            end
        end
        %% plotting FAP vs ESP curves 
        figure('visible','off')
        set(gcf,'PaperSize',[10 6])
        set(gcf,'PaperPosition',[0 0 10 6])
        x = .25;
        y = .75;
        plot(linspace(0,1),linspace(0,1),'--')
        for ind = 1:length(vartest)
            hold on
            curve = plot(s.(char(site)).(dir).(vartest{ind}).FAP,s.(char(site)).(dir).(vartest{ind}).ESP);
            set(curve,{'DisplayName'},{vartest{ind}})
            str = [vartest{ind}(1:4) 'Star: ' num2str(length(s.(char(site)).(dir).(vartest{ind}).p_star))];
            text(x,y,str)
            x = x -.05;
            y = y - .05;
        end
        title([char(site) ' ' dir ' FAP vs. ESP'])
        xlabel('FAP')
        ylabel('ESP')
        legend('show','Location','southeast')
        if ispc
            saveas(gcf,[fs char(site) '\' char(site) '_mat_' dir '_pv_FAP_ESP.pdf'])
            savefig(gcf,[fs char(site) '\figures\' char(site) '_mat_' dir '_pv_FAP_ESP.fig'])
        else
            saveas(gcf,[fs char(site) '_mat_' dir '_FAP_ESP.pdf'])
        end
        %% Calculating The Logistic Regression Of Data And Plotting The Curves
        p_training_set = [];
        for indx = 1:length(vartest)
            p_training_set = [p_training_set s.(char(site)).(dir).(vartest{indx}).combined(:,1)];
        end
        y_training_set = [s.(char(site)).(dir).(vartest{1}).combined(:,2)];
        name = '';
        name2 = '';
        alpha_step_size = 9.5;
        fig = figure('visible','off');
        set(fig,'PaperSize',[10 6])
        set(fig,'PaperPosition',[0 0 10 6])
        plot(linspace(0,1),linspace(0,1),'--')
        hold on
        for indx2 = 1:length(vartest)
            name = [name vartest{indx2}(1:4) ','];
            name2 = [name2  vartest{indx2} '_'];
            save('test3.mat','dir','site','fs')
            thetas = glmfit(p_training_set(:,1:indx2),[y_training_set ones(length(y_training_set),1)],'binomial','link','logit');
            alpha_step_size = alpha_step_size - .3;
            s.(char(site)).(dir).(name2).thetas_matrix = thetas;
            s.(char(site)).(dir).(name2).effiency_curve_1 = thetas(1);
            s.(char(site)).(dir).(name2).effiency_curve_2 = thetas(1);
            s.(char(site)).(dir).(name2).effiency_curve_combined = [];
            s.(char(site)).(dir).(name2).ESP = [];
            s.(char(site)).(dir).(name2).FAP = [];
            for indx5 = 1:indx2
                s.(char(site)).(dir).(name2).effiency_curve_1 = s.(char(site)).(dir).(name2).effiency_curve_1 + (thetas(indx5+1)* ...
                    s.(char(site)).(dir).(vartest{indx5}).lockloss(:,1));
                s.(char(site)).(dir).(name2).effiency_curve_2 = s.(char(site)).(dir).(name2).effiency_curve_2 + (thetas(indx5+1)* ...
                    s.(char(site)).(dir).(vartest{indx5}).lock(:,1));
            end
            s.(char(site)).(dir).(name2).effiency_curve_combined = [s.(char(site)).(dir).(name2).effiency_curve_combined ...
                    s.(char(site)).(dir).(name2).effiency_curve_1];
            s.(char(site)).(dir).(name2).effiency_curve_combined = [s.(char(site)).(dir).(name2).effiency_curve_combined; ...
                    s.(char(site)).(dir).(name2).effiency_curve_2];
            p_star = min(s.(char(site)).(dir).(name2).effiency_curve_combined):0.005:max(s.(char(site)).(dir).(name2).effiency_curve_combined);
            for p = p_star
                count3 = length(find(s.(char(site)).(dir).(name2).effiency_curve_1 < p));
                count3 = count3/length(s.(char(site)).(dir).(name2).effiency_curve_1);
                s.(char(site)).(dir).(name2).FAP = [s.(char(site)).(dir).(name2).FAP count3];
                count4 = length(find(s.(char(site)).(dir).(name2).effiency_curve_2 < p));
                count4 = count4/length(s.(char(site)).(dir).(name2).effiency_curve_2);
                s.(char(site)).(dir).(name2).ESP = [s.(char(site)).(dir).(name2).ESP count4];
            end
            set(groot,'CurrentFigure',fig)
            ec = plot(s.(char(site)).(dir).(name2).FAP,s.(char(site)).(dir).(name2).ESP);
            set(ec,{'DisplayName'},{[name]})
            hold on
        end
        xlabel('FAP')
        ylabel('ESP')
        title([char(site) ' ' dir ' Efficiency Curves'])
        legend('show','Location','southeast')
        if ispc
            saveas(gcf,[fs char(site) '\' char(site) '_mat_' dir '_pv_efficiency_curves.pdf'])
        else
            saveas(gcf,[fs char(site) '_mat_' dir '_efficiency_curves.pdf'])
        end
    end
    for ind = 1:length(vartest)
        figure('visible','off')
        set(gcf,'PaperSize',[10 6])
        set(gcf,'PaperPosition',[0 0 10 6])
        for dir = ['Z']
            curve2 = plot(s.(char(site)).(dir).(vartest{ind}).FAP, s.(char(site)).(dir).(vartest{ind}).ESP,'o');
            set(curve2,{'DisplayName'},{[dir ' ' vartest{ind}]})
            hold on
        end
        title([char(site) ' ' vartest{ind} ' FAP vs. ESP'])
        xlabel('FAP')
        ylabel('ESP')
        legend('show','Location','southeast')
        if ispc
            saveas(gcf,[fs char(site) '\' char(site) '_mat_' vartest{ind}(1:4) ...
                '_FAP_ESP.pdf'])
            savefig([fs char(site) '\' 'figures\' char(site) '_mat_' vartest{ind}(1:4) ...
                '_FAP_ESP.fig'])
        else
            saveas(gcf,[fs char(site) '_mat_' dir ' ' vartest{ind}(1:3) '_FAP_ESP.pdf'])
        end
    end
end
save('seismon_logistic_regression.mat','s')