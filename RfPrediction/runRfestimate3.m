O1thresh = 0.5e-6;
S5thresh = 0.3e-6;
S6thresh = 1.3e-6;
for site = {'LLO' 'LHO'}
    for data = {'O1_RMS_Z' 'O1_RMS_X' 'O1_RMS_Y'}
        if strcmp(data,'S5')
           % run /home/eric.coughlin/gitrepo/seismon/RfPrediction/Rfestimate3(S5thresh, site, data)
           Rfestimate3(S5thresh, site{1}, data{1})
        elseif strcmp(data,'S6')
           % run /home/eric.coughlin/gitrepo/seismon/RfPrediction/Rfestimate3(S6thresh, site, data)
           Rfestimate3(S6thresh, site{1}, data{1})
        elseif strcmp(data,'O1')
           % run /home/eric.coughlin/gitrepo/seismon/RfPrediction/Rfestimate3(O1thresh, site, data)
           Rfestimate3(O1thresh, site{1}, data{1})
        else
           Rfestimate3(O1thresh, site{1}, data{1})
        end
    end
end
