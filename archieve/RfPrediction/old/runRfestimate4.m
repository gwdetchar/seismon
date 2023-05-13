O1thresh = 0.5e-6;
S5thresh = 0.3e-6;
S6thresh = 1.3e-6;
for site = {'LLO' 'LHO'}
    % run /home/eric.coughlin/gitrepo/seismon/RfPrediction/Rfestimate3(S5thresh, site, data)
    Rfestimate4(S5thresh, S6thresh, O1thresh, site{1})
end
