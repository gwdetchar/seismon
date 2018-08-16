% [robust_prediction, outlier_FLAG_1,outlier_FLAG_2 ] =  robustPrediction4(testFile,trainFile,predictionFile)
%
% where robust_prediction = [robust_Rfamp_prediction robust_lockloss_prediction robust_Rfamp_prediction_sigma robust_lockloss_prediction_sigma];
% and  test2.csv = [mag lat lon log10(dist) depth azimuth]
%
% This function aims to predict ground motion & lockloss
% from historic data. Uses Mahalanobis distance to find
% the closest matching past events and generates a weighted
% average prediction. Compared to robustPrediction2.m this
% function directly tried to read to from the input data file
% and aims to remove the dependency on analytic prediction
% variable. This function also computes the associated 1-sigma
% uncertainity on the predicted quantities.
%
% Example:
% [robust_prediction] =  robustPrediction4('test2.csv','H1O1O2_GPR_earthquakes.txt','prediction.csv');
%
%
% Author: Nikhil Mukund Menon (17th July 2018)

function[robust_prediction, outlier_FLAG_1,outlier_FLAG_2 ] =  robustPrediction4(testFile,trainFile,predictionFile)

Debug = 0;

% Load Past Earthquake Data
trainData = importfile_asMatrix(trainFile);
%trainData = table2array(importfile_asMatrix(trainFile));

% Boost trainData to better fit newer observations
tempData = trainData;
noise_level = 1e-3;
copy_num = 4;
for i = 1:copy_num
    tempData = [ tempData; tempData + tempData.*noise_level];
end
 tempData(:,end) = round(tempData(:,end));
trainData = tempData;



% Select only those EQs that caused more than one micron motion
% Thresh = 5e-8;
% idx = trainData(:,end-1) > Thresh;
% trainData = trainData(idx,:);

% Take log tranform for Distance & Measured Rf Amplitude
trainData(:,[4,end-1]) = real(log10(trainData(:,[4,end-1])));

% Remove unlocked states for Lockloss Prediction Part
trainData_2 = trainData;
idx2 = trainData_2(:,end) == 0 ;
trainData_2 = trainData_2(~idx2,:);

if isempty(trainData_2)
    trainData_2 = trainData;
end


testingData = csvread(char(testFile));


robust_Rfamp_prediction =           zeros(size(testingData,1),1);
robust_lockloss_prediction =        zeros(size(testingData,1),1);
robust_Rfamp_prediction_sigma =     zeros(size(testingData,1),1);
robust_lockloss_prediction_sigma =  zeros(size(testingData,1),1);
outlier_FLAG_1     =                zeros(size(testingData,1),1);
outlier_FLAG_2 =                    zeros(size(testingData,1),1);
Orig =                              zeros(size(testingData,1),1);
Orig2 =                             zeros(size(testingData,1),1);


for IDX = 1 : size(testingData,1)
    
  
    
    % Get Mahalanobis Dist of test point from the training points
    P = pdist2(trainData(:,1:6), testingData(IDX,1:6) ,'mahal');
    
    % Sort as per minmimum distance
    [Val,ID] = sort(P);
    
    % Select events within a threshold
    Val_thresh = 0.2;
    [Val_idx] = find(Val <= Val_thresh);
    

    
    
    % Check if outlier
    if ~isempty(Val_idx)
    
        Num = numel(Val_idx);
        trainSimilarOrig = zeros(Num,1);
        count = 1;
        % Get Prediction from nearby training points
        for ijk = 1:Num
            trainSimilarOrig(count) = 10.^trainData(ID(Val_idx(ijk)),7);
            count = count + 1;
        end
        if Num> 1
            invScoreNorm = ( 1./Val(Val_idx(1:Num)) -min(1./Val(Val_idx(1:Num)))  )./( max(1./Val(Val_idx(1:Num))) - min(1./Val(Val_idx(1:Num))) );
        elseif Num == 1
            invScoreNorm = 1./Val(Val_idx(1));
        end
        invScoreNorm = erf(invScoreNorm);
        Orig(IDX) = sum(trainSimilarOrig.*invScoreNorm)./sum(invScoreNorm);
        % Combined Prediction based on Mahalanobis Similarity
        robust_Rfamp_prediction(IDX) = log10(Orig(IDX))';
        robust_Rfamp_prediction_sigma(IDX) = std(trainSimilarOrig);
        
        
      
    else
        outlier_FLAG_1(IDX) = 1;
       
        if Debug; disp(' '); disp('OUTLIER DETECTED for RfAmp, switching to analytic model');end
     
       
        % If no nearby point found, then switch to older analytic model(LLO-LHO Optimized parameters)
        Rf0 = 0.16;
        Rfs =  1.31;
        cd  =  4672.83;
        rs  =  0.82;
        Mag = testingData(IDX,1);
        Dist = 10.^testingData(IDX,4);
        Depth = testingData(IDX,5);
        Rf = ampRfNew(Mag,Dist,Depth,Rf0,Rfs,cd,rs);
      
        % Calculate Rfamp for the closest match
        Rf0 = 0.16;
        Rfs =  1.31;
        cd  =  4672.83;
        rs  =  0.82;
      
                Mag = trainData(ID(1),1);
                Dist = 10.^trainData(ID(1),4);
                Depth = trainData(ID(1),5);
                Rf_ref = ampRfNew(Mag,Dist,Depth,Rf0,Rfs,cd,rs)    ;
                Scale_fac = Rf./Rf_ref;
                
        Scaled_Rf = Scale_fac * 10.^(trainData(ID(1),end-1))   ;            
        
        % Combined Prediction based on Mahalanobis Similarity
        robust_Rfamp_prediction(IDX) = log10(Scaled_Rf);
        robust_Rfamp_prediction_sigma(IDX) = 0;% Needs to be changed
      
        
    end
    
    
    
    
    
    % Get Mahalanobis Dist of test point from the training points
    P2 = pdist2(real(trainData_2(:,1:7)), real([testingData(IDX,1:6) robust_Rfamp_prediction(IDX)]) ,'mahal');
    
    % Sort as per minmimum distance
    [Val,ID] = sort(P2);
    
    % Select events within a threshold
    Val_thresh = 0.2;
    [Val_idx] = find(Val <= Val_thresh);
    
    % Check if outlier
    if ~isempty(Val_idx)
        
        Num = numel(Val_idx);
        trainSimilarOrig = zeros(Num,1);
        count = 1;
        
        % Get Prediction from nearby training points
        for ijk = 1:Num
            trainSimilarOrig(count) = trainData_2(ID(Val_idx(ijk)),8);
            % trainSimilarPred(count) = trainedModel.predictFcn(trainData(ID(Val_idx(ijk)),1:7));
            count = count + 1;
        end
        % PRED(IDX) = trainedModel.predictFcn(testingData(IDX,1:7));
        if Num> 1
            invScoreNorm = ( 1./Val(Val_idx(1:Num)) -min(1./Val(Val_idx(1:Num)))  )./( max(1./Val(Val_idx(1:Num))) - min(1./Val(Val_idx(1:Num))) );
        elseif Num == 1
            invScoreNorm = 1./Val(Val_idx(1));
        end
        invScoreNorm = erf(invScoreNorm);
        Orig2(IDX) = sum(trainSimilarOrig.*invScoreNorm)./sum(invScoreNorm);
        % Combined Prediction based on GPR Prediction & Mahalanobis Similarity
        robust_lockloss_prediction(IDX) = Orig2(IDX);
        robust_lockloss_prediction_sigma(IDX) = std(trainSimilarOrig);
        
       
    else
       
        outlier_FLAG_2(IDX) = 1;
        
        if Debug; disp(' '); disp('OUTLIER DETECTED for Lockloss !');end
        
        Rf_limit = 50e-6;
       
        if 10^(robust_Rfamp_prediction(IDX)) > Rf_limit         
        robust_lockloss_prediction(IDX) = 2;
        else
            robust_lockloss_prediction(IDX) = 1;
        end
        robust_lockloss_prediction_sigma(IDX) = 0;
        
                
    end
    
    
end


% Combine Rfamp & Lockloss results
robust_prediction = [robust_Rfamp_prediction robust_lockloss_prediction robust_Rfamp_prediction_sigma robust_lockloss_prediction_sigma];


% Save Precitions
csvwrite(predictionFile,robust_prediction);

%% Auxiliary Functions

%% Import Earthquake Data File as Matrix
    function GPRearthquakes = importfile_asMatrix(filename, startRow, endRow)
        %IMPORTFILE1 Import numeric data from a text file as a matrix.
        %   GPREARTHQUAKES = IMPORTFILE1(FILENAME) Reads data from text file
        %   FILENAME for the default selection.
        %
        %   GPREARTHQUAKES = IMPORTFILE1(FILENAME, STARTROW, ENDROW) Reads
        %   data from rows STARTROW through ENDROW of text file FILENAME.
        %
        % Example:
        %   GPRearthquakes = importfile_asMatrix('H1O1O2_GPR_earthquakes.txt', 1, 2105);
        %
        %    See also TEXTSCAN.
        
        % Auto-generated by MATLAB on 2018/04/14 02:43:24
        
        %% Initialize variables.
        delimiter = ' ';
        if nargin<=2
            startRow = 1;
            endRow = inf;
        end
        
        %% Format for each line of text:
        %   column2: double (%f)
        %	column11: double (%f)
        %   column12: double (%f)
        %	column13: double (%f)
        %   column14: double (%f)
        %	column15: double (%f)
        %   column30: double (%f)
        %	column32: double (%f)
        % For more information, see the TEXTSCAN documentation.
        formatSpec = '%*q%f%*q%*q%*q%*q%*q%*q%*q%*q%f%f%f%f%f%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%f%*q%f%[^\n\r]';
        
        %% Open the text file.
        fileID = fopen(filename,'r');
        
        %% Read columns of data according to the format.
        % This call is based on the structure of the file used to generate this
        % code. If an error occurs for a different file, try regenerating the code
        % from the Import Tool.
        dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true, 'TextType', 'string', 'EmptyValue', NaN, 'HeaderLines', startRow(1)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
        for block=2:length(startRow)
            frewind(fileID);
            dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true, 'TextType', 'string', 'EmptyValue', NaN, 'HeaderLines', startRow(block)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
            for col=1:length(dataArray)
                dataArray{col} = [dataArray{col};dataArrayBlock{col}];
            end
        end
        
        %% Close the text file.
        fclose(fileID);
        
        %% Post processing for unimportable data.
        % No unimportable data rules were applied during the import, so no post
        % processing code is included. To generate code which works for
        % unimportable data, select unimportable cells in a file and regenerate the
        % script.
        
        %% Create output variable
        GPRearthquakes = [dataArray{1:end-1}];
        
    end

    function Rf = ampRfNew(M,r,h,Rf0,Rfs,cd,rs)
        % Ref: https://dcc.ligo.org/LIGO-T1400487
        % function Rf = ampRf(M,r,h,Rf0,Rfs,Q0,Qs,cd,ch)
        % M = magnitude
        % r = distance [km]
        % h = depth [km]
        %
        % Rf0 = Rf amplitude parameter
        % Rfs = exponent of power law for f-dependent Rf amplitude
        % Q0 = Q-value of Earth for Rf waves at 1Hz
        % Qs = exponent of power law for f-dependent Q
        % cd = speed parameter for surface coupling  [km/s]
        % ch = speed parameter for horizontal propagation  [km/s]
        %
        % exp(-2*pi*h.*fc./cd), coupling of source to surface waves
        % exp(-2*pi*r.*fc./ch./Q), dissipation
        
        fc = 10.^(2.3-M/2);
        Af = Rf0./fc.^Rfs;
        
        % Rf = Rf0.*(M./fc).*exp(-2*pi*h.*fc/c).*exp(-2*pi*r.*fc/c./Q)./r*1e-3;
        % Rf = M.*Af.*exp(-2*pi*h.*fc./cd).*exp(-2*pi*r.*fc./ch./Q)./r.^rs*1e-3;
        Rf = M.*Af.*exp(-2*pi*h.*fc./cd)./r.^rs*1e-3;
        
    end



end
