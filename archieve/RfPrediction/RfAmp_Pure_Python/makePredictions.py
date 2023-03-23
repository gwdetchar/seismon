######################################################
## SEISMON Rayleigh Ground Motion Prediction  Code
## 
##
## Input Parameters : ifo, earthquake mag, latitude,longitude,distance, depth, azimuth
## Output file : predicted amplitude (m/s), lockloss_prediction(value btw 1&2 --> no lockloss to lockloss), uncertainity in both the predictions
##
## Example:
##   python makePredictions.py -ifo 'H1' -mag 5.5 -lat -6.2 -lon 130.6 -dist 10690548.79 -depth 126.5 -azi 42.9
##
##
## Nikhil Mukund Menon (Last Edited : 16/8/2018)
## nikhil@iucaa.in, nikhil.mukund@LIGO.ORG
######################################################





from __future__ import division
import pandas as pd
import numpy as np
import argparse
import configparser
import sys
from scipy.spatial.distance import cdist
from scipy.special import  erf


##########################################################

class helpfulParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('Error: %s\n' % message)
        self.print_help()
        sys.exit(2)

parser = helpfulParser()

# Filename : Parameter .ini file
#parser.add_argument('-filename','--filename', type=str, default='test.csv' , help="Filename of parameter file. Defaults to '%(default)s'. ")

# SEISMON EQ test data params

parser.add_argument('-ifo','--ifo', type=str,  default='H1', help='Interferometer')

parser.add_argument('-mag','--mag', type=float,  default=5.5, help='Magnitude of the earthquake ')

parser.add_argument('-lat','--lat' , type=float, default=30,help='Latitude of the earthquake ')

parser.add_argument('-lon','--lon' , type=float, default=-90, help='Longitude of the earthquake ')

parser.add_argument('-dist','--dist' , type=float, default=1000,help='Distance of the earthquake ')

parser.add_argument('-depth','--depth' , type=float, default=30, help='Depth of the earthquake ')

parser.add_argument('-azi','--azi' , type=float, default=30,help='Azimuth of the earthquake ')

parser.add_argument('-v','--verbose' , type=int, default=1,help='Verbose')


# Get parameters into global namespace
args   = parser.parse_args()

ifo              = args.ifo
mag              = args.mag
lat              = args.lat
lon              = args.lon
dist             = args.dist
depth            = args.depth
azi              = args.azi
verbose          = args.verbose
#filename         = args.filename

# Log transform
dist             = np.log10(dist)

# Select Appropriate Input Files
if   ifo=='H1':
     trainFile      = 'H1O1O2_GPR_earthquakes.txt'
     testFile       = 'H1_test.csv'
     predictionFile = 'H1_prediction.csv'
elif ifo=='L1':
     trainFile      = 'L1O1O2_GPR_earthquakes.txt'
     testFile       = 'L1_test.csv'
     predictionFile = 'L1_prediction.csv'
elif ifo=='V1':
     trainFile      = 'V1O1O2_GPR_earthquakes.txt'
     testFile       = 'V1_test.csv'
     predictionFile = 'V1_prediction.csv'


# Create & Save Test file
train_data = [[mag,lat,lon,dist,depth,azi]]
my_df = pd.DataFrame(train_data)
my_df.to_csv(testFile, index=False, header=False)

# Analtic Formula for Rf amp
def ampRfnew(M,r,h,Rf0,Rfs,cd,rs):
    fc = 10**(2.3 - M/2)
    Af = Rf0/fc**Rfs
    Rf = M*Af*np.exp(-2*np.pi*h*fc/cd)/r**rs*1e-3
    return Rf

Debug = 0

# Load Past EQ Dara
trainData = pd.read_csv(trainFile,delimiter=' ',header=None,usecols=[1,10,11,12,13,14,29,31])

# Boost trainData to better fit newer observations
tempData = trainData
noise_level = 1e-1
copy_num = 1



"""for i in range(copy_num):
    tempData = np.vstack([tempData, tempData])

tempData = tempData +  tempData*np.random.randn(*tempData.shape)*noise_level    
  
# Convert back to dataframe
trainData = pd.DataFrame(tempData)"""


#  Take log tranform for Distance & Measured Rf Amplitude
trainData.iloc[:,[3,-2]] = np.log10(np.abs(trainData.iloc[:,[3,-2]]))



# Remove unlocked states for Lockloss Prediction Part
trainData_2 = trainData
idx2 = trainData_2.iloc[:,-1]==0
trainData_2 = trainData_2.loc[~idx2]

if trainData_2.empty:
     trainData_2 = trainData
        
# Read Test Data
testingData = pd.read_csv(testFile,delimiter=',',header=None)

# Predefine 
robust_Rfamp_prediction =          pd.DataFrame(index = testingData.index)
robust_lockloss_prediction =       pd.DataFrame(index = testingData.index)
robust_Rfamp_prediction_sigma =    pd.DataFrame(index = testingData.index)
robust_lockloss_prediction_sigma = pd.DataFrame(index = testingData.index)
outlier_FLAG_1  =                  pd.DataFrame(index = testingData.index)
outlier_FLAG_2  =                  pd.DataFrame(index = testingData.index)
Orig =                             pd.DataFrame(index = testingData.index)
Orig2 =                            pd.DataFrame(index = testingData.index)


for IDX in range(testingData.shape[0]):


    # Rayleigh Amplitude Prediction

    # Get Mahalanobis Dist of test point from the training points
    P = cdist(trainData.iloc[:,[1,2]],testingData.iloc[:,[1,2]],'mahalanobis')

    # Sort as per minimum distance
    Val = np.sort(P,axis=0)
    ID  = np.argsort(P,axis=0)

    # Select events within a threshold [LOCATION]
    Val_thresh = 0.05
    Val_idx = np.where(Val <= Val_thresh)[0]

    TD   = trainData.iloc[pd.DataFrame(ID[Val_idx])[0]]

    # Select events within a threshold [EQ Magnitude]
    Ldist = cdist(TD.iloc[:,[0]],testingData.iloc[:,[0]].values.reshape(1,-1))


    # Sort as per minimum distance
    CVal = np.sort(Ldist,axis=0); 
    CID  = np.argsort(Ldist,axis=0)
   
    NearbyEvents = pd.DataFrame()

    # Select events within a threshold [EQ MAGNITUDE]
    Val_thresh_2 = 0.5
    Val_idx_2 = np.where(CVal <= Val_thresh_2)[0]
    CTD   = TD.iloc[pd.DataFrame(CID[Val_idx_2])[0]]


    if (len(Val_idx_2)  == 0):
        # Incerease the threshold and try again
        print('Increasing the threshold...')
        Val_thresh_2 = 1
        Val_idx_2 = np.where(CVal <= Val_thresh_2)[0]
        CTD   = TD.iloc[pd.DataFrame(CID[Val_idx_2])[0]]

   
    

    if (len(Val_idx_2)  != 0):
   
        Num  = len(Val_idx_2)
        trainSimilarOrig = np.zeros([Num,1])
        count = 0
        # Get predictions from nearby training points
        KEY = list(CID[Val_idx_2][:].flatten())
        NearbyEvents = TD.iloc[KEY]
      
        for ijk in range(Num):
            trainSimilarOrig[ijk] = TD.iloc[KEY[ijk],-2]


        Orig[IDX]   = 10**(np.median(trainSimilarOrig))


        robust_Rfamp_prediction[IDX] = np.log10(Orig[IDX])
        
        robust_Rfamp_prediction_sigma[IDX] = np.std(10**(trainSimilarOrig))

    else :
        outlier_FLAG_1[IDX] = 1
        

        # If no nearby point found, then use a scaled version of old analytic formula
        Rf0 = 0.16
        Rfs = 1.31
        cd  = 4672.83
        rs  = 0.82
        Mag  = testingData.iloc[IDX,0]
        Dist = 10**testingData.iloc[IDX,3] 
        Depth = testingData.iloc[IDX,4]
        Rf = ampRfnew(Mag,Dist,Depth,Rf0,Rfs,cd,rs)

        # Calculate Rfamp for the closest match
        Mag = trainData.iloc[ID[0],0].values
        Dist = 10**trainData.iloc[ID[0],3].values
        Depth = trainData.iloc[ID[0],4].values    
        Rf_ref = ampRfnew(Mag,Dist,Depth,Rf0,Rfs,cd,rs)
        Measured_Rf = 10**trainData.iloc[ID[0],6].values

        Scale_fac = Rf/Rf_ref 

        Scaled_Rf = Scale_fac*Measured_Rf

        robust_Rfamp_prediction[IDX] = np.log10(Scaled_Rf)
        robust_Rfamp_prediction_sigma[IDX] = 0 # Needs to be changed

    # LOCKLOSS Prediction    
    # Get Mahalanobis Dist of test point from the training points    
    P2 = cdist(trainData_2.iloc[:,[1,2]],testingData.iloc[:,[1,2]],'mahalanobis')

    # Sort as per minimum distance
    Val = np.sort(P2,axis=0)
    ID  = np.argsort(P2,axis=0)

    # Select events within a threshold
    Val_thresh = 0.08
    Val_idx = np.where(Val <= Val_thresh)[0]

    TD   = trainData_2.iloc[pd.DataFrame(ID[Val_idx])[0]]

    Ldist = cdist(TD.iloc[:,[0]],testingData.iloc[:,[0]].values.reshape(1,-1))

    # Sort as per minimum distance
    CVal = np.sort(Ldist,axis=0)
    CID  = np.argsort(Ldist,axis=0)

    # Select events within a threshold [EQ PARAMETERS]
    Val_thresh_2 = 0.5
    Val_idx_2 = np.where(CVal <= Val_thresh_2)[0]    
    CTD   = TD.iloc[pd.DataFrame(CID[Val_idx_2])[0]]
    if (len(Val_idx_2)  == 0):
        # Incerease the threshold and try again        
        Val_thresh_2 = 1
        Val_idx_2 = np.where(CVal <= Val_thresh_2)[0]
        CTD   = TD.iloc[pd.DataFrame(CID[Val_idx_2])[0]]    


       
    if (len(Val_idx_2)  != 0):
        Num  = len(Val_idx_2)
        trainSimilarOrig = np.zeros([Num,1])
        count = 0
        # Get predictions from nearby training points
        KEY = list(CID[Val_idx_2][:].flatten())
      



        for ijk in range(Num):
            trainSimilarOrig[ijk] = TD.iloc[KEY[ijk],-1]
         
        """
        if (Num > 1):
               invScoreNorm = np.true_divide((1./CVal[Val_idx_2]) - min(1./CVal[Val_idx_2]), (max(1./CVal[Val_idx_2]) - min(1./CVal[Val_idx_2]) ) )
        else:
               invScoreNorm = 1               
        invScoreNorm = erf(invScoreNorm)   
        Orig2[IDX]   = np.round(np.sum(trainSimilarOrig*invScoreNorm))
        """

        Orig2[IDX]   = np.round(np.median(trainSimilarOrig))

        robust_lockloss_prediction[IDX] = Orig2[IDX]
        robust_lockloss_prediction_sigma[IDX] = np.std(trainSimilarOrig) 

    else:
        outlier_FLAG_2[IDX] = 1
        Rf_limit = 50*1e-6
        if (10**(robust_Rfamp_prediction[IDX].values) > Rf_limit):
            robust_lockloss_prediction[IDX] = 2
        else:
            robust_lockloss_prediction[IDX] = 1
        robust_lockloss_prediction_sigma[IDX] = 0


# Combine Rfamp & Lockloss results        
robust_prediction = {'robust_Rfamp_prediction':robust_Rfamp_prediction.values[0][0],'robust_lockloss_prediction':robust_lockloss_prediction.values[0][0],'robust_Rfamp_prediction_sigma':robust_Rfamp_prediction_sigma.values[0][0],'robust_lockloss_prediction_sigma':robust_lockloss_prediction_sigma.values[0][0]}


# Save Results
np.save("robust_prediction",robust_prediction)

# Display Results
Rfamp             = 10**robust_Rfamp_prediction.values[0][0]
LocklossTag       = robust_lockloss_prediction.values[0][0]
Rfamp_sigma       = robust_Rfamp_prediction_sigma.values[0][0]
LocklossTag_sigma = robust_lockloss_prediction_sigma.values[0][0]

if (verbose != 0):
    print("#################################################")
    print("#############  SEISMON Results ##################")
    print("Input EQ Parameters (Mag, Lat, Lon, Dist, Depth, Azi)")
    print(testingData)
    print("")
    print('Nearby events picked up from the database...')
    print(NearbyEvents)  
    print('-----------------------------------------------')
    print('-----------------------------------------------')    
    print("Rayleigh Ground Motion (m/s): {}".format(Rfamp) )
    print("LocklossTag (1-No Lockloss / 2-Lockloss): {}".format(LocklossTag))
    print("Rfamp_sigma: {}".format(Rfamp_sigma))
    print("LocklossTag_sigma: {}".format(LocklossTag_sigma))
    print("###############################################")

            

