######################################################
## SEISMON RfAmp Prediction  Code
## 
##
## Uses PYTHON package robustLocklossPredictionPkg3 & MATLAB 2016b shared libraries,
## Make sure to run the script set_shared_library_paths.sh prior to running this script. 
## To re-install the package go through readme.txt 
##
## Input Parameters : ifo, earthquake mag, latitude,longitude,distance, depth, azimuth
## Output file : predicted amplitude, lockloss_prediction(value btw 1&2 --> no lockloss to lockloss)
##
##Example:
##   python makePredictions.py -ifo 'H1' -mag 7.5 -lat -6.2 -lon 130.6 -dist 10690548.79 -depth 126.5 -azi 42.9
##
## To embed the same functionality in another code as a function use the commented lines of code at the end
##    Rfamp,LocklossTag = makePredictions('H1',5.1,-18.2,-174.9,1.048178e+07,197.7,59.4)
##
## Nikhil Mukund Menon (Last Edited : 14/4/2018)
## nikhil@iucaa.in, nikhil.mukund@LIGO.ORG
######################################################




#######################################################
## Prediction Code
#######################################################



import robustLocklossPredictionPkg3


import argparse
import configparser
import sys
import pandas as pd
import numpy as np

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


# Get parameters into global namespace
args   = parser.parse_args()

ifo              = args.ifo
mag              = args.mag
lat              = args.lat
lon              = args.lon
dist             = args.dist
depth            = args.depth
azi              = args.azi
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





robust = robustLocklossPredictionPkg3.initialize()

# Do prediction
robust.robustPrediction3(testFile,trainFile,predictionFile)
Result            = pd.read_csv(predictionFile)
Rfamp             = float(Result.keys()[0])
LocklossTag       = float(Result.keys()[1])
Rfamp_sigma       = float(Result.keys()[2])
LocklossTag_sigma =  float(Result.keys()[3])

print("Rfamp,LocklossTag,Rfamp_sigma,LocklossTag_sigma")
print(Rfamp,LocklossTag,Rfamp_sigma,LocklossTag_sigma)



'''
#######################################################################
# To call as a function use the code below
#
# Example:
#  Rfamp,LocklossTag = makePredictions('H1',5.1,-18.2,-174.9,1.048178e+07,197.7,59.4)
#

import robustLocklossPredictionPkg3
import numpy as np
import pandas as pd


robust = robustLocklossPredictionPkg3.initialize()


def makePredictions(ifo,mag,lat,lon,dist,depth,azi):
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

    # Log transform
    dist             = np.log10(dist)
    # Save to file
    train_data = [[mag,lat,lon,dist,depth,azi]]
    my_df = pd.DataFrame(train_data)
    my_df.to_csv(testFile, index=False, header=False)
    # Do prediction
    robust.robustPrediction3(testFile,trainFile,predictionFile)
    Result = pd.read_csv(predictionFile)
    Rfamp  = float(Result.keys()[0])
    LocklossTag  = float(Result.keys()[1])
    return (Rfamp,LocklossTag)


Rfamp,LocklossTag = makePredictions('H1',5.1,-18.2,-174.9,1.048178e+07,197.7,59.4)
print(Rfamp,LocklossTag)


'''


