######################################################
## SEISMON RfAmp Prediction  Code
## 
##
## Uses PYTHON package robustLocklossPrediction & MATLAB 2016b shared libraries,
## Make sure to run the script set_shared_library_paths.sh prior to running this script. 
## To re-install the package go through readme.txt 
##
## Input Parameters : earthquake mag,analtic predicted ground motion (m/s), latitude,longitude,distance, depth, azimuth
## Output file : predicted amplitude, lockloss_prediction(value btw 1&2 --> no lockloss to lockloss)
##
##Example:
##   python makePredictions.py -mag 7.5 -analyticPred 2.674237657088992e-07 -lat -6.2 -lon 130.6 -dist 10690548.79 -depth 126.5 -azi 42.9
##
## To embed the same functionality in another code as a function use the commented lines of code at the end
##    Rfamp,LocklossTag = makePredictions(5.1,1.88191e-07,-18.2,-174.9,1.048178e+07,197.7,59.4)
##
## Nikhil Mukund Menon (Last Edited : 26/3/2018)
## nikhil@iucaa.in, nikhil.mukund@LIGO.ORG
######################################################




#######################################################
## Prediction Code
#######################################################
from __future__ import division
import argparse
import configparser
import sys
import pandas as pd
import numpy as np

import robustLocklossPredictionPkg

testFile = 'test.csv'
trainFile = 'train.csv'
predictionFile = 'prediction.csv'
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
parser.add_argument('-mag','--mag', type=float,  default=5.5, help='Magnitude of the earthquake ')

parser.add_argument('-analyticPred','--analyticPred' , type=float ,default=0.00001,help='Depth of the earthquake ')

parser.add_argument('-lat','--lat' , type=float, default=30,help='Latitude of the earthquake ')

parser.add_argument('-lon','--lon' , type=float, default=-90, help='Longitude of the earthquake ')

parser.add_argument('-dist','--dist' , type=float, default=1000,help='Distance of the earthquake ')
     
parser.add_argument('-depth','--depth' , type=float, default=30, help='Depth of the earthquake ')

parser.add_argument('-azi','--azi' , type=float, default=30,help='Azimuth of the earthquake ')


# Get parameters into global namespace
args   = parser.parse_args()

mag              = args.mag
analyticPred     = args.analyticPred
lat              = args.lat
lon              = args.lon
dist             = args.dist
depth            = args.depth
azi              = args.azi
#filename         = args.filename

# Log transform
analyticPred     = np.log10(analyticPred)
dist             = np.log10(dist)


# Create & Save Test file
train_data = [[mag,analyticPred,lat,lon,dist,depth,azi]]
my_df = pd.DataFrame(train_data)
my_df.to_csv(testFile, index=False, header=False)



######################################################
robust = robustLocklossPredictionPkg.initialize()

# Do prediction
robust.robustPrediction2(testFile,trainFile,predictionFile)
Result = pd.read_csv(predictionFile)
Rfamp  = 10**float(Result.keys()[0])
LocklossTag  = float(Result.keys()[1])
print(Rfamp,LocklossTag)


'''
#######################################################################
# To call as a function use the code below
#
# Example:
#  Rfamp,LocklossTag = makePredictions(5.1,1.88191e-07,-18.2,-174.9,1.048178e+07,197.7,59.4)
#
import numpy as np
import pandas as pd
import robustLocklossPredictionPkg
robust = robustLocklossPredictionPkg.initialize()

def makePredictions(mag,analyticPred,lat,lon,dist,depth,azi):
    trainFile = 'train.csv'
    testFile = 'test.csv'
    predictionFile = 'prediction.csv'
    # Log transform
    analyticPred     = np.log10(analyticPred)
    dist             = np.log10(dist)
    # Save to file
    train_data = [[mag,analyticPred,lat,lon,dist,depth,azi]]
    my_df = pd.DataFrame(train_data)
    my_df.to_csv(testFile, index=False, header=False)
    # Do prediction
    robust.robustPrediction2(testFile,trainFile,predictionFile)
    Result = pd.read_csv(predictionFile)
    Rfamp  = 10**float(Result.keys()[0])
    LocklossTag  = float(Result.keys()[1])
    return (Rfamp,LocklossTag)
'''
