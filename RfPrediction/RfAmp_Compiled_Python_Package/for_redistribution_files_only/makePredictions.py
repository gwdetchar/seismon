######################################################
## SEISMON RfAmp Prediction Example Code
## 
##
## Uses PYTHON package robustLocklossPrediction & MATLAB shared libraries,
## Make sure to run the script set_shared_library_paths.sh prior to running this script. 
## To re-install the package go through readme.txt 
##
## The input files : train.csv & test.csv
## Input Parameters : earthquake mag,log10(predicted ground motion (m/s) from emperical formula), latitude,longitude,log10(distance), depth
## Output file : log10(predicted amplitude), lockloss_prediction(value btw 1&2 -- no lockloss to lockloss)
##
## Nikhil Mukund Menon (1/3/2018)
## nikhil@iucaa.in, nikhil.mukund@LIGO.ORG
######################################################




#######################################################
## Prediction Code
#######################################################
import robustLocklossPredictionPkg

testFile = 'test.csv'
trainFile = 'train.csv'
predictionFile = 'prediction.csv'

robust = robustLocklossPredictionPkg.initialize()
print(robust.robustPrediction2(testFile,trainFile,predictionFile))
print('Data saved to {}'.format(predictionFile))

