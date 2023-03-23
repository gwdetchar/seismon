# Stacked Ensemble RfAmp Prediction Model
# Multiple ML regressors are individually trained and then combined via meta-regressor. 
# Hyperparameters are tuned via GridSearchCV

# coding: utf-8

from __future__ import division

import optparse

import numpy as np
import pandas as pd
import os

if not os.getenv("DISPLAY", None):
    import matplotlib
    matplotlib.use("agg", warn=False)

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.externals import joblib
from mlxtend.regressor import StackingCVRegressor
from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam
from keras import losses
from keras import callbacks
from keras.utils import plot_model

import pickle

__author__ = "Nikhil Mukund <nikhil.mukund@ligo.org>, Michael Coughlin <michael.coughlin.ligo.org>"
__version__ = 1.0
__date__ = "11/26/2017"

def parse_commandline():
    """@parse the options given on the command-line.
    """
    parser = optparse.OptionParser(usage=__doc__,version=__version__)

    parser.add_option("-f", "--earthquakesFile", help="Seismon earthquakes file.",default ="/home/mcoughlin/Seismon/Predictions/L1O1O2_CMT_GPR/earthquakes.txt")
    parser.add_option("-o", "--outputDirectory", help="output folder.",default ="/home/mcoughlin/Seismon/MLA/L1O1O2/")
    parser.add_option("-r", "--runType", help="run type (original, lowlatency, cmt)", default ="lowlatency")
    parser.add_option("-m", "--minMagnitude", help="Minimum earthquake magnitude.", default=5.0,type=float)
    parser.add_option("-N", "--Nepoch", help="number of epochs", default =10, type=int)
    parser.add_option("--doPlots",  action="store_true", default=False)
    parser.add_option("-v", "--verbose", action="store_true", default=False,
                      help="Run verbosely. (Default: False)")

    opts, args = parser.parse_args()

    # show parameters
    if opts.verbose:
        print >> sys.stderr, ""
        print >> sys.stderr, "running network_eqmon..."
        print >> sys.stderr, "version: %s"%__version__
        print >> sys.stderr, ""
        print >> sys.stderr, "***************** PARAMETERS ********************"
        for o in opts.__dict__.items():
          print >> sys.stderr, o[0]+":"
          print >> sys.stderr, o[1]
        print >> sys.stderr, ""

    return opts

'''
0: earthquake gps time
1: earthquake mag
2: p gps time
3: s gps time
4: r (2 km/s)
5: r (3.5 km/s)
6: r (5 km/s)
7: predicted ground motion (m/s)
8: lower bounding time
9: upper bounding time
10: latitude
11: longitude
12: distance
13: depth (m)
14: azimuth (deg)
15: nodalPlane1_strike
16: nodalPlane1_rake
17: nodalPlane1_dip
18: momentTensor_Mrt
19: momentTensor_Mtp
20: momentTensor_Mrp
21: momentTensor_Mtt
22: momentTensor_Mrr
23: momentTensor_Mpp
24: peak ground velocity gps time
25: peak ground velocity (m/s)
26: peak ground acceleration gps time
27: peak ground acceleration (m/s^2)
28: peak ground displacement gps time
29: peak ground displacement (m)
30: Lockloss time
31: Detector Status
'''


# Parse command line
opts = parse_commandline()

outputDirectory = os.path.join(opts.outputDirectory,opts.runType)
if not os.path.isdir(outputDirectory):
    os.makedirs(outputDirectory)

data = pd.read_csv(opts.earthquakesFile,delimiter=' ',header=None)
neqs, ncols = data.shape
if ncols == 32:
    fileType = "seismon"
elif ncols == 27:
    fileType = "usarray"
    data = data.drop(data.columns[[24]], 1)
    data = data.rename(index=int, columns={25: 24, 26: 25})
else:
    print("I do not understand the file type...")
    exit(0)

# find magnitudes greater than minimum magnitude
index = data[1] > opts.minMagnitude
data = data[:][index]

# find depth = 0
index = np.where(data[[13]] == 0)[0]
data.iloc[index,13] = 1.0

# shuffle data
data = data.reindex(np.random.permutation(data.index))

Mag_idx = 1
Dist_idx = 12
Depth_idx = 13
Rf_Amp_idx = 25


# Mag threshold
Rf_Amp_thresh = 1e-8; 
index = data[Rf_Amp_idx] > Rf_Amp_thresh
data = data[:][index]

if opts.runType == "cmt":
    # Select features
    FeatSet_index = [1,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
elif opts.runType == "lowlatency":
    #FeatSet_index = [1,7,10,11,12,13,14,15,16,17] #  these lower set paramaters makes  sense
    FeatSet_index = [1,10,11,12,13,14] #  these lower set paramaters makes  sense
elif opts.runType == "original":
    FeatSet_index = [1,12,13] #  Just Mag, Dist, Depth
else:
    print("--runType must be original, lowlatency, and cmt")
    exit(0)


Target_index = [Rf_Amp_idx]


# Artificially increase samples
data_temp = data
copy_num =  6
noise_level = 1e-2 # 1e-2


Rfamp_orig = data_temp[Target_index];
data_orig =  data_temp   


def boost_samples(x_samples,y_samples,copy_num=3,noise_level=1e-2):
    # Artificially increase samples
    data_x_temp = x_samples
    data_y_temp = y_samples

    for i in range(copy_num):
        data_x_temp = np.vstack((data_x_temp,data_x_temp))
        data_y_temp = np.vstack((data_y_temp,data_y_temp))


    data_x_orig =  data_x_temp   
    data_y_orig =  data_y_temp   


    x1 = data_x_temp
    x2 = np.random.randn(*data_x_temp.shape)*noise_level
    x_samples_boosted = x1 + np.multiply(x1,x2)
    
    y1 = data_y_temp
    y2 = np.random.randn(*data_y_temp.shape)*noise_level
    y_samples_boosted = y1 + np.multiply(y1,y2)   
    
    # Shuffle samples
    #IDX = np.random.permutation(y_samples_boosted.index)
    IDX = np.random.permutation(np.arange(0,len(y_samples_boosted)))
    x_samples_boosted = x_samples_boosted[IDX,:]
    y_samples_boosted = y_samples_boosted[IDX,:]
    
    return x_samples_boosted, y_samples_boosted

data = data_temp

# Take Log10 of certain features (Mag, Dist, Depth)
data[[Dist_idx, Depth_idx]] = np.log10(data[[Dist_idx, Depth_idx]])
data[Target_index] = np.log10(data[Target_index])

X = np.asarray(data[FeatSet_index])
Y = np.asarray(data[Target_index])

# Normalize samples
x_scaler = preprocessing.MinMaxScaler()
#x_scaler = preprocessing.data.QuantileTransformer()
X = x_scaler.fit_transform(X)
y_scaler = preprocessing.MinMaxScaler()
#y_scaler = preprocessing.data.QuantileTransformer()
Y = y_scaler.fit_transform(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3,random_state=42)

# boost_samples + normalize + shuffle them
TUPLE1 = boost_samples(x_train,y_train,copy_num,noise_level)
TUPLE2 = boost_samples(x_val,y_val,copy_num,noise_level)  
TUPLE3 = boost_samples(x_test,y_test,copy_num,noise_level)  

x_train = TUPLE1[0]
y_train = TUPLE1[1]

x_val   = TUPLE2[0]
y_val   = TUPLE2[1]

x_test  = TUPLE3[0] 
y_test  = TUPLE3[1]

#############################################
# Construct Stacked Ensemble Model #
#############################################

RANDOM_SEED = 42


ridge = Ridge()
lasso = Lasso()
svr_lin = SVR(kernel='linear')
svr_rbf = SVR(kernel='rbf')
lr = LinearRegression()
rf = RandomForestRegressor(random_state=RANDOM_SEED)
                   
np.random.seed(RANDOM_SEED)


regressors = [svr_lin, svr_rbf, lr, ridge, lasso]

stack = StackingCVRegressor(regressors=regressors,
                            meta_regressor=rf, 
                            use_features_in_secondary=True)

'''params = {'lasso__alpha': [0.1, 1.0, 10.0],
          'ridge__alpha': [0.1, 1.0, 10.0],
          'svr__C': [0.1, 1.0, 10.0],
          'meta-svr__C': [0.1, 1.0, 10.0, 100.0],
          'meta-svr__gamma': [0.1, 1.0, 10.0]}


params = {'lasso__alpha': [0.1, 1.0, 10.0],
          'ridge__alpha': [0.1, 1.0, 10.0]}'''

model = GridSearchCV(
    estimator=stack, 
    param_grid={
        'lasso__alpha': [x/5.0 for x in range(1, 10)],
        'ridge__alpha': [x/20.0 for x in range(1, 10)],
        'meta-randomforestregressor__n_estimators': [10, 100]
    }, 
    cv=5,
    refit=True,
    verbose=10,
    n_jobs=8,
)


###################################################

model.fit(x_train, y_train.ravel())

print("Best: %f using %s" % (grid.best_score_, grid.best_params_))

###################################################
y_pred = model.predict(x_test)
y_pred = np.expand_dims(y_pred,axis=1)


# Rescale Back
y_pred = 10**y_scaler.inverse_transform(y_pred)
y_test = 10**y_scaler.inverse_transform(y_test)

# Reject test samples below certain threshold
Rf_thresh = 0.5*1e-7 # 0.5*1e-6
ijk = y_test > Rf_thresh
y_test = y_test[ijk]
y_pred = y_pred[ijk]
x_test = x_test[ijk.flatten(),:]

# Add bias
#y_pred = y_pred + 0.1*y_pred


# sort results in Ascending order
y_test_sort = np.sort(y_test,axis=0)
y_pred_sort = y_pred[np.argsort(y_test,axis=0)]


## Percentage within the specified factor
Fac = 2
IDX = y_pred_sort/(y_test_sort+np.finfo(float).eps) >= 1
K = y_pred_sort[IDX]
Q = y_test_sort[IDX]
L = y_pred_sort[~IDX]
M = y_test_sort[~IDX]
Upper_indices = [i for i, x in enumerate(K <= Fac*Q) if x == True]
Lower_indices = [i for i, x in enumerate(L >= M/Fac) if x == True]
Percent_within_Fac = (len(Upper_indices) + len(Lower_indices))/len(y_pred)*100
print("Percentage captured within a factor of {} = {:.2f}".format(Fac,Percent_within_Fac))       

Diff = abs(y_pred_sort - y_test_sort)

# Errorbar values
yerr_lower = y_test_sort - y_test_sort/Fac
yerr_upper = Fac*y_test_sort - y_test_sort


idx = np.arange(0,len(y_test_sort))



if opts.doPlots:
    font = {'weight' : 'bold',
        'size'   : 15}
    plt.rc('font', **font)
    plt.rc('legend',**{'fontsize':15})

    plt.figure(figsize=(10,8))
    plt.style.use('dark_background')
    #plt.style.use('ggplot')

    diff_plt = plt.scatter(idx,Diff,color='lightgreen',alpha=0.1)
 
    errorbar_plt = plt.errorbar(idx,y_test_sort,yerr=[yerr_lower,yerr_upper], alpha=0.05 ,color='lightgrey')
    actual_plt = plt.scatter(idx,y_test_sort,color='#1f77b4',alpha=0.9)

    idx2 = np.arange(0,len(y_pred_sort))
    pred_plt = plt.scatter(idx2,y_pred_sort,color='#d62728',alpha=0.2)      

    plt.yscale('log')
    plt.grid()
    plt.ylim([1e-7, 1e-3])
    #plt.ylim([0, 1])
    #plt.ylabel('Rf Amplitude (m/s) \n (Normalized to 1)',fontsize=25)
    plt.ylabel('Rf Amplitude (m/s) ',fontsize=25)
    plt.xlabel('Samples',fontsize=25)
    plt.title("Percentage captured within a factor of {} = {:.2f}".format(Fac,Percent_within_Fac))
    legend_plt = plt.legend([pred_plt,actual_plt, diff_plt],['Prediction', 'Actual', 'Difference'],loc=2,markerscale=2., scatterpoints=100)

    plt.autoscale(enable=True, axis='x', tight=True)
    plt.grid(linestyle=':')

    plt.savefig(os.path.join(outputDirectory,'performance.pdf'),bbox_inches='tight')
    plt.close()
    
 
# Save Model
# serialize model & pickle
pickle.dump(model, open("%s/model.p"%outputDirectory, "wb"))
print("Saved model to disk")

'''
# Load Saved Model
# load pickle
pickle_file = open('%s/model.p'%outputDirectory, 'rb')
loaded_model_pickle = pickle.load(pickle_file)
print("Loaded model from disk")
'''

