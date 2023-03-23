# Python GPR code for Rf Amplitude Prediction from Earthquake Parameters
# Nikhil Mukund Menon (nikhil@iucaa.in)
# 16th July 2017

from __future__ import division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel, RationalQuadratic, ExpSineSquared, Matern, ConstantKernel as C


filename = '/home/mcoughlin/Seismon/Predictions/L1O1O2_CMT/earthquakes.txt' 

'''
1: earthquake gps time
2: earthquake mag
3: p gps time
4: s gps time
5: r (2 km/s)
6: r (3.5 km/s)
7: r (5 km/s)
8: predicted ground motion (m/s)
9: lower bounding time
10: upper bounding time
11: latitude
12: longitude
13: distance
14: depth (m)
15: azimuth (deg)
16: nodalPlane1_strike
17: nodalPlane1_rake
18: nodalPlane1_dip
19: momentTensor_Mrt
20: momentTensor_Mtp
21: momentTensor_Mrp
22: momentTensor_Mtt
23: momentTensor_Mrr
24: momentTensor_Mpp
25: peak ground velocity gps time
26: peak ground velocity (m/s)
27: peak ground acceleration gps time
28: peak ground acceleration (m/s^2)
29: peak ground displacement gps time
30: peak ground displacement (m)
31: Lockloss time
32: Detector Status
'''

data = pd.read_csv(filename,delimiter=' ',header=None)

Rf_Amp_thresh = 1e-7; 
index = data[25] > Rf_Amp_thresh

data = data[:][index]

X = np.asarray(data[[1,10,7,11,12,13,14,15,16,17,18,19,20,21,22,23]])
Y = np.asarray(data[[25]])[:,np.newaxis]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

##############################################################################
# Instanciate a Gaussian Process model
# Choose Kernel [Tricky]

kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
kernel =  Matern(length_scale=0.2, nu=0.5) + WhiteKernel(noise_level=0.1) + C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

##############################################################################

gp = GaussianProcessRegressor(alpha=1e-3, copy_X_train=True,
kernel=kernel,
n_restarts_optimizer=10, normalize_y=False,
optimizer='fmin_l_bfgs_b', random_state=None)

'''
OKish Parameter Values 
gp = GaussianProcessRegressor(alpha=1e-7, copy_X_train=True,
kernel=1**2 + Matern(length_scale=0.2, nu=0.5) + WhiteKernel(noise_level=0.1),
n_restarts_optimizer=10, normalize_y=False,
optimizer='fmin_l_bfgs_b', random_state=None)
'''


# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(x_train,y_train)


#x = np.linspace(min(X),max(X),len(X))[:,np.newaxis]
y_pred, sigma = gp.predict(x_test, return_std=True)


## Percentage within the specified factor
Fac = 5
IDX = y_pred/y_test >= 1
K = y_pred[IDX]
Q = y_test[IDX]
L = y_pred[~IDX]
M = y_test[~IDX]
Upper_indices = [i for i, x in enumerate(K <= Fac*Q) if x == True]
Lower_indices = [i for i, x in enumerate(L >= M/Fac) if x == True]
Percent_within_Fac = (len(Upper_indices) + len(Lower_indices))/len(y_pred)*100
print("Percentage captured within a factor of {} = {:.2f}".format(Fac,Percent_within_Fac))



# sort results in Ascending order
y_test_sort = np.sort(y_test,axis=0)
y_pred_sort = y_pred[np.argsort(y_test,axis=0)]

# Errorbar values
yerr_lower = y_test_sort - y_test_sort/Fac
yerr_upper = Fac*y_test_sort - y_test_sort


idx = np.arange(0,len(y_test_sort))
#plt.scatter(idx,y_test_sort,color='red',alpha=0.3)
plt.errorbar(idx,y_test_sort,yerr=[yerr_lower,yerr_upper], alpha=0.3 )


idx2 = np.arange(0,len(y_pred_sort))
plt.scatter(idx2,y_pred_sort,color='green',alpha=0.7)      
plt.yscale('log')
plt.grid()
plt.ylim([1e-8, 1e-3])
plt.title("Percentage captured within a factor of {} = {:.2f}".format(Fac,Percent_within_Fac))
plt.savefig('GPR.png')


