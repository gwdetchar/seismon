import numpy as np
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

plt.figure(1)
mu, sigma = 1.5e-06, 1.0e-02
mu2, sigma2 = 1.0e-07,1.5e-02
hist_array = np.random.normal(mu, sigma, 1000)
hist2_array = np.random.normal(mu2,sigma2,1000)
prob = np.divide(hist_array,hist2_array)
count, bins,ignored = plt.hist(hist_array,30, normed=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu) ** 2 / (2 * sigma**2) ),linewidth=2,color='r',label='lockloss')
count3, bins3,ignored3 = plt.hist(hist2_array,30, normed=True)
plt.plot(bins3, 1/(sigma2 * np.sqrt(2 * np.pi)) * np.exp( - (bins3 - mu2) ** 2 / (2 * sigma2**2) ),linewidth=2,color='k',label='locked')
plt.title('Velocity Histogram(generated data)')
plt.legend(loc='best')
plt.savefig('/home/eric.coughlin/public_html/hist_test.png')
plt.figure(2)
acceleration_array = np.diff(hist_array)
acceleration2_array = np.diff(hist2_array)
prob_acceleration = np.divide(acceleration_array, acceleration2_array)
count, bins,ignored = plt.hist(acceleration_array,30, normed=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu) ** 2 / (2 * sigma**2) ),linewidth=2,color='r',label='lockloss')
count, bins,ignored = plt.hist(acceleration2_array,30, normed=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu) ** 2 / (2 * sigma**2) ),linewidth=2,color='k',label='locked')
plt.title('Acceleration Histogram(generated data)')
plt.legend(loc='best')
plt.savefig('/home/eric.coughlin/public_html/hist_acc_test.png')

plt.figure(3)
count, bins,ignored = plt.hist(hist_array,30, normed=True)
count2, bins2,ignored2 = plt.hist(acceleration_array,30, normed=True)
plt.plot((bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu) ** 2 / (2 * sigma**2) )), (bins2, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins2 - mu) ** 2 / (2 * sigma**2) )),linewidth=2,color='k',label='acc/vel')
plt.title('acceleration versus velocity(generated data)')
plt.xlabel('Velocity')
plt.ylabel('Acceleration')
plt.legend(loc='best')
plt.savefig('/home/eric.coughlin/public_html/hist_acc_vel_test.png')

plt.figure(4)

