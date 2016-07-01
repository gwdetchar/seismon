import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mu, sigma = 1.5e-06, 1.0e-02
hist_array = np.random.normal(mu, sigma, 1000)
count, bins,ignored = plt.hist(hist_array, normed=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu) ** 2 / (2 * sigma**2) ),linewidth=2,color='r')
plt.savefig('/home/eric.coughlin/public_html/hist_test.png') 
