
import os, sys, time, glob, math, matplotlib, random, string, optparse
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 18})
from matplotlib import pyplot as plt
from matplotlib import cm

import scipy.signal, scipy.stats
from scipy import optimize
import numpy as np

import gwpy.time, gwpy.timeseries
import gwpy.frequencyseries, gwpy.spectrogram
import gwpy.plotter

__author__ = "Michael Coughlin <michael.coughlin@ligo.org>"
__version__ = 1.0
__date__    = "9/22/2013"

# =============================================================================
#
#                               DEFINITIONS
#
# =============================================================================

def parse_commandline():
    """@parse the options given on the command-line.
    """
    parser = optparse.OptionParser(usage=__doc__,version=__version__)

    parser.add_option("-i", "--ifo", help="ifo.",
                      default ="H1")
    parser.add_option("-s", "--gpsStart", help="GPS Start Time.", default=1000000000,type=int)
    parser.add_option("-e", "--gpsEnd", help="GPS End Time.", default=1200000000,type=int)

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

def get_data(ifo, gpsStart, gpsEnd):
    channel = '%s:ISI-GND_STS_HAM2_Z_DQ'%ifo
    dataFull = gwpy.timeseries.TimeSeries.fetch(channel, gpsStart, gpsEnd, verbose=True)
    dataFull = dataFull.resample(16)
    # convert to m/s
    dataFull *= 1e-9
    # convert to nm/s
    dataFull *= 1e6
    fs = 1/dataFull.dx.value

    return dataFull, fs

def filter_data(dataFull,cutoff_low,cutoff_high):

    cutoff_band = np.array([cutoff_low,cutoff_high])
    
    n = 3
    if cutoff_band[0] == 0:
        B, A = scipy.signal.butter(n, cutoff_band[1] / (fs / 2.0), btype='lowpass')
    elif cutoff_band[1] == 0:
        B, A = scipy.signal.butter(n, cutoff_band[0] / (fs / 2.0), btype='highpass')
    else:
        B, A = scipy.signal.butter(n, cutoff_band / (fs / 2.0), btype='bandpass')
    
    data = scipy.signal.lfilter(B, A, dataFull,
                                        axis=0).view(dataFull.__class__)
    data.data[np.isnan(data.data)] = 0.0
    data.data[:2*fs] = data.data[2*fs]
    data.data[-2*fs:] = data.data[-2*fs]
    data.dx =  dataFull.dx
    data.epoch = dataFull.epoch

    return data

def get_snr(data,data_off):

    on_max = np.max(np.abs(data.data))
    off_max = np.max(np.abs(data_off.data))
    snr_max = on_max / off_max

    return on_max, off_max, snr_max

# Parse command line
opts = parse_commandline()

ifo = opts.ifo
gpsStart = opts.gpsStart
gpsEnd = opts.gpsEnd
gpsStart_off = opts.gpsStart - 600
gpsEnd_off = opts.gpsStart

dataFull, fs = get_data(ifo,gpsStart,gpsEnd)
dataFull_off, fs = get_data(ifo,gpsStart_off,gpsEnd_off)

cutoff_low = 0.01
cutoff_high = 0.5
cutoff_low = 0.0
cutoff_high = 8.0
dataLowpass = filter_data(dataFull,cutoff_low,cutoff_high)
dataLowpass_off = filter_data(dataFull_off,cutoff_low,cutoff_high)

cutoffs = np.logspace(-2,np.log10(8.0),50)
#cutoffs_log10 = np.log10(cutoffs)
#[X,Y] = np.meshgrid(cutoffs_log10,cutoffs_log10)
[X,Y] = np.meshgrid(cutoffs,cutoffs)

cutoffs[0] = 0.0
Z = np.zeros(X.shape)
Z_snr = np.zeros(X.shape)

for ii in xrange(len(cutoffs)):
    for jj in xrange(len(cutoffs)):
        cutoff_low = cutoffs[ii]
        cutoff_high = cutoffs[jj]
        if ii+2 >= jj: continue

        data = filter_data(dataFull,cutoff_low,cutoff_high)
        data_off = filter_data(dataFull_off,cutoff_low,cutoff_high)
        on_max, off_max, snr_max = get_snr(data,data_off)

        if on_max > 1e4: continue
        if off_max > 1e4: continue

        #print "%.2e %.2e %.1f %.1f %.1f"%(cutoff_low,cutoff_high, on_max, off_max, snr_max)      
        Z[ii,jj] = on_max
        Z_snr[ii,jj] = snr_max

ii,jj = np.unravel_index(Z.argmax(), Z.shape)
ii,jj = np.unravel_index(Z_snr.argmax(), Z_snr.shape)
cutoff_low = cutoffs[ii]
cutoff_high = cutoffs[jj]
data = filter_data(dataFull,cutoff_low,cutoff_high)
data_off = filter_data(dataFull_off,cutoff_low,cutoff_high)
on_max, off_max, snr_max = get_snr(data,data_off)

print "Best: %.2e %.2e %.1f %.1f %.1f"%(cutoff_low,cutoff_high, on_max, off_max, snr_max)

plotDirectory = "/home/mcoughlin/Seismon/FilterStudy/%s/"%(ifo)
if not os.path.isdir(plotDirectory): os.mkdir(plotDirectory)
plotDirectory = "/home/mcoughlin/Seismon/FilterStudy/%s/%d_%d/"%(ifo,gpsStart,gpsEnd)
if not os.path.isdir(plotDirectory): os.mkdir(plotDirectory)

plot = gwpy.plotter.TimeSeriesPlot(figsize=[14,8])
kwargs = {"linestyle":"-","color":"k"}
plot.add_timeseries(dataLowpass,**kwargs)
plot.ylabel = r"Velocity [$\mu$m/s]"
pdfFile = os.path.join(plotDirectory,"timeseries_raw.pdf")
plot.save(pdfFile)
plot.close()

xmin, xmax = 0.01,8.0
ymin, ymax = 10**-3, 10**2

asdLowpass = dataLowpass.asd(128, 64)
asd = data.asd(128, 64)
asdLowpass_off = dataLowpass_off.asd(128, 64)
asd_off = data_off.asd(128, 64)

plot = gwpy.plotter.FrequencySeriesPlot(figsize=[14,8])
kwargs = {"linestyle":"-","color":"k"}
plot.add_spectrum(asdLowpass,label="on source",**kwargs)
kwargs = {"linestyle":"-.","color":"r"}
plot.add_spectrum(asdLowpass_off,label="off source",**kwargs)
kwargs = {"linestyle":":","color":"b"}
plot.add_spectrum(asd,label="on source filtered",**kwargs)
kwargs = {"linestyle":"--","color":"g"}
plot.add_spectrum(asd_off,label="off source filtered",**kwargs)
plot.xlabel = "Frequency [Hz]"
plot.ylabel = r"Velocity [$\mu$m/s / \rtHz]"
plot.xlim = [xmin, xmax]
plot.ylim = [ymin, ymax]
plot.axes[0].set_xscale("log")
plot.axes[0].set_yscale("log")
pdfFile = os.path.join(plotDirectory,"psd.pdf")
plot.save(pdfFile)
plot.close()

plot = gwpy.plotter.TimeSeriesPlot(figsize=[14,8])
kwargs = {"linestyle":"-","color":"k"}
plot.add_timeseries(data,**kwargs)
plot.ylabel = r"Velocity [$\mu$m/s]"
pdfFile = os.path.join(plotDirectory,"timeseries.pdf")
plot.save(pdfFile)
plot.close()

plot = gwpy.plotter.TimeSeriesPlot(figsize=[14,8])
kwargs = {"linestyle":"-","color":"k"}
plot.add_timeseries(data_off,**kwargs)
plot.ylabel = r"Velocity [$\mu$m/s]"
pdfFile = os.path.join(plotDirectory,"timeseries_off.pdf")
plot.save(pdfFile)
plot.close()

plt.figure()
ax = plt.subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
plt.pcolor(X,Y,Z)
#plt.imshow(Z)
plt.xlabel("Upper knee Cutoff [Hz]")
plt.ylabel("Lower Knee Cutoff [Hz]")
cbar = plt.colorbar()
cbar.set_label(r"Max Velocity [$\mu$m/s]")
pdfFile = os.path.join(plotDirectory,"vel.pdf")
plt.savefig(pdfFile)
plt.close('all')

plt.figure()
ax = plt.subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
plt.pcolor(X,Y,Z_snr)
#plt.imshow(Z)
plt.xlabel("Upper Knee Cutoff [Hz]")
plt.ylabel("Lower Knee Cutoff [Hz]")
cbar = plt.colorbar()
cbar.set_label("SNR")
pdfFile = os.path.join(plotDirectory,"snr.pdf")
plt.savefig(pdfFile)
plt.close('all')

