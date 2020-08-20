#!/usr/bin/python

import sys, os, glob
import numpy as np
import scipy.linalg

import seismon.utils

try:
    import gwpy.time, gwpy.timeseries
    import gwpy.frequencyseries, gwpy.spectrogram
except:
    print("gwpy import fails... no plotting possible.")


__author__ = "Michael Coughlin <michael.coughlin@ligo.org>"
__date__ = "2012/8/26"
__version__ = "0.1"

# =============================================================================
#
#                               DEFINITIONS
#
# =============================================================================

def wiener(params, target_channel, segment):
    """@calculates wiener filter for given channel and segment.

    @param params
        seismon params dictionary
    @param target_channel
        seismon channel structure
    @param segment
        [start,end] gps
    """

    ifo = seismon.utils.getIfo(params)

    gpsStart = segment[0]
    gpsEnd = segment[1]

    # set the times
    duration = np.ceil(gpsEnd-gpsStart)

    dataAll = []

    if params["wienerFilterSampleRate"] > 0:
        samplef =params["wienerFilterSampleRate"]
    else:
        samplef = target_channel.samplef
    N = params["wienerFilterOrder"]

    for channel in params["channels"]:
        # make timeseries
        dataFull = seismon.utils.retrieve_timeseries(params, channel, segment)
        if dataFull == []:
            continue

        dataFull = dataFull / channel.calibration
        indexes = np.where(np.isnan(dataFull.data))[0]
        meanSamples = np.mean(np.ma.masked_array(dataFull.data,np.isnan(dataFull.data)))
        for index in indexes:
            dataFull[index] = meanSamples
        dataFull -= np.mean(dataFull.data)

        if np.mean(dataFull.data) == 0.0:
            print("data only zeroes... continuing\n")
            continue
        if len(dataFull.data) < 2*channel.samplef:
            print("timeseries too short for analysis... continuing\n")
            continue

        if params["wienerFilterSampleRate"] > 0:
            dataFull = dataFull.resample(params["wienerFilterSampleRate"])

        if (params["wienerFilterLowFreq"] > 0) and (params["wienerFilterHighFreq"] == 0):
            dataFull = dataFull.highpass(params["wienerFilterLowFreq"], amplitude=0.9, order=3, method='scipy')
        elif (params["wienerFilterLowFreq"] == 0) and (params["wienerFilterHighFreq"] > 0):
            dataFull = dataFull.lowpass(params["wienerFilterHighFreq"], amplitude=0.9, order=3, method='scipy')
        elif (params["wienerFilterLowFreq"] > 0) and (params["wienerFilterHighFreq"] > 0):
            dataFull = dataFull.bandpass(params["wienerFilterLowFreq"],params["wienerFilterHighFreq"], amplitude=0.9, order=3, method='scipy')

        indexes = np.where(np.isnan(dataFull.data))[0]
        meanSamples = np.mean(np.ma.masked_array(dataFull.data,np.isnan(dataFull.data)))
        for index in indexes:
            dataFull[index] = meanSamples
        dataFull -= np.mean(dataFull.data)

        dataAll.append(dataFull)

    X = []
    y = []
    for dataFull in dataAll:
        if dataFull.channel.name == target_channel.station:
            tt = np.array(dataFull.times)
            y = dataFull.data
        else:
            if X == []:
                X = dataFull.data
            else:
                try:
                    X = np.vstack([X,dataFull.data])
                except:
                    continue

    if len(y) == 0:
        print("No data for target channel... continuing")
        return

    originalASD = []
    residualASD = []
    FFASD = []

    gpss = np.arange(gpsStart,gpsEnd,params["fftDuration"])
    create_filter = True
    for i in range(len(gpss)-1):
        tt = np.array(dataFull.times)
        indexes = np.intersect1d(np.where(tt >= gpss[i])[0],np.where(tt <= gpss[i+1]+N)[0])

        if len(indexes) == 0:
            continue

        indexMin = np.min(indexes)
        indexMax = np.max(indexes)

        ttCut = tt[indexMin:indexMax] 
        yCut = y[indexMin:indexMax]

        if len(X.shape) == 1:
            XCut = np.reshape(X[indexMin:indexMax],(-1,1))
        else:
            XCut = X[:,indexMin:indexMax]
            XCut = XCut.T
        if create_filter:
            print("Generating filter")
            W,R,P = miso_firwiener(N,XCut,yCut)
            create_filter = False
            print("finished generating filter")
            continue
            
        residual, FF = subtractFF(W,XCut,yCut,samplef)

        thisGPSStart = tt[indexMin]
        dataOriginal = gwpy.timeseries.TimeSeries(yCut, epoch=thisGPSStart, sample_rate=samplef,name="Original")
        dataResidual = gwpy.timeseries.TimeSeries(residual, epoch=thisGPSStart, sample_rate=samplef,name="Residual")
        dataFF = gwpy.timeseries.TimeSeries(FF, epoch=thisGPSStart, sample_rate=samplef,name="FF")

        # calculate spectrum
        NFFT = params["fftDuration"]
        #window = None
        dataOriginalASD = dataOriginal.asd(NFFT,NFFT,'welch')
        dataResidualASD = dataResidual.asd(NFFT,NFFT,'welch')
        dataFFASD = dataFF.asd(NFFT,NFFT,'welch')

        freq = np.array(dataOriginalASD.frequencies)
        indexes = np.where((freq >= params["fmin"]) & (freq <= params["fmax"]))[0]
        freq = freq[indexes]

        dataOriginalASD = np.array(dataOriginalASD.data)
        dataOriginalASD = dataOriginalASD[indexes]
        dataOriginalASD = gwpy.frequencyseries.Spectrum(dataOriginalASD, f0=np.min(freq), df=(freq[1]-freq[0]))

        dataResidualASD = np.array(dataResidualASD.data)
        dataResidualASD = dataResidualASD[indexes]
        dataResidualASD = gwpy.frequencyseries.Spectrum(dataResidualASD, f0=np.min(freq), df=(freq[1]-freq[0]))

        dataFFASD = np.array(dataFFASD.data)
        dataFFASD = dataFFASD[indexes]
        dataFFASD = gwpy.frequencyseries.Spectrum(dataFFASD, f0=np.min(freq), df=(freq[1]-freq[0]))

        originalASD.append(dataOriginalASD)
        residualASD.append(dataResidualASD)
        FFASD.append(dataFFASD)

    dt = gpss[1] - gpss[0]
    epoch = gwpy.time.Time(gpss[0], format='gps')
    originalSpecgram = gwpy.spectrogram.Spectrogram.from_spectra(*originalASD, dt=dt,epoch=epoch)
    residualSpecgram = gwpy.spectrogram.Spectrogram.from_spectra(*residualASD, dt=dt,epoch=epoch)
    FFSpecgram = gwpy.spectrogram.Spectrogram.from_spectra(*FFASD, dt=dt,epoch=epoch)

    freq = np.array(originalSpecgram.frequencies)

    kwargs = {'log':True,'nbins':500,'norm':True}
    originalSpecvar = gwpy.frequencyseries.hist.SpectralVariance.from_spectrogram(originalSpecgram,**kwargs)
    bins = originalSpecvar.bins[:-1]
    originalSpecvar = originalSpecvar * 100
    original_spectral_variation_50per = originalSpecvar.percentile(50)
    residualSpecvar = gwpy.frequencyseries.hist.SpectralVariance.from_spectrogram(residualSpecgram,**kwargs)
    bins = residualSpecvar.bins[:-1]
    residualSpecvar = residualSpecvar * 100
    residual_spectral_variation_50per = residualSpecvar.percentile(50)
    FFSpecvar = gwpy.frequencyseries.hist.SpectralVariance.from_spectrogram(FFSpecgram,**kwargs)
    bins = FFSpecvar.bins[:-1]
    FFSpecvar = FFSpecvar * 100
    FF_spectral_variation_50per = FFSpecvar.percentile(50)

    psdDirectory = params["dirPath"] + "/Text_Files/Wiener/" + target_channel.station_underscore + "/" + str(params["fftDuration"]) + "/" + str(params["wienerFilterOrder"])
    seismon.utils.mkdir(psdDirectory)

    freq = np.array(residual_spectral_variation_50per.frequencies)

    psdFile = os.path.join(psdDirectory,"%d-%d.txt"%(gpsStart,gpsEnd))
    f = open(psdFile,"wb")
    for i in range(len(freq)):
        f.write("%e %e\n"%(freq[i],residual_spectral_variation_50per[i]))
    f.close()

    if params["doPlots"]:

        plotDirectory = params["path"] + "/Wiener/" + target_channel.station_underscore + "/" + str(params["fftDuration"]) + "/" + str(params["wienerFilterOrder"])
        seismon.utils.mkdir(plotDirectory)

        fl, low, fh, high = seismon.NLNM.NLNM(2)

        pngFile = os.path.join(plotDirectory,"psd.png")

        plot = gwpy.plotter.Plot(figsize=[14,8])
        kwargs = {"linestyle":"-","color":"b","label":"Original"}
        plot.add_line(freq, original_spectral_variation_50per, **kwargs)
        kwargs = {"linestyle":"-","color":"g","label":"Residual"}
        plot.add_line(freq, residual_spectral_variation_50per, **kwargs)
        kwargs = {"linestyle":"-","color":"r","label":"FF"}
        plot.add_line(freq, FF_spectral_variation_50per, **kwargs)
        kwargs = {"linestyle":"-.","color":"k"}
        plot.add_line(fl, low, **kwargs)
        plot.add_line(fh, high, **kwargs)
        plot.axes[0].set_yscale("log")
        plot.axes[0].set_xscale("log")
        plot.xlim = [params["fmin"],params["fmax"]]
        #plot.ylim = [np.min(bins), np.max(bins)]
        plot.xlabel = "Frequency [Hz]"
        plot.ylabel = "Amplitude Spectrum [(m/s)/rtHz]"
        plot.add_legend(loc=1,prop={'size':10})
        plot.save(pngFile,dpi=200)
        plot.close()

        pngFile = os.path.join(plotDirectory,"ratio.png")

        plot = gwpy.plotter.Plot(figsize=[14,8])
        kwargs = {"linestyle":"-","color":"b"}
        plot.add_line(freq, residual_spectral_variation_50per/original_spectral_variation_50per, **kwargs)
        plot.axes[0].set_xscale("log")
        plot.xlim = [params["fmin"],params["fmax"]]
        #plot.ylim = [np.min(bins), np.max(bins)]
        plot.xlabel = "Frequency [Hz]"
        plot.ylabel = "Amplitude Ratio"
        plot.save(pngFile,dpi=200)
        plot.close()

def wiener_hilbert(params, segment):
    """@calculates wiener filter for given channel and segment.

    @param params
        seismon params dictionary
    @param target_channel
        seismon channel structure
    @param segment
        [start,end] gps
    """

    ifo = seismon.utils.getIfo(params)

    gpsStart = segment[0]
    gpsEnd = segment[1]

    # set the times
    duration = np.ceil(gpsEnd-gpsStart)

    dataAll = []

    if params["wienerFilterSampleRate"] > 0:
        samplef =params["wienerFilterSampleRate"]
    else:
        samplef = params["channels"][0].samplef
    N = params["wienerFilterOrder"]

    for channel in params["channels"]:
        # make timeseries
        dataFull = seismon.utils.retrieve_timeseries(params, channel, segment)
        if dataFull == []:
            continue

        dataFull = dataFull / channel.calibration
        indexes = np.where(np.isnan(dataFull.data))[0]
        meanSamples = np.mean(np.ma.masked_array(dataFull.data,np.isnan(dataFull.data)))
        for index in indexes:
            dataFull[index] = meanSamples
        dataFull -= np.mean(dataFull.data)

        if np.mean(dataFull.data) == 0.0:
            print("data only zeroes... continuing\n")
            continue
        if len(dataFull.data) < 2*channel.samplef:
            print("timeseries too short for analysis... continuing\n")
            continue

        if params["wienerFilterSampleRate"] > 0:
            dataFull = dataFull.resample(samplef)

        if (params["wienerFilterLowFreq"] > 0) and (params["wienerFilterHighFreq"] == 0):
            dataFull = dataFull.highpass(params["wienerFilterLowFreq"], amplitude=0.9, order=3, method='scipy')
        elif (params["wienerFilterLowFreq"] == 0) and (params["wienerFilterHighFreq"] > 0):
            dataFull = dataFull.lowpass(params["wienerFilterHighFreq"], amplitude=0.9, order=3, method='scipy')
        elif (params["wienerFilterLowFreq"] > 0) and (params["wienerFilterHighFreq"] > 0):
            dataFull = dataFull.bandpass(params["wienerFilterLowFreq"],params["wienerFilterHighFreq"], amplitude=0.9, order=3, method='scipy')

        indexes = np.where(np.isnan(dataFull.data))[0]
        meanSamples = np.mean(np.ma.masked_array(dataFull.data,np.isnan(dataFull.data)))
        for index in indexes:
            dataFull[index] = meanSamples
        dataFull -= np.mean(dataFull.data)

        dataAll.append(dataFull)

    for dataFull in dataAll:
        if "X" in dataFull.channel.name or "E" == dataFull.channel.name[-1]:
            tsx = dataFull.data
        if "Y" in dataFull.channel.name or "N" == dataFull.channel.name[-1]:
            tsy = dataFull.data
        if "Z" in dataFull.channel.name:
            tsz = dataFull.data

    tt = np.array(dataAll[0].times)

    tszhilbert = scipy.signal.hilbert(tsz).imag
    tszhilbert = -tszhilbert

    angles = np.linspace(0,2*np.pi,60)
    xcorrs = []
    for angle in angles:
        rot = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    
        twodarray = np.vstack([tsx,tsy])
        z = rot.dot(twodarray)
        tsxy = np.sum(z.T,axis=1)
    
        tt = np.array(dataFull.times)
        indexes = np.arange(0,len(tt))

        indexMin = np.min(indexes)
        indexMax = np.max(indexes)

        tszhilbertCut = tszhilbert[indexMin:indexMax]
        tsxyCut = tsxy[indexMin:indexMax]

        xcorr,lags = seismon.utils.xcorr(tszhilbertCut,tsxyCut,maxlags=1)
        xcorrs.append(xcorr[1])
    xcorrs = np.array(xcorrs)

    angleMax = angles[np.argmax(xcorrs)]
    rot = np.array([[np.cos(angleMax), -np.sin(angleMax)],[np.sin(angleMax),np.cos(angleMax)]])

    print("Using angle: %f"%(angleMax))

    #angle = 0
    #rot = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle),np.cos(angle)]])

    twodarray = np.vstack([tsx,tsy])
    z = rot.dot(twodarray)
    tsxy = np.sum(z.T,axis=1)

    dataHilbert = tszhilbert.view(tsz.__class__)
    dataHilbert = gwpy.timeseries.TimeSeries(dataHilbert)
    dataHilbert.sample_rate = samplef 
    dataHilbert.epoch = tt[0]
    dataHilbert = dataHilbert.resample(1)

    dataXY = tsxy.view(tsz.__class__)
    dataXY = gwpy.timeseries.TimeSeries(tsxy)
    dataXY.sample_rate = samplef
    dataXY.epoch = tt[0]
    dataXY = dataXY.resample(1)

    dataZ = gwpy.timeseries.TimeSeries(tsz)
    dataZ.sample_rate = samplef
    dataZ.epoch = tt[0]
    dataZ = dataZ.resample(1)

    if params["doPlots"]:

        plotDirectory = params["path"] + "/Wiener_Hilbert/" + params["channels"][0].station_underscore + "/" + str(params["fftDuration"]) + "/" + str(params["wienerFilterOrder"])
        seismon.utils.mkdir(plotDirectory)

        pngFile = os.path.join(plotDirectory,"timeseries.png")

        dataHilbert *= 1e6
        dataXY *= 1e6
        dataZ *= 1e6

        plot = gwpy.plotter.TimeSeriesPlot(figsize=[14,8])

        kwargs = {"linestyle":"-","color":"b"}
        plot.add_timeseries(dataHilbert,label="Hilbert",**kwargs)
        kwargs = {"linestyle":"-","color":"g"}
        plot.add_timeseries(dataXY,label="XY",**kwargs)
        kwargs = {"linestyle":"-","color":"r"}
        #plot.add_timeseries(dataZ,label="Z",**kwargs)
        plot.ylabel = r"Velocity [$\mu$m/s]"
        plot.add_legend(loc=1,prop={'size':10})

        plot.save(pngFile)
        plot.close()

    y = tsxy
    X = np.vstack([tszhilbert,tsz])

    if len(y) == 0:
        print("No data for target channel... continuing")
        return

    originalASD = []
    residualASD = []
    FFASD = []

    gpss = np.arange(gpsStart,gpsEnd,params["fftDuration"])
    create_filter = True
    for i in range(len(gpss)-2):
        tt = np.array(dataFull.times)
        indexes = np.intersect1d(np.where(tt >= gpss[i])[0],np.where(tt <= gpss[i+1]+params["fftDuration"])[0])

        if len(indexes) == 0:
            continue

        indexMin = np.min(indexes)
        indexMax = np.max(indexes)

        ttCut = tt[indexMin:indexMax]
        yCut = y[indexMin:indexMax]
        XCut = X[:,indexMin:indexMax]

        XCut = XCut.T
        if create_filter:
            print("Generating filter")
            W,R,P = miso_firwiener(N,XCut,yCut)
            create_filter = False
            print("finished generating filter")
            continue

        residual, FF = subtractFF(W,XCut,yCut,samplef)

        thisGPSStart = tt[indexMin]
        dataOriginal = gwpy.timeseries.TimeSeries(yCut, epoch=thisGPSStart, sample_rate=samplef,name="Original")
        dataResidual = gwpy.timeseries.TimeSeries(residual, epoch=thisGPSStart, sample_rate=samplef,name="Residual")
        dataFF = gwpy.timeseries.TimeSeries(FF, epoch=thisGPSStart, sample_rate=samplef,name="FF")

        # calculate spectrum
        NFFT = params["fftDuration"]
        #window = None
        dataOriginalASD = dataOriginal.asd(NFFT,NFFT,'welch')
        dataResidualASD = dataResidual.asd(NFFT,NFFT,'welch')
        dataFFASD = dataFF.asd(NFFT,NFFT,'welch')

        freq = np.array(dataOriginalASD.frequencies)
        indexes = np.where((freq >= params["fmin"]) & (freq <= params["fmax"]))[0]
        freq = freq[indexes]

        dataOriginalASD = np.array(dataOriginalASD.data)
        dataOriginalASD = dataOriginalASD[indexes]
        dataOriginalASD = gwpy.frequencyseries.Spectrum(dataOriginalASD, f0=np.min(freq), df=(freq[1]-freq[0]))

        dataResidualASD = np.array(dataResidualASD.data)
        dataResidualASD = dataResidualASD[indexes]
        dataResidualASD = gwpy.frequencyseries.Spectrum(dataResidualASD, f0=np.min(freq), df=(freq[1]-freq[0]))

        dataFFASD = np.array(dataFFASD.data)
        dataFFASD = dataFFASD[indexes]
        dataFFASD = gwpy.frequencyseries.Spectrum(dataFFASD, f0=np.min(freq), df=(freq[1]-freq[0]))

        originalASD.append(dataOriginalASD)
        residualASD.append(dataResidualASD)
        FFASD.append(dataFFASD)

    dt = gpss[1] - gpss[0]
    epoch = gwpy.time.Time(gpss[0], format='gps')
    originalSpecgram = gwpy.spectrogram.Spectrogram.from_spectra(*originalASD, dt=dt,epoch=epoch)
    residualSpecgram = gwpy.spectrogram.Spectrogram.from_spectra(*residualASD, dt=dt,epoch=epoch)
    FFSpecgram = gwpy.spectrogram.Spectrogram.from_spectra(*FFASD, dt=dt,epoch=epoch)

    freq = np.array(originalSpecgram.frequencies)

    kwargs = {'log':True,'nbins':500,'norm':True}
    originalSpecvar = gwpy.frequencyseries.hist.SpectralVariance.from_spectrogram(originalSpecgram,**kwargs)
    bins = originalSpecvar.bins[:-1]
    originalSpecvar = originalSpecvar * 100
    original_spectral_variation_50per = originalSpecvar.percentile(50)
    residualSpecvar = gwpy.frequencyseries.hist.SpectralVariance.from_spectrogram(residualSpecgram,**kwargs)
    bins = residualSpecvar.bins[:-1]
    residualSpecvar = residualSpecvar * 100
    residual_spectral_variation_50per = residualSpecvar.percentile(50)
    FFSpecvar = gwpy.frequencyseries.hist.SpectralVariance.from_spectrogram(FFSpecgram,**kwargs)
    bins = FFSpecvar.bins[:-1]
    FFSpecvar = FFSpecvar * 100
    FF_spectral_variation_50per = FFSpecvar.percentile(50)

    psdDirectory = params["dirPath"] + "/Text_Files/Wiener_Hilbert/" + params["channels"][0].station_underscore + "/" + str(params["fftDuration"]) + "/" + str(params["wienerFilterOrder"])
    seismon.utils.mkdir(psdDirectory)

    freq = np.array(residual_spectral_variation_50per.frequencies)

    psdFile = os.path.join(psdDirectory,"%d-%d.txt"%(gpsStart,gpsEnd))
    f = open(psdFile,"wb")
    for i in range(len(freq)):
        f.write("%e %e\n"%(freq[i],residual_spectral_variation_50per[i]))
    f.close()

    if params["doPlots"]:

        plotDirectory = params["path"] + "/Wiener_Hilbert/" + params["channels"][0].station_underscore + "/" + str(params["fftDuration"]) + "/" + str(params["wienerFilterOrder"])
        seismon.utils.mkdir(plotDirectory)

        fl, low, fh, high = seismon.NLNM.NLNM(2)

        pngFile = os.path.join(plotDirectory,"psd.png")

        plot = gwpy.plotter.Plot(figsize=[14,8])
        kwargs = {"linestyle":"-","color":"b","label":"Original"}
        plot.add_line(freq, original_spectral_variation_50per, **kwargs)
        kwargs = {"linestyle":"-","color":"g","label":"Residual"}
        plot.add_line(freq, residual_spectral_variation_50per, **kwargs)
        kwargs = {"linestyle":"-","color":"r","label":"FF"}
        plot.add_line(freq, FF_spectral_variation_50per, **kwargs)
        kwargs = {"linestyle":"-.","color":"k"}
        plot.add_line(fl, low, **kwargs)
        plot.add_line(fh, high, **kwargs)
        plot.axes[0].set_yscale("log")
        plot.axes[0].set_xscale("log")
        plot.xlim = [params["fmin"],params["fmax"]]
        #plot.ylim = [np.min(bins), np.max(bins)]
        plot.xlabel = "Frequency [Hz]"
        plot.ylabel = "Amplitude Spectrum [(m/s)/rtHz]"
        plot.add_legend(loc=1,prop={'size':10})
        plot.save(pngFile,dpi=200)
        plot.close()

def miso_firwiener(N,X,y):

    # MISO_FIRWIENER Optimal FIR Wiener filter for multiple inputs.
    # MISO_FIRWIENER(N,X,Y) computes the optimal FIR Wiener filter of order
    # N, given any number of (stationary) random input signals as the columns
    # of matrix X, and one output signal in column vector Y.
    # Author: Keenan Pepper
    # Last modified: 2007/08/02
    # References:
    # [1] Y. Huang, J. Benesty, and J. Chen, Acoustic MIMO Signal
    # Processing, SpringerVerlag, 2006, page 48

    # Number of input channels.
    try:
        junk, M = X.shape
    except:
        M = 1

    # Input covariance matrix.
    R = np.zeros([M*(N+1),M*(N+1)])
    for m in range(M):
        for i in range(m,M):
            rmi,lags = seismon.utils.xcorr(X[:,m]-np.mean(X[:,m]),X[:,i]-np.mean(X[:,i]),maxlags=N,normed=False)
            Rmi = scipy.linalg.toeplitz(np.flipud(rmi[range(N+1)]),r=rmi[range(N,2*N+1)])
            top = m*(N+1)
            bottom = (m+1)*(N+1)
            left = i*(N+1)
            right = (i+1)*(N+1)
            #R[range(top,bottom),range(left,right)] = Rmi

            for j in range(top,bottom):
                for k in range(left,right):
                    R[j,k] = Rmi[j-top,k-left]
         
            if not i == m:
                #R[range(left,right),range(top,bottom)] = Rmi  # Take advantage of hermiticity.

                RmiT = Rmi.T
                for j in range(left,right):
                    for k in range(top,bottom):
                        R[j,k] = RmiT[j-left,k-top]

    # Crosscorrelation vector.
    P = np.zeros([M*(N+1),])
    for i in range(M):
        top = i*(N+1)
        bottom = (i+1)*(N+1)

        p, lags = seismon.utils.xcorr(y-np.mean(y),X[:,i]-np.mean(X[:,i]),maxlags=N,normed=False)

        P[range(top,bottom)] = p[range(N,2*N+1)]

    # The following step is very inefficient because it fails to exploit the
    # block Toeplitz structure of R. Its done the same way in the builtin
    # function "firwiener".
    # P / R

    Z = np.linalg.lstsq(R.T, P.T)[0].T
    W = Z.reshape(M,N+1).T

    return W,R,P

def subtractFF(W,SS,S,samplef):

    # Subtracts the filtered SS from S using FIR filter coefficients W.
    # Routine written by Jan Harms. Routine modified by Michael Coughlin.
    # Modified: August 17, 2012
    # Contact: michael.coughlin@ligo.org

    N = len(W)-1
    ns = len(S)

    FF = np.zeros([ns-N,])

    for k in range(N,ns):
        tmp = SS[k-N:k+1,:] * W
        FF[k-N] = np.sum(tmp)

    cutoff = 1.0
    dataFF = gwpy.timeseries.TimeSeries(FF, sample_rate=samplef)
    #dataFFLowpass = dataFF.lowpass(cutoff, amplitude=0.9, order=3, method='scipy')
    #FF = np.array(dataFFLowpass)
    FF = np.array(dataFF)

    residual = S[range(ns-N)]-FF
    residual = residual - np.mean(residual)

    return residual, FF

def wiener_summary(params, target_channel, segment):
    """@calculates wiener filter for given channel and segment.

    @param params
        seismon params dictionary
    @param target_channel
        seismon channel structure
    @param segment
        [start,end] gps
    """

    gpsStart = segment[0]
    gpsEnd = segment[1]

    psdDirectory = params["dirPath"] + "/Text_Files/Wiener/" + target_channel.station_underscore + "/" + str(params["fftDuration"])

    directories = glob.glob(os.path.join(psdDirectory,"*"))

    data = {}

    for directory in directories:

        directorySplit = directory.split("/")
        N = int(directorySplit[-1])

        file = os.path.join(directory,"%d-%d.txt"%(gpsStart,gpsEnd))

        if not os.path.isfile(file):
            continue

        spectra_out = gwpy.frequencyseries.Spectrum.read(file)
        spectra_out.unit = 'counts/Hz^(1/2)'

        if np.sum(spectra_out.data) == 0.0:
            continue

        data[str(N)] = {}
        data[str(N)]["data"] = spectra_out

    if data == {}:
        return

    if params["doPlots"]:

        plotDirectory = params["path"] + "/wiener_summary/" + target_channel.station_underscore
        seismon.utils.mkdir(plotDirectory)

        fl, low, fh, high = seismon.NLNM.NLNM(2)

        pngFile = os.path.join(plotDirectory,"psd.png")
        lowBin = np.inf
        highBin = -np.inf
        plot = gwpy.plotter.Plot(figsize=[14,8])
        for key in data.iterkeys():

            label = key.replace("_","\_")

            plot.add_spectrum(data[key]["data"], label=label)
            lowBin = np.min([lowBin,np.min(data[key]["data"])])
            highBin = np.max([highBin,np.max(data[key]["data"])])

        kwargs = {"linestyle":"-.","color":"k"}
        plot.add_line(fl, low, **kwargs)
        plot.add_line(fh, high, **kwargs)
        plot.xlim = [params["fmin"],params["fmax"]]
        plot.ylim = [lowBin, highBin]
        plot.xlabel = "Frequency [Hz]"
        plot.ylabel = "Amplitude Spectrum [(m/s)/rtHz]"
        plot.add_legend(loc=1,prop={'size':10})
        plot.axes[0].set_xscale("log")
        plot.axes[0].set_yscale("log")

        plot.save(pngFile,dpi=200)
        plot.close()


