import os, glob, optparse, shutil, warnings, pickle, math, copy, pickle, matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal, scipy.stats
from lxml import etree
import seismon.NLNM, seismon.html
import seismon.eqmon, seismon.utils

try:
    import gwpy.time, gwpy.timeseries
    import gwpy.frequencyseries, gwpy.spectrogram
    import gwpy.plotter
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

def hilbert(params, segment):
    """@calculates hilbert transform for given segment.

    @param params
        seismon params dictionary
    @param segment
        [start,end] gps

    """

    from obspy.core.util.geodetics import gps2DistAzimuth

    ifo = seismon.utils.getIfo(params)
    ifolat,ifolon = seismon.utils.getLatLon(params)

    gpsStart = segment[0]
    gpsEnd = segment[1]

    # set the times
    duration = np.ceil(gpsEnd-gpsStart)

    dataAll = []
    print("Loading data...")

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

        #cutoff = 0.01
        #dataFull = dataFull.lowpass(cutoff, amplitude=0.9, order=3, method='scipy')

        dataAll.append(dataFull)

    if len(dataAll) == 0:
        print("No data... returning")
        return

    if params["doEarthquakes"]:
        earthquakesDirectory = os.path.join(params["path"],"earthquakes")
        earthquakesXMLFile = os.path.join(earthquakesDirectory,"earthquakes.xml")
        attributeDics = seismon.utils.read_eqmons(earthquakesXMLFile)

    else:
        attributeDics = []

    print("Performing Hilbert transform")
    for attributeDic in attributeDics:

        if params["ifo"] == "IRIS":
            attributeDic = seismon.eqmon.ifotraveltimes(attributeDic, "IRIS", channel.latitude, channel.longitude)
            traveltimes = attributeDic["traveltimes"]["IRIS"]
        else:
            traveltimes = attributeDic["traveltimes"][ifo]

        doTilt = 0
        for dataFull in dataAll:
            if "_X_" in dataFull.channel.name or "E" == dataFull.channel.name[-1]:
                tsx = dataFull.data
            if "_Y_" in dataFull.channel.name or "N" == dataFull.channel.name[-1]:
                tsy = dataFull.data
            if "_Z_" in dataFull.channel.name:
                tsz = dataFull.data
            if "_RY_" in dataFull.channel.name:
                tttilt = np.array(dataFull.times)
                tstilt = dataFull.data
                doTilt = 1

        tt = np.array(dataAll[0].times)
        fs = 1.0/(tt[1]-tt[0])     
 
        if doTilt:   
            tstilt = np.interp(tt,tttilt,tstilt)

        Ptime = max(traveltimes["Ptimes"])
        Stime = max(traveltimes["Stimes"])
        Rtwotime = max(traveltimes["Rtwotimes"])
        RthreePointFivetime = max(traveltimes["RthreePointFivetimes"])
        Rfivetime = max(traveltimes["Rfivetimes"])
        distance = max(traveltimes["Distances"])

        indexes = np.intersect1d(np.where(tt >= Rfivetime)[0],np.where(tt <= Rtwotime)[0])
        indexes = np.intersect1d(np.where(tt >= tt[0])[0],np.where(tt <= tt[-1])[0])

        if len(indexes) == 0:
            continue

        indexMin = np.min(indexes)
        indexMax = np.max(indexes)
        tt = tt[indexes]
        tsx = tsx[indexes]
        tsy = tsy[indexes]
        tsz = tsz[indexes]

        if doTilt:
            tstilt = tstilt[indexes]

            cutoff_high = 0.01 # 10 MHz
            cutoff_low = 0.3
            n = 3
            worN = 16384
            B_low, A_low = scipy.signal.butter(n, cutoff_low / (fs / 2.0), btype='lowpass')
            #w_low, h_low = scipy.signal.freqz(B_low,A_low)
            w_low, h_low = scipy.signal.freqz(B_low,A_low,worN=worN)
            B_high, A_high = scipy.signal.butter(n, cutoff_high / (fs / 2.0), btype='highpass')
            w_high, h_high = scipy.signal.freqz(B_high,A_high,worN=worN)

            dataLowpass = scipy.signal.lfilter(B_low, A_low, tstilt,
                                        axis=0).view(dataFull.__class__)
            dataHighpass = scipy.signal.lfilter(B_high, A_high, tstilt,
                                        axis=0).view(dataFull.__class__)

            dataTilt = dataHighpass.view(dataHighpass.__class__)
            #dataTilt = tstilt.view(tstilt.__class__)
            dataTilt = gwpy.timeseries.TimeSeries(dataTilt)
            dataTilt.sample_rate = dataFull.sample_rate
            dataTilt.epoch = Rfivetime

        tszhilbert = scipy.signal.hilbert(tsz).imag
        tszhilbert = -tszhilbert

        dataHilbert = tszhilbert.view(tsz.__class__)
        dataHilbert = gwpy.timeseries.TimeSeries(dataHilbert)
        dataHilbert.sample_rate =  dataFull.sample_rate
        dataHilbert.epoch = Rfivetime

        distance,fwd,back = gps2DistAzimuth(attributeDic["Latitude"],attributeDic["Longitude"],ifolat,ifolon)
        xazimuth,yazimuth = seismon.utils.getAzimuth(params)

        angle1 = fwd
        rot1 = np.array([[np.cos(angle1), -np.sin(angle1)],[np.sin(angle1),np.cos(angle1)]])
        angle2 = xazimuth * (np.pi/180.0)
        rot2 = np.array([[np.cos(angle2), -np.sin(angle2)],[np.sin(angle2),np.cos(angle2)]])

        angleEQ = np.mod(angle1+angle2,2*np.pi)
        rot = np.array([[np.cos(angleEQ), -np.sin(angleEQ)],[np.sin(angleEQ),np.cos(angleEQ)]])

        twodarray = np.vstack([tsx,tsy])
        z = rot.dot(twodarray)
        tsxy = np.sum(z.T,axis=1)
     
        dataXY = tsxy.view(tsz.__class__)
        dataXY = gwpy.timeseries.TimeSeries(tsxy)
        dataXY.sample_rate = dataFull.sample_rate
        dataXY.epoch = Rfivetime

        dataHilbert = dataHilbert.resample(16)
        dataXY = dataXY.resample(16)

        if doTilt:
            dataTilt = dataTilt.resample(16)

        if params["doPlots"]:

            plotDirectory = params["path"] + "/Hilbert"
            seismon.utils.mkdir(plotDirectory)

            dataHilbert *= 1e6
            dataXY *= 1e6

            pngFile = os.path.join(plotDirectory,"%s.png"%(attributeDic["eventName"]))
            plot = gwpy.plotter.TimeSeriesPlot(figsize=[14,8])

            kwargs = {"linestyle":"-","color":"b"}
            plot.add_timeseries(dataHilbert,label="Hilbert",**kwargs)
            kwargs = {"linestyle":"-","color":"g"}
            plot.add_timeseries(dataXY,label="XY",**kwargs)
            plot.ylabel = r"Velocity [$\mu$m/s]"
            plot.add_legend(loc=1,prop={'size':10})

            plot.save(pngFile)
            plot.close()

        dataHilbert = tszhilbert.view(tsz.__class__)
        dataHilbert = gwpy.timeseries.TimeSeries(dataHilbert)
        dataHilbert.sample_rate =  dataFull.sample_rate
        dataHilbert.epoch = Rfivetime
        dataHilbert = dataHilbert.resample(16)

        angles = np.linspace(0,2*np.pi,10)
        xcorrs = []
        for angle in angles:
            rot = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle),np.cos(angle)]])

            twodarray = np.vstack([tsx,tsy])
            z = rot.dot(twodarray)
            tsxy = np.sum(z.T,axis=1)

            dataXY = tsxy.view(tsz.__class__)
            dataXY = gwpy.timeseries.TimeSeries(tsxy)
            dataXY.sample_rate = dataFull.sample_rate
            dataXY.epoch = Rfivetime
            dataXY = dataXY.resample(16)

            xcorr,lags = seismon.utils.xcorr(dataHilbert.data,dataXY.data,maxlags=1)
            xcorrs.append(xcorr[1])
        xcorrs = np.array(xcorrs)

        angleMax = angles[np.argmax(xcorrs)]
        rot = np.array([[np.cos(angleMax), -np.sin(angleMax)],[np.sin(angleMax),np.cos(angleMax)]])

        twodarray = np.vstack([tsx,tsy])
        z = rot.dot(twodarray)
        tsxy = np.sum(z.T,axis=1)

        dataXY = tsxy.view(tsz.__class__)
        dataXY = gwpy.timeseries.TimeSeries(tsxy)
        dataXY.sample_rate = dataFull.sample_rate
        dataXY.epoch = Rfivetime
        dataXY = dataXY.resample(16)

        if params["doPlots"]:

            plotDirectory = params["path"] + "/Hilbert"
            seismon.utils.mkdir(plotDirectory)

            dataHilbert *= 1e6
            dataXY *= 1e6

            if doTilt:
                dataTilt *= 1e6 * (9.81)/(2*np.pi*0.01)

            pngFile = os.path.join(plotDirectory,"%s-max.png"%(attributeDic["eventName"]))
            plot = gwpy.plotter.TimeSeriesPlot(figsize=[14,8])

            kwargs = {"linestyle":"-","color":"b"}
            plot.add_timeseries(dataHilbert,label="Hilbert",**kwargs)
            kwargs = {"linestyle":"-","color":"g"}
            plot.add_timeseries(dataXY,label="XY",**kwargs)
            plot.ylabel = r"Velocity [$\mu$m/s]"
            plot.add_legend(loc=1,prop={'size':10})

            plot.save(pngFile)
            plot.close()

            pngFile = os.path.join(plotDirectory,"%s-rot.png"%(attributeDic["eventName"]))
            plot = gwpy.plotter.Plot(figsize=[14,8])

            kwargs = {"linestyle":"-","color":"b"}
            plot.add_line(angles,xcorrs,label="Xcorrs",**kwargs)
            ylim = [plot.ylim[0],plot.ylim[1]]
            kwargs = {"linestyle":"--","color":"r"}
            plot.add_line([angleEQ,angleEQ],ylim,label="EQ",**kwargs)
            plot.ylabel = r"XCorr"
            plot.xlabel = r"Angle [rad]"
            plot.add_legend(loc=1,prop={'size':10})

            plot.save(pngFile)
            plot.close()

        dataHilbert = dataHilbert.resample(1.0)
        dataXY = dataXY.resample(1.0)

        if doTilt:
            dataTilt = dataTilt.resample(1.0)

        dataHilbertAbs = np.absolute(dataHilbert)
        dataXYAbs = np.absolute(dataXY)
        dataRatio = dataXYAbs / dataHilbertAbs

        meanValue = np.median(np.absolute(dataRatio))
        dataXYScale = dataXY/meanValue

        dataXYScale = gwpy.timeseries.TimeSeries(dataXYScale)
        dataXYScale.sample_rate = dataXY.sample_rate
        dataXYScale.epoch = dataXY.epoch

        if params["doPlots"]:

            plotDirectory = params["path"] + "/Hilbert"
            seismon.utils.mkdir(plotDirectory)

            pngFile = os.path.join(plotDirectory,"%s-max-abs.png"%(attributeDic["eventName"]))
            plot = gwpy.plotter.TimeSeriesPlot(figsize=[14,8])

            kwargs = {"linestyle":"-","color":"b"}
            plot.add_timeseries(dataHilbertAbs,label="Hilbert",**kwargs)
            kwargs = {"linestyle":"-","color":"g"}
            plot.add_timeseries(dataXYAbs,label="XY",**kwargs)
            plot.ylabel = r"Velocity [$\mu$m/s]"
            plot.add_legend(loc=1,prop={'size':10})

            plot.save(pngFile)
            plot.close()

            pngFile = os.path.join(plotDirectory,"%s-max-scale.png"%(attributeDic["eventName"]))
            plot = gwpy.plotter.TimeSeriesPlot(figsize=[14,8])

            kwargs = {"linestyle":"-","color":"b"}
            plot.add_timeseries(dataHilbert,label="Hilbert",**kwargs)
            kwargs = {"linestyle":"-","color":"g"}
            plot.add_timeseries(dataXYScale,label="XY Scaled",**kwargs)
            plot.ylabel = r"Velocity [$\mu$m/s]"
            plot.add_legend(loc=1,prop={'size':10})

            plot.save(pngFile)
            plot.close()

            pngFile = os.path.join(plotDirectory,"%s-ratio.png"%(attributeDic["eventName"]))
            plot = gwpy.plotter.TimeSeriesPlot(figsize=[14,8])

            kwargs = {"linestyle":"-","color":"k"}
            plot.add_timeseries(np.absolute(dataRatio),label="Ratio",**kwargs)
            plot.ylabel = r"Ratio [XY / Hilbert(Z)]"
          
            xlim = [plot.xlim[0],plot.xlim[1]]
            kwargs = {"linestyle":"-","color":"r"}
            plot.add_line(xlim,[0.78,0.78],label="PREM",**kwargs)

            plot.title = "Median Ratio: %.3f"%(meanValue)

            #plot.ylim = [0,100]

            plot.axes[0].set_yscale("log")
            plot.add_legend(loc=1,prop={'size':10})

            plot.save(pngFile)
            plot.close()

            if doTilt:
                pngFile = os.path.join(plotDirectory,"%s-max-scale-tilt.png"%(attributeDic["eventName"]))
                plot = gwpy.plotter.TimeSeriesPlot(figsize=[14,8])

                kwargs = {"linestyle":"-","color":"b"}
                plot.add_timeseries(dataHilbert,label="Hilbert",**kwargs)
                kwargs = {"linestyle":"-","color":"g"}
                plot.add_timeseries(dataXYScale,label="XY Scaled",**kwargs)
                kwargs = {"linestyle":"-","color":"k"}
                plot.add_timeseries(dataTilt,label="Tiltmeter",**kwargs)
                plot.ylabel = r"Velocity [$\mu$m/s]"
                plot.add_legend(loc=1,prop={'size':10})

                plot.save(pngFile)
                plot.close()

