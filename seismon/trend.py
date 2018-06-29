#!/usr/bin/python

import os, glob, optparse, shutil, warnings, pickle, math, copy, pickle, matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal, scipy.stats
import scipy.io
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

def trend(params, channel, segment):
    """@calculates spectral data for given channel and segment.

    @param params
        seismon params dictionary
    @param channel
        seismon channel structure
    @param segment
        [start,end] gps
    """

    ifo = seismon.utils.getIfo(params)

    gpsStart = segment[0]
    gpsEnd = segment[1]
   
    # set the times
    duration = np.ceil(gpsEnd-gpsStart)

    #matDir = "/home/mcoughlin/Seismon/seismicdata/STSA_BLRMS_Z_30M_100M"
    #mats = glob.glob(os.path.join(matDir,"*.mat"))

    t = []
    data = []

    #for mat in mats:
    #    mat_data = scipy.io.loadmat(mat) 
    #    if t == []:
    #        t = mat_data["timevect"]
    #        data = mat_data["datavect"]
    #    else:
    #        t = np.concatenate((t,mat_data["timevect"]))
    #        data = np.concatenate((data,mat_data["datavect"]))

    mat = "/home/mcoughlin/Seismon/seismicdata/STSA_BLRMS_Z_30M_100M_raw.mat" 
    #mat = "/home/mcoughlin/Seismon/seismicdata/STSA_BLRMS_Z_30M_100M.mat"
    mat_data = scipy.io.loadmat(mat)

    t = mat_data["time"]
    data = mat_data["data"]

    p = t[:,0].argsort() 
    t = t[p,0]
    data = data[p,0]

    sample_rate = 1.0/(t[1]-t[0])

    dataFull = gwpy.timeseries.TimeSeries(data, times=None, epoch=t[0], channel=channel.station, unit=None,sample_rate=sample_rate, name=channel.station)

    # make timeseries
    #dataFull = seismon.utils.retrieve_timeseries(params, channel, segment)
    if dataFull == []:
        return 

    dataFull = dataFull / channel.calibration
    indexes = np.where(np.isnan(dataFull.data))[0]
    meanSamples = np.mean(np.ma.masked_array(dataFull.data,np.isnan(dataFull.data)))
    for index in indexes:
        dataFull[index] = meanSamples
    dataFull -= np.mean(dataFull.data)

    if np.mean(dataFull.data) == 0.0:
        print("data only zeroes... continuing\n")
        return
    if len(dataFull.data) < 2*channel.samplef:
        print("timeseries too short for analysis... continuing\n")
        return

    if params["doEarthquakes"]:
        earthquakesDirectory = os.path.join(params["path"],"earthquakes")
        earthquakesXMLFile = os.path.join(earthquakesDirectory,"earthquakes.xml")
        attributeDics = seismon.utils.read_eqmons(earthquakesXMLFile)

    else:
        attributeDics = []

    if params["doPlots"]:

        plotDirectory = params["path"] + "/" + channel.station_underscore
        seismon.utils.mkdir(plotDirectory)        

        pngFile = os.path.join(plotDirectory,"trend.png")
        plot = gwpy.plotter.TimeSeriesPlot(figsize=[14,8])

        dataFull = dataFull.resample(16)

        plot.add_timeseries(dataFull,label="data")

        xlim = [plot.xlim[0],plot.xlim[1]]
        ylim = [plot.ylim[0],plot.ylim[1]]

        times = []
        velocities = []

        count = 0
        for attributeDic in attributeDics:

            if params["ifo"] == "IRIS":
                attributeDic = seismon.eqmon.ifotraveltimes(attributeDic, "IRIS", channel.latitude, channel.longitude)
                traveltimes = attributeDic["traveltimes"]["IRIS"]
            else:
                traveltimes = attributeDic["traveltimes"][ifo]

            Ptime = max(traveltimes["Ptimes"])
            Stime = max(traveltimes["Stimes"])
            Rtwotime = max(traveltimes["Rtwotimes"])
            RthreePointFivetime = max(traveltimes["RthreePointFivetimes"])
            Rfivetime = max(traveltimes["Rfivetimes"])

            if params["doEarthquakesVelocityMap"]:
                Rvelocitymaptimes = traveltimes["Rvelocitymaptimes"]
                Rvelocitymaptime = max(traveltimes["Rvelocitymaptimes"])
                Rvelocitymapvelocities = traveltimes["Rvelocitymapvelocities"]

            peak_velocity = traveltimes["Rfamp"][0]
            peak_velocity = peak_velocity * 1e6

            if peak_velocity > ylim[1]:
               ylim[1] = peak_velocity*1.1
            if -peak_velocity < ylim[0]:
               ylim[0] = -peak_velocity*1.1

            times.append(RthreePointFivetime)
            velocities.append(peak_velocity)

        kwargs = {"color":"k"}
        plot.add_scatter(times,velocities,label="pred. vel.",**kwargs)

        plot.ylabel = r"RMS Velocity [$\mu$m/s]"
        plot.title = channel.station.replace("_","\_")
        #plot.xlim = xlim
        #plot.ylim = ylim
        plot.add_legend(loc=1,prop={'size':10})

        #plot.axes[0].set_xscale("log")
        #plot.axes[0].set_yscale("log")

        plot.save(pngFile)
        plot.close()


