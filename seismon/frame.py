#!/usr/bin/python

import sys, os, glob, matplotlib
import numpy as np
import scipy.spatial

from mpl_toolkits.basemap import Basemap
matplotlib.use("AGG")
matplotlib.rcParams.update({'font.size': 18})
from matplotlib import pyplot as plt
from matplotlib import cm

import seismon.utils
import pylal.Fr

import glue.datafind, glue.segments, glue.segmentsUtils, glue.lal

import gwpy.time, gwpy.timeseries, gwpy.spectrum, gwpy.plotter
import gwpy.segments

__author__ = "Michael Coughlin <michael.coughlin@ligo.org>"
__date__ = "2012/8/26"
__version__ = "0.1"

# =============================================================================
#
#                               DEFINITIONS
#
# =============================================================================

def worldmap_channel_plot(params,plotName):
    """@worldmap plot

    @param params
        seismon params dictionary
    @param plotName
        name of plot
    """

    ifo = seismon.utils.getIfo(params)

    plt.figure(figsize=(10,5))
    plt.axes([0,0,1,1])

    # lon_0 is central longitude of robinson projection.
    # resolution = 'c' means use crude resolution coastlines.
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    #set a background colour
    m.drawmapboundary(fill_color='#85A6D9')

    # draw coastlines, country boundaries, fill continents.
    m.fillcontinents(color='white',lake_color='#85A6D9')
    m.drawcoastlines(color='#6D5F47', linewidth=.4)
    m.drawcountries(color='#6D5F47', linewidth=.4)

    # draw lat/lon grid lines every 30 degrees.
    m.drawmeridians(np.arange(-180, 180, 30), color='#bbbbbb')
    m.drawparallels(np.arange(-90, 90, 30), color='#bbbbbb')

    for channel in params["channels"]:
        label = channel.station.replace("_","\_")

        x,y = m(channel.longitude, channel.latitude)

        m.scatter(
                x,
                y,
                s=20, #size
                marker='o', #symbol
                alpha=0.5, #transparency
                zorder = 3, #plotting order
                c='k'
        )

        plt.text(
                x,
                y+5,
                label,
                color = 'black',
                size='small',
                horizontalalignment='center',
                verticalalignment='center',
                zorder = 3,
        )

    plt.show()
    plt.savefig(plotName,dpi=200)
    plt.close('all')

def do_kdtree(combined_x_y_arrays,points):
    """@calculate nearest points.

    @param combined_x_y_arrays
        list of x,y map points
    @param points
        list of x,y points
    """

    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
    dist, indexes = mytree.query(points)
    return indexes

def make_frames(params, segment):
    """@calculates wiener filter for given channel and segment.

    @param params
        seismon params dictionary
    @param segment
        [start,end] gps
    """

    ifo = seismon.utils.getIfo(params)

    gpsStart = segment[0]
    gpsEnd = segment[1]

    # set the times
    duration = np.ceil(gpsEnd-gpsStart)

    out_dicts = []
    channels_keep = []

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

        if np.mean(dataFull.data) == 0.0:
            print "data only zeroes... continuing\n"
            continue

        if params["framesSampleRate"] > 0:
            dataFull = dataFull.resample(params["framesSampleRate"])
            samplef = params["framesSampleRate"]
        else:
            samplef = channel.samplef

        dx = 1.0/samplef
        out_dict = {'name'  : '%s:%s' %(params["ifo"],dataFull.name) ,
            'data'  : np.array(dataFull.data),
            'start' : dataFull.epoch.vals[0],
            'dx'    : dx,
            'kind'  : 'ADC'}

        out_dicts.append(out_dict)
        channels_keep.append(channel)

    frameFile = "%s/%s-%s-%d-%d.gwf"%(params["framesFolder"],params["ifo"],params["ifo"],gpsStart,duration)
    seismon.utils.mkdir(params["framesFolder"])
    # out_f is a destination filename
    pylal.Fr.frputvect(frameFile,out_dicts)

    frameDirectory = params["path"] + "/frame/"
    seismon.utils.mkdir(frameDirectory)

    channelFile = os.path.join(frameDirectory,"channels.txt")
    f = open(channelFile,"wb")
    for channel in channels_keep:
        f.write("%s %.10f %.10f\n"%(channel.station,channel.latitude,channel.longitude))
    f.close()

    if params["doPlots"]:

        plotDirectory = params["path"] + "/frame/" 
        seismon.utils.mkdir(plotDirectory)

        pngFile = os.path.join(plotDirectory,"map.png")
        worldmap_channel_plot(params,pngFile)

def make_frames_calibrated(params, segment):
    """@calculates wiener filter for given channel and segment.

    @param params
        seismon params dictionary
    @param segment
        [start,end] gps
    """

    ifo = seismon.utils.getIfo(params)

    gpsStart = segment[0]
    gpsEnd = segment[1]

    # set the times
    duration = np.ceil(gpsEnd-gpsStart)

    frameDir = params["framesFolder"]
    frameList = [os.path.join(root, name)
        for root, dirs, files in os.walk(frameDir)
        for name in files]

    datacache = []
    for frame in frameList:
        thisFrame = frame.replace("file://localhost","")
        if thisFrame == "":
            continue

        thisFrameSplit = thisFrame.split(".")
        if thisFrameSplit[-1] == "log":
            continue

        thisFrameSplit = thisFrame.split("-")
        gps = float(thisFrameSplit[-2])
        dur = float(thisFrameSplit[-1].replace(".gwf",""))

        if gps+dur < gpsStart:
            continue
        if gps > gpsEnd:
            continue

        #cacheFile = glue.lal.CacheEntry("%s %s %d %d %s"%("XG","Homestake",gps,dur,frame))
        datacache.append(frame)
    datacache = glue.lal.Cache(map(glue.lal.CacheEntry.from_T050017, datacache))
    params["frame"] = datacache

    out_dicts = []
    channels_keep = []
    channels_calibration = []

    beta = 3200.0
    alpha = 6000.0

    velocityFileR = '/home/mcoughlin/Seismon/velocity_maps/R025_1_GDM52.pix'
    velocityFileL = '/home/mcoughlin/Seismon/velocity_maps/L025_0_GDM52.pix'
    velocity_map_R = np.loadtxt(velocityFileR)
    velocity_map_L = np.loadtxt(velocityFileL)
    base_velocity_R = 3.85019
    base_velocity_L = 4.23685
    combined_x_y_arrays_R = np.dstack([velocity_map_R[:,0],velocity_map_R[:,1]])[0]
    combined_x_y_arrays_L = np.dstack([velocity_map_L[:,0],velocity_map_L[:,1]])[0]

    for channel in params["channels"]:

        channel_station = '%s:%s' %(params["ifo"],channel.station)
        # make timeseries

        try:
            dataFull = gwpy.timeseries.TimeSeries.read(params["frame"], channel_station, start=gpsStart, end=gpsEnd)
        except:
            print "data read from frames failed... continuing\n"
            continue

        points_list = np.dstack([channel.latitude, channel.longitude])
        indexes = do_kdtree(combined_x_y_arrays_R,points_list)[0]
        index_R = indexes[0]
        indexes = do_kdtree(combined_x_y_arrays_L,points_list)[0]
        index_L = indexes[0]

        cR = 1000 * (1 + 0.01*velocity_map_R[index_R,3])*base_velocity_R
        beta = 1000 * (1 + 0.01*velocity_map_L[index_L,3])*base_velocity_L
        alpha = (4*(beta**3)*np.sqrt(beta**2 - cR**2))/np.sqrt(16*(beta**6) - 24*(beta**4)*(cR**2) + 8*(beta**2)*(cR**4) - (cR**6))

        eps = 0.01
        alpha = (1.7549+eps*2.25427)*cR
        beta = (1.01319 - eps*0.049422)*cR

        eps = 0
        strain_calibration = 1.0/( cR*0.5682*(1-1.5377*eps))

        if params["ifo"] == "LUNAR":
            alpha = 6000.0
            beta = 3500.0
            strain_calibration = alpha/(beta**2)
        #elif params["ifo"] == "CZKHC":
        #    strain_calibration = 1

        print "%s, cR: %.3f, alpha: %.3f, beta: %.3f, calib: %.3e"%(channel.station,cR,alpha,beta,strain_calibration)
        dataFull = dataFull * strain_calibration

        indexes = np.where(np.isnan(dataFull.data))[0]
        meanSamples = np.mean(np.ma.masked_array(dataFull.data,np.isnan(dataFull.data)))
        for index in indexes:
            dataFull[index] = meanSamples

        if np.mean(dataFull.data) == 0.0:
            print "data only zeroes... continuing\n"
            continue

        if params["framesSampleRate"] > 0:
            dataFull = dataFull.resample(params["framesSampleRate"])
            samplef = params["framesSampleRate"]
        else:
            samplef = channel.samplef

        dx = 1.0/samplef

        if params["ifo"] == "LUNAR":
            this_data = np.diff(dataFull.data) * samplef
            this_data = np.append(this_data,this_data[-1])
        elif params["ifo"] == "CZKHC":
            this_data = np.diff(dataFull.data) * samplef
            this_data = np.append(this_data,this_data[-1])
        else:
            this_data = np.array(dataFull.data)

        out_dict = {'name'  : '%s' %(dataFull.name) ,
            'data'  : this_data,
            'start' : dataFull.epoch.vals[0],
            'dx'    : dx,
            'kind'  : 'ADC'}

        out_dicts.append(out_dict)
        channels_keep.append(channel)
        channels_calibration.append(strain_calibration)

    frameFile = "%s/%s-%s-%d-%d.gwf"%(params["framesFolderCalibrated"],params["ifo"],params["ifo"],gpsStart,duration)
    seismon.utils.mkdir(params["framesFolderCalibrated"])
    # out_f is a destination filename
    pylal.Fr.frputvect(frameFile,out_dicts)

    frameDirectory = params["path"] + "/frame_calibrated/"
    seismon.utils.mkdir(frameDirectory)

    channelFile = os.path.join(frameDirectory,"channels.txt")
    f = open(channelFile,"wb")
    for channel,strain_calibration in zip(channels_keep,channels_calibration):
        f.write("%s %.10f %.10f %.10e\n"%(channel.station,channel.latitude,channel.longitude,strain_calibration))
    f.close()

    if params["doPlots"]:

        plotDirectory = params["path"] + "/frame_calibrated/"
        seismon.utils.mkdir(plotDirectory)

        pngFile = os.path.join(plotDirectory,"map.png")
        worldmap_channel_plot(params,pngFile)
