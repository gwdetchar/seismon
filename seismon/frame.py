#!/usr/bin/python

import sys, os, glob, matplotlib
import numpy as np

from mpl_toolkits.basemap import Basemap
matplotlib.use("AGG")
matplotlib.rcParams.update({'font.size': 18})
from matplotlib import pyplot as plt
from matplotlib import cm

import seismon.utils
import pylal.Fr

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
        dx = 1.0/len(np.array(dataFull.data))
        out_dict = {'name'  : '%s:%s' %(params["ifo"],dataFull.name.replace(":","_")) ,
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

