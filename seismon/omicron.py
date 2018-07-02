#!/usr/bin/python

import os, glob, optparse, shutil, warnings
import numpy as np
from subprocess import Popen, PIPE, STDOUT

import seismon.utils

try:
    import gwpy.time, gwpy.timeseries
    import gwpy.frequencyseries, gwpy.spectrogram
    import gwpy.plotter, gwpy.table
    from gwpy.table.lsctables import SnglBurstTable
except:
    print("gwpy import fails... no plotting possible.")


#import laldetchar.triggers

__author__ = "Michael Coughlin <michael.coughlin@ligo.org>"
__date__ = "2012/8/26"
__version__ = "0.1"

# =============================================================================
#
#               DEFINITIONS
#
# =============================================================================

def plot_triggers(params,channel,segment):
    """plot omicron triggers for given channel and segment.

    @param params
        seismon params dictionary
    @param channel
        seismon channel structure
    @param segment
        [start,end] gps
    """

    gpsStart = segment[0]
    gpsEnd = segment[1]

    omicronDirectory = os.path.join(params["path"],"omicron")
    omicronPath = os.path.join(omicronDirectory,channel.station)
    omicronXMLs = glob.glob(os.path.join(omicronPath,"*.xml"))

    table = []
    for ii in range(len(omicronXMLs)):
        tabletmp = SnglBurstTable.read(omicronXMLs[0])
        for jj in range(len(tabletmp)):
            table.append(tabletmp[jj])
    if table == []:
       return
   
    peak_times = gwpy.plotter.table.get_table_column(table, "peak_time")
    central_freqs = gwpy.plotter.table.get_table_column(table, "central_freq")
    snrs = gwpy.plotter.table.get_table_column(table, "snr")

    textLocation = params["path"] + "/" + channel.station_underscore
    seismon.utils.mkdir(textLocation)

    f = open(os.path.join(textLocation,"triggers.txt"),"w")
    for peak_time,central_freq,snr in zip(peak_times,central_freqs,snrs):
        f.write("%.1f %e %e\n"%(peak_time,central_freq,snr))
    f.close()

    if params["doPlots"]:

        plotLocation = params["path"] + "/" + channel.station_underscore
        seismon.utils.mkdir(plotLocation)

        if params["doEarthquakesAnalysis"]:
            pngFile = os.path.join(plotLocation,"omicron-%d-%d.png"%(gpsStart,gpsEnd))
        else:
            pngFile = os.path.join(plotLocation,"omicron.png")

        epoch = gwpy.time.Time(gpsStart, format='gps')
       
        #plot = gwpy.plotter.EventTablePlot(table, 'time', 'central_freq', 'snr', figsize=[14,8],epoch=gpsStart,size_by_log='snr', size_range=[6, 20],edgecolor='none')
        plot = table.plot('time', 'central_freq', color='snr', edgecolor='none', epoch=gpsStart)
        plot.add_colorbar(log=True, clim=[6, 20],label='Signal-to-noise ratio (SNR)')
        plot.xlim = [gpsStart, gpsEnd]
        plot.ylabel = 'Frequency [Hz]'
        plot.ylim = [params["fmin"],params["fmax"]]
        plot.axes[0].set_yscale("log")
        plot.save(pngFile)
        plot.close()

def generate_triggers(params):
    """@generate omicron triggers.

    @param params
        seismon params dictionary
    """

    omicronDirectory = os.path.join(params["path"],"omicron")
    seismon.utils.mkdir(omicronDirectory)

    gpsStart = 1e20
    gpsEnd = -1e20

    gpss = []
    for frame in params["frame"]:
        gpss.append(frame.segment[0])
    indexes = np.argsort(gpss)
   
    f = open(os.path.join(omicronDirectory,"frames.ffl"),"w")
    for ii in indexes: 
        frame = params["frame"][ii]
        f.write("%s %d %d 0 0\n"%(frame.path, frame.segment[0], frame.segment[1]-frame.segment[0]))
        gpsStart = min(gpsStart,frame.segment[0])
        gpsEnd = max(gpsEnd,frame.segment[1])
    f.close()

    paramsFile = omicron_params(params)
    f = open(os.path.join(omicronDirectory,"params.txt"),"w")
    f.write("%s"%(paramsFile))
    f.close()

    f = open(os.path.join(omicronDirectory,"segments.txt"),"w")
    f.write("%d %d\n"%(gpsStart,gpsEnd))
    f.close()

    omicron = "/home/detchar/opt/virgosoft/Omicron/v2r1/Linux-x86_64/omicron.exe"
    environmentSetup = "CMTPATH=/home/detchar/opt/virgosoft; export CMTPATH; source /home/detchar/opt/virgosoft/Omicron/v2r1/cmt/setup.sh"
    omicronCommand = "%s; %s %s %s"%(environmentSetup, omicron, os.path.join(omicronDirectory,"segments.txt"),os.path.join(omicronDirectory,"params.txt"))

    p = Popen(omicronCommand,shell=True,stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
    output = p.stdout.read()

def omicron_params(params):
    """@generate omicron params file.

    @param params
        seismon params dictionary
    """

    omicronDirectory = os.path.join(params["path"],"omicron")

    channelList = ""
    samplerateList = ""
    for channel in params["channels"]:
        channelList = "%s %s"%(channelList,channel.station)
        samplerateList = "%s %d"%(samplerateList,channel.samplef)

    paramsFile = """

    DATA    FFL     %s/frames.ffl
    
    //** list of channels you want to process
    DATA    CHANNELS %s
    
    //** native sampling frequency (Hz) of working channels (as many as
    //the number of input channels)
    DATA    NATIVEFREQUENCY %s
    
    //** working sampling (one value for all channels)
    DATA    SAMPLEFREQUENCY 32
    
    //*************************************************************************************
    //************************        SEARCH PARAMETERS
    //*****************************
    //*************************************************************************************
    
    //** chunk duration in seconds (must be an integer)
    PARAMETER       CHUNKDURATION   512
    
    //** segment duration in seconds (must be an integer)
    PARAMETER       SEGMENTDURATION   512
    
    //** overlap duration between segments in seconds (must be an integer)
    PARAMETER       OVERLAPDURATION  160
    
    //** search frequency range
    PARAMETER       FREQUENCYRANGE  0.1      10
    
    //** search Q range
    PARAMETER       QRANGE          3.3166  141
    
    //** maximal mismatch between 2 consecutive tiles (0<MM<1)
    PARAMETER       MISMATCHMAX     0.2
    
    //*************************************************************************************
    //************************            TRIGGERS
    //*****************************
    //*************************************************************************************
    
    //** tile SNR threshold
    TRIGGER         SNRTHRESHOLD    5
    
    //** maximum number of triggers per file
    TRIGGER         NMAX            500000
    
    //*************************************************************************************
    //************************             OUTPUT
    //*****************************
    //*************************************************************************************
    
    //** full path to output directory
    OUTPUT  DIRECTORY       %s/
    
    OUTPUT  FORMAT   xml
    
    """%(omicronDirectory,channelList,samplerateList,omicronDirectory)

    return paramsFile

