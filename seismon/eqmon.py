#!/usr/bin/python

import os, sys, time, glob, math, matplotlib, random, string
import pickle
import calendar

matplotlib.use('Agg') 
matplotlib.rcParams.update({'font.size': 18})
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
from datetime import datetime
from operator import itemgetter

try:
    import glue.datafind, glue.segments, glue.segmentsUtils, glue.lal
except:
    print("Glue import fails... no datafind possible.")

from lxml import etree
import scipy.spatial
import smtplib, email.mime.text

import astropy.time
#import lal.gpstime

import seismon.utils, seismon.eqmon_plot

try:
    from sklearn import gaussian_process
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel
    from sklearn import preprocessing
    from sklearn import svm
except:
    print("sklearn import fails... no prediction possible.")

try:
    import gwpy.time, gwpy.timeseries, gwpy.frequencyseries
    import gwpy.plotter
except:
    print("gwpy import fails... no plotting possible.")

try:
    from pylal import Fr
except:
    print("No pylal installed...")

try:
    from geopy.geocoders import Nominatim
except:
    print("No geopy installed...")

def run_earthquakes(params,segment):
    """@run earthquakes prediction.

    @param params
        seismon params dictionary
    @param segment
        [start,end] gps
    """

    gpsStart = segment[0]
    gpsEnd = segment[1]

    timeseriesDirectory = os.path.join(params["path"],"timeseries")
    seismon.utils.mkdir(timeseriesDirectory)
    earthquakesDirectory = os.path.join(params["path"],"earthquakes")
    seismon.utils.mkdir(earthquakesDirectory)
    noticesDirectory = os.path.join(params["path"],"notices")
    seismon.utils.mkdir(noticesDirectory)
    segmentsDirectory = os.path.join(params["path"],"segments")
    seismon.utils.mkdir(segmentsDirectory)
    predictionDirectory = params["dirPath"] + "/Text_Files/Prediction/%s/"%params["ifo"]
    seismon.utils.mkdir(predictionDirectory)

    ifo = seismon.utils.getIfo(params)

    attributeDics = retrieve_earthquakes(params,gpsStart,gpsEnd)
    attributeDics = sorted(attributeDics, key=itemgetter("Magnitude"), reverse=True)

    earthquakesFile = os.path.join(earthquakesDirectory,"earthquakes.txt")
    earthquakesXMLFile = os.path.join(earthquakesDirectory,"earthquakes.xml")
    timeseriesFile = os.path.join(timeseriesDirectory,"amp.txt")
    noticesFile = os.path.join(noticesDirectory,"notices.txt")
    segmentsFile = os.path.join(segmentsDirectory,"segments.txt")
    predictionFile = os.path.join(predictionDirectory,"%d-%d.txt"%(gpsStart,gpsEnd))

    f = open(earthquakesFile,"w+")
    g = open(noticesFile,"w+")
    h = open(segmentsFile,"w+")

    threshold = 10**(-7)
    threshold = 0

    amp = 0
    segmentlist = glue.segments.segmentlist()

    for attributeDic in attributeDics:

        if params["doEarthquakesVelocityMap"]:
            attributeDic = calculate_traveltimes_velocitymap(attributeDic)
        if params["doEarthquakesLookUp"]:
            attributeDic = calculate_traveltimes_lookup(attributeDic)        
        if not "Arbitrary" in attributeDic["traveltimes"]:
            continue

        if params["ifo"] == "IRIS":
            distances = []
            for channel in params["channels"]:
                attributeDic = seismon.eqmon.ifotraveltimes_loc(attributeDic, "IRIS", channel.latitude, channel.longitude)
                traveltimes = attributeDic["traveltimes"]["IRIS"]
                distances.append(traveltimes["Distances"][0])
            index = np.argmax(distances)
            channel = params["channels"][index]
            attributeDic = seismon.eqmon.ifotraveltimes_loc(attributeDic, "IRIS", channel.latitude, channel.longitude)
            traveltimes = attributeDic["traveltimes"]["IRIS"]

        else:
            attributeDic = seismon.eqmon.eqmon_loc(attributeDic,ifo)
            traveltimes = attributeDic["traveltimes"][ifo]

        #if "Arbitrary" in attributeDic["traveltimes"]:
        #    attributeDic = eqmon_loc(attributeDic,ifo)

        if params["doEarthquakesChile"]:
            minlatitude = -80.0
            maxlatitude = -10.0
            minlongitude = -80.0
            maxlongitude = -60.0

            if attributeDic["Latitude"] < minlatitude:
                continue
            if attributeDic["Latitude"] > maxlatitude:
                continue
            if attributeDic["Longitude"] < minlongitude:
                continue
            if attributeDic["Longitude"] > maxlongitude:
                continue

            #if attributeDic["Magnitude"] < 6.0:
            #    continue

        #traveltimes = attributeDic["traveltimes"][ifo]

        arrival = np.min([max(traveltimes["Rtwotimes"]),max(traveltimes["RthreePointFivetimes"]),max(traveltimes["Rfivetimes"]),max(traveltimes["Stimes"]),max(traveltimes["Ptimes"])])
        departure = np.max([max(traveltimes["Rtwotimes"]),max(traveltimes["RthreePointFivetimes"]),max(traveltimes["Rfivetimes"]),max(traveltimes["Stimes"]),max(traveltimes["Ptimes"])])

        arrival_floor = np.floor(arrival / 100.0) * 100.0
        departure_ceil = np.ceil(departure / 100.0) * 100.0

        check_intersect = (arrival >= gpsStart) and (departure <= gpsEnd)

        if check_intersect:
            amp += traveltimes["Rfamp"][0]

            if traveltimes["Rfamp"][0] >= threshold:

                if "nodalPlane1_strike" in attributeDic:     

                    f.write("%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.5e %d %d %.1f %.1f %e %.1f %.1f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n"%(attributeDic["GPS"],attributeDic["Magnitude"],max(traveltimes["Ptimes"]),max(traveltimes["Stimes"]),max(traveltimes["Rtwotimes"]),max(traveltimes["RthreePointFivetimes"]),max(traveltimes["Rfivetimes"]),traveltimes["Rfamp"][0],arrival_floor,departure_ceil,attributeDic["Latitude"],attributeDic["Longitude"],max(traveltimes["Distances"]),attributeDic["Depth"],traveltimes["Azimuth"][0],attributeDic["nodalPlane1_strike"],attributeDic["nodalPlane1_rake"],attributeDic["nodalPlane1_dip"],attributeDic["momentTensor_Mrt"],attributeDic["momentTensor_Mtp"],attributeDic["momentTensor_Mrp"],attributeDic["momentTensor_Mtt"],attributeDic["momentTensor_Mrr"],attributeDic["momentTensor_Mpp"]))

                    print "%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.5e %d %d %.1f %.1f %e %.1f %.1f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n"%(attributeDic["GPS"],attributeDic["Magnitude"],max(traveltimes["Ptimes"]),max(traveltimes["Stimes"]),max(traveltimes["Rtwotimes"]),max(traveltimes["RthreePointFivetimes"]),max(traveltimes["Rfivetimes"]),traveltimes["Rfamp"][0],arrival_floor,departure_ceil,attributeDic["Latitude"],attributeDic["Longitude"],max(traveltimes["Distances"]),attributeDic["Depth"],traveltimes["Azimuth"][0],attributeDic["nodalPlane1_strike"],attributeDic["nodalPlane1_rake"],attributeDic["nodalPlane1_dip"],attributeDic["momentTensor_Mrt"],attributeDic["momentTensor_Mtp"],attributeDic["momentTensor_Mrp"],attributeDic["momentTensor_Mtt"],attributeDic["momentTensor_Mrr"],attributeDic["momentTensor_Mpp"])

                else:
                    f.write("%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.5e %d %d %.1f %.1f %e %.1f %.1f\n"%(attributeDic["GPS"],attributeDic["Magnitude"],max(traveltimes["Ptimes"]),max(traveltimes["Stimes"]),max(traveltimes["Rtwotimes"]),max(traveltimes["RthreePointFivetimes"]),max(traveltimes["Rfivetimes"]),traveltimes["Rfamp"][0],arrival_floor,departure_ceil,attributeDic["Latitude"],attributeDic["Longitude"],max(traveltimes["Distances"]),attributeDic["Depth"],traveltimes["Azimuth"][0]))

                    print "%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.5e %d %d %.1f %.1f %e %.1f %.1f\n"%(attributeDic["GPS"],attributeDic["Magnitude"],max(traveltimes["Ptimes"]),max(traveltimes["Stimes"]),max(traveltimes["Rtwotimes"]),max(traveltimes["RthreePointFivetimes"]),max(traveltimes["Rfivetimes"]),traveltimes["Rfamp"][0],arrival_floor,departure_ceil,attributeDic["Latitude"],attributeDic["Longitude"],max(traveltimes["Distances"]),attributeDic["Depth"],traveltimes["Azimuth"][0])

                g.write("%.1f %.1f %.5e\n"%(arrival,departure-arrival,traveltimes["Rfamp"][0]))
                h.write("%.0f %.0f\n"%(arrival_floor,departure_ceil))

                segmentlist.append(glue.segments.segment(arrival_floor,departure_ceil))
    
    f.close()
    g.close()
    h.close()

    f = open(timeseriesFile,"w+")
    f.write("%e\n"%(amp))
    f.close()

    f = open(predictionFile,"w+")
    f.write("%e\n"%(amp))
    f.close()

    write_info(earthquakesXMLFile,attributeDics)

    if params["doEarthquakesOnline"]:
        sender = params["userEmail"]
        receivers = [params["userEmail"]]

        lines = [line for line in open(earthquakesFile)]
        if lines == []:
            return segmentlist

        message = ""
        for line in lines:
            message = "%s\n%s"%(message,line)

        s = smtplib.SMTP('localhost')
        s.sendmail(sender,receivers, message)         
        s.quit()
        print "mail sent"

    return segmentlist

def caput(channel, value, **kwargs):
    """caput with logging

    Parameters
    ----------
    channel : `str`
        name of channel to put
    value : `float`, `str`
        target value for channel
    **kwargs
        other keyword arguments to pass to `epics.caput`
    """
    epics.caput(channel, value, **kwargs)
    logger.debug("caput %s = %s" % (channel, value))


EPICS_RETRY = 0

def caget(channel, log=True, retry=0, **kwargs):
    """caget with logging

    Parameters
    ----------
    channel : `str`
        name of channel to get
    log : `bool`, optional, default: `True`
        use verbose logging
    retry : `int`, optional, defauilt: `0`
        catch failed cagets and try again this many times
    **kwargs
        other keyword arguments to pass to `epics.caget`

    Returns
    -------
    value : `float`, str`
        the value retrieved from that channel

    Raises
    ------
    ValueError
        if the channel failed to respond to a caget request
    """
    global EPICS_RETRY
    value = epics.caget(channel, **kwargs)
    if value is None:
        if log:
            logger.critical("Failed to caget %s" % channel)
        if EPICS_RETRY < retry:
            logger.warning('Retrying [%d]' % EPICS_RETRY)
            return caget(channel, log=log, retry=retry, **kwargs)
        else:
            raise ValueError("Failed to caget %s" % channel)
    if log:
        logger.debug("caget %s = %s" % (channel, value))
    EPICS_RETRY = 0
    return value

def run_earthquakes_info(params,segment):
    """@run earthquakes prediction.

    @param params
        seismon params dictionary
    @param segment
        [start,end] gps
    """

    gpsStart = segment[0]
    gpsEnd = segment[1]

    seismonpath = os.path.dirname(seismon.__file__)
    scriptpath = os.path.join(seismonpath,'..','EGG-INFO','scripts')

    params["earthquakesMinMag"] = 2.0
    attributeDics = retrieve_earthquakes(params,gpsStart,gpsEnd)
    attributeDics = sorted(attributeDics, key=itemgetter("Magnitude"), reverse=True)

    #ifos = ["H1","L1","G1","V1","MIT"]
    ifos = params["ifos"].split(",")
    amp = 0

    if params["doEPICs"]:

       epics_dicts = {}
       for ifo in ifos:
            ifoShort = ifo
            params["ifo"] = ifo
            ifo = seismon.utils.getIfo(params)

            epics_dicts[ifoShort] = {}
            epics_dicts[ifoShort]["prob"] = 0
            epics_dicts[ifoShort]["eta"] = 0
            epics_dicts[ifoShort]["amp"] = 0
            epics_dicts[ifoShort]["mult"] = 0

    params["path_temp"] = "%s_temp"%params["path"]
    for attributeDic in attributeDics:
        if attributeDic["eventID"] == "None":
            eventID = "%.0f"%attributeDic['GPS']
            eventName = ''.join(["iris",str(eventID)])
            attributeDic["eventID"] = eventName

        earthquakesDirectory = os.path.join(params["path_temp"],"earthquakes")
        earthquakesDirectory = os.path.join(earthquakesDirectory,str(attributeDic["eventID"]))

        earthquakesFile = os.path.join(earthquakesDirectory,"earthquakes.txt")
        earthquakesXMLFile = os.path.join(earthquakesDirectory,"earthquakes.xml")
        seismon.utils.mkdir(earthquakesDirectory)

        f = open(earthquakesFile,"w+")
        for ifo in ifos:
            ifoShort = ifo
            params["ifo"] = ifo
            ifo = seismon.utils.getIfo(params) 

            attributeDic = seismon.eqmon.eqmon_loc(attributeDic,ifo)
            traveltimes = attributeDic["traveltimes"][ifo]

            arrival = np.min([max(traveltimes["Rtwotimes"]),max(traveltimes["RthreePointFivetimes"]),max(traveltimes["Rfivetimes"]),max(traveltimes["Stimes"]),max(traveltimes["Ptimes"])])
            departure = np.max([max(traveltimes["Rtwotimes"]),max(traveltimes["RthreePointFivetimes"]),max(traveltimes["Rfivetimes"]),max(traveltimes["Stimes"]),max(traveltimes["Ptimes"])])

            arrival_floor = np.floor(arrival / 100.0) * 100.0
            departure_ceil = np.ceil(departure / 100.0) * 100.0

            try:
                geolocator = Nominatim()
                locationstr = "%.6f, %.6f"%(attributeDic["Latitude"],attributeDic["Longitude"])
                location = geolocator.reverse(locationstr, language='en')
                locationstr = location.address.encode('utf-8')
                locationstr = locationstr.replace(", ",",").replace(" ","_")
            except:
                locationstr = "Unknown"

            try:
                if ifoShort == "L1":
                    svmfile = os.path.join(scriptpath,'svm_llo.pickle')
                else:
                    svmfile = os.path.join(scriptpath,'svm_lho.pickle')

                with open(svmfile, 'rb') as fid:
                    scaler,clf = pickle.load(fid)

                X = np.vstack((attributeDic["Magnitude"],attributeDic["Latitude"],attributeDic["Longitude"],max(traveltimes["Distances"])/1000.0,attributeDic["Depth"],traveltimes["Azimuth"][0],np.log10(traveltimes["Rfamp"][0]))).T
                X = scaler.transform(X)

                lockloss_prediction = clf.predict(X)[0]
                lockloss_prob = clf.predict_proba(X)
                lockloss_probability = lockloss_prob[0,1]
            except:
                lockloss = -1
                lockloss_probability = -1

            f.write("%s %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.5e %d %d %.1f %.1f %e %.1f %.1f %.3f %s %s\n"%(attributeDic["eventID"],attributeDic["GPS"],attributeDic["Magnitude"],max(traveltimes["Ptimes"]),max(traveltimes["Stimes"]),max(traveltimes["Rtwotimes"]),max(traveltimes["RthreePointFivetimes"]),max(traveltimes["Rfivetimes"]),traveltimes["Rfamp"][0],arrival_floor,departure_ceil,attributeDic["Latitude"],attributeDic["Longitude"],max(traveltimes["Distances"]),attributeDic["Depth"],traveltimes["Azimuth"][0],lockloss_probability,locationstr,ifoShort))

            print "%s %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.5e %d %d %.1f %.1f %e %.1f %.1f %.3f %s %s\n"%(attributeDic["eventID"],attributeDic["GPS"],attributeDic["Magnitude"],max(traveltimes["Ptimes"]),max(traveltimes["Stimes"]),max(traveltimes["Rtwotimes"]),max(traveltimes["RthreePointFivetimes"]),max(traveltimes["Rfivetimes"]),traveltimes["Rfamp"][0],arrival_floor,departure_ceil,attributeDic["Latitude"],attributeDic["Longitude"],max(traveltimes["Distances"]),attributeDic["Depth"],traveltimes["Azimuth"][0],lockloss_probability,locationstr,ifoShort)

            if params["doEPICs"]:
                #indexes = np.intersect1d(np.where(arrival <= tt)[0],np.where(departure >= tt)[0])
                #eqs = np.zeros(tt.shape)
                #eqs[indexes] = 1

                eta = arrival - gpsEnd
                if eta < 0:
                    continue 

                epics_dicts[ifoShort]["eta"] = arrival - gpsEnd
                epics_dicts[ifoShort]["amp"] = traveltimes["Rfamp"][0]
                if len(attributeDics) > 1:
                    epics_dicts[ifoShort]["mult"] = 1
                else:
                    epics_dicts[ifoShort]["mult"] = 0
                if epics_dicts[ifoShort]["amp"] < 1e-6:
                    epics_dicts[ifoShort]["prob"] = 1
                elif epics_dicts[ifoShort]["amp"] < 5e-6:
                    epics_dicts[ifoShort]["prob"] = 2
                else:
                    epics_dicts[ifoShort]["prob"] = 3     

                #tstart = (max(traveltimes["Rfivetimes"]) + max(traveltimes["RthreePointFivetimes"]))/2.0
                #tend = max(traveltimes["Rtwotimes"]) 
                #x = np.arange(tstart,tend,1)
                #x = tt
                #y = (x - np.min(x))/600.0
                #lamb = 1 + y[-1]/5.0
                #vals = scipy.stats.gamma.pdf(y, lamb)
                #vals = np.absolute(vals)
                #vals = vals / np.max(vals)
                #vals = vals * traveltimes["Rfamp"][0]
                #amps = np.interp(tt,x,vals,left=0,right=0)

                #epics_dicts[ifoShort]["amp"] = epics_dicts[ifoShort]["amp"] + amps*traveltimes["Rfamp"][0]
                #epics_dicts[ifoShort]["amp"] = epics_dicts[ifoShort]["amp"] + amps
                #epics_dicts[ifoShort]["eq"] = epics_dicts[ifoShort]["eq"] + eqs

        f.close()
        write_info(earthquakesXMLFile,[attributeDic])
   
    if not attributeDics:
        earthquakesDirectory = os.path.join(params["path_temp"],"earthquakes")
        earthquakesDirectory = os.path.join(earthquakesDirectory,"tmp")

        earthquakesFile = os.path.join(earthquakesDirectory,"earthquakes.txt")
        earthquakesXMLFile = os.path.join(earthquakesDirectory,"earthquakes.xml")
        seismon.utils.mkdir(earthquakesDirectory)

        f = open(earthquakesFile,"w+")
        for ifo in ifos:
            ifoShort = ifo
            params["ifo"] = ifo
            ifo = seismon.utils.getIfo(params)

            f.write("%s %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.5e %d %d %.1f %.1f %e %.1f %.1f %.3f %s %s\n"%("tmp",-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,"tmp",ifoShort))
        f.close()
 
    if os.path.isdir(params["path_temp"]):

        sys_command = "cp -r %s/* %s"%(params["path_temp"],params["path"])
        os.system(sys_command)

        sys_command = "rm -rf %s"%(params["path_temp"])
        os.system(sys_command)

        if not os.path.isdir(params["currentpath"]):
            sys_command = "mkdir %s"%(params["currentpath"])
            os.system(sys_command)
        sys_command = "find %s/* -mtime +1 -exec rm -rf {} \;"%params["currentpath"]
        os.system(sys_command)
        sys_command = "cp -r %s/* %s"%(params["path"],params["currentpath"])
        os.system(sys_command)

    if params["doEPICs"]:

        #fileName = "%s/S-SEISMON-%d-%d.txt"%(params["epicsDirectory"],gpsStart,gpsEnd-gpsStart)
        fileName = "%s/epics.txt"%(params["epicsDirectory"])
        fid = open(fileName,'w')
        #out_dicts = []
        #for ifo in ifos:
        
        #    channel_name =  "%s:AMP"%ifo
        #    out_dict = {'name'  : channel_name,
        #        'data'  : epics_dicts[ifo]["amp"],
        #        'start' : tt[0],
        #        'dx'    : tt[1]-tt[0],
        #        'kind'  : 'ADC'}
        #    out_dicts.append(out_dict)

        #    channel_name =  "%s:EQ"%ifo
        #    out_dict = {'name'  : channel_name,
        #        'data'  : epics_dicts[ifo]["eq"],
        #        'start' : tt[0],
        #        'dx'    : tt[1]-tt[0],
        #        'kind'  : 'ADC'}
        #    out_dicts.append(out_dict)

        #Fr.frputvect(frameName,out_dicts)
        #print frameName, "completed"

        for ifo in ifos:
            ifoShort = ifo
            params["ifo"] = ifo
            ifo = seismon.utils.getIfo(params)

            #filenamedir = "%s/txt/%s"%(params["epicsDirectory"],ifo)
            #if not os.path.isdir(filenamedir):
            #    os.mkdir(filenamedir)
            fid.write('%s %d %.5f %.5e %d\n'%(ifoShort,epics_dicts[ifoShort]["prob"],epics_dicts[ifoShort]["eta"],epics_dicts[ifoShort]["amp"],epics_dicts[ifoShort]["mult"]))
        fid.close()

        if params["doPlots"]:
            plotsDirectory = os.path.join(params["path"],"plots")
            seismon.utils.mkdir(plotsDirectory)            

            t0 = tt[0]
            samplef = 1.0/(tt[1]-tt[0])
            ttplot = tt - t0
            plot = gwpy.plotter.TimeSeriesPlot(figsize=[14,8])

            for ifo in ifos:
                dataFull = gwpy.timeseries.TimeSeries(epics_dicts[ifo]["amp"], times=None, epoch=t0, channel=ifo, unit=None,sample_rate=samplef, name=ifo)
                plot.add_timeseries(dataFull)
            plot.add_legend(loc=1,prop={'size':10})

            #plot.ylim = [-0.05,0.05]
            plt.show()
            plotName = "%s/amp.png"%(plotsDirectory)
            plot.save(plotName)
            plotName = "%s/amp.eps"%(plotsDirectory)
            plot.save(plotName)
            plotName = "%s/amp.pdf"%(plotsDirectory)
            plot.save(plotName)
            plt.close()

    if params["doReadEPICs"]:

        params["channel"] = None
        params["referenceChannel"] = None
        params = seismon.utils.channel_struct(params,params["epicsChannelList"])
        epicsDirectory = os.path.join(params["epicsDirectory"],"frames")
        frameList = [os.path.join(root, name)
            for root, dirs, files in os.walk(epicsDirectory)
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

        data = []
        for channel in params["channels"]:
            dataFull = seismon.utils.retrieve_timeseries(params, channel, segment)
            data.append(dataFull)

        if params["doTimeseries"]:
            params = seismon.utils.channel_struct(params,params["channelList"])
            datands = []
            for channel in params["channels"]:
                dataFull = gwpy.timeseries.TimeSeries.fetch(channel.station, segment[0], segment[1], verbose=True, host='nds.ligo.caltech.edu')
                dataFull = dataFull / channel.calibration
                dataFull = dataFull.resample(1)
                dataFull = dataFull - np.mean(dataFull)
                datands.append(dataFull)

        if params["doPlots"]:
            plotsDirectory = os.path.join(params["path"],"plots")
            seismon.utils.mkdir(plotsDirectory)

            plot = gwpy.plotter.TimeSeriesPlot(figsize=[14,8])
            for dataFull in data:
                plot.add_timeseries(dataFull)
            plot.add_legend(loc=1,prop={'size':10})

            #plot.ylim = [-0.05,0.05]
            plt.show()
            plotName = "%s/amp.png"%(plotsDirectory)
            plot.save(plotName)
            plotName = "%s/amp.eps"%(plotsDirectory)
            plot.save(plotName)
            plotName = "%s/amp.pdf"%(plotsDirectory)
            plot.save(plotName)
            plt.close()

            if params["doTimeseries"]:
                plot = gwpy.plotter.TimeSeriesPlot(figsize=[14,8])
                for dataFull in datands:
                    plot.add_timeseries(dataFull)
                plot.add_legend(loc=1,prop={'size':10})
                plt.show()
                plotName = "%s/timeseries.png"%(plotsDirectory)
                plot.save(plotName)
                plotName = "%s/timeseries.eps"%(plotsDirectory)
                plot.save(plotName)
                plotName = "%s/timeseries.pdf"%(plotsDirectory)
                plot.save(plotName)
                plt.close()

            sys_command = "cp -r %s/* %s/plots"%(plotsDirectory,params["currentpath"])
            os.system(sys_command)

def run_earthquakes_analysis(params,segment):
    """@run earthquakes analysis.

    @param params
        seismon params dictionary
    @param segment
        [start,end] gps
    """

    gpsStart = segment[0]
    gpsEnd = segment[1]

    ifo = seismon.utils.getIfo(params)

    earthquakesDirectory = os.path.join(params["path"],"earthquakes")
    earthquakesXMLFile = os.path.join(earthquakesDirectory,"earthquakes.xml")
    attributeDics = seismon.utils.read_eqmons(earthquakesXMLFile)

    minDiff = 10*60
    coincident = []
    for i in xrange(len(attributeDics)):
        attributeDic1 = attributeDics[i]
        for j in xrange(len(attributeDics)):
            if j <= i:
                continue
            attributeDic2 = attributeDics[j]
            gpsDiff = attributeDic1["GPS"] - attributeDic2["GPS"]
            if np.absolute(gpsDiff) < minDiff:
                coincident.append(j)
    coincident = list(set(coincident))
    print "%d coincident earthquakes"%len(coincident)
    indexes = list(set(range(len(attributeDics))) - set(coincident))
    attributeDicsKeep = []
    for index in indexes:
        attributeDicsKeep.append(attributeDics[index])
    attributeDics = attributeDicsKeep

    data = {}
    data["prediction"] = loadPredictions(params,segment)
    data["earthquakes_all"] = loadEarthquakes(params,attributeDics)

    data["channels"] = {}
    for channel in params["channels"]:
        data["channels"][channel.station_underscore] = {}
        data["channels"][channel.station_underscore]["info"] = channel
        data["channels"][channel.station_underscore]["psd"] = loadChannelPSD(params,channel,segment)
        data["channels"][channel.station_underscore]["timeseries"] = loadChannelTimeseries(params,channel,segment)
        data["channels"][channel.station_underscore]["earthquakes"] = loadChannelEarthquakes(params,channel,attributeDics)
        data["channels"][channel.station_underscore]["trips"] = loadChannelTrips(params,channel)
        if params["doPowerLawFit"]:
            data["channels"][channel.station_underscore]["powerlaw"] = loadChannelEarthquakesPowerLaw(params,channel,attributeDics)

    data["earthquakes"] = {}
    for attributeDic in attributeDics:

        if not "Arbitrary" in attributeDic["traveltimes"]:
            continue

        data["earthquakes"][attributeDic["eventName"]] = {}
        data["earthquakes"][attributeDic["eventName"]]["attributeDic"] = attributeDic
        data["earthquakes"][attributeDic["eventName"]]["data"] = loadEarthquakeChannels(params,attributeDic)

    if params["doKML"]:
        kmlName = os.path.join(earthquakesDirectory,"earthquakes_time.kml")
        create_kml(params,attributeDics,data,"time",kmlName)
        kmlName = os.path.join(earthquakesDirectory,"earthquakes_amplitude.kml") 
        create_kml(params,attributeDics,data,"amplitude",kmlName)

    save_predictions(params,data)

    if params["doPlots"]:

        print "Creating plots..."

        plotName = os.path.join(earthquakesDirectory,"earthquakes_timedelay.png")
        seismon.eqmon_plot.timedelay_plot(params,data,plotName)

        plotName = os.path.join(earthquakesDirectory,"earthquakes_timedelay_distance.png")
        seismon.eqmon_plot.timedelay_distance_plot(params,data,plotName)

        plotName = os.path.join(earthquakesDirectory,"prediction.png")
        seismon.eqmon_plot.prediction(data,plotName)
        plotName = os.path.join(earthquakesDirectory,"residual.png")
        seismon.eqmon_plot.residual(data,plotName)

        plotName = os.path.join(earthquakesDirectory,"earthquakes_timeseries.png")
        seismon.eqmon_plot.earthquakes_station(params,data,"timeseries",plotName)
        plotName = os.path.join(earthquakesDirectory,"earthquakes_psd.png")
        seismon.eqmon_plot.earthquakes_station(params,data,"psd",plotName)

        plotName = os.path.join(earthquakesDirectory,"earthquakes_distance_amplitude.png")
        seismon.eqmon_plot.earthquakes_station_distance(params,data,"amplitude",plotName)
        plotName = os.path.join(earthquakesDirectory,"earthquakes_distance_time.png")
        seismon.eqmon_plot.earthquakes_station_distance(params,data,"time",plotName)
        plotName = os.path.join(earthquakesDirectory,"earthquakes_distance_residual.png")
        seismon.eqmon_plot.earthquakes_station_distance(params,data,"residual",plotName)

        name = ""
        for key in data["channels"].iterkeys():
            if name == "":
               name = key
            else:
               name = "%s_%s"%(name,key)

        plotName = os.path.join(earthquakesDirectory,"earthquakes_distance_heatmap_time_%s.png"%name)
        seismon.eqmon_plot.earthquakes_station_distance_heatmap(params,data,"time",plotName)

        plotName = os.path.join(earthquakesDirectory,"station_amplitude.png")
        seismon.eqmon_plot.station_plot(params,attributeDics,data,"amplitude",plotName)
        plotName = os.path.join(earthquakesDirectory,"station_time.png")
        seismon.eqmon_plot.station_plot(params,attributeDics,data,"time",plotName)

        if params["doEarthquakesTrips"]:
            plotName = os.path.join(earthquakesDirectory,"trip_time_plot.png")
            seismon.eqmon_plot.trip_time_plot(params,attributeDics,data,"amplitude",plotName)

            plotName = os.path.join(earthquakesDirectory,"trip_plot.png")
            seismon.eqmon_plot.trip_plot(params,attributeDics,data,"amplitude",plotName)

            plotName = os.path.join(earthquakesDirectory,"trip_plot_diff.png")
            seismon.eqmon_plot.trip_plot(params,attributeDics,data,"time",plotName)

            plotName = os.path.join(earthquakesDirectory,"trip_amp_platform_plot.png")
            seismon.eqmon_plot.trip_amp_platform_plot(params,attributeDics,data,"amplitude",plotName)

            plotName = os.path.join(earthquakesDirectory,"trip_amp_stage_plot.png")
            seismon.eqmon_plot.trip_amp_stage_plot(params,attributeDics,data,"amplitude",plotName)

        plotName = os.path.join(earthquakesDirectory,"worldmap_station_amplitude.png")
        seismon.eqmon_plot.worldmap_station_plot(params,attributeDics,data,"amplitude",plotName)
        plotName = os.path.join(earthquakesDirectory,"worldmap_station_time.png")
        seismon.eqmon_plot.worldmap_station_plot(params,attributeDics,data,"time",plotName)

        plotName = os.path.join(earthquakesDirectory,"worldmap_magnitudes.png")
        seismon.eqmon_plot.worldmap_plot(params,attributeDics,"Magnitude",plotName)
        plotName = os.path.join(earthquakesDirectory,"worldmap_traveltimes.png")
        seismon.eqmon_plot.worldmap_plot(params,attributeDics,"Traveltimes",plotName)
        plotName = os.path.join(earthquakesDirectory,"worldmap_restimates.png")
        seismon.eqmon_plot.worldmap_plot(params,attributeDics,"Restimates",plotName)

        plotName = os.path.join(earthquakesDirectory,"worldmap_channel_traveltimes.png")
        seismon.eqmon_plot.worldmap_channel_plot(params,data,"time",plotName)

        plotName = os.path.join(earthquakesDirectory,"restimates.png")
        seismon.eqmon_plot.restimates(params,attributeDics,plotName)
        plotName = os.path.join(earthquakesDirectory,"magnitudes.png")
        seismon.eqmon_plot.magnitudes(params,attributeDics,plotName)
        plotName = os.path.join(earthquakesDirectory,"magnitudes_latencies.png")
        seismon.eqmon_plot.magnitudes_latencies(params,attributeDics,plotName)
        plotName = os.path.join(earthquakesDirectory,"latencies_sent.png")
        seismon.eqmon_plot.latencies_sent(params,attributeDics,plotName)
        plotName = os.path.join(earthquakesDirectory,"latencies_written.png")

        seismon.eqmon_plot.latencies_written(params,attributeDics,plotName)
        plotName = os.path.join(earthquakesDirectory,"traveltimes%s.png"%params["ifo"])
        seismon.eqmon_plot.traveltimes(params,attributeDics,ifo,gpsEnd,plotName)
        plotName = os.path.join(earthquakesDirectory,"worldmap.png")
        seismon.eqmon_plot.worldmap_wavefronts(params,attributeDics,gpsEnd,plotName)

        if params["doEarthquakesVelocityMap"]:
            plotName = os.path.join(earthquakesDirectory,"earthquakes_distance_velocitymap.png")
            seismon.eqmon_plot.earthquakes_station_distance(params,data,"velocitymap",plotName)
            plotName = os.path.join(earthquakesDirectory,"earthquakes_distance_velocitymapresidual.png")
            seismon.eqmon_plot.earthquakes_station_distance(params,data,"velocitymapresidual",plotName)
            plotName = os.path.join(earthquakesDirectory,"velocitymap.png")
            seismon.eqmon_plot.worldmap_velocitymap(params,plotName)

        if params["doEarthquakesLookUp"]:
            plotName = os.path.join(earthquakesDirectory,"earthquakes_distance_lookup.png")
            seismon.eqmon_plot.earthquakes_station_distance(params,data,"lookup",plotName)
            plotName = os.path.join(earthquakesDirectory,"earthquakes_distance_lookupresidual.png")
            seismon.eqmon_plot.earthquakes_station_distance(params,data,"lookupresidual",plotName)

        if params["doPowerLawFit"]:
            plotName = os.path.join(earthquakesDirectory,"earthquakes_powerlaw.png")
            seismon.eqmon_plot.powerlaw_plot(params,data,plotName)
            plotName = os.path.join(earthquakesDirectory,"earthquakes_powerlaw_timedelay.png")
            seismon.eqmon_plot.powerlaw_timedelay_plot(params,data,plotName)

def save_predictions(params,data):
    """@save file for generating predictions

    @param params
        seismon params dictionary
    @param data
        channel data dictionary
    """

    earthquakesDirectory = os.path.join(params["path"],"earthquakes")
    predictionsDirectory = os.path.join(earthquakesDirectory,"predictions")
    seismon.utils.mkdir(predictionsDirectory)

    threshold = 1.5e-6

    for key in data["channels"].iterkeys():
        channel_data = data["channels"][key]["earthquakes"]

        predictionFile = os.path.join(predictionsDirectory,"%s.txt"%key)
        f = open(predictionFile,"w")
        for gps,arrival,departure,latitude, longitude, distance, magnitude, depth,ampMax,ampPrediction,ttDiff in zip(channel_data["gps"],channel_data["arrival"],channel_data["departure"],channel_data["latitude"],channel_data["longitude"],channel_data["distance"],channel_data["magnitude"],channel_data["depth"],channel_data["ampMax"],channel_data["ampPrediction"],channel_data["ttDiff"]):

            if (ampMax < threshold) and (ampPrediction<threshold):
                continue

            f.write("%.0f %.0f %.0f %.2f %.2f %e %.2f %.2f %e %.2f\n"%(gps,arrival,departure,latitude, longitude, distance, magnitude, depth,ampMax,ttDiff))
        f.close()

def create_kml(params,attributeDics,data,type,kmlFile):
    """@create kml

    @param params
        seismon params dictionary
    @param attributeDics
        list of eqmon structures
    @param data
        channel data dictionary
    @param type
        type of worldmap plot
    @param kmlFile
        name of file
    """

    import simplekml

    # Create an instance of Kml
    kml = simplekml.Kml(open=1)

    for attributeDic in attributeDics:
        pnt = kml.newpoint(coords = [(attributeDic["Longitude"],attributeDic["Latitude"])])
        pnt.name = attributeDic["eventName"]
        
        pnt.lookat.longitude = attributeDic["Longitude"]
        pnt.lookat.latitude = attributeDic["Latitude"]

    for channel in params["channels"]:
        channel_data = data["channels"][channel.station_underscore]

        if len(channel_data["timeseries"]["data"]) == 0:
            continue

        pnt = kml.newpoint(coords = [(channel_data["info"].longitude,channel_data["info"].latitude)])
        pnt.name = channel_data["info"].station

        pnt.lookat.longitude = channel_data["info"].longitude
        pnt.lookat.latitude = channel_data["info"].latitude

        #pnt.lookat.longitude = 0
        #pnt.lookat.latitude = 0

        if type == "amplitude":
            z = channel_data["timeseries"]["data"][0] * 1e6
            array = np.linspace(1,2000,1000)
        elif type == "time":
            z = channel_data["timeseries"]["ttMax"][0] - attributeDic["GPS"]
            array = np.linspace(0,10000,1000)

        snrSig,hexcolor = seismon.utils.html_hexcolor(z,array)

        #pnt.style.labelstyle.scale = 0.1  # Text twice as big
        pnt.style.iconstyle.color = hexcolor  # Blue
        pnt.style.iconstyle.scale = 3  # Icon thrice as big
        pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/road_shield3.png'
        #pnt.style.iconstyle.icon.href = None

    kml.save(kmlFile)

def loadPredictions(params,segment):
    """@load earthquakes predictions.

    @param params
        seismon params dictionary
    @param segment
        [start,end] gps
    """

    gpsStart = segment[0]
    gpsEnd = segment[1]

    predictionDirectory = params["dirPath"] + "/Text_Files/Prediction/"
    files = glob.glob(os.path.join(predictionDirectory,"*.txt"))
    files = sorted(files)

    ttStart = []
    ttEnd = []
    amp = []

    for file in files:

        fileSplit = file.split("/")

        txtFile = fileSplit[-1].replace(".txt","")
        txtFileSplit = txtFile.split("-")
        thisTTStart = int(txtFileSplit[0])
        thisTTEnd = int(txtFileSplit[1])

        if (thisTTStart < gpsStart) or (thisTTEnd > gpsEnd):
            continue

        ttStart.append(thisTTStart)
        ttEnd.append(thisTTEnd)

        data_out = np.loadtxt(file)
        thisAmp = data_out

        amp.append(thisAmp)

    ttStart = np.array(ttStart)
    ttEnd = np.array(ttEnd)
    amp = np.array(amp)

    data = {}
    data["ttStart"] = ttStart
    data["ttEnd"] = ttEnd
    data["data"] = amp

    return data

def loadEarthquakes(params,attributeDics):
    """@load earthquakes dictionaries.

    @param params
        seismon params dictionary
    @param attributeDics
        list of seismon earthquake structures
    """

    ifo = seismon.utils.getIfo(params)

    tt = []
    ttArrival = []
    amp = []
    distance = []
    names = []

    for attributeDic in attributeDics:

        if not "Arbitrary" in attributeDic["traveltimes"]:
            continue

        if params["ifo"] == "IRIS":
            distances = []
            for channel in params["channels"]:

                attributeDic = seismon.eqmon.ifotraveltimes_loc(attributeDic, "IRIS", channel.latitude, channel.longitude)
                traveltimes = attributeDic["traveltimes"]["IRIS"]
                distances.append(traveltimes["Distances"][0])
            index = np.argmax(distances)
            channel = params["channels"][index]

            attributeDic = seismon.eqmon.ifotraveltimes_loc(attributeDic, "IRIS", channel.latitude, channel.longitude)
            traveltimes = attributeDic["traveltimes"]["IRIS"]

        else:
            traveltimes = attributeDic["traveltimes"][ifo]

        names.append(attributeDic["eventName"])
        tt.append(attributeDic["GPS"])
        ttArrival.append(max(traveltimes["RthreePointFivetimes"]))
        amp.append(traveltimes["Rfamp"][0])
        distance.append(max(traveltimes["Distances"]))

    tt = np.array(tt)
    ttArrival = np.array(ttArrival)
    amp = np.array(amp)
    distance = np.array(distance)

    data = {}
    data["tt"] = tt
    data["ttArrival"] = ttArrival
    data["data"] = amp
    data["distance"] = distance
    data["names"] = names

    return data

def loadEarthquakeChannels(params,attributeDic):
    """@load earthquakes dictionaries.

    @param params
        seismon params dictionary
    @param attributeDics
        list of seismon earthquake structures
    """

    ifo = seismon.utils.getIfo(params)

    if params["ifo"] == "IRIS":
        for channel in params["channels"]:
            attributeDic = seismon.eqmon.ifotraveltimes_loc(attributeDic, "IRIS", channel.latitude, channel.longitude)
            traveltimes = attributeDic["traveltimes"]["IRIS"]
    else:
        traveltimes = attributeDic["traveltimes"][ifo]

    ttMax = []
    ttDiff = []
    distance = []
    velocity = []
    ampMax = []
    ampPrediction = []
    residual = []

    for channel in params["channels"]:
        earthquakesDirectory = params["dirPath"] + "/Text_Files/Earthquakes/" + channel.station_underscore + "/" + str(params["fftDuration"])
        earthquakesFile = os.path.join(earthquakesDirectory,"%s.txt"%(attributeDic["eventName"]))

        if not os.path.isfile(earthquakesFile):
            continue

        data_out = np.loadtxt(earthquakesFile)
        ttMax.append(data_out[0])
        ttDiff.append(data_out[1])
        distance.append(data_out[2])
        velocity.append(data_out[3])
        ampMax.append(data_out[4])
        ampPrediction.append(traveltimes["Rfamp"][0])
        thisResidual = (data_out[4] - traveltimes["Rfamp"][0])/traveltimes["Rfamp"][0]
        residual.append(thisResidual)

    ttMax = np.array(ttMax)
    ttDiff = np.array(ttDiff)
    distance = np.array(distance)
    velocity = np.array(velocity)
    ampMax = np.array(ampMax)
    ampPrediction = np.array(ampPrediction)
    residual = np.array(residual)

    data = {}
    data["tt"] = ttMax
    data["ttDiff"] = ttDiff
    data["distance"] = distance
    data["velocity"] = velocity
    data["ampMax"] = ampMax
    data["ampPrediction"] = ampPrediction
    data["residual"] = residual

    return data

def loadChannelEarthquakesPowerLaw(params,channel,attributeDics):
    """@load earthquakes dictionaries.

    @param params
        seismon params dictionary
    @param attributeDics
        list of seismon earthquake structures
    """

    ifo = seismon.utils.getIfo(params)

    amp = []
    index = []
    distance = []
    depth = []
    magnitude = []
    Ptimes = []
    Stimes = []
    RthreePointFivetimes = []
    Rtwotimes = []
    Rfivetimes = []

    for attributeDic in attributeDics:

        if not "Arbitrary" in attributeDic["traveltimes"]:
            continue

        if params["ifo"] == "IRIS":
            attributeDic = seismon.eqmon.ifotraveltimes_loc(attributeDic, "IRIS", channel.latitude, channel.longitude)
            traveltimes = attributeDic["traveltimes"]["IRIS"]
        else:
            traveltimes = attributeDic["traveltimes"][ifo]

        EQpowerlawDirectory = params["dirPath"] + "/Text_Files/EQPowerlaw/" + channel.station_underscore + "/" + str(params["fftDuration"])
        EQPowerLawFile = os.path.join(EQpowerlawDirectory,"%s.txt"%(attributeDic["eventName"]))

        if not os.path.isfile(EQPowerLawFile):
            continue

        data_out = np.loadtxt(EQPowerLawFile)
        index.append(data_out[0])
        amp.append(data_out[1])
        distance.append(traveltimes["Distances"][0])
        depth.append(attributeDic["Depth"])
        magnitude.append(attributeDic["Magnitude"])
        Ptimes.append(max(traveltimes["Ptimes"]))
        Stimes.append(max(traveltimes["Stimes"]))
        RthreePointFivetimes.append(max(traveltimes["RthreePointFivetimes"]))
        Rfivetimes.append(max(traveltimes["Rfivetimes"]))
        Rtwotimes.append(max(traveltimes["Rtwotimes"]))

    Ptimes = np.array(Ptimes)
    Stimes = np.array(Stimes)
    RthreePointFivetimes = np.array(RthreePointFivetimes)
    Rfivetimes = np.array(Rfivetimes)
    Rtwotimes = np.array(Rtwotimes)

    data = {}
    data["index"] = np.array(index)
    data["amp"] = np.array(amp)
    data["distance"] = np.array(distance)
    data["depth"] = np.array(depth)
    data["magnitude"] = np.array(magnitude)
    data["Ptimes"] = Ptimes
    data["Stimes"] = Stimes
    data["RthreePointFivetimes"] = RthreePointFivetimes
    data["Rfivetimes"] = Rfivetimes
    data["Rtwotimes"] = Rtwotimes

    return data

def loadChannelEarthquakes(params,channel,attributeDics):
    """@load earthquakes dictionaries.

    @param params
        seismon params dictionary
    param attributeDics
        list of seismon earthquake structures
    """

    ifo = seismon.utils.getIfo(params)

    ttMax = []
    ttDiff = []
    distance = []
    velocity = []
    ampMax = [] 
    ampPrediction = []
    depth = []
    magnitude = []
    latitude = []
    longitude = []
    arrival = []
    departure = []
    gps = []
    residual = []
    Rvelocitymaptime = []
    RvelocitymaptimeDiff = []
    RvelocitymaptimeResidual = []
    Rlookuptime = []
    RlookuptimeDiff = []
    RlookuptimeResidual = [] 
    Ptimes = []
    Stimes = []
    RthreePointFivetimes = []
    Rtwotimes = []
    Rfivetimes = []
 
    print "Large residuals"
    for attributeDic in attributeDics:

        if not "Arbitrary" in attributeDic["traveltimes"]:
            continue

        if params["ifo"] == "IRIS":
            attributeDic = seismon.eqmon.ifotraveltimes_loc(attributeDic, "IRIS", channel.latitude, channel.longitude)
            traveltimes = attributeDic["traveltimes"]["IRIS"]
        else:
            traveltimes = attributeDic["traveltimes"][ifo]

        earthquakesDirectory = params["dirPath"] + "/Text_Files/Earthquakes/" + channel.station_underscore + "/" + str(params["fftDuration"])
        earthquakesFile = os.path.join(earthquakesDirectory,"%s.txt"%(attributeDic["eventName"]))

        if not os.path.isfile(earthquakesFile):
            continue

        thisArrival = np.min([max(traveltimes["Rtwotimes"]),max(traveltimes["RthreePointFivetimes"]),max(traveltimes["Rfivetimes"]),max(traveltimes["Stimes"]),max(traveltimes["Ptimes"])])
        thisDeparture = np.max([max(traveltimes["Rtwotimes"]),max(traveltimes["RthreePointFivetimes"]),max(traveltimes["Rfivetimes"]),max(traveltimes["Stimes"]),max(traveltimes["Ptimes"])])

        arrival_floor = np.floor(thisArrival / 100.0) * 100.0
        departure_ceil = np.ceil(thisDeparture / 100.0) * 100.0

        data_out = np.loadtxt(earthquakesFile)
        ttMax.append(data_out[0])
        ttDiff.append(data_out[1])
        distance.append(data_out[2])
        velocity.append(data_out[3])
        ampMax.append(data_out[4])
        ampPrediction.append(traveltimes["Rfamp"][0])
        depth.append(attributeDic["Depth"])
        magnitude.append(attributeDic["Magnitude"])
        latitude.append(attributeDic["Latitude"])
        longitude.append(attributeDic["Longitude"])
        arrival.append(arrival_floor)
        departure.append(departure_ceil)
        gps.append(attributeDic["GPS"])
        thisResidual = (data_out[4] - traveltimes["Rfamp"][0])/traveltimes["Rfamp"][0]
        residual.append(thisResidual)
        Ptimes.append(max(traveltimes["Ptimes"]))
        Stimes.append(max(traveltimes["Stimes"]))
        RthreePointFivetimes.append(max(traveltimes["RthreePointFivetimes"]))
        Rfivetimes.append(max(traveltimes["Rfivetimes"]))
        Rtwotimes.append(max(traveltimes["Rtwotimes"]))

        if params["doEarthquakesVelocityMap"]:
            thisRvelocitymaptime = max(traveltimes["Rvelocitymaptimes"])
            Rvelocitymaptime.append(thisRvelocitymaptime)
            thisRvelocitymaptimeDiff = thisRvelocitymaptime - attributeDic["GPS"]
            RvelocitymaptimeDiff.append(thisRvelocitymaptimeDiff)
            thisRvelocitymaptimeResidual = (thisRvelocitymaptimeDiff - data_out[1]) / data_out[1]
            RvelocitymaptimeResidual.append(thisRvelocitymaptimeResidual)
        if params["doEarthquakesLookUp"]:
            thisRlookuptime = traveltimes["Rlookuptime"][0]
            Rlookuptime.append(thisRlookuptime)
            thisRlookuptimeDiff = thisRlookuptime - attributeDic["GPS"]
            RlookuptimeDiff.append(thisRlookuptimeDiff)
            thisRlookuptimeResidual = (thisRlookuptimeDiff - data_out[1]) / data_out[1]
            RlookuptimeResidual.append(thisRlookuptimeResidual)

        if thisResidual > 100:
            print "%.0f %.0f %.0f %e"%(attributeDic["GPS"],arrival_floor,departure_ceil,thisResidual)

    ttMax = np.array(ttMax)
    ttDiff = np.array(ttDiff)
    distance = np.array(distance)
    velocity = np.array(velocity)
    ampMax = np.array(ampMax)
    ampPrediction = np.array(ampPrediction)
    depth = np.array(depth)
    magnitude = np.array(magnitude)
    latitude = np.array(latitude)
    longitude = np.array(longitude)
    arrival = np.array(arrival)
    departure = np.array(departure)
    gps = np.array(gps)
    residual = np.array(residual)
    Rvelocitymaptime = np.array(Rvelocitymaptime)
    RvelocitymaptimeDiff = np.array(RvelocitymaptimeDiff)
    RvelocitymaptimeResidual = np.array(RvelocitymaptimeResidual)
    Rlookuptime = np.array(Rlookuptime)
    RlookuptimeDiff = np.array(RlookuptimeDiff)
    RlookuptimeResidual = np.array(RlookuptimeResidual)
    Ptimes = np.array(Ptimes)
    Stimes = np.array(Stimes)
    RthreePointFivetimes = np.array(RthreePointFivetimes)
    Rfivetimes = np.array(Rfivetimes)
    Rtwotimes = np.array(Rtwotimes)

    data = {}
    data["tt"] = ttMax
    data["ttDiff"] = ttDiff
    data["distance"] = distance
    data["velocity"] = velocity
    data["ampMax"] = ampMax
    data["ampPrediction"] = ampPrediction
    data["depth"] = depth
    data["magnitude"] = magnitude
    data["latitude"] = latitude
    data["longitude"] = longitude
    data["arrival"] = arrival
    data["departure"] = departure 
    data["gps"] = gps
    data["residual"] = residual
    data["Rvelocitymaptime"] = Rvelocitymaptime
    data["RvelocitymaptimeDiff"] = RvelocitymaptimeDiff
    data["RvelocitymaptimeResidual"] = RvelocitymaptimeResidual
    data["Rlookuptime"] = Rlookuptime
    data["RlookuptimeDiff"] = RlookuptimeDiff
    data["RlookuptimeResidual"] = RlookuptimeResidual
    data["Ptimes"] = Ptimes
    data["Stimes"] = Stimes
    data["RthreePointFivetimes"] = RthreePointFivetimes
    data["Rfivetimes"] = Rfivetimes
    data["Rtwotimes"] = Rtwotimes

    return data

def loadChannelPSD(params,channel,segment):
    """@load channel PSDs.

    @param params
        seismon params dictionary
    @param channel
        seismon channel structure
    @param segment
        [start,end] gps
    """

    gpsStart = segment[0]
    gpsEnd = segment[1]

    # Break up entire frequency band into 6 segments
    ff_ave = [1/float(128), 1/float(64),  0.1, 1, 3, 5, 10]

    psdDirectory = params["dirPath"] + "/Text_Files/PSD/" + channel.station_underscore + "/" + str(params["fftDuration"])

    files = glob.glob(os.path.join(psdDirectory,"*.txt"))
    files = sorted(files)

    ttStart = []
    ttEnd = []
    amp = []

    for file in files:

        fileSplit = file.split("/")
        txtFile = fileSplit[-1].replace(".txt","")
        txtFileSplit = txtFile.split("-")
        thisTTStart = int(txtFileSplit[0])
        thisTTEnd = int(txtFileSplit[1])

        if (thisTTStart < gpsStart) or (thisTTEnd > gpsEnd):
            continue

        ttStart.append(thisTTStart)
        ttEnd.append(thisTTEnd)

        data_out = np.loadtxt(file)
        thisSpectra_out = data_out[:,1]
        thisFreq_out = data_out[:,0]

        freqAmps = []
        for i in xrange(len(ff_ave)-1):
            newSpectraNow = []
            for j in xrange(len(thisFreq_out)):
                if ff_ave[i] <= thisFreq_out[j] and thisFreq_out[j] <= ff_ave[i+1]:
                    newSpectraNow.append(thisSpectra_out[j])
                    freqAmps.append(np.mean(newSpectraNow))

        thisAmp = freqAmps[1]
        amp.append(thisAmp)

    ttStart = np.array(ttStart)
    ttEnd = np.array(ttEnd)
    amp = np.array(amp)

    data = {}
    data["ttStart"] = ttStart
    data["ttEnd"] = ttEnd
    data["data"] = amp

    return data

def loadChannelTimeseries(params,channel,segment):
    """@load channel timeseries.

    @param params
        seismon params dictionary
    @param channel
        seismon channel structure
    @param segment
        [start,end] gps
    """

    gpsStart = segment[0]
    gpsEnd = segment[1]

    timeseriesDirectory = params["dirPath"] + "/Text_Files/Timeseries/" + channel.station_underscore + "/" + str(params["fftDuration"])

    files = glob.glob(os.path.join(timeseriesDirectory,"*.txt"))
    files = sorted(files)

    ttStart = []
    ttEnd = []
    ttMax = []
    amp = []

    for file in files:

        fileSplit = file.split("/")
        txtFile = fileSplit[-1].replace(".txt","")
        txtFileSplit = txtFile.split("-")
        thisTTStart = int(txtFileSplit[0])
        thisTTEnd = int(txtFileSplit[1])

        if (thisTTStart < gpsStart) or (thisTTEnd > gpsEnd):
            continue

        ttStart.append(thisTTStart)
        ttEnd.append(thisTTEnd)

        data_out = np.loadtxt(file)

        thisttMax = data_out[1,0]
        thisAmp = data_out[1,1]
        ttMax.append(thisttMax)
        amp.append(thisAmp)

    data = {}
    data["ttStart"] = ttStart
    data["ttEnd"] = ttEnd
    data["ttMax"] = ttMax
    data["data"] = amp

    return data

def loadChannelTrips(params,channel):
    """@load channel timeseries.

    @param params
        seismon params dictionary
    @param channel
        seismon channel structure
    @param segment
        [start,end] gps
    """

    tripsDirectory = params["dirPath"] + "/Text_Files/Trips/" + channel.station_underscore + "/" + str(params["fftDuration"])

    data = {}

    platforms = glob.glob(os.path.join(tripsDirectory,"*"))
    for platformDirectory in platforms:
        platform = platformDirectory.split("/")[-1]
        data[platform] = {}
        stages = glob.glob(os.path.join(platformDirectory,"*"))
        for stageDirectory in stages:
            stage = stageDirectory.split("/")[-1]
            data[platform][stage] = {}
            data[platform][stage]["gps"] = []
            data[platform][stage]["velocity"] = []

            files = glob.glob(os.path.join(stageDirectory,"*.txt"))
            for file in files:
                data_out = np.loadtxt(file)
                data[platform][stage]["gps"].append(data_out[0])
                data[platform][stage]["velocity"].append(data_out[1])

    return data

def write_info(file,attributeDics):
    """@write eqmon file

    @param file
        eqmon file
    @param attributeDics
        list of eqmon structures
    """

    baseroot = etree.Element('eqmon')
    for attributeDic in attributeDics:
        root = etree.SubElement(baseroot,attributeDic["eventName"])
        for key, value in attributeDic.items():
            if not key == "traveltimes":
                element = etree.SubElement(root,key)
                element.text = str(value)
        element = etree.SubElement(root,'traveltimes')
        for key, value in attributeDic["traveltimes"].items():
            subelement = etree.SubElement(element,key)
            for category in value:
                subsubelement = etree.SubElement(subelement,category)
                subsubelement.text = write_array(value[category])

    tree = etree.ElementTree(baseroot)
    tree.write(file, pretty_print=True, xml_declaration=True)

def write_array(array):
    """@create string of array values

    @param array
        array of values
    """

    if isinstance(array, float):
        text = str(array)
    else:
        text = ' '.join([str(x) for x in array])
    return text

def parse_xml(element):
    """@parse xml element.

    @param element
        xml element
    """

    subdic = {}

    numChildren = 0
    for subelement in element.iterchildren():
        tag = str(subelement.tag)
        tag = tag.replace("{http://www.usgs.gov/ansseqmsg}","")
        tag = tag.replace("{http://quakeml.org/xmlns/quakeml/1.2}","")
        tag = tag.replace("{http://quakeml.org/xmlns/bed/1.2}","")
        subdic[tag] = parse_xml(subelement)
        numChildren += 1

    if numChildren == 0:
        value = str(element.text)
        value = value.replace("{http://www.usgs.gov/ansseqmsg}","")
    else:
        value = subdic

    tag = str(element.tag)
    tag = tag.replace("{http://www.usgs.gov/ansseqmsg}","")
    tag = tag.replace("{http://quakeml.org/xmlns/quakeml/1.2}","")
    tag = tag.replace("{http://quakeml.org/xmlns/bed/1.2}","")
    dic = value

    return dic

def cmtread(event):
    """@read cmt event.

    @param event
        cmt event
    """

    attributeDic = {}

    attributeDic['Time'] = str(event.origins[0].time)
    timeString = attributeDic['Time'].replace("T"," ").replace("Z","")
    dt = datetime.strptime(timeString, "%Y-%m-%d %H:%M:%S.%f")
    tm = time.struct_time(dt.timetuple())

    attributeDic["Longitude"] = event.origins[0].longitude
    attributeDic["Latitude"] = event.origins[0].latitude
    if not event.origins[0].depth == None:
        attributeDic["Depth"] = event.origins[0].depth / 1000.0
    else:
        attributeDic["Depth"] = 0
    attributeDic["eventID"] = event.origins[0].region

    td = dt - datetime(1970, 1, 1)
    thistime =  td.microseconds * 1e-6 + td.seconds + td.days * 24 * 3600

    if thistime < 300000000:
        attributeDic['GPS'] = thistime
    else:
        attributeDic['GPS'] = astropy.time.Time(dt, format='datetime', scale='utc').gps
        #attributeDic['GPS'] = float(lal.gpstime.utc_to_gps(dt))

    attributeDic['UTC'] = float(dt.strftime("%s"))

    eventID = "%.0f"%attributeDic['GPS']
    eventName = ''.join(["cmt",str(eventID)])

    attributeDic["eventName"] = eventName
    attributeDic["Magnitude"] = event.magnitudes[0].mag
    attributeDic["MomentMagnitude"] = (attributeDic["Magnitude"] - 9.1)/1.5
    attributeDic["DataSource"] = "CMT"
    attributeDic["Version"] = 1.0
    attributeDic["Type"] = 1.0
    attributeDic['Region'] = event.origins[0].region

    if event.origins[0].evaluation_status == "AUTOMATIC":
        attributeDic["Review"] = "Automatic"
    else:
        attributeDic["Review"] = "Manual"

    SentTime = time.gmtime()
    dt = datetime.utcfromtimestamp(calendar.timegm(SentTime))
    attributeDic['SentGPS'] = astropy.time.Time(dt, format='datetime', scale='utc').gps
    #attributeDic['SentGPS'] = float(lal.gpstime.utc_to_gps(dt))
    attributeDic['SentUTC'] = time.time()
    attributeDic['Sent'] = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", SentTime)

    attributeDic = calculate_traveltimes(attributeDic)
    tm = time.struct_time(time.gmtime())
    dt = datetime.utcfromtimestamp(calendar.timegm(tm))

    attributeDic['WrittenGPS'] = astropy.time.Time(dt, format='datetime', scale='utc').gps
    #attributeDic['WrittenGPS'] = float(lal.gpstime.utc_to_gps(dt))
    attributeDic['WrittenUTC'] = float(time.time())

    nodalPlane1 = event["focal_mechanisms"][0]["nodal_planes"]["nodal_plane_1"]
    nodalPlane2 = event["focal_mechanisms"][0]["nodal_planes"]["nodal_plane_2"]

    attributeDic["nodalPlane1_strike"] = float(nodalPlane1["strike"])
    attributeDic["nodalPlane1_rake"] = float(nodalPlane1["rake"])
    attributeDic["nodalPlane1_dip"] = float(nodalPlane1["dip"])
    attributeDic["nodalPlane2_strike"] = float(nodalPlane2["strike"])
    attributeDic["nodalPlane2_rake"] = float(nodalPlane2["rake"])
    attributeDic["nodalPlane2_dip"] = float(nodalPlane2["dip"])

    momentTensor = event["focal_mechanisms"][0]["moment_tensor"]
    tensor = momentTensor["tensor"]
    attributeDic["momentTensor_Mrt"] = float(tensor["m_rt"]) / float(momentTensor["scalar_moment"])
    attributeDic["momentTensor_Mtp"] = float(tensor["m_tp"]) / float(momentTensor["scalar_moment"])
    attributeDic["momentTensor_Mrp"] = float(tensor["m_rp"]) / float(momentTensor["scalar_moment"])
    attributeDic["momentTensor_Mtt"] = float(tensor["m_tt"]) / float(momentTensor["scalar_moment"])
    attributeDic["momentTensor_Mrr"] = float(tensor["m_rr"]) / float(momentTensor["scalar_moment"])
    attributeDic["momentTensor_Mpp"] = float(tensor["m_pp"]) / float(momentTensor["scalar_moment"])

    return attributeDic

def read_product(file,eventName):
    """@read eqxml file.

    @param file
        eqxml file
    @param eventName
        name of earthquake event
    """

    tree = etree.parse(file)
    root = tree.getroot()
    dic = parse_xml(root)

    attributeDic = []

    if not "event" in dic["eventParameters"]:
        return attributeDic

    if "origin" not in dic["eventParameters"]["event"]:
        return attributeDic

    if "nodalPlanes" not in dic["eventParameters"]["event"]["focalMechanism"]:
        return attributeDic

    attributeDic = {}

    #triggeringOriginID = dic["eventParameters"]["event"]["focalMechanism"]["triggeringOriginID"]
    #eventID = triggeringOriginID.split("/")[-1]
    #creationInfo = dic["eventParameters"]["event"]["focalMechanism"]["creationInfo"]
    #agencyID = creationInfo["agencyID"]

    eventNameSplit = eventName.split("_")
    if len(eventNameSplit) == 3:
        eventID = "%s%s"%(eventNameSplit[0],eventNameSplit[1])
    else:
        eventID = eventNameSplit[0]

    attributeDic["eventID"] = eventID
    attributeDic["eventName"] = eventID

    attributeDic["Longitude"] = float(dic["eventParameters"]["event"]["origin"]["longitude"]["value"])
    attributeDic["Latitude"] = float(dic["eventParameters"]["event"]["origin"]["latitude"]["value"])
    attributeDic["Depth"] = float(dic["eventParameters"]["event"]["origin"]["depth"]["value"]) / 1000

    attributeDic["Time"] = dic["eventParameters"]["event"]["origin"]["time"]["value"]
    timeString = attributeDic["Time"].replace("T"," ").replace("Z","")
    try:
        dt = datetime.strptime(timeString, "%Y-%m-%d %H:%M:%S.%f")
        tm = time.struct_time(dt.timetuple())
        attributeDic['GPS'] = astropy.time.Time(dt, format='datetime', scale='utc').gps
        attributeDic['UTC'] = float(dt.strftime("%s"))
    except:
        dt = datetime.strptime(timeString, "%Y-%m-%d %H:%M:%S")
        tm = time.struct_time(dt.timetuple())
        attributeDic['GPS'] = astropy.time.Time(dt, format='datetime', scale='utc').gps
        attributeDic['UTC'] = float(dt.strftime("%s"))

    if "creationInfo" in dic["eventParameters"]["event"]:
        attributeDic["Sent"] = dic["eventParameters"]["event"]["creationInfo"]["creationTime"]
        timeString = attributeDic["Sent"].replace("T"," ").replace("Z","")

        try:
            dt = datetime.strptime(timeString, "%Y-%m-%d %H:%M:%S.%f")
            tm = time.struct_time(dt.timetuple())
            attributeDic['SentGPS'] = astropy.time.Time(dt, format='datetime', scale='utc').gps
            attributeDic['SentUTC'] = float(dt.strftime("%s"))
        except:
            dt = datetime.strptime(timeString, "%Y-%m-%d %H:%M:%S")
            tm = time.struct_time(dt.timetuple())
            attributeDic['SentGPS'] = astropy.time.Time(dt, format='datetime', scale='utc').gps
            attributeDic['SentUTC'] = float(dt.strftime("%s"))
    else:
        tm = time.struct_time(time.gmtime())
        dt = datetime.utcfromtimestamp(calendar.timegm(tm))

        attributeDic['SentGPS'] = astropy.time.Time(dt, format='datetime', scale='utc').gps
        attributeDic['SentUTC'] = float(time.time())

    if "magnitude" in dic["eventParameters"]["event"]:
        attributeDic["Magnitude"] = float(dic["eventParameters"]["event"]["magnitude"]["mag"]["value"])
    else:
        attributeDic["Magnitude"] = 0

    nodalPlanes = dic["eventParameters"]["event"]["focalMechanism"]["nodalPlanes"]
    nodalPlane1 = nodalPlanes["nodalPlane1"]
    nodalPlane2 = nodalPlanes["nodalPlane2"]

    attributeDic["nodalPlane1_strike"] = float(nodalPlane1["strike"]["value"])
    attributeDic["nodalPlane1_rake"] = float(nodalPlane1["rake"]["value"])
    attributeDic["nodalPlane1_dip"] = float(nodalPlane1["dip"]["value"])
    attributeDic["nodalPlane2_strike"] = float(nodalPlane2["strike"]["value"])
    attributeDic["nodalPlane2_rake"] = float(nodalPlane2["rake"]["value"])
    attributeDic["nodalPlane2_dip"] = float(nodalPlane2["dip"]["value"])

    momentTensor = dic["eventParameters"]["event"]["focalMechanism"]["momentTensor"]
    tensor = momentTensor["tensor"]
    attributeDic["momentTensor_Mrt"] = float(tensor["Mrt"]["value"]) / float(momentTensor["scalarMoment"]["value"])
    attributeDic["momentTensor_Mtp"] = float(tensor["Mtp"]["value"]) / float(momentTensor["scalarMoment"]["value"])
    attributeDic["momentTensor_Mrp"] = float(tensor["Mrp"]["value"]) / float(momentTensor["scalarMoment"]["value"])
    attributeDic["momentTensor_Mtt"] = float(tensor["Mtt"]["value"]) / float(momentTensor["scalarMoment"]["value"])
    attributeDic["momentTensor_Mrr"] = float(tensor["Mrr"]["value"]) / float(momentTensor["scalarMoment"]["value"])
    attributeDic["momentTensor_Mpp"] = float(tensor["Mpp"]["value"]) / float(momentTensor["scalarMoment"]["value"])

    return attributeDic

def read_eqxml(file,eventName):
    """@read eqxml file.

    @param file
        eqxml file
    @param eventName
        name of earthquake event
    """

    tree = etree.parse(file)
    root = tree.getroot()
    dic = parse_xml(root)

    attributeDic = {}

    if not "Origin" in dic["Event"] or not "Magnitude" in dic["Event"]["Origin"]:
        return attributeDic

    attributeDic["Longitude"] = float(dic["Event"]["Origin"]["Longitude"])
    attributeDic["Latitude"] = float(dic["Event"]["Origin"]["Latitude"])
    attributeDic["Depth"] = float(dic["Event"]["Origin"]["Depth"])
    attributeDic["eventID"] = dic["Event"]["EventID"]
    attributeDic["eventName"] = eventName
    attributeDic["Magnitude"] = float(dic["Event"]["Origin"]["Magnitude"]["Value"])
    attributeDic["MomentMagnitude"] = (attributeDic["Magnitude"] - 9.1)/1.5

    if "Region" in dic["Event"]["Origin"]:
        attributeDic["Region"] = dic["Event"]["Origin"]["Region"]
    else:
        attributeDic["Region"] = "N/A"

    attributeDic["Time"] = dic["Event"]["Origin"]["Time"]
    timeString = attributeDic["Time"].replace("T"," ").replace("Z","")
    dt = datetime.strptime(timeString, "%Y-%m-%d %H:%M:%S.%f")
    tm = time.struct_time(dt.timetuple())

    attributeDic['GPS'] = astropy.time.Time(dt, format='datetime', scale='utc').gps
    #attributeDic['GPS'] = float(lal.gpstime.utc_to_gps(dt))
    attributeDic['UTC'] = float(dt.strftime("%s"))

    attributeDic["Sent"] = dic["Sent"]
    timeString = attributeDic["Sent"].replace("T"," ").replace("Z","")
    dt = datetime.strptime(timeString, "%Y-%m-%d %H:%M:%S.%f")
    tm = time.struct_time(dt.timetuple())
    attributeDic['SentGPS'] = astropy.time.Time(dt, format='datetime', scale='utc').gps
    #attributeDic['SentGPS'] = float(lal.gpstime.utc_to_gps(dt))
    attributeDic['SentUTC'] = float(dt.strftime("%s"))

    attributeDic["DataSource"] = dic["Source"]
    attributeDic["Version"] = dic["Event"]["Version"]

    if "Type" in dic["Event"]:
        attributeDic["Type"] = dic["Event"]["Type"]
    else:
        attributeDic["Type"] = "N/A"  

    if dic["Event"]["Origin"]["Status"] == "Automatic":
        attributeDic["Review"] = "Automatic"
    else:
        attributeDic["Review"] = "Manual"

    attributeDic = calculate_traveltimes(attributeDic)
    tm = time.struct_time(time.gmtime())
    dt = datetime.utcfromtimestamp(calendar.timegm(tm))

    attributeDic['WrittenGPS'] = astropy.time.Time(dt, format='datetime', scale='utc').gps
    #attributeDic['WrittenGPS'] = float(lal.gpstime.utc_to_gps(dt))
    attributeDic['WrittenUTC'] = float(time.time())

    return attributeDic

def read_quakeml(file,eventName):
    """@read quakeml file.

    @param file
        quakeml file
    @param eventName
        name of earthquake event
    """

    tree = etree.parse(file)
    root = tree.getroot()
    dic = parse_xml(root)

    attributeDic = {}

    if "origin" not in dic["eventParameters"]["event"]:
        return attributeDic
    if "type" not in dic["eventParameters"]["event"]:
        dic["eventParameters"]["event"]["type"] = "None"

    attributeDic["Longitude"] = float(dic["eventParameters"]["event"]["origin"]["longitude"]["value"])
    attributeDic["Latitude"] = float(dic["eventParameters"]["event"]["origin"]["latitude"]["value"])
    attributeDic["Depth"] = float(dic["eventParameters"]["event"]["origin"]["depth"]["value"]) / 1000
    attributeDic["eventID"] = eventName
    attributeDic["eventName"] = eventName

    if "magnitude" in dic["eventParameters"]["event"]:
        attributeDic["Magnitude"] = float(dic["eventParameters"]["event"]["magnitude"]["mag"]["value"])
    else:
        attributeDic["Magnitude"] = 0
    attributeDic["MomentMagnitude"] = (attributeDic["Magnitude"] - 9.1)/1.5

    attributeDic["Time"] = dic["eventParameters"]["event"]["origin"]["time"]["value"]
    timeString = attributeDic["Time"].replace("T"," ").replace("Z","")
    dt = datetime.strptime(timeString, "%Y-%m-%d %H:%M:%S.%f")
    tm = time.struct_time(dt.timetuple())
    attributeDic['GPS'] = astropy.time.Time(dt, format='datetime', scale='utc').gps
    #attributeDic['GPS'] = float(lal.gpstime.utc_to_gps(dt))
    attributeDic['UTC'] = float(dt.strftime("%s"))

    attributeDic["Sent"] = dic["eventParameters"]["event"]["creationInfo"]["creationTime"]
    timeString = attributeDic["Sent"].replace("T"," ").replace("Z","")
    dt = datetime.strptime(timeString, "%Y-%m-%d %H:%M:%S.%f")
    tm = time.struct_time(dt.timetuple())
    attributeDic['SentGPS'] = astropy.time.Time(dt, format='datetime', scale='utc').gps
    #attributeDic['SentGPS'] = float(lal.gpstime.utc_to_gps(dt))
    attributeDic['SentUTC'] = float(dt.strftime("%s"))

    attributeDic["DataSource"] = dic["eventParameters"]["event"]["creationInfo"]["agencyID"]
    #attributeDic["Version"] = float(dic["eventParameters"]["event"]["creationInfo"]["version"])
    attributeDic["Type"] = dic["eventParameters"]["event"]["type"]

    if "evalulationMode" in dic["eventParameters"]["event"]:
        if dic["eventParameters"]["event"]["origin"]["evaluationMode"] == "automatic":
            attributeDic["Review"] = "Automatic"
        else:
            attributeDic["Review"] = "Manual"
    else:
        attributeDic["Review"] = "Unknown"

    attributeDic = calculate_traveltimes(attributeDic)
    tm = time.struct_time(time.gmtime())
    dt = datetime.utcfromtimestamp(calendar.timegm(tm))

    attributeDic['WrittenGPS'] = astropy.time.Time(dt, format='datetime', scale='utc').gps
    #attributeDic['WrittenGPS'] = float(lal.gpstime.utc_to_gps(dt))
    attributeDic['WrittenUTC'] = float(time.time())

    return attributeDic

def read_eqmon(params,file):
    """@read eqmon file.

    @param params
        seismon params struct
    @param file
        name of eqmon file
    """

    attributeDic = {}
    tree = etree.parse(file)
    root = tree.getroot()       # get the document root
    for element in root.iterchildren(): # now iter through it and print the text
        if element.tag == "traveltimes":
            attributeDic[element.tag] = {}
            for subelement in element.iterchildren():
                attributeDic[element.tag][subelement.tag] = {}
                for subsubelement in subelement.iterchildren():
                    textlist = subsubelement.text.replace("\n","").split(" ")
                    floatlist = [float(x) for x in textlist]
                    attributeDic[element.tag][subelement.tag][subsubelement.tag] = floatlist
        else:
            try:
                attributeDic[element.tag] = float(element.text)
            except:
                attributeDic[element.tag] = element.text

    magThreshold = 0
    if not "Magnitude" in attributeDic or attributeDic["Magnitude"] < magThreshold:
        return attributeDic
    if not "traveltimes" in attributeDic:
        attributeDic = calculate_traveltimes(attributeDic)

    attributeDic["doPlots"] = 0
    for ifoName, traveltimes in attributeDic["traveltimes"].items():
        #arrivalMin = min([max(traveltimes["Rtimes"]),max(traveltimes["Stimes"]),max(traveltimes["Ptimes"])])
        #arrivalMax = max([max(traveltimes["Rtimes"]),max(traveltimes["Stimes"]),max(traveltimes["Ptimes"])])
        arrivalMin = min([max(traveltimes["Rtwotimes"]),max(traveltimes["RthreePointFivetimes"]),max(traveltimes["Rfivetimes"]),max(traveltimes["Stimes"]),max(traveltimes["Ptimes"])])
        arrivalMax = max([max(traveltimes["Rtwotimes"]),max(traveltimes["RthreePointFivetimes"]),max(traveltimes["Rfivetimes"]),max(traveltimes["Stimes"]),max(traveltimes["Ptimes"])])

        attributeDic["traveltimes"][ifoName]["arrivalMin"] = arrivalMin
        attributeDic["traveltimes"][ifoName]["arrivalMax"] = arrivalMax
        #if params["gps"] <= attributeDic["traveltimes"][ifoName]["arrivalMax"]:
        #    attributeDic["doPlots"] = 1
    return attributeDic

def jsonread(event):
    """@read json event.

    @param event
        json event
    """

    attributeDic = {}
    attributeDic["Longitude"] = event["geometry"]["coordinates"][0]
    attributeDic["Latitude"] = event["geometry"]["coordinates"][1]
    attributeDic["Depth"] = event["geometry"]["coordinates"][2]
    attributeDic["eventID"] = event["properties"]["code"]
    attributeDic["eventName"] = event["properties"]["ids"].split(",")[1]
    attributeDic["Magnitude"] = event["properties"]["mag"]

    attributeDic["MomentMagnitude"] = (attributeDic["Magnitude"] - 9.1)/1.5
    attributeDic["UTC"] = float(event["properties"]["time"]) / 1000.0
    attributeDic["DataSource"] = event["properties"]["sources"].replace(",","")
    attributeDic["Version"] = 1.0
    attributeDic["Type"] = 1.0
    attributeDic['Region'] = event["properties"]["place"]

    if event["properties"]["status"] == "AUTOMATIC":
        attributeDic["Review"] = "Automatic"
    else:
        attributeDic["Review"] = "Manual"

    Time = time.gmtime(attributeDic["UTC"])
    dt = datetime.utcfromtimestamp(calendar.timegm(Time))

    attributeDic['GPS'] = astropy.time.Time(dt, format='datetime', scale='utc').gps
    #attributeDic['GPS'] = float(lal.gpstime.utc_to_gps(dt))
    SentTime = time.gmtime()
    dt = datetime.utcfromtimestamp(calendar.timegm(SentTime))
    attributeDic['SentGPS'] = astropy.time.Time(dt, format='datetime', scale='utc').gps
    #attributeDic['SentGPS'] = float(lal.gpstime.utc_to_gps(dt))
    attributeDic['SentUTC'] = time.time()

    attributeDic['Time'] = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", Time)
    attributeDic['Sent'] = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", SentTime)

    attributeDic = calculate_traveltimes(attributeDic)
    tm = time.struct_time(time.gmtime())
    dt = datetime.utcfromtimestamp(calendar.timegm(tm))
    attributeDic['WrittenGPS'] = astropy.time.Time(dt, format='datetime', scale='utc').gps
    #attributeDic['WrittenGPS'] = float(lal.gpstime.utc_to_gps(dt))
    attributeDic['WrittenUTC'] = float(time.time())

    return attributeDic

def irisread(event):
    """@read iris event.

    @param event
        iris event
    """

    attributeDic = {}

    attributeDic['Time'] = str(event.origins[0].time)
    timeString = attributeDic['Time'].replace("T"," ").replace("Z","")
    dt = datetime.strptime(timeString, "%Y-%m-%d %H:%M:%S.%f")
    tm = time.struct_time(dt.timetuple())

    attributeDic["Longitude"] = event.origins[0].longitude
    attributeDic["Latitude"] = event.origins[0].latitude
    if not event.origins[0].depth == None:
        attributeDic["Depth"] = event.origins[0].depth / 1000.0
    else:
        attributeDic["Depth"] = 0
    attributeDic["eventID"] = event.origins[0].region

    td = dt - datetime(1970, 1, 1)
    thistime =  td.microseconds * 1e-6 + td.seconds + td.days * 24 * 3600

    if thistime < 300000000:
        attributeDic['GPS'] = thistime
    else:
        attributeDic['GPS'] = astropy.time.Time(dt, format='datetime', scale='utc').gps
        #attributeDic['GPS'] = float(lal.gpstime.utc_to_gps(dt))

    attributeDic['UTC'] = float(dt.strftime("%s"))

    eventID = "%.0f"%attributeDic['GPS']
    eventName = ''.join(["iris",str(eventID)])

    attributeDic["eventName"] = eventName
    attributeDic["Magnitude"] = event.magnitudes[0].mag
    attributeDic["MomentMagnitude"] = (attributeDic["Magnitude"] - 9.1)/1.5
    attributeDic["DataSource"] = "IRIS"
    attributeDic["Version"] = 1.0
    attributeDic["Type"] = 1.0
    attributeDic['Region'] = event.origins[0].region

    if event.origins[0].evaluation_status == "AUTOMATIC":
        attributeDic["Review"] = "Automatic"
    else:
        attributeDic["Review"] = "Manual"

    SentTime = time.gmtime()
    dt = datetime.utcfromtimestamp(calendar.timegm(SentTime))
    attributeDic['SentGPS'] = astropy.time.Time(dt, format='datetime', scale='utc').gps
    #attributeDic['SentGPS'] = float(lal.gpstime.utc_to_gps(dt))
    attributeDic['SentUTC'] = time.time()
    attributeDic['Sent'] = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", SentTime)

    attributeDic = calculate_traveltimes(attributeDic)
    tm = time.struct_time(time.gmtime())
    dt = datetime.utcfromtimestamp(calendar.timegm(tm))

    attributeDic['WrittenGPS'] = astropy.time.Time(dt, format='datetime', scale='utc').gps
    #attributeDic['WrittenGPS'] = float(lal.gpstime.utc_to_gps(dt))
    attributeDic['WrittenUTC'] = float(time.time())

    return attributeDic

def databaseread(event):

    attributeDic = {}
    eventSplit = event.split(",")

    date = eventSplit[0]

    year = int(date[0:4])
    month = int(date[5:7])
    day = int(date[8:10])
    hour = int(date[11:13])
    minute = int(date[14:16])
    second = int(date[17:19])

    timeString = "%d-%02d-%02d %02d:%02d:%02d"%(year,month,day,hour,minute,second)
    dt = datetime.strptime(timeString, "%Y-%m-%d %H:%M:%S")

    eventID = int(eventSplit[11])
    eventName = ''.join(["db",str(eventID)])

    attributeDic["Longitude"] = float(eventSplit[2])
    attributeDic["Latitude"] = float(eventSplit[1])
    attributeDic["Depth"] = float(eventSplit[3])
    attributeDic["eventID"] = float(eventID)
    attributeDic["eventName"] = eventName
    try:
        attributeDic["Magnitude"] = float(eventSplit[4])
    except:
        attributeDic["Magnitude"] = 0
    attributeDic["MomentMagnitude"] = (attributeDic["Magnitude"] - 9.1)/1.5
    tm = time.struct_time(dt.timetuple())

    attributeDic['GPS'] = astropy.time.Time(dt, format='datetime', scale='utc').gps
    #attributeDic['GPS'] = float(lal.gpstime.utc_to_gps(tm))
    attributeDic['UTC'] = float(dt.strftime("%s"))
    attributeDic["DataSource"] = "DB"
    attributeDic["Version"] = 1.0
    attributeDic["Type"] = 1.0
    attributeDic['Region'] = "N/A"
    attributeDic["Review"] = "Manual"

    SentTime = time.gmtime()
    attributeDic['SentGPS'] = astropy.time.Time(SentTime, format='datetime', scale='utc').gps
    #attributeDic['SentGPS'] = float(lal.gpstime.utc_to_gps(SentTime))
    attributeDic['SentUTC'] = time.time()

    attributeDic['Time'] = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", tm)
    attributeDic['Sent'] = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", SentTime)

    attributeDic = calculate_traveltimes(attributeDic)
    tm = time.struct_time(time.gmtime())
    attributeDic['WrittenGPS'] = astropy.time.Time(tm, format='datetime', scale='utc').gps
    #attributeDic['WrittenGPS'] = float(lal.gpstime.utc_to_gps(tm))
    attributeDic['WrittenUTC'] = float(time.time())

    return attributeDic

def fakeeventread(params):

    attributeDic = {}

    attributeDic['GPS'] = astropy.time.Time(params["gps"], format='gps', scale='utc').gps
    timeString = astropy.time.Time(params["gps"], format='gps', scale='utc').isot

    dt = datetime.strptime(timeString, "%Y-%m-%dT%H:%M:%S.%f")
    tm = time.struct_time(dt.timetuple())
   
    attributeDic['UTC'] = float(dt.strftime("%s"))
    attributeDic['Time'] = timeString

    attributeDic["Longitude"] = params["longitude"]
    attributeDic["Latitude"] = params["latitude"]
    attributeDic["Depth"] = params["depth"]

    eventID = "%.0f"%attributeDic['GPS']
    eventName = ''.join(["fake",str(eventID)])

    attributeDic["eventName"] = eventName
    attributeDic["eventID"] = eventID

    attributeDic["Magnitude"] = params["magnitude"]
    attributeDic["MomentMagnitude"] = (attributeDic["Magnitude"] - 9.1)/1.5
    attributeDic["DataSource"] = "FAKE"
    attributeDic["Version"] = 1.0
    attributeDic["Type"] = 1.0
    attributeDic['Region'] = "FAKE"
    attributeDic["Review"] = "FAKE"

    SentTime = time.gmtime()
    dt = datetime.utcfromtimestamp(calendar.timegm(SentTime))

    attributeDic['SentGPS'] = astropy.time.Time(dt, format='datetime', scale='utc').gps
    #attributeDic['SentGPS'] = float(lal.gpstime.utc_to_gps(dt))
    attributeDic['SentUTC'] = time.time()
    attributeDic['Sent'] = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", SentTime)

    #attributeDic = calculate_traveltimes(attributeDic)
    tm = time.struct_time(time.gmtime())
    dt = datetime.utcfromtimestamp(calendar.timegm(tm))

    attributeDic['WrittenGPS'] = astropy.time.Time(dt, format='datetime', scale='utc').gps
    #attributeDic['WrittenGPS'] = float(lal.gpstime.utc_to_gps(dt))
    attributeDic['WrittenUTC'] = float(time.time())

    return attributeDic

def calculate_traveltimes_lookup(attributeDic):
    """@calculate travel times of earthquake

    @param attributeDic
        earthquake stucture
    """

    if not "traveltimes" in attributeDic:
        attributeDic["traveltimes"] = {}

    if not "Latitude" in attributeDic and not "Longitude" in attributeDic:
        return attributeDic

    attributeDic = ifotraveltimes_lookup(attributeDic, "LHO", 46.6475, -119.5986)
    attributeDic = ifotraveltimes_lookup(attributeDic, "LLO", 30.4986, -90.7483)
    attributeDic = ifotraveltimes_lookup(attributeDic, "GEO", 52.246944, 9.808333)
    attributeDic = ifotraveltimes_lookup(attributeDic, "VIRGO", 43.631389, 10.505)
    attributeDic = ifotraveltimes_lookup(attributeDic, "FortyMeter", 34.1391, -118.1238)
    attributeDic = ifotraveltimes_lookup(attributeDic, "Homestake", 44.3465, -103.7574)

    return attributeDic

def calculate_traveltimes_velocitymap(attributeDic):
    """@calculate travel times of earthquake

    @param attributeDic
        earthquake stucture
    """

    if not "traveltimes" in attributeDic:
        attributeDic["traveltimes"] = {}

    if not "Latitude" in attributeDic and not "Longitude" in attributeDic:
        return attributeDic

    attributeDic = ifotraveltimes_velocitymap(attributeDic, "LHO", 46.6475, -119.5986)
    attributeDic = ifotraveltimes_velocitymap(attributeDic, "LLO", 30.4986, -90.7483)
    attributeDic = ifotraveltimes_velocitymap(attributeDic, "GEO", 52.246944, 9.808333)
    attributeDic = ifotraveltimes_velocitymap(attributeDic, "VIRGO", 43.631389, 10.505)
    attributeDic = ifotraveltimes_velocitymap(attributeDic, "FortyMeter", 34.1391, -118.1238)
    attributeDic = ifotraveltimes_velocitymap(attributeDic, "Homestake", 44.3465, -103.7574)

    return attributeDic

def calculate_traveltimes(attributeDic): 
    """@calculate travel times of earthquake

    @param attributeDic
        earthquake stucture
    """

    if not "traveltimes" in attributeDic:
        attributeDic["traveltimes"] = {}

    if not "Latitude" in attributeDic and not "Longitude" in attributeDic:
        return attributeDic

    attributeDic = ifotraveltimes_lookup(attributeDic, "Arbitrary", 0.0, 0.0)
    #attributeDic = ifotraveltimes(attributeDic, "Arbitrary", 0.0, 0.0)
    #attributeDic = ifotraveltimes(attributeDic, "LHO", 46.6475, -119.5986)
    #attributeDic = ifotraveltimes(attributeDic, "LLO", 30.4986, -90.7483)
    #attributeDic = ifotraveltimes(attributeDic, "GEO", 52.246944, 9.808333)
    #attributeDic = ifotraveltimes(attributeDic, "VIRGO", 43.631389, 10.505)
    #attributeDic = ifotraveltimes(attributeDic, "FortyMeter", 34.1391, -118.1238)
    #attributeDic = ifotraveltimes(attributeDic, "Homestake", 44.3465, -103.7574)

    return attributeDic

def eqmon_loc(attributeDic,ifo):
    """@calculate travel times of earthquake

    @param attributeDic
        earthquake stucture
    """

    if not "traveltimes" in attributeDic:
        attributeDic["traveltimes"] = {}

    if not "Latitude" in attributeDic and not "Longitude" in attributeDic:
        return attributeDic

    #attributeDic = ifotraveltimes(attributeDic, "Arbitrary", 0.0, 0.0)

    if ifo == "LHO":
        attributeDic = ifotraveltimes_loc(attributeDic, "LHO", 46.6475, -119.5986)
    elif ifo == "LLO":
        attributeDic = ifotraveltimes_loc(attributeDic, "LLO", 30.4986, -90.7483)
    elif ifo == "GEO":
        attributeDic = ifotraveltimes_loc(attributeDic, "GEO", 52.246944, 9.808333)
    elif ifo == "VIRGO":
        attributeDic = ifotraveltimes_loc(attributeDic, "VIRGO", 43.631389, 10.505)
    elif ifo == "FortyMeter":
        attributeDic = ifotraveltimes_loc(attributeDic, "FortyMeter", 34.1391, -118.1238)
    elif ifo == "Homestake":
        attributeDic = ifotraveltimes_loc(attributeDic, "Homestake", 44.3465, -103.7574)
    elif ifo == "LSST":
        attributeDic = ifotraveltimes_loc(attributeDic, "LSST", -30.2446, -70.7494)
    elif ifo == "MIT":
        attributeDic = ifotraveltimes_loc(attributeDic, "MIT", 42.3598, -71.0921)
    else:
        ifoSplit = ifo.split(",")
        site = ifoSplit[0]
        latitude = float(ifoSplit[1])
        longitude = float(ifoSplit[2])
        attributeDic = ifotraveltimes_loc(attributeDic, site, latitude, longitude)

    return attributeDic

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

def ampRf(M,r,h,Rf0,Rfs,cd,rs):
    #def ampRf(M,r,h,Rf0,Rfs,Q0,Qs,cd,ch,rs):
    # Rf amplitude estimate
    # M = magnitude
    # r = distance [km]
    # h = depth [km]

    # Rf0 = Rf amplitude parameter
    # Rfs = exponent of power law for f-dependent Rf amplitude
    # Q0 = Q-value of Earth for Rf waves at 1Hz
    # Qs = exponent of power law for f-dependent Q
    # cd = speed parameter for surface coupling  [km/s]
    # ch = speed parameter for horizontal propagation  [km/s]
    # rs

    # exp(-2*pi*h.*fc./cd), coupling of source to surface waves
    # exp(-2*pi*r.*fc./ch./Q), dissipation

    fc = 10**(2.3-M/2)
    #Q = Q0/(fc**Qs)
    Af = Rf0/(fc**Rfs)

    #Rf = 1e-3 * M*Af*np.exp(-2*np.pi*h*fc/cd)*np.exp(-2*np.pi*r*(fc/ch)/Q)/(r**(rs))
    Rf = 1e-3 * M*Af*np.exp(-2*np.pi*h*fc/cd)/(r**(rs))

    return Rf

def ifotraveltimes_lookup(attributeDic,ifo,ifolat,ifolon):
    """@calculate travel times of earthquake

    @param attributeDic
        earthquake stucture
    @param ifo
        ifo name
    @param ifolat
        ifo latitude
    @param ifolon
        ifo longitude
    """

    try:
        from obspy.geodetics.base import gps2dist_azimuth
        from obspy.taup import TauPyModel
    except:
        print "Enable ObsPy if traveltimes information desired...\n"
        return attributeDic

    seismonpath = os.path.dirname(seismon.__file__)
    scriptpath = os.path.join(seismonpath,'input')

    if ifo == "LLO":
        gpfile = os.path.join(scriptpath,'gp_llo.pickle')
    elif ifo == "Virgo":
        gpfile = os.path.join(scriptpath,'gp_virgo.pickle')
    else:
        gpfile = os.path.join(scriptpath,'gp_lho.pickle')

    with open(gpfile, 'rb') as fid:
        scaler,gp = pickle.load(fid)

    if ifo == "Arbitrary":
        degrees = np.linspace(1,180,180)
        distances = degrees*(np.pi/180)*6370000
        fwd = 0
        back = 0

        M = attributeDic["Magnitude"]*np.ones(distances.shape)
        lat = attributeDic["Latitude"]*np.ones(distances.shape)
        lon = attributeDic["Longitude"]*np.ones(distances.shape)
        h = attributeDic["Depth"]*np.ones(distances.shape)
        az = fwd*np.ones(distances.shape)

        X = np.vstack((M,lat,lon,distances/1000.0,h,az)).T
        X = scaler.transform(X)

        pred, pred_std = gp.predict(X, return_std=True)
        pred = 10**pred
        pred_std = pred*np.log(10)*pred_std

        Rfamp = pred

    else:
        distance,fwd,back = gps2dist_azimuth(attributeDic["Latitude"],attributeDic["Longitude"],ifolat,ifolon)
        distances = np.linspace(0,distance,100)
        degrees = (distances/6370000)*(180/np.pi)

        M = attributeDic["Magnitude"]*np.ones(distances.shape)
        lat = attributeDic["Latitude"]*np.ones(distances.shape)
        lon = attributeDic["Longitude"]*np.ones(distances.shape)
        h = attributeDic["Depth"]*np.ones(distances.shape)
        az = fwd*np.ones(distances.shape)

        X = np.vstack((M,lat,lon,distances[-1]/1000.0,h,az)).T
        X = scaler.transform(X)

        pred, pred_std = gp.predict(X, return_std=True)
        pred = 10**pred
        pred_std = pred*np.log(10)*pred_std
        
        Rfamp = pred[0]

    Pamp = 1e-6
    Samp = 1e-5

    lats = []
    lons = []
    Ptimes = []
    Stimes = []
    Rtwotimes = []
    RthreePointFivetimes = []
    Rfivetimes = []
    Rfamps = []

    if attributeDic["Depth"] >= 2.0:
        depth = attributeDic["Depth"]
    else:
        depth = 2.0

    pfile = os.path.join(scriptpath,'p.dat')
    sfile = os.path.join(scriptpath,'s.dat')
    parrivals = np.loadtxt(pfile)
    sarrivals = np.loadtxt(sfile)
    depths = np.linspace(1,100,100)
    index = np.argmin(np.abs(depths-depth))
    parrivals = parrivals[:,index]
    sarrivals = sarrivals[:,index]

    for distance, degree, parrival, sarrival in zip(distances, degrees,parrivals,sarrivals):

        lon, lat, baz = shoot(attributeDic["Longitude"], attributeDic["Latitude"], fwd, distance/1000)
        lats.append(lat)
        lons.append(lon)

        Ptime = attributeDic["GPS"]+parrival
        Stime = attributeDic["GPS"]+sarrival
        Rtwotime = attributeDic["GPS"]+distance/2000.0
        RthreePointFivetime = attributeDic["GPS"]+distance/3500.0
        Rfivetime = attributeDic["GPS"]+distance/5000.0

        Ptimes.append(Ptime)
        Stimes.append(Stime)
        Rtwotimes.append(Rtwotime)
        RthreePointFivetimes.append(RthreePointFivetime)
        Rfivetimes.append(Rfivetime)

    traveltimes = {}
    traveltimes["Latitudes"] = lats
    traveltimes["Longitudes"] = lons
    traveltimes["Distances"] = distances
    traveltimes["Degrees"] = degrees
    traveltimes["Ptimes"] = Ptimes
    traveltimes["Stimes"] = Stimes
    #traveltimes["Rtimes"] = Rtimes
    traveltimes["Rtwotimes"] = Rtwotimes
    traveltimes["RthreePointFivetimes"] = RthreePointFivetimes
    traveltimes["Rfivetimes"] = Rfivetimes

    if ifo == "Arbitrary":
        traveltimes["Rfamp"] = Rfamp
    else:
        traveltimes["Rfamp"] = [Rfamp]
    traveltimes["Pamp"] = [Pamp]
    traveltimes["Samp"] = [Samp]

    attributeDic["traveltimes"][ifo] = traveltimes

    return attributeDic

def ifotraveltimes_velocitymap(attributeDic,ifo,ifolat,ifolon):
    """@calculate travel times of earthquake

    @param attributeDic
        earthquake stucture
    @param ifo
        ifo name
    @param ifolat
        ifo latitude
    @param ifolon
        ifo longitude
    """

    try:
        from obspy.core.util.geodetics import gps2DistAzimuth
    except:
        print "Enable ObsPy if traveltimes information desired...\n"
        return attributeDic

    distance,fwd,back = gps2DistAzimuth(attributeDic["Latitude"],attributeDic["Longitude"],ifolat,ifolon)
    distances = np.linspace(0,distance,1000)
    degrees = (distances/6370000)*(180/np.pi)

    distance_delta = distances[1] - distances[0]

    periods = [25.0,27.0,30.0,32.0,35.0,40.0,45.0,50.0,60.0,75.0,100.0,125.0,150.0,200.0,250.0]
    frequencies = 1 / np.array(periods)
   
    fc = 10**(2.3-attributeDic["Magnitude"]/2)
    index = np.argmin(np.absolute(frequencies - fc))

    lats = []
    lons = []
    Rvelocitytimes = []
    velocities = []

    velocityFile = '/home/mcoughlin/Seismon/velocity_maps/GR025_1_GDM52.pix'
    velocity_map = np.loadtxt(velocityFile)
    base_velocity = 3.59738 

    for distance, degree in zip(distances, degrees):

        lon, lat, baz = shoot(attributeDic["Longitude"], attributeDic["Latitude"], fwd, distance/1000)
        lats.append(lat)
        lons.append(lon)

    combined_x_y_arrays = np.dstack([velocity_map[:,0],velocity_map[:,1]])[0]
    points_list = np.dstack([lats, lons])

    indexes = do_kdtree(combined_x_y_arrays,points_list)[0]

    time = attributeDic["GPS"]

    for distance, degree, index in zip(distances, degrees,indexes):

        velocity = 1000 * (1 + 0.01*velocity_map[index,3])*base_velocity

        time_delta = distance_delta / velocity
        time = time + time_delta

        Rvelocitytimes.append(time)
        velocities.append(velocity/1000)

    attributeDic["traveltimes"][ifo]["Rvelocitymaptimes"] = Rvelocitytimes
    attributeDic["traveltimes"][ifo]["Rvelocitymapvelocities"] = velocities

    return attributeDic

def ifotraveltimes(attributeDic,ifo,ifolat,ifolon):
    """@calculate travel times of earthquake

    @param attributeDic
        earthquake stucture
    @param ifo
        ifo name
    @param ifolat
        ifo latitude
    @param ifolon
        ifo longitude
    """

    try:
        from obspy.core.util.geodetics import gps2DistAzimuth
        from obspy.taup import TauPyModel
    except:
        print "Enable ObsPy if traveltimes information desired...\n"
        return attributeDic

    if ifo == "LLO":
        gpfile = os.path.join(scriptpath,'gp_llo.pickle')
    elif ifo == "Virgo":
        gpfile = os.path.join(scriptpath,'gp_virgo.pickle')
    else:
        gpfile = os.path.join(scriptpath,'gp_lho.pickle')

    with open(gpfile, 'rb') as fid:
        scaler,gp = pickle.load(fid)

    if ifo == "Arbitrary":
        degrees = np.linspace(1,180,180)
        distances = degrees*(np.pi/180)*6370000
        fwd = 0
        back = 0

        M = attributeDic["Magnitude"]*np.ones(distances.shape)
        lat = attributeDic["Latitude"]*np.ones(distances.shape)
        lon = attributeDic["Longitude"]*np.ones(distances.shape) 
        h = attributeDic["Depth"]*np.ones(distances.shape)
        az = fwd*np.ones(distances.shape)

        X = np.vstack((M,lat,lon,distances/1000.0,h,az)).T
        X = scaler.transform(X)

        pred, pred_std = gp.predict(X, return_std=True)
        pred = 10**pred
        pred_std = pred*np.log(10)*pred_std

        Rfamp = pred

    else:
        distance,fwd,back = gps2DistAzimuth(attributeDic["Latitude"],attributeDic["Longitude"],ifolat,ifolon)
        distances = np.linspace(0,distance,100)
        degrees = (distances/6370000)*(180/np.pi)

        M = attributeDic["Magnitude"]*np.ones(distances.shape)
        lat = attributeDic["Latitude"]*np.ones(distances.shape)
        lon = attributeDic["Longitude"]*np.ones(distances.shape)
        h = attributeDic["Depth"]*np.ones(distances.shape)
        az = fwd*np.ones(distances.shape)

        X = np.vstack((attributeDic["Magnitude"],attributeDic["Latitude"],attributeDic["Longitude"],distances[-1]/1000.0,attributeDic["Depth"],az)).T
        X = scaler.transform(X)

        pred, pred_std = gp.predict(X, return_std=True)
        pred = 10**pred
        pred_std = pred*np.log(10)*pred_std

        Rfamp = pred[0]

    Pamp = 1e-6
    Samp = 1e-5

    lats = []
    lons = []
    Ptimes = []
    Stimes = []
    #Rtimes = []
    Rtwotimes = []
    RthreePointFivetimes = []
    Rfivetimes = []
    Rfamps = []

    # Pmag = T * 10^(Mb - 5.9 - 0.01*dist)
    # Rmag = T * 10^(Ms - 3.3 - 1.66*log_10(dist))
    T = 20

    model = TauPyModel(model="iasp91")

    for distance, degree in zip(distances, degrees):

        lon, lat, baz = shoot(attributeDic["Longitude"], attributeDic["Latitude"], fwd, distance/1000)
        lats.append(lat)
        lons.append(lon)

        if attributeDic["Depth"] >= 2.0:
            depth = attributeDic["Depth"]
        else:
            depth = 2.0

        #tt = getTravelTimes(delta=degree, depth=depth)

        #arrivals = model.get_travel_times(source_depth_in_km=depth,
        #                          distance_in_degree=degree,phase_list=('P','S'))
        arrivals = model.get_travel_times(source_depth_in_km=depth,distance_in_degree=degree)
        #tt.append({'phase_name': 'R', 'dT/dD': 0, 'take-off angle': 0, 'time': distance/3500, 'd2T/dD2': 0, 'dT/dh': 0})
        #tt.append({'phase_name': 'Rtwo', 'dT/dD': 0, 'take-off angle': 0, 'time': distance/2000, 'd2T/dD2': 0, 'dT/dh': 0})
        #tt.append({'phase_name': 'RthreePointFive', 'dT/dD': 0, 'take-off angle': 0, 'time': distance/3500, 'd2T/dD2': 0, 'dT/dh': 0})
        #tt.append({'phase_name': 'Rfive', 'dT/dD': 0, 'take-off angle': 0, 'time': distance/5000, 'd2T/dD2': 0, 'dT/dh': 0})

        Ptime = -1
        Stime = -1
        Rtime = -1
        for phase in arrivals:
            if Ptime == -1 and phase.name.lower()[0] == "p":
                Ptime = attributeDic["GPS"]+phase.time
            if Stime == -1 and phase.name.lower()[0] == "s":
                Stime = attributeDic["GPS"]+phase.time
        Rtwotime = attributeDic["GPS"]+distance/2000.0
        RthreePointFivetime = attributeDic["GPS"]+distance/3500.0
        Rfivetime = attributeDic["GPS"]+distance/5000.0

        Ptimes.append(Ptime)
        Stimes.append(Stime)
        #Rtimes.append(Rtime)
        Rtwotimes.append(Rtwotime)
        RthreePointFivetimes.append(RthreePointFivetime)
        Rfivetimes.append(Rfivetime)


    traveltimes = {}
    traveltimes["Latitudes"] = lats
    traveltimes["Longitudes"] = lons
    traveltimes["Distances"] = distances
    traveltimes["Degrees"] = degrees
    traveltimes["Azimuth"] = [fwd]
    traveltimes["Ptimes"] = Ptimes
    traveltimes["Stimes"] = Stimes
    #traveltimes["Rtimes"] = Rtimes
    traveltimes["Rtwotimes"] = Rtwotimes
    traveltimes["RthreePointFivetimes"] = RthreePointFivetimes
    traveltimes["Rfivetimes"] = Rfivetimes

    if ifo == "Arbitrary":
        traveltimes["Rfamp"] = Rfamp
    else:
        traveltimes["Rfamp"] = [Rfamp]
    traveltimes["Pamp"] = [Pamp]
    traveltimes["Samp"] = [Samp]

    attributeDic["traveltimes"][ifo] = traveltimes

    return attributeDic

def distance_latlon(lat1,lon1,lat2,lon2):
    R = 6373.0

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    lon1 = math.radians(lon1)
    lon2 = math.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (math.sin(dlat/2))**2 + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon/2))**2
    c = 2 * math.atan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c

    return distance

def ifotraveltimes_loc(attributeDic,ifo,ifolat,ifolon):
    """@calculate travel times of earthquake

    @param attributeDic
        earthquake stucture
    @param ifo
        ifo name
    @param ifolat
        ifo latitude
    @param ifolon
        ifo longitude
    """

    try:
        from obspy.geodetics.base import gps2dist_azimuth
    except:
        print "Enable ObsPy if traveltimes information desired...\n"
        return attributeDic

    if not "traveltimes" in attributeDic:
        print "This analysis missing traveltimes... returning.\n"
        return attributeDic

    seismonpath = os.path.dirname(seismon.__file__)
    scriptpath = os.path.join(seismonpath,'input')

    #print attributeDic["Latitude"],attributeDic["Longitude"],ifolat,ifolon
    #if (np.absolute(attributeDic["Latitude"]-ifolat)**2 + np.absolute(attributeDic["Longitude"]-ifolon)**2) < 5:
    #    distance = distance_latlon(attributeDic["Latitude"],attributeDic["Longitude"],ifolat,ifolon) 
    #else:       
    distance,fwd,back = gps2dist_azimuth(attributeDic["Latitude"],attributeDic["Longitude"],ifolat,ifolon)
    degree = (distance/6370000)*(180/np.pi)

    ptime_interp = np.interp(distance, attributeDic["traveltimes"]["Arbitrary"]["Distances"],\
        attributeDic["traveltimes"]["Arbitrary"]["Ptimes"])
    stime_interp = np.interp(distance, attributeDic["traveltimes"]["Arbitrary"]["Distances"],\
        attributeDic["traveltimes"]["Arbitrary"]["Stimes"])
    rtwotime_interp = np.interp(distance, attributeDic["traveltimes"]["Arbitrary"]["Distances"],\
        attributeDic["traveltimes"]["Arbitrary"]["Rtwotimes"])
    rthreePointFivetime_interp = np.interp(distance, attributeDic["traveltimes"]["Arbitrary"]["Distances"],\
        attributeDic["traveltimes"]["Arbitrary"]["RthreePointFivetimes"])
    rfivetime_interp = np.interp(distance, attributeDic["traveltimes"]["Arbitrary"]["Distances"],\
        attributeDic["traveltimes"]["Arbitrary"]["Rfivetimes"])
    rfamp_interp = np.interp(distance, attributeDic["traveltimes"]["Arbitrary"]["Distances"],\
        attributeDic["traveltimes"]["Arbitrary"]["Rfamp"])

    if ifo == "LLO":
        gpfile = os.path.join(scriptpath,'gp_llo.pickle')
    elif ifo == "Virgo":
        gpfile = os.path.join(scriptpath,'gp_virgo.pickle')
    else:
        gpfile = os.path.join(scriptpath,'gp_lho.pickle')

    with open(gpfile, 'rb') as fid:
        scaler,gp = pickle.load(fid)

    X = np.vstack((attributeDic["Magnitude"],attributeDic["Latitude"],attributeDic["Longitude"],distance/1000.0,attributeDic["Depth"],fwd)).T
    X = scaler.transform(X)

    pred, pred_std = gp.predict(X, return_std=True)
    pred = 10**pred
    pred_std = pred*np.log(10)*pred_std

    Rfamp = pred[0]

    traveltimes = {}
    traveltimes["Latitudes"] = ifolat
    traveltimes["Longitudes"] = ifolon
    traveltimes["Distances"] = [distance]
    traveltimes["Degrees"] = [degree]
    traveltimes["Azimuth"] = [fwd]
    traveltimes["Ptimes"] = [ptime_interp]
    traveltimes["Stimes"] = [stime_interp]
    #traveltimes["Rtimes"] = Rtimes
    traveltimes["Rtwotimes"] = [rtwotime_interp]
    traveltimes["RthreePointFivetimes"] = [rthreePointFivetime_interp]
    traveltimes["Rfivetimes"] = [rfivetime_interp]
    #traveltimes["Rfamp"] = [rfamp_interp]
    traveltimes["Rfamp"] = [Rfamp]
    traveltimes["Pamp"] = [attributeDic["traveltimes"]["Arbitrary"]["Pamp"][0]]
    traveltimes["Samp"] = [attributeDic["traveltimes"]["Arbitrary"]["Samp"][0]]

    attributeDic["traveltimes"][ifo] = traveltimes

    return attributeDic

def eventDiff(attributeDics, magnitudeDiff, latitudeDiff, longitudeDiff):
    """@calculate difference between two events

    @param attributeDics
        list of earthquake stuctures
    @param magnitudeDiff
        difference in magnitudes
    @param latitudeDiff
        difference in latitudes
    @param longitudeDiff
        difference in longitudes
    """

    if len(attributeDics) > 1:
        for i in xrange(len(attributeDics)-1):
            if "Magnitude" in attributeDics[i] and "Magnitude" in attributeDics[i+1] and \
                "Latitude" in attributeDics[i] and "Latitude" in attributeDics[i+1] and\
                "Longitude" in attributeDics[i] and "Longitude" in attributeDics[i+1]:

                magnitudeDiff.append(attributeDics[i]["Magnitude"]-attributeDics[i+1]["Magnitude"])
                latitudeDiff.append(attributeDics[i]["Latitude"]-attributeDics[i+1]["Latitude"])
                longitudeDiff.append(attributeDics[i]["Longitude"]-attributeDics[i+1]["Longitude"])
    return magnitudeDiff, latitudeDiff, longitudeDiff

def great_circle_distance(latlong_a, latlong_b):
    """@calculate distance between two points

    @param latlong_a
        first point
    @param latlong_b
        second point
    """

    EARTH_CIRCUMFERENCE = 6378.137 # earth circumference in kilometers

    lat1, lon1 = latlong_a
    lat2, lon2 = latlong_b

    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat / 2) * math.sin(dLat / 2) +
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
            math.sin(dLon / 2) * math.sin(dLon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = EARTH_CIRCUMFERENCE * c
    
    return d

def retrieve_earthquakes(params,gpsStart,gpsEnd):
    """@retrieve earthquakes information.

    @param params
        seismon params dictionary
    """

    attributeDics = []
    eventfilesTypes = params["eventfilesType"].split(",")
    for eventfilesType in eventfilesTypes:

        eventfilesLocation = os.path.join(params["eventfilesLocation"],eventfilesType)
        files = glob.glob(os.path.join(eventfilesLocation,"*.xml"))

        for numFile in xrange(len(files)):

            file = files[numFile]

            fileSplit = file.replace(".xml","").split("-")
            gps = float(fileSplit[-1])
            if (gps < gpsStart - 3600) or (gps > gpsEnd):
                continue

            attributeDic = read_eqmon(params,file)

            if attributeDic["Magnitude"] >= params["earthquakesMinMag"]:
                attributeDics.append(attributeDic)

    return attributeDics

def equi(m, centerlon, centerlat, radius):
    """@calculate circle around specified point

    @param m
        basemap projection
    @param centerlon
        longitude of center
    @param centerlat
        latitude of center
    @param radius
        radius of circle
    """

    glon1 = centerlon
    glat1 = centerlat
    X = []
    Y = []
    for azimuth in range(0, 360):
        glon2, glat2, baz = shoot(glon1, glat1, azimuth, radius)
        X.append(glon2)
        Y.append(glat2)
    X.append(X[0])
    Y.append(Y[0])

    #m.plot(X,Y,**kwargs) #Should work, but doesn't...
    X,Y = m(X,Y)
    return X,Y

def shoot(lon, lat, azimuth, maxdist=None):
    """@Shooter Function
    Original javascript on http://williams.best.vwh.net/gccalc.htm
    Translated to python by Thomas Lecocq

    @param lon
        longitude
    @param lat
        latitude
    @param azimuth
        azimuth
    @param maxdist
        maximum distance

    """
    glat1 = lat * np.pi / 180.
    glon1 = lon * np.pi / 180.
    s = maxdist / 1.852
    faz = azimuth * np.pi / 180.

    EPS= 0.00000000005
    if ((np.abs(np.cos(glat1))<EPS) and not (np.abs(np.sin(faz))<EPS)):
        alert("Only N-S courses are meaningful, starting at a pole!")

    a=6378.13/1.852
    f=1/298.257223563
    r = 1 - f
    tu = r * np.tan(glat1)
    sf = np.sin(faz)
    cf = np.cos(faz)
    if (cf==0):
        b=0.
    else:
        b=2. * np.arctan2 (tu, cf)

    cu = 1. / np.sqrt(1 + tu * tu)
    su = tu * cu
    sa = cu * sf
    c2a = 1 - sa * sa
    x = 1. + np.sqrt(1. + c2a * (1. / (r * r) - 1.))
    x = (x - 2.) / x
    c = 1. - x
    c = (x * x / 4. + 1.) / c
    d = (0.375 * x * x - 1.) * x
    tu = s / (r * a * c)
    y = tu
    c = y + 1
    while (np.abs (y - c) > EPS):

        sy = np.sin(y)
        cy = np.cos(y)
        cz = np.cos(b + y)
        e = 2. * cz * cz - 1.
        c = y
        x = e * cy
        y = e + e - 1.
        y = (((sy * sy * 4. - 3.) * y * cz * d / 6. + x) *
              d / 4. - cz) * sy * d + tu

    b = cu * cy * cf - su * sy
    c = r * np.sqrt(sa * sa + b * b)
    d = su * cy + cu * sy * cf
    glat2 = (np.arctan2(d, c) + np.pi) % (2*np.pi) - np.pi
    c = cu * cy - su * sy * cf
    x = np.arctan2(sy * sf, c)
    c = ((-3. * c2a + 4.) * f + 4.) * c2a * f / 16.
    d = ((e * cy * c + cz) * sy * c + y) * sa
    glon2 = ((glon1 + x - (1. - c) * d * f + np.pi) % (2*np.pi)) - np.pi

    baz = (np.arctan2(sa, b) + np.pi) % (2 * np.pi)

    glon2 *= 180./np.pi
    glat2 *= 180./np.pi
    baz *= 180./np.pi

    return (glon2, glat2, baz)

