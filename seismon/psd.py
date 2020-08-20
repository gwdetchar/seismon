#!/usr/bin/python

import os, glob, optparse, shutil, warnings, pickle, math, copy, pickle, matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal, scipy.stats
from scipy import optimize
import seismon.NLNM, seismon.html
import seismon.eqmon, seismon.utils
from matplotlib import cm

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

def save_data(params,channel,gpsStart,gpsEnd,data,attributeDics):
    """@saves spectral data in text files.

    @param params
        seismon params dictionary
    @param channel
        seismon channel structure
    @param gpsStart
        start gps
    @param gpsStart
        start gps
    @param gpsEnd
        end gps
    @param data
        spectral data structure 
    """

    ifo = seismon.utils.getIfo(params)

    psdDirectory = params["dirPath"] + "/Text_Files/PSD/" + channel.station_underscore + "/" + str(params["fftDuration"])
    seismon.utils.mkdir(psdDirectory)

    powerlawDirectory = params["dirPath"] + "/Text_Files/Powerlaw/" + channel.station_underscore + "/" + str(params["fftDuration"])
    seismon.utils.mkdir(powerlawDirectory)

    EQpowerlawDirectory = params["dirPath"] + "/Text_Files/EQPowerlaw/" + channel.station_underscore + "/" + str(params["fftDuration"])
    seismon.utils.mkdir(EQpowerlawDirectory)

    fftDirectory = params["dirPath"] + "/Text_Files/FFT/" + channel.station_underscore + "/" + str(params["fftDuration"])
    seismon.utils.mkdir(fftDirectory)

    timeseriesDirectory = params["dirPath"] + "/Text_Files/Timeseries/" + channel.station_underscore + "/" + str(params["fftDuration"])
    seismon.utils.mkdir(timeseriesDirectory)

    accelerationDirectory = params["dirPath"] + "/Text_Files/Acceleration/" + channel.station_underscore + "/" + str(params["fftDuration"])
    seismon.utils.mkdir(accelerationDirectory)

    displacementDirectory = params["dirPath"] + "/Text_Files/Displacement/" + channel.station_underscore + "/" + str(params["fftDuration"])
    seismon.utils.mkdir(displacementDirectory)

    earthquakesDirectory = params["dirPath"] + "/Text_Files/Earthquakes/" + channel.station_underscore + "/" + str(params["fftDuration"])
    seismon.utils.mkdir(earthquakesDirectory)

    spectrogramDirectory = params["dirPath"] + "/Text_Files/Spectrogram/" + channel.station_underscore + "/" + str(params["fftDuration"])
    seismon.utils.mkdir(spectrogramDirectory)

    freq = np.array(data["dataASD"].frequencies)

    psdFile = os.path.join(psdDirectory,"%d-%d.txt"%(gpsStart,gpsEnd))
    f = open(psdFile,"wb")
    for i in range(len(freq)):
        f.write("%e %e\n"%(freq[i],data["dataASD"][i].value))
    f.close()

    freq = np.array(data["dataFFT"].frequencies)

    fftFile = os.path.join(fftDirectory,"%d-%d.txt"%(gpsStart,gpsEnd))
    f = open(fftFile,"wb")
    for i in range(len(freq)):
        f.write("%e %e %e\n"%(freq[i],data["dataFFT"][i].value.real,data["dataFFT"][i].value.imag))
    f.close()

    tt = np.array(data["dataLowpass"].times)
    timeseriesFile = os.path.join(timeseriesDirectory,"%d-%d.txt"%(gpsStart,gpsEnd))
    f = open(timeseriesFile,"wb")
    f.write("%.10f %e\n"%(tt[np.argmin(data["dataLowpass"].value)],np.min(data["dataLowpass"].value)))
    f.write("%.10f %e\n"%(tt[np.argmax(data["dataLowpass"].value)],np.max(data["dataLowpass"].value)))
    f.close()

    ttacc = np.array(data["dataLowpassAcc"].times)
    accelerationFile = os.path.join(accelerationDirectory,"%d-%d.txt"%(gpsStart,gpsEnd))
    f = open(accelerationFile,"wb")
    f.write("%.10f %e\n"%(ttacc[np.argmin(data["dataLowpassAcc"].value)],np.min(data["dataLowpassAcc"].value)))
    f.write("%.10f %e\n"%(ttacc[np.argmax(data["dataLowpassAcc"].value)],np.max(data["dataLowpassAcc"].value)))
    f.close()

    ttdisp = np.array(data["dataLowpassDisp"].times)
    displacementFile = os.path.join(displacementDirectory,"%d-%d.txt"%(gpsStart,gpsEnd))
    f = open(displacementFile,"wb")    
    f.write("%.10f %e\n"%(ttdisp[np.argmin(data["dataLowpassDisp"].value)],np.min(data["dataLowpassDisp"].value)))
    f.write("%.10f %e\n"%(ttdisp[np.argmax(data["dataLowpassDisp"].value)],np.max(data["dataLowpassDisp"].value)))
    f.close()

    specgram = data["dataSpecgram"]
    freq = np.array(specgram.frequencies)
    times = np.array(specgram.times)

    spectrogramFile = os.path.join(spectrogramDirectory,"%d-%d.txt"%(gpsStart,gpsEnd))
    f = open(spectrogramFile,"wb")
    f.write("-1")
    for jj in range(len(times)):
        f.write(" %.10f "%times[jj])
    f.write("\n")
    for ii in range(len(freq)):
        f.write("%.10f"%freq[ii])
        for jj in range(len(times)):
                f.write(" %e "%(specgram[jj,ii].value))
        f.write("\n")
    f.close()

    if params["doPowerLawFit"]:
        xdata = freq
        ydata = np.array(data["dataASD"])

        indexes = np.where((xdata >= 0.05) & (xdata<=1))[0]
        xdata = xdata[indexes]
        ydata = ydata[indexes]

        logx = np.log10(xdata)
        logy = np.log10(ydata)

        fitfunc = lambda p, x: p[0] + p[1] * x
        errfunc = lambda p, x, y: (y - fitfunc(p, x))

        out,success = optimize.leastsq(errfunc, [1,-1],args=(logx,logy),maxfev=3000)

        index = out[1]
        amp = 10.0**out[0]

        powerlawFile = os.path.join(powerlawDirectory,"%d-%d.txt"%(gpsStart,gpsEnd))
        f = open(powerlawFile,"wb")
        f.write("%e %e\n"%(index,amp))
        f.close()


    if params["doEarthquakesTrips"]:

        tripsDirectory = params["dirPath"] + "/Text_Files/Trips/" + channel.station_underscore + "/" + str(params["fftDuration"])
        seismon.utils.mkdir(tripsDirectory)

        trips = []
        platforms = []
        stages = []
        trip_lines = [line.strip() for line in open(params["tripsTextFile"])]
        for line in trip_lines:
            lineSplit = line.split(" ")
            trips.append(float(lineSplit[0]))
            platforms.append(lineSplit[1])
            stages.append(lineSplit[2])

        count = 1
        for trip,platform,stage in zip(trips,platforms,stages):
            if (trip >= gpsStart) and (trip <= gpsEnd):
                    tt_diff = np.absolute(tt - trip)
                    index = np.argmin(tt_diff)

                    tripsDirectory = params["dirPath"] + "/Text_Files/Trips/" + channel.station_underscore + "/" + str(params["fftDuration"]) + "/" + platform + "/" + stage
                    seismon.utils.mkdir(tripsDirectory)

                    tripsFile = os.path.join(tripsDirectory,"%d.txt"%(trip))
                    f = open(tripsFile,"wb")
                    f.write("%d %e\n"%(tt[index],data["dataLowpass"].value[index]))
                    f.close()

    for attributeDic in attributeDics:

        if not "Arbitrary" in attributeDic["traveltimes"]:
            continue  

        if params["ifo"] == "IRIS":
            attributeDic = seismon.eqmon.ifotraveltimes_loc(attributeDic, "IRIS", channel.latitude, channel.longitude)
            traveltimes = attributeDic["traveltimes"]["IRIS"]
        else:
            traveltimes = attributeDic["traveltimes"][ifo]

        Ptime = max(traveltimes["Ptimes"])
        Stime = max(traveltimes["Stimes"])
        Rtwotime = max(traveltimes["Rtwotimes"])
        RthreePointFivetime = max(traveltimes["RthreePointFivetimes"])
        Rfivetime = max(traveltimes["Rfivetimes"])
        distance = max(traveltimes["Distances"])

        indexes = np.intersect1d(np.where(tt >= Rfivetime)[0],np.where(tt <= Rtwotime)[0])

        if len(indexes) == 0:
            continue

        indexMin = np.min(indexes)
        indexMax = np.max(indexes)
        ttCut = tt[indexes]
        dataCut = data["dataLowpass"][indexMin:indexMax]

        ampMax = np.max(dataCut.value)
        ttMax = ttCut[np.argmax(dataCut.value)]
        ttDiff = ttMax - attributeDic["GPS"] 
        velocity = distance / ttDiff
        velocity = velocity / 1000.0
 
        earthquakesFile = os.path.join(earthquakesDirectory,"%s.txt"%(attributeDic["eventName"]))
        f = open(earthquakesFile,"wb")
        f.write("%.10f %e %e %e %e %.1f %.1f %.1f\n"%(ttMax,ttDiff,distance,velocity,ampMax,Ptime,Stime,RthreePointFivetime))
        f.close()

        if params["doPowerLawFit"]:
            xdata = freq
            ydata = np.array(data["dataASD"])

            indexes = np.where((xdata >= 0.05) & (xdata<=1))[0]
            xdata = xdata[indexes]
            ydata = ydata[indexes]

            logx = np.log10(xdata)
            logy = np.log10(ydata)

            fitfunc = lambda p, x: p[0] + p[1] * x
            errfunc = lambda p, x, y: (y - fitfunc(p, x))

            out,success = optimize.leastsq(errfunc, [1,-1],args=(logx,logy),maxfev=3000)

            index = out[1]
            amp = 10.0**out[0]

            powerlawFile = os.path.join(EQpowerlawDirectory,"%s.txt"%(attributeDic["eventName"]))
            f = open(powerlawFile,"wb")
            f.write("%e %e\n"%(index,amp))
            f.close()

def calculate_spectra(params,channel,dataFull):
    """@calculate spectral data

    @param params
        seismon params dictionary
    @param channel
        seismon channel structure
    @param dataFull
        timeseries data structure
    """

    fs = channel.samplef # 1 ns -> 1 GHz

    if channel.station == "H1:ISI-GND_BRS_ETMX_RY_OUT_DQ":
        cutoff_high = 0.05 # 10 MHz
        cutoff_low = 0.3
    else:
        cutoff_high = 0.5 # 10 MHz
        cutoff_low = 0.1
        cutoff_band = 0.01
    n = 3
    worN = 16384
    B_low, A_low = scipy.signal.butter(n, cutoff_low / (fs / 2.0), btype='lowpass')
    #B_low, A_low = scipy.signal.butter(n, [cutoff_band/(fs / 2.0), cutoff_low/(fs / 2.0)], btype='band')
    #w_low, h_low = scipy.signal.freqz(B_low,A_low)
    w_low, h_low = scipy.signal.freqz(B_low,A_low,worN=worN)
    B_high, A_high = scipy.signal.butter(n, cutoff_high / (fs / 2.0), btype='highpass') 
    w_high, h_high = scipy.signal.freqz(B_high,A_high,worN=worN)

    w = w_high * (fs / (2.0*np.pi))

    B_band, A_band = scipy.signal.butter(n, cutoff_band / (fs / 2.0), btype='high')

    if params["doPlots"]:

        plotDirectory = params["path"] + "/" + channel.station_underscore
        seismon.utils.mkdir(plotDirectory)

        pngFile = os.path.join(plotDirectory,"bode.png")
        kwargs = {'logx':True}
        plot = gwpy.plotter.BodePlot(figsize=[14,8],**kwargs)
        plot.add_filter((B_low,A_low),frequencies=w_high,label="lowpass")
        plot.add_filter((B_high,A_high),frequencies=w_high,label="highpass")
        plot.add_legend(loc=1,prop={'size':10})
        plot.save(pngFile,dpi=200)
        plot.close()

    #dataLowpass = dataFull.copy()
    #lowpassdata = scipy.signal.filtfilt(B_low, A_low, np.array(dataFull.value))
    #lowpassdata[:2*channel.samplef] = lowpassdata[2*channel.samplef]
    #lowpassdata[-2*channel.samplef:] = lowpassdata[-2*channel.samplef]
    #lowpassdata[np.isnan(lowpassdata)] = 0.0
    dataLowpass = dataFull.lowpass(cutoff_low)

    accdata = np.diff(dataLowpass.value)/dataLowpass.dx.value
    dataLowpassAcc = gwpy.timeseries.TimeSeries(accdata, unit=dataFull.unit, sample_rate = 1/dataFull.dx.value, epoch = dataFull.epoch, dtype=float)

    dispdata = scipy.integrate.cumtrapz(dataLowpass.value, dx=dataLowpass.dx.value)
    dataLowpassDisp = gwpy.timeseries.TimeSeries(dispdata, unit=dataFull.unit, sample_rate = 1/dataFull.dx.value, epoch = dataFull.epoch, dtype=float)

    dataHighpass = dataFull.highpass(1.0)

    # calculate spectrum
    NFFT = params["fftDuration"]
    #window = None
    #dataASD = dataFull.asd()
    dataASD = dataFull.asd(fftlength=NFFT,overlap=None,method='welch')
    freq = np.array(dataASD.frequencies)
    indexes = np.where((freq >= params["fmin"]) & (freq <= params["fmax"]))[0]
    dataASD = np.array(dataASD.value)
    freq = freq[indexes]
    dataASD = dataASD[indexes]
    dataASD = gwpy.frequencyseries.FrequencySeries(dataASD, f0=np.min(freq), df=(freq[1]-freq[0]))

    nfft = len(dataFull.value)
    dataFFT = dataFull.fft(nfft=nfft)
    freqFFT = np.array(dataFFT.frequencies)
    dataFFT = np.array(dataFFT)
    dataFFTreal = np.interp(freq,freqFFT,dataFFT.real)
    dataFFTimag = np.interp(freq,freqFFT,dataFFT.imag)
    dataFFT = dataFFTreal + 1j*dataFFTimag
    dataFFT = gwpy.frequencyseries.FrequencySeries(dataFFT, f0=np.min(freqFFT), df=(freqFFT[1]-freqFFT[0]))

    # manually set units (units in CIS aren't correct)
    #dataASD.unit = 'counts/Hz^(1/2)'
    #dataFFT.unit = 'counts/Hz^(1/2)'
    dataASD.override_unit('count/Hz^(1/2)')
    dataFFT.override_unit('count/Hz^(1/2)')

    data = {}
    data["dataFull"] = dataFull
    data["dataLowpass"] = dataLowpass
    data["dataLowpassAcc"] = dataLowpassAcc
    data["dataLowpassDisp"] = dataLowpassDisp
    data["dataHighpass"] = dataHighpass
    data["dataASD"] = dataASD
    data["dataFFT"] = dataFFT

    if params["doEarthquakesHilbert"]:

        dataHilbert = scipy.signal.hilbert(dataLowpass).imag
        dataHilbert = dataHilbert.view(dataLowpass.__class__)
        dataHilbert.sample_rate =  dataFull.sample_rate
        dataHilbert.epoch = dataFull.epoch
        data["dataHilbert"] = dataHilbert

    return data

def apply_calibration(params,channel,data):
    """@applies calibration to necessary channels

    @param params
        seismon params dictionary
    @param channel
        seismon channel structure
    @param data
        spectral data structure
    """  

    if False:
    #if ("L4C" in channel.station) or ("GS13" in channel.station):

        zeros = [0,0,0]
        poles = [0.70711 + 0.70711*1j , 0.70711 - 0.70711*1j]
        gain = 1

        #poles = -2*np.pi*poles
        #zeros = -2*np.pi*zeros

        filt = [[(-1)*np.complex(0.70711,0.70711),(-1)*np.complex(0.70711,-0.70711)],[0,0],1]
        b, a = scipy.signal.zpk2tf(*filt)

        #b = [1,0,0,0];
        #a = [0,1,-1.414,1];
        w, h = scipy.signal.freqz(b, a)

        f = data["dataASD"].frequencies.value
      
        # Divide by f to get to displacement
        data["dataASD"]/=f
        # Filter spectrum
        filt = [b,a] 
        data["dataASD"].filter(*filt,inplace=True)
        fresp = abs(scipy.signal.freqs(b, a, f)[1])
        # Multiply by f to get to velocity
        data["dataASD"]*=f

        if params["doPlots"]:

            plotDirectory = params["path"] + "/" + channel.station_underscore
            seismon.utils.mkdir(plotDirectory)

            pngFile = os.path.join(plotDirectory,"calibration.png")
            plot = gwpy.plotter.Plot(figsize=[14,8])
            plot.add_line(f,fresp)
            plot.xlim = [params["fmin"],params["fmax"]]
            plot.xlabel = "Frequency [Hz]"
            plot.ylabel = "Response"
            plot.title = channel.station.replace("_","\_")
            plot.axes[0].set_xscale("log")
            plot.axes[0].set_yscale("log")
            plot.save(pngFile,dpi=200)
            plot.close()

    elif channel.station == "H1:ISI-GND_BRS_ETMX_RY_OUT_DQ":

        f = data["dataASD"].frequencies.value
        # Multiply by f to get to velocity
        data["dataASD"]/=(2*np.pi*f)

    elif "IRIS:II:BFO:00:LA1" == channel.station:

        z = []
        p = [4.56775E-03,-2.06901E-04]
        p = 1
        z = [-4.56775E-03,-2.06901E-04]
        k = 2.51043E-03
        k = 1

        b, a = scipy.signal.zpk2tf(z, p, k)

        f = data["dataASD"].frequencies.value

        # Filter spectrum
        data["dataASD"].filterba(a,b,inplace=True)
        data["dataASD"]*=9.81
        data["dataASD"]/=f
        fresp = abs(scipy.signal.freqs(a, b, f)[1])

        if params["doPlots"]:

            plotDirectory = params["path"] + "/" + channel.station_underscore
            seismon.utils.mkdir(plotDirectory)

            pngFile = os.path.join(plotDirectory,"calibration.png")
            plot = gwpy.plotter.Plot(figsize=[14,8])
            plot.add_line(f,fresp)
            plot.xlim = [params["fmin"],params["fmax"]]
            plot.xlabel = "Frequency [Hz]"
            plot.ylabel = "Response"
            plot.title = channel.station.replace("_","\_")
            plot.axes[0].set_xscale("log")
            plot.axes[0].set_yscale("log")
            plot.save(pngFile,dpi=200)
            plot.close()

    return data

def calculate_picks(params,channel,data):

    if params["doEarthquakesPicks"]:
        import obspy.signal

        nsta = int(2.5 * channel.samplef)
        nlta = int(10.0 * channel.samplef)
        cft = obspy.signal.trigger.recSTALTA(data["dataFull"].value, nsta, nlta)

        thres1 = 0.9 * np.max(cft)
        thres2 = 0.5
        on_off = obspy.signal.triggerOnset(cft, thres1, thres2)
        on_off = np.array(on_off) / channel.samplef
        on_off = data["dataFull"].epoch.val + on_off

    else:
        on_off = []

    data["on_off"] = on_off
   
    return data

def spectra(params, channel, segment):
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

    # make timeseries
    dataFull = seismon.utils.retrieve_timeseries(params, channel, segment)
    print("data read in...")
    if dataFull == []:
        return 

    dataFull = dataFull / channel.calibration
    indexes = np.where(np.isnan(dataFull.value))[0]
    meanSamples = np.median(np.ma.masked_array(dataFull.value,np.isnan(dataFull.value)))
    for index in indexes:
        dataFull[index] = meanSamples
    meanSamples = np.median(np.ma.masked_array(dataFull.value,np.isnan(dataFull.value)))
    dataFull -= meanSamples * dataFull.unit

    if np.mean(dataFull.value) == 0.0:
        print("data only zeroes... continuing\n")
        return
    if len(dataFull.value) < 2*channel.samplef:
        print("timeseries too short for analysis... continuing\n")
        return

    if channel.station == "H1:ISI-GND_BRS_ETMX_RY_OUT_DQ":
        dataFull *= 9.81

    print("calculating spectra...")
    data = calculate_spectra(params,channel,dataFull)
    print("calculating calibration...")
    data = apply_calibration(params,channel,data)
    data = calculate_picks(params,channel,data)

    # calculate spectrogram
    specgram = dataFull.spectrogram(params["fftDuration"])
    specgram **= 1/2.
    data["dataSpecgram"] = specgram

    #speclog = specgram.to_logf()
    #medratio = speclog.ratio('median')
    medratio = specgram.ratio('median')

    if params["doEarthquakes"]:
        earthquakesDirectory = os.path.join(params["path"],"earthquakes")
        earthquakesXMLFile = os.path.join(earthquakesDirectory,"earthquakes.xml")
        attributeDics = seismon.utils.read_eqmons(earthquakesXMLFile)
    else:
        attributeDics = []
    #attributeDics = [attributeDics[0]]

    save_data(params,channel,gpsStart,gpsEnd,data,attributeDics)

    if params["doPlots"]:

        plotDirectory = params["path"] + "/" + channel.station_underscore
        seismon.utils.mkdir(plotDirectory)        

        pngFile = os.path.join(plotDirectory,"timeseries.png")
        plot = gwpy.plotter.TimeSeriesPlot(figsize=[14,8])

        dataHighpass = data["dataHighpass"].resample(16)
        dataFull = data["dataFull"].resample(16)
        dataLowpass = data["dataLowpass"].resample(16)
        dataLowpassAcc = data["dataLowpassAcc"].resample(16)     
        dataLowpassDisp = data["dataLowpassDisp"].resample(16) 

        #dataHighpass = data["dataHighpass"]
        #dataFull = data["dataFull"]
        #dataLowpass = data["dataLowpass"]

        dataHighpass *= 1e6
        #dataFull *= 1e6
        dataLowpass *= 1e6
        dataLowpassAcc *= 1e6
        dataLowpassDisp *= 1e6

        if channel.station == "H1:ISI-GND_BRS_ETMX_RY_OUT_DQ":
            #plot.add_timeseries(dataHighpass,label="data")
            #dataFull *= 1e6*9.81/(2*np.pi*0.1)
            #plot.add_timeseries(dataFull,label="data")

            dataHighpass *= (9.81)/(2*np.pi*0.05)

            plot.add_timeseries(dataHighpass,label="highpass")

            xlim = [plot.xlim[0],plot.xlim[1]]
            ylim = [plot.ylim[0],plot.ylim[1]]

            #ylim = [-0.5*(9.81)/(2*np.pi*0.05),0.5*(9.81)/(2*np.pi*0.05)]

        else:

            plot.add_timeseries(dataHighpass,label="highpass")
            #plot.add_timeseries(dataFull,label="data")
            kwargs = {"linestyle":"-","color":"k"}
            plot.add_timeseries(dataLowpass,label="lowpass",**kwargs)

            xlim = [plot.xlim[0],plot.xlim[1]]
            ylim = [plot.ylim[0],plot.ylim[1]]
        count = 0
        for attributeDic in attributeDics:

            if not "Arbitrary" in attributeDic["traveltimes"]:
                continue

            #if channel.station == "H1:ISI-GND_BRS_ETMX_RY_OUT_DQ":
            #    continue

            if attributeDic["Magnitude"] < params["minMagnitude"]:
                continue

            if params["ifo"] == "IRIS":
                attributeDic = seismon.eqmon.ifotraveltimes_loc(attributeDic, "IRIS", channel.latitude, channel.longitude)
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

            if count ==  0:
                kwargs = {"linestyle":"--","color":"r"}
                plot.add_line([Ptime,Ptime],ylim,label="P Est. Arrival",**kwargs)
                plt.text(Ptime+30,ylim[1]-10,'P')
                kwargs = {"linestyle":"--","color":"g"}
                plot.add_line([Stime,Stime],ylim,label="S Est. Arrival",**kwargs)
                plt.text(Stime+30,ylim[1]-10,'S')
                kwargs = {"linestyle":"--","color":"b"}
                plot.add_line([RthreePointFivetime,RthreePointFivetime],ylim,label="3.5 km/s R Est. Arrival",**kwargs)            
                plt.text(RthreePointFivetime+30,ylim[1]-10,'Rf')
                plot.add_line([Rtwotime,Rtwotime],ylim,label="2 km/s R Est. Arrival",**kwargs)
                plot.add_line([Rfivetime,Rfivetime],ylim,label="5 km/s R Est. Arrival",**kwargs)

                if params["doEarthquakesVelocityMap"]:
                    kwargs = {"linestyle":"--","color":"m"}
                    plot.add_line([Rvelocitymaptime,Rvelocitymaptime],ylim,label="Velocity map R Est. Arrival",**kwargs)
                kwargs = {"linestyle":"--","color":"k"}
                plot.add_line(xlim,[peak_velocity,peak_velocity],label="pred. vel.",**kwargs)
                plot.add_line(xlim,[-peak_velocity,-peak_velocity],**kwargs)

                tstart = (Rfivetime + RthreePointFivetime)/2.0
                #tstart = RthreePointFivetime
                tend = Rtwotime
                x = np.arange(tstart,tend,1)
                y = (x - np.min(x))/600.0
                mu = RthreePointFivetime
                lamb = 1 + y[-1]/5.0
                vals = scipy.stats.gamma.pdf(y, lamb)
                vals = np.absolute(vals)
                vals = vals / np.max(vals)
                vals = vals * peak_velocity
                kwargs = {"linestyle":"--","color":"g"}
                #plot.add_line(x,vals,label="Envelope",**kwargs)

            else:
                kwargs = {"linestyle":"--","color":"r"}
                plot.add_line([Ptime,Ptime],ylim,**kwargs)
                kwargs = {"linestyle":"--","color":"g"}
                plot.add_line([Stime,Stime],ylim,**kwargs)
                kwargs = {"linestyle":"--","color":"b"}
                plot.add_line([RthreePointFivetime,RthreePointFivetime],ylim,**kwargs)
                plot.add_line([Rtwotime,Rtwotime],ylim,**kwargs)
                plot.add_line([Rfivetime,Rfivetime],ylim,**kwargs)

                if params["doEarthquakesVelocityMap"]:
                    kwargs = {"linestyle":"--","color":"m"}
                    plot.add_line([Rvelocitymaptime,Rvelocitymaptime],ylim,**kwargs)
                kwargs = {"linestyle":"--","color":"k"}
                plot.add_line(xlim,[peak_velocity,peak_velocity],**kwargs)
                plot.add_line(xlim,[-peak_velocity,-peak_velocity],**kwargs)

            earthquakesDirectory = params["dirPath"] + "/Text_Files/Earthquakes/" + channel.station_underscore + "/" + str(params["fftDuration"])
            earthquakesFile = os.path.join(earthquakesDirectory,"%s.txt"%(attributeDic["eventName"]))
 
            if not os.path.isfile(earthquakesFile):
                continue

            data_out = np.loadtxt(earthquakesFile)
            ttMax = data_out[0]
            kwargs = {"linestyle":"-","color":"k"}
            #plot.add_line([ttMax,ttMax],ylim,label="Max amplitude",**kwargs)

            count = count + 1

        count = 0
        for on_off in data["on_off"]:

            ontime = on_off[0]
            offtime = on_off[1]

            if count ==  0:
                kwargs = {"linestyle":"-.","color":"r"}
                plot.add_line([ontime,ontime],ylim,label="On trigger",**kwargs)
                kwargs = {"linestyle":"-.","color":"b"}
                plot.add_line([offtime,offtime],ylim,label="Off trigger",**kwargs)
            else:
                kwargs = {"linestyle":"-.","color":"r"}
                plot.add_line([ontime,ontime],ylim,**kwargs)
                kwargs = {"linestyle":"-.","color":"b"}
                plot.add_line([offtime,offtime],ylim,**kwargs)
            count = count + 1
 
        if params["doEarthquakesTrips"]:    
            trips = []
            platforms = []
            stages = [] 
            trip_lines = [line.strip() for line in open(params["tripsTextFile"])]
            for line in trip_lines:
                lineSplit = line.split(" ")
                trips.append(float(lineSplit[0]))
                platforms.append(lineSplit[1])
                stages.append(lineSplit[2])

            ylim_diff = ylim[1] - ylim[0]
            perc_diff = ylim_diff * 0.025

            count = 1
            for trip,platform,stage in zip(trips,platforms,stages):
                if (trip >= xlim[0]) and (trip <= xlim[1]):
                    if count ==  1:
                        kwargs = {"linestyle":"-.","color":"m","linewidth":4}
                        plot.add_line([trip,trip],ylim,label="Trip time",**kwargs)
                    else:
                        kwargs = {"linestyle":"-.","color":"m","linewidth":4}
                        plot.add_line([trip,trip],ylim,**kwargs)
                    plt.text(trip+15,ylim[1]-count*perc_diff,'%s %s'%(platform,stage),fontsize=12)
                    count = count + 1

        if channel.station == "H1:ISI-GND_BRS_ETMX_RY_OUT_DQ":
            plot.ylabel = r"Angle [$\mu$rad]"
        elif channel.station == "H1:ISI-GND_BRS_ETMX_DAMPCTRLMON":
            plot.ylabel = r"Counts"
        else:
            plot.ylabel = r"Velocity [$\mu$m/s]"
        plot.title = channel.station.replace("_","\_")
        plot.xlim = xlim
        plot.ylim = ylim
        plot.add_legend(loc=1,prop={'size':10})

        #plot.ylim = [-0.05,0.05]
        plot.save(pngFile)
        #plot.grid = 0
        #pdfFile = os.path.join(plotDirectory,"timeseries.pdf")
        #plot.save(pdfFile)

        try:
            plot.save(pngFile)

            epsFile = os.path.join(plotDirectory,"timeseries.eps")
            plot.save(epsFile)

            pdfFile = os.path.join(plotDirectory,"timeseries.pdf")
            plot.save(pdfFile)

            pngFile = os.path.join(plotDirectory,"timeseries_30.png")
            plot.ylim = [-30,30]
            plot.save(pngFile)

        except:
            pass
        plot.close()

        if params["doEarthquakesHilbert"]:
            dataHilbert = data["dataHilbert"].resample(16)
            dataHilbert *= 1e6

            pngFile = os.path.join(plotDirectory,"hilbert.png")
            plot = gwpy.plotter.TimeSeriesPlot(figsize=[14,8])

            plot.add_timeseries(dataHilbert,label="highpass")

            plot.ylabel = r"Velocity [$\mu$m/s]"
            plot.title = channel.station.replace("_","\_")
            plot.xlim = xlim
            plot.ylim = ylim

            plot.save(pngFile)
            plot.close()

        pngFile = os.path.join(plotDirectory,"acc.png")
        plot = gwpy.plotter.TimeSeriesPlot(figsize=[14,8])

        plot.add_timeseries(dataLowpassAcc,label="highpass")

        plot.ylabel = r"Acceleration [$\mu$m/s$^2$]"
        plot.title = channel.station.replace("_","\_")
        #plot.xlim = xlim
        #plot.ylim = ylim

        plot.save(pngFile)
        plot.close()

        pngFile = os.path.join(plotDirectory,"disp.png")
        plot = gwpy.plotter.TimeSeriesPlot(figsize=[14,8])

        plot.add_timeseries(dataLowpassDisp,label="highpass")

        plot.ylabel = r"Displacement [$\mu$m]"
        plot.title = channel.station.replace("_","\_")
        #plot.xlim = xlim
        #plot.ylim = ylim

        plot.save(pngFile)
        plot.close()

        fl, low, fh, high = seismon.NLNM.NLNM(2)


        pngFile = os.path.join(plotDirectory,"psd.png")
        label = channel.station.replace("_","\_")

        plot = gwpy.plotter.Plot(figsize=[14,8])
        plot.add_frequencyseries(data["dataASD"],label=label)
        kwargs = {"linestyle":"-.","color":"k"}
        plot.add_line(fl, low, label="HNM/LNM", **kwargs)
        plot.add_line(fh, high, **kwargs)
        plot.xlim = [params["fmin"],params["fmax"]]
        plot.ylim = [10**-10, 10**-4]
        #plot.ylim = [10**-8, 10**-2]
        plot.xlabel = "Frequency [Hz]"

        if channel.station == "H1:ISI-GND_BRS_ETMX_RY_OUT_DQ":
            plot.ylabel = r"Angle [$\mu$rad/rtHz]"
        else:
            plot.ylabel = "Amplitude Spectrum [(m/s)/rtHz]"
        plot.title = channel.station.replace("_","\_")
        plot.axes[0].set_xscale("log")
        plot.axes[0].set_yscale("log")
        plot.add_legend(loc=1,prop={'size':10})
        plot.save(pngFile,dpi=200)
        plot.close()

        pngFile = os.path.join(plotDirectory,"psd_linear.png")
        label = channel.station.replace("_","\_")

        if not np.sum(data["dataASD"]) == 0:
            plot = gwpy.plotter.Plot(figsize=[14,8])
            plot.add_spectrum(data["dataASD"],label=label)
            kwargs = {"linestyle":"-.","color":"k"}
            #plot.add_line(fl, low, label="HNM/LNM", **kwargs)
            #plot.add_line(fh, high, **kwargs)
            plot.xlim = [params["fmin"],params["fmax"]]
            #plot.ylim = [10**-10, 10**-4]
            plot.xlabel = "Frequency [Hz]"
            plot.ylabel = "Amplitude Spectrum [(m/s)/rtHz]"
            plot.title = channel.station.replace("_","\_")
            #plot.axes[0].set_xscale("log")
            plot.axes[0].set_yscale("log")
            plot.add_legend(loc=1,prop={'size':10})
            plot.save(pngFile,dpi=200)
            plot.close()
  
        if not len(medratio) == 0:
           pngFile = os.path.join(plotDirectory,"tf.png")
           plot = medratio.plot(figsize=[14,8])
           plot.add_colorbar(log=True, clim=[0.1, 10], label='ASD ratio to median average')
           plot.ylabel = "Frequency [Hz]"
           plot.ylim = [params["fmin"],params["fmax"]]
           plot.axes[0].set_yscale("log")
           plot.xlim = [gpsStart,gpsEnd]
           plot.axes[0].auto_gps_scale()
           plot.show()
           plot.save(pngFile,dpi=200)
           plot.close()                                       

        if params["doEarthquakesVelocityMap"]:
            pngFile = os.path.join(plotDirectory,"velocitymaps.png")
            plot = gwpy.plotter.TimeSeriesPlot(figsize=[14,8])

            for attributeDic in attributeDics:
       
                Rvelocitymaptimes = traveltimes["Rvelocitymaptimes"]
                Rvelocitymaptime = max(traveltimes["Rvelocitymaptimes"])
                Rvelocitymapvelocities = traveltimes["Rvelocitymapvelocities"]

                kwargs = {"linestyle":"-","color":"k"}
                plot.add_line(Rvelocitymaptimes,Rvelocitymapvelocities,label="Velocity Map",**kwargs)

            xlim = [plot.xlim[0],plot.xlim[1]]
            ylim = [plot.ylim[0],plot.ylim[1]]

            kwargs = {"linestyle":"-","color":"b"}
            plot.add_line(xlim,[2,2],**kwargs)
            plot.add_line(xlim,[3.5,3.5],**kwargs)
            plot.add_line(xlim,[5,5],**kwargs)

            plot.ylim = [1.75,5.25]
            plot.xlabel = "Time [s]"
            plot.ylabel = "Velocity [km/s]"
            plot.show()
            plot.save(pngFile,dpi=200)
            plot.close()

def freq_analysis(params,channel,tt,freq,spectra):
    """@frequency analysis of spectral data.

    @param params
        seismon params dictionary
    @param channel
        seismon channel structure
    @param tt
        array of start times
    @param freq
        frequency vector
    @param spectra
        spectrogram structure
    """

    if params["doPlots"]:
        plotDirectory = params["path"] + "/" + channel.station_underscore + "/freq"
        seismon.utils.mkdir(plotDirectory)
 
    indexes = np.logspace(0,np.log10(len(freq)-1),num=100)
    indexes = list(np.unique(np.ceil(indexes)))
    indexes = range(len(freq))
    #indexes = range(16)

    indexes = np.where(10.0 >= freq)[0]

    deltaT = tt[1] - tt[0]

    n_dist = []
    for j in range(1000):
        n_dist.append(scipy.stats.chi2.rvs(2))

    p_chi2_vals = []
    p_ks_vals = []
    ttCoh_vals = []

    for i in indexes:
        vals = spectra[:,i]

        meanPSD = np.mean(vals) 
        stdPSD = np.std(vals)

        vals_norm = 2 * vals / meanPSD

        bins = np.arange(0,10,1)
        (n,bins) = np.histogram(vals_norm,bins=bins)
        n_total = np.sum(n)

        bins = (bins[1:] + bins[:len(bins)-1])/2

        n_expected = []
        for bin in bins:
            expected_val = n_total * scipy.stats.chi2.pdf(bin, 2)
            n_expected.append(expected_val)
        n_expected = np.array(n_expected)

        (stat_chi2,p_chi2) = scipy.stats.mstats.chisquare(n, f_exp=n_expected)
        p_chi2_vals.append(p_chi2)

        (stat_ks,p_ks) = scipy.stats.ks_2samp(vals_norm, n_dist)
        p_ks_vals.append(p_ks)

        acov = np.correlate(vals,vals,"full")
        acov = acov / np.max(acov)

        ttCov = (np.arange(len(acov)) - len(acov)/2) * float(deltaT)

        #ttLimitMin = - 5/freq[i]
        #ttLimitMax = 5 /freq[i]

        ttLimitMin = - float('inf')
        ttLimitMax = float('inf')

        ttIndexes = np.intersect1d(np.where(ttCov >= ttLimitMin)[0],np.where(ttCov <= ttLimitMax)[0])
        #ttCov = ttCov / (60)

        acov_minus_05 = np.absolute(acov[ttIndexes] - 0.66)
        index_min = acov_minus_05.argmin()

        ttCoh = np.absolute(ttCov[ttIndexes[index_min]])
        ttCoh_vals.append(ttCoh)

        if len(ttIndexes) == 0:
            continue

        #if freq[i] > 0:
        #    continue

        if params["doPlots"]:
            ax = plt.subplot(111)
            plt.plot(bins,n,label='true')
            plt.plot(bins,n_expected,'k*',label='expected')
            plt.xlabel("2 * data / mean")
            plt.ylabel("Counts")
            plot_title = "p-value: %f"%p_chi2
            plt.title(plot_title)
            plt.legend()
            plt.show()
            plt.savefig(os.path.join(plotDirectory,"%s_dist.png"%str(freq[i])),dpi=200)
            plt.savefig(os.path.join(plotDirectory,"%s_dist.eps"%str(freq[i])),dpi=200)
            plt.close('all')

            ax = plt.subplot(111)
            plt.semilogy(ttCov[ttIndexes],acov[ttIndexes])
            plt.vlines(ttCoh,10**(-3),1,color='r')
            plt.vlines(-ttCoh,10**(-3),1,color='r')
            plt.xlabel("Time [Seconds]")
            plt.ylabel("Correlation")
            plt.show()
            plt.savefig(os.path.join(plotDirectory,"%s_cov.png"%str(freq[i])),dpi=200)
            plt.savefig(os.path.join(plotDirectory,"%s_cov.eps"%str(freq[i])),dpi=200)
            plt.close('all')

    if params["doPlots"]:
        ax = plt.subplot(111)
        plt.loglog(freq[indexes],p_chi2_vals,label='chi2')
        plt.loglog(freq[indexes],p_ks_vals,label='k-s')
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("p-value")
        plt.legend(loc=3)
        plt.show()
        plt.savefig(os.path.join(plotDirectory,"freq_analysis.png"),dpi=200)
        plt.savefig(os.path.join(plotDirectory,"freq_analysis.eps"),dpi=200)
        plt.close('all')      

        ax = plt.subplot(111)
        plt.semilogx(freq[indexes],ttCoh_vals)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Coherence Time [s]")
        plt.show()
        plt.savefig(os.path.join(plotDirectory,"ttCohs.png"),dpi=200)
        plt.savefig(os.path.join(plotDirectory,"ttCohs.eps"),dpi=200)
        plt.close('all')

def analysis(params, channel):
    """@analysis of spectral data.

    @param params
        seismon params dictionary
    @param channel
        seismon channel structure
    """

    psdDirectory = params["dirPath"] + "/Text_Files/PSD/" + channel.station_underscore + "/" + str(params["fftDuration"])
    files = glob.glob(os.path.join(psdDirectory,"*.txt"))

    files = sorted(files)

    if not params["doFreqAnalysis"]:
        if len(files) > 1000:
            files = files[:1000]

    #files = files[-10:]

    tts = []
    spectra = []

    for file in files:

        fileSplit = file.split("/")
        txtFile = fileSplit[-1].replace(".txt","")
        txtFileSplit = txtFile.split("-")
        thisTTStart = int(txtFileSplit[0])
        thisTTEnd = int(txtFileSplit[1])
        tt = thisTTStart

        if tt in tts:
            continue

        tts.append(tt)

        spectra_out = gwpy.frequencyseries.Spectrum.read(file,format='dat')
        spectra_out.unit = 'counts/Hz^(1/2)'
        spectra.append(spectra_out)

        if tt == params["gpsStart"]:
            spectraNow = spectra_out

    if not 'spectraNow' in locals():
        print("no data at requested time... continuing\n")
        return

    if np.mean(spectraNow.value) == 0.0:
        print("data only zeroes... continuing\n")
        return

    dt = tts[1] - tts[0]
    epoch = gwpy.time.Time(tts[0], format='gps')
    specgram = gwpy.spectrogram.Spectrogram.from_spectra(*spectra, dt=dt,epoch=epoch)

    freq = np.array(specgram.frequencies)

    # Define bins for the spectral variation histogram
    kwargs = {'log':True,'nbins':500,'norm':True}
    #kwargs = {'log':True,'nbins':500}
    specvar = gwpy.frequencyseries.hist.SpectralVariance.from_spectrogram(specgram,**kwargs) 
    bins = specvar.bins[:-1]
    specvar = specvar * 100

    if params["doFreqAnalysis"]:
        freq_analysis(params,channel,ttStart,freq,specgram)

    # Calculate percentiles
    spectral_variation_1per = specvar.percentile(1)
    spectral_variation_10per = specvar.percentile(10)
    spectral_variation_50per = specvar.percentile(50)
    spectral_variation_90per = specvar.percentile(90)
    spectral_variation_99per = specvar.percentile(99)

    textDirectory = params["path"] + "/" + channel.station_underscore
    seismon.utils.mkdir(textDirectory)

    f = open(os.path.join(textDirectory,"spectra.txt"),"w")
    for i in range(len(freq)):
        f.write("%e %e %e %e %e %e %e\n"%(freq[i],spectral_variation_1per[i].value,\
            spectral_variation_10per[i].value,spectral_variation_50per[i].value,\
            spectral_variation_90per[i].value,spectral_variation_99per[i].value,\
            spectraNow[i].value))
    f.close()

    sigDict = {}
    # Break up entire frequency band into 6 segments
    ff_ave = [1/float(128), 1/float(64),  0.1, 1, 3, 5, 10]

    f = open(os.path.join(textDirectory,"sig.txt"),"w")
    for i in range(len(ff_ave)-1):
        newSpectra = []
        newSpectraNow = []
        newFreq = []

        for j in range(len(freq)):
            if ff_ave[i] <= freq[j] and freq[j] <= ff_ave[i+1]:
                newFreq.append(freq[j])
                newSpectraNow.append(spectraNow.value[j])
                if newSpectra == []:
                    newSpectra = specgram.value[:,j]
                else:                 
                    newSpectra = np.vstack([newSpectra,specgram.value[:,j]])

        newSpectra = np.array(newSpectra)
        if len(newSpectra.shape) > 1:
            newSpectra = np.mean(newSpectra, axis = 0)
        try:
            sig, bgcolor = seismon.utils.html_bgcolor(np.mean(newSpectraNow),newSpectra)
        except:
            sig = 0
            bgcolor = 'nan'

        f.write("%e %e %e %e %s\n"%(ff_ave[i],ff_ave[i+1],np.mean(newSpectraNow),sig,bgcolor))

        key = "%s-%s"%(ff_ave[i],ff_ave[i+1])

        dt = tts[-1] - tts[-2]
        epoch = gwpy.time.Time(tts[0], format='gps')

        timeseries = gwpy.timeseries.TimeSeries(newSpectra, epoch=epoch, sample_rate=1.0/dt)
        
        sigDict[key] = {}
        sigDict[key]["data"] = timeseries

    f.close()

    if params["doPlots"]:

        plotDirectory = params["path"] + "/" + channel.station_underscore
        seismon.utils.mkdir(plotDirectory)

        fl, low, fh, high = seismon.NLNM.NLNM(2)

        pngFile = os.path.join(plotDirectory,"psd.png")

        #plot = spectraNow.plot()
        plot = gwpy.plotter.Plot(figsize=[14,8])
        plot.add_spectrum(spectraNow,label='now')
        kwargs = {"linestyle":"-","color":"k"}
        plot.add_line(freq, spectral_variation_10per, **kwargs)
        plot.add_line(freq, spectral_variation_50per, **kwargs)
        plot.add_line(freq, spectral_variation_90per, **kwargs)
        kwargs = {"linestyle":"-.","color":"k"}
        plot.add_line(fl, low, **kwargs)
        plot.add_line(fh, high, **kwargs)
        plot.xlim = [params["fmin"],params["fmax"]]
        plot.ylim = [np.min(bins), np.max(bins)]
        plot.xlabel = "Frequency [Hz]"
        plot.ylabel = "Amplitude Spectrum [(m/s)/rtHz]"
        plot.axes[0].set_xscale("log")
        plot.axes[0].set_yscale("log")
        plot.save(pngFile,dpi=200)
        plot.close()

        pngFile = os.path.join(plotDirectory,"disp.png")

        spectraNowDisplacement = spectraNow / freq
        #plot = spectraNowDisplacement.plot()

        plot = gwpy.plotter.Plot(figsize=[14,8])
        plot.add_spectrum(spectraNowDisplacement,label='now')
        kwargs = {"linestyle":"-","color":"w"}
        plot.add_line(freq, spectral_variation_10per/freq, **kwargs)
        plot.add_line(freq, spectral_variation_50per/freq, **kwargs)
        plot.add_line(freq, spectral_variation_90per/freq, **kwargs)
        kwargs = {"linestyle":"-.","color":"k"}
        plot.add_line(fl, low/fl, **kwargs)
        plot.add_line(fh, high/fh, **kwargs)
        plot.xlim = [params["fmin"],params["fmax"]]
        plot.ylim = [np.min(bins)/np.max(freq), np.max(bins)/np.min(freq)]
        plot.xlabel = "Frequency [Hz]"
        plot.ylabel = "Displacement Spectrum [m/rtHz]"
        plot.axes[0].set_xscale("log")
        plot.axes[0].set_yscale("log")
        plot.save(pngFile,dpi=200)
        plot.close()

        pngFile = os.path.join(plotDirectory,"tf.png")
        #specgramLog = specgram.to_logf(fmin=np.min(freq),fmax=np.max(freq))
        #plot = specgramLog.plot()
        plot = specgram.plot()
        plot.ylim = [params["fmin"],params["fmax"]]
        plot.ylabel = "Frequency [Hz]"
        colorbar_label = "Amplitude Spectrum [(m/s)/rtHz]"
        kwargs = {}
        plot.axes[0].set_xscale("log")
        plot.axes[0].set_yscale("log")
        plot.add_colorbar(location='right', log=True, label=colorbar_label, clim=None, visible=True, **kwargs)
        plot.save(pngFile,dpi=200)
        plot.close()

        pngFile = os.path.join(plotDirectory,"specvar.png")
        kwargs = {"linestyle":"-","color":"w"}
        #plot = specvar.plot(**kwargs)
        plot = gwpy.plotter.Plot(figsize=[14,8])
        plot.add_spectrum(spectraNow,label='now')
        #plot = spectraNow.plot(**kwargs)
        kwargs = {"linestyle":"-","color":"k"}
        plot.add_line(freq, spectral_variation_10per, **kwargs)
        plot.add_line(freq, spectral_variation_50per, **kwargs)
        plot.add_line(freq, spectral_variation_90per, **kwargs)
        extent = [np.min(freq), np.max(freq),
                   np.min(bins), np.max(bins)]
        kwargs = {}
        #plot.plot_variance(specvar, extent=extent, **kwargs)
        plot.axes[0].set_xscale("log")
        plot.axes[0].set_yscale("log")
        kwargs = {"linestyle":"-.","color":"k"}
        plot.add_line(fl, low, **kwargs)
        plot.add_line(fh, high, **kwargs)
        plot.xlim = [params["fmin"],params["fmax"]]
        plot.ylim = [np.min(bins), np.max(bins)]
        plot.xlabel = "Frequency [Hz]"
        plot.ylabel = "Amplitude Spectrum [(m/s)/rtHz]"

        plot.save(pngFile,dpi=200)
        plot.close()

        X,Y = np.meshgrid(freq, bins)
        ax = plt.subplot(111)
        #im = plt.pcolor(X,Y,np.transpose(spectral_variation_norm), cmap=plt.cm.jet)

        im = plt.pcolor(X,Y,np.transpose(specvar.value), cmap=plt.cm.jet)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.semilogx(freq,spectraNow, 'k', label='Current')
        plt.semilogx(freq,spectral_variation_10per,'w',label='10')
        plt.semilogx(freq,spectral_variation_50per,'w',label='50')
        plt.semilogx(freq,spectral_variation_90per,'w',label='90')
        plt.loglog(fl,low,'k-.')
        plt.loglog(fh,high,'k-.',label='LNM/HNM')
        plt.xlim([params["fmin"],params["fmax"]])
        plt.ylim([np.min(bins), np.max(bins)])
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude Spectrum [(m/s)/rtHz]")
        plt.clim(0,5)
        plt.grid()
        plt.show()
        plt.savefig(pngFile,dpi=200)
        plt.close('all')

        pngFile = os.path.join(plotDirectory,"bands.png")
        plot = gwpy.plotter.TimeSeriesPlot()
        for key in sigDict.iterkeys():
            label = key
            plot.add_timeseries(sigDict[key]["data"], label=label)
        plot.axes[0].set_yscale("log")
        plot.ylabel = "Average Amplitude Spectrum log10[(m/s)/rtHz]"
        plot.add_legend(loc=1,prop={'size':10})
        plot.save(pngFile,dpi=200)
        plot.close()

    htmlPage = seismon.html.seismon_page(channel,textDirectory)
    if htmlPage is not None:
        f = open(os.path.join(textDirectory,"psd.html"),"w")
        f.write(htmlPage)
        f.close()

def channel_summary(params, segment):
    """@summary of channels of spectral data.

    @param params
        seismon params dictionary
    @param segment
        [start,end] gps
    """

    gpsStart = segment[0]
    gpsEnd = segment[1]

    data = {}
    for channel in params["channels"]:

        psdDirectory = params["dirPath"] + "/Text_Files/PSD/" + channel.station_underscore + "/" + str(params["fftDuration"])
        file = os.path.join(psdDirectory,"%d-%d.txt"%(gpsStart,gpsEnd))

        if not os.path.isfile(file):
            continue

        spectra_out = gwpy.frequencyseries.Spectrum.read(file,format='dat')
        spectra_out.unit = 'counts/Hz^(1/2)'

        if np.sum(spectra_out.value) == 0.0:
            continue

        data[channel.station_underscore] = {}
        data[channel.station_underscore]["data"] = spectra_out

        if params["doPowerLawFit"]:

            powerlawDirectory = params["dirPath"] + "/Text_Files/Powerlaw/" + channel.station_underscore + "/" + str(params["fftDuration"])
            seismon.utils.mkdir(powerlawDirectory)

            powerlawFile = os.path.join(powerlawDirectory,"%d-%d.txt"%(gpsStart,gpsEnd))
            data_out = np.loadtxt(powerlawFile)
            data[channel.station_underscore]["powerlaw"] = data_out

    if data == {}:
        return

    if params["doPlots"]:

        plotDirectory = params["path"] + "/summary"
        seismon.utils.mkdir(plotDirectory)

        fl, low, fh, high = seismon.NLNM.NLNM(2)

        pngFile = os.path.join(plotDirectory,"psd.png")
        lowBin = np.inf
        highBin = -np.inf
        plot = gwpy.plotter.Plot(figsize=[14,8])
        for key in data.iterkeys():

            label = key.replace("_","\_")

            plot.add_spectrum(data[key]["data"], label=label)

            lowBin = np.min([lowBin,np.min(np.array(data[key]["data"]))])
            highBin = np.max([highBin,np.max(np.array(data[key]["data"]))])

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

        pngFile = os.path.join(plotDirectory,"ratio.png")
        lowBin = np.inf
        highBin = -np.inf
        ref = params["referenceChannel"].replace(":","_")

        if not ref in data:
            return

        plot = gwpy.plotter.Plot(figsize=[14,8])
        for key in data.iterkeys():

            label = key.replace("_","\_")

            plot.add_spectrum(data[key]["data"] / data[ref]["data"], label=label)
            lowBin = np.min([lowBin,np.min(np.array(data[key]["data"]))])
            highBin = np.max([highBin,np.max(np.array(data[key]["data"]))])

        kwargs = {"linestyle":"-.","color":"k"}
        #plot.add_line(fl, low, **kwargs)
        #plot.add_line(fh, high, **kwargs)
        plot.xlim = [params["fmin"],params["fmax"]]
        #plot.ylim = [lowBin, highBin]
        plot.xlabel = "Frequency [Hz]"
        label_ref = params["referenceChannel"].replace("_","\_")
        plot.ylabel = "Spectrum / Reference [%s]"%(label_ref)
        plot.add_legend(loc=1,prop={'size':10})
        plot.axes[0].set_xscale("log")
        plot.axes[0].set_yscale("log")

        plot.save(pngFile,dpi=200)
        plot.close()

        if params["doPowerLawFit"]:

            pngFile = os.path.join(plotDirectory,"powerlaw.png")
            colors = cm.rainbow(np.linspace(0, 1, len(data)))
            count = 0

            plot = gwpy.plotter.Plot(figsize=[14,8])
            for key in data.iterkeys():

                label = key.replace("_","\_")

                color = colors[count]
                kwargs = {"linestyle":"-","color":color}

                plot.add_spectrum(data[key]["data"], label=label, **kwargs)

                powerlaw = lambda x, amp, index: amp * (x**index)

                xdata = data[key]["data"].frequencies
                index = data[key]["powerlaw"][0]
                amp = data[key]["powerlaw"][1]

                kwargs = {"linestyle":"--","color":color}
                plot.add_line(xdata,powerlaw(xdata,amp,index),**kwargs)
                count = count + 1

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

