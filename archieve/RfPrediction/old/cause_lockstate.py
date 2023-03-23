#!/usr/bin/python

from __future__ import division
import os, sys, glob, optparse, warnings, time, json
import numpy as np
import subprocess
from subprocess import Popen
from lxml import etree
from scipy.interpolate import interp1d
from gwpy.timeseries import TimeSeries
import lal.gpstime

#from seismon import (eqmon, utils)

import matplotlib
matplotlib.use("AGG")
matplotlib.rcParams.update({'font.size': 18})
from matplotlib import pyplot as plt
from matplotlib import cm

def parse_commandline():
    """@parse the options given on the command-line.
    """
    parser = optparse.OptionParser(usage=__doc__)

    parser.add_option("-t", "--time_after_p_wave", help="time to check for lockloss status after p wave arrival.",
                      default = 3600)

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
## LHO
rms_toggle = ''
os.system('mkdir -p /home/eric.coughlin/H1O1/')
os.system('mkdir -p /home/eric.coughlin/public_html/lockloss_threshold_plots/LHO/')
for direction in ['Z','X','Y']:
    if rms_toggle == "":
        channel = 'H1:ISI-GND_STS_HAM2_{0}_DQ'.format(direction)
    elif rms_toggle == "RMS_":
        channel = 'H1:ISI-GND_STS_HAM5_{0}_BLRMS_30M_100M'.format(direction)
    H1_lock_time_list = []
    H1_lockloss_time_list = []
    H1_peak_ground_velocity_list = []
    hdir = os.environ["HOME"]
    options = parse_commandline()
    predicted_peak_ground_velocity_list = []
    datafileH1 = open('{0}/gitrepo/seismon/RfPrediction/data/LHO_O1_{1}{2}.txt'.format(hdir, rms_toggle, direction), 'r')
    resultfileH1 = open('/home/eric.coughlin/gitrepo/seismon/RfPrediction/data/LHO_lockstatus_{0}{1}.txt'.format(rms_toggle, direction), 'w')
    H1_channel_lockstatus_data = open('/home/eric.coughlin/gitrepo/seismon/RfPrediction/data/segs_Locked_H_1126569617_1136649617.txt', 'r')
    # This next section of code is where the data is seperated into two lists to make this data easier to search through and process.
    for item in (line.strip().split() for line in H1_channel_lockstatus_data):
        H1_lock_time = item[0]
        H1_lockloss_time = item[1]
        H1_lock_time_list.append(float(H1_lock_time))
        H1_lockloss_time_list.append(float(H1_lockloss_time))
    #resultfileH1.write('{0:^20} {1:^20} {2:^20} {3:^20} \n'.format('eq arrival time','pw arrival time','peak ground velocity','lockloss'))
    for column in ( line.strip().split() for line in datafileH1):
            eq_time = column[0] # This is the time that the earthquake was detected
            eq_mag = column[1]
            pw_arrival_time = column[2] #this is the arrival time of the pwave
            sw_arrival_time = column[3]
            eq_distance = column[12]
            eq_depth = column[13]
            peak_acceleration = column[17]
            peak_displacement = column[19]
            rw_arrival_time = column[5] #this is the arrival time of rayleigh wave
            peak_ground_velocity = column[15] # this is the peak ground velocity during the time of the earthquake.
            predicted_peak_ground_velocity = column[7]
            predicted_peak_ground_velocity_list.append(float(predicted_peak_ground_velocity))
            # The next function is designed to to take a list and find the first item in that list that matches the conditions. If an item is not in the list that matches the condition it is possible to set a default value which will prevent the program from raising an error otherwise.
            #H1_lock_time = next((item for item in H1_lock_time_list if min(H1_lock_time_list, key=lambda x:abs(x-float(pw_arrival_time)))),[None])
            #H1_lockloss_time = next((item for item in H1_lockloss_time_list if min(H1_lockloss_time_list, key=lambda x:abs(x-float(float(pw_arrival_time)+float(options.time_after_p_wave))))),[None])
            H1_lock_time = min(H1_lock_time_list, key=lambda x:abs(x-float(pw_arrival_time)))
            H1_lockloss_time = min(H1_lockloss_time_list, key=lambda x:abs(x-float(float(pw_arrival_time) + float(options.time_after_p_wave))))
            lockloss = ""
            if (H1_lock_time <= float(pw_arrival_time) and H1_lockloss_time <= float(float(pw_arrival_time) + float(options.time_after_p_wave))): # The if statements are designed to check if the interferometer is in lock or not and if it is. Did it lose lock around the time of the earthquake? 
                lockloss = "Y"
                resultfileH1.write('{0:^20} {1:^20} {2:^20} {3:^20} {4:^20} {5:^20} {6:^20} {7:^20} {8:^20} \n'.format(eq_time,pw_arrival_time,peak_ground_velocity,peak_acceleration,peak_displacement,eq_mag,eq_distance,eq_depth,lockloss))
            elif (H1_lock_time <= float(pw_arrival_time) and H1_lockloss_time > float(float(pw_arrival_time) + float(options.time_after_p_wave))):
                lockloss = "N"
                resultfileH1.write('{0:^20} {1:^20} {2:^20} {3:^20} {4:^20} {5:^20} {6:^20} {7:^20} {8:^20} \n'.format(eq_time,pw_arrival_time,peak_ground_velocity,peak_acceleration,peak_displacement,eq_mag,eq_distance,eq_depth,lockloss))
            elif (H1_lock_time > float(pw_arrival_time)):
                lockloss = "Z"
                resultfileH1.write('{0:^20} {1:^20} {2:^20} {3:^20} {4:^20} {5:^20} {6:^20} {7:^20} {8:^20} \n'.format(eq_time,pw_arrival_time,peak_ground_velocity,peak_acceleration,peak_displacement,eq_mag,eq_distance,eq_depth,lockloss))
    datafileH1.close()
    resultfileH1.close()
    H1_channel_lockstatus_data.close()
    eq_time_list = []
    locklosslist = []
    pw_arrival_list = []
    peak_acceleration_list = []
    peak_displacement_list = []
    eq_mag_list = []
    eq_distance_list = []
    eq_depth_list = []
    
    resultfileplotH1 = open('/home/eric.coughlin/gitrepo/seismon/RfPrediction/data/LHO_lockstatus_{0}{1}.txt'.format(rms_toggle, direction), 'r')
    for item in (line.strip().split() for line in resultfileplotH1):
        eq_time = item[0]
        pw_arrival = item[1]
        peakgroundvelocity = item[2]
        peak_acceleration = item[3]
        peak_displacement = item[4]
        eq_mag = item[5]
        eq_distance = item[6]
        eq_depth = item[7]
        lockloss = item[8]
        H1_peak_ground_velocity_list.append(float(peakgroundvelocity))
        locklosslist.append(lockloss)
        eq_time_list.append(eq_time)
        pw_arrival_list.append(pw_arrival)
        peak_acceleration_list.append(peak_acceleration)
        peak_displacement_list.append(peak_displacement)
        eq_mag_list.append(eq_mag)
        eq_distance_list.append(eq_distance)
        eq_depth_list.append(eq_depth)
    
    H1_binary_file = open('/home/eric.coughlin/gitrepo/seismon/RfPrediction/data/LHO_O1_binary_{0}{1}.txt'.format(direction, rms_toggle), 'w')

    for eq_time, pw_arrival, peakgroundvelocity, peak_acceleration,peak_displacement,eq_mag,eq_distance,eq_depth, lockloss in zip(eq_time_list, pw_arrival_list, H1_peak_ground_velocity_list, peak_acceleration_list,peak_displacement_list,eq_mag_list,eq_distance_list,eq_depth_list, locklosslist):
        if lockloss == "Y":
            lockloss_binary = '1'
            H1_binary_file.write('{0:^20} {1:^20} {2:^20} {3:^20} {4:^20} {5:^20} {6:^20} {7:^20} {8:^20} \n'.format(eq_time,pw_arrival,peakgroundvelocity,peak_acceleration,peak_displacement,eq_mag,eq_distance,eq_depth,lockloss_binary))
        elif lockloss == "N":
            lockloss_binary = '0'
            H1_binary_file.write('{0:^20} {1:^20} {2:^20} {3:^20} {4:^20} {5:^20} {6:^20} {7:^20} {8:^20} \n'.format(eq_time,pw_arrival,peakgroundvelocity,peak_acceleration,peak_displacement,eq_mag,eq_distance,eq_depth,lockloss_binary))
        else:
            pass
    H1_binary_file.close()

    locklosslistZ = []
    locklosslistY = []
    locklosslistN = []
    eq_time_list_Z = []
    eq_time_list_N = []
    eq_time_list_Y = []
    H1_peak_ground_velocity_list_Z = []
    H1_peak_ground_velocity_list_N = []
    H1_peak_ground_velocity_list_Y = []
    peak_ground_acceleration_list_Z = []
    peak_ground_acceleration_list_N = []
    peak_ground_acceleration_list_Y = []
    H1_peak_ground_velocity_sorted_list, locklosssortedlist, predicted_peak_ground_velocity_sorted_list = (list(t) for t in zip(*sorted(zip(H1_peak_ground_velocity_list, locklosslist, predicted_peak_ground_velocity_list))))
    num_lock_list = []
    YN_peak_list = []
    for sortedpeak, sortedlockloss in zip(H1_peak_ground_velocity_sorted_list, locklosssortedlist):
        if sortedlockloss == "Y":
            YN_peak_list.append(sortedpeak)
            num_lock_list.append(1)
        elif sortedlockloss == "N":
            YN_peak_list.append(sortedpeak)
            num_lock_list.append(0)
    num_lock_prob_cumsum = np.divide(np.cumsum(num_lock_list), np.cumsum(np.ones(len(num_lock_list))))

    f, axarr = plt.subplots(1)
    for t,time,peak, peak_acc, lockloss in zip(range(len(eq_time_list)),eq_time_list,H1_peak_ground_velocity_list,peak_acceleration_list,locklosslist):
            if lockloss == "Z":
                eq_time_list_Z.append(t)
                H1_peak_ground_velocity_list_Z.append(peak)
                locklosslistZ.append(lockloss)
                peak_ground_acceleration_list_Z.append(peak_acc)
            elif lockloss == "N":
                eq_time_list_N.append(t)
                H1_peak_ground_velocity_list_N.append(peak)
                locklosslistN.append(lockloss)
                peak_ground_acceleration_list_N.append(peak_acc)
            elif lockloss == "Y":
                eq_time_list_Y.append(t)
                H1_peak_ground_velocity_list_Y.append(peak)
                locklosslistY.append(lockloss)
                peak_ground_acceleration_list_Y.append(peak_acc)
    axarr.plot(eq_time_list_N, H1_peak_ground_velocity_list_N, 'go', label='locked at earthquake(eq)')
    axarr.plot(eq_time_list_Y, H1_peak_ground_velocity_list_Y, 'ro', label='lockloss at earthquake(eq)')
    axarr.set_title('H1 Lockstatus Plot')
    axarr.set_yscale('log')
    axarr.set_xlabel('earthquake count(eq)')
    axarr.set_ylabel('peak ground velocity(m/s)')
    axarr.legend(loc='best')
    #f.savefig('/home/eric.coughlin/public_html/lockloss_threshold_plots/LHO/lockstatus_LHO_{0}{1}.png'.format(rms_toggle, direction))
    f.savefig('/home/eric.coughlin/gitrepo/seismon/RfPrediction/plots/lockstatus_LHO_{0}{1}.png'.format(rms_toggle, direction))
    plt.figure(2)
    plt.plot(eq_time_list_N, peak_ground_acceleration_list_N, 'go', label='locked at earthquake(eq)')
    plt.plot(eq_time_list_Y, peak_ground_acceleration_list_Y, 'ro', label='lockloss at earthquake(eq)')
    plt.title('H1 Lockstatus Plot(acceleration)')
    plt.yscale('log')
    plt.xlabel('earthquake count(eq)')
    plt.ylabel('peak ground acceleration(m/s)')
    plt.legend(loc='best')
    plt.savefig('/home/eric.coughlin/public_html/lockstatus_acceleration_LHO_{0}{1}.png'.format(rms_toggle, direction))
    #plt.savefig('/home/eric.coughlin/gitrepo/seismon/RfPrediction/plots/lockstatus_acceleration_LHO_{0}{1}.png'.format(rms_toggle, direction))
    plt.clf()
    plt.figure(3)
    plt.plot(H1_peak_ground_velocity_sorted_list, predicted_peak_ground_velocity_sorted_list, 'o', label='actual vs predicted')
    plt.title('H1 actual vs predicted ground velocity')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('peak ground velocity(m/s)')
    plt.ylabel('predicted peak ground velocity(m/s)')
    plt.legend(loc='best')
    #plt.savefig('/home/eric.coughlin/public_html/lockloss_threshold_plots/LHO/check_prediction_LHO_{0}{1}.png'.format(rms_toggle, direction))
    plt.savefig('/home/eric.coughlin/gitrepo/seismon/RfPrediction/plots/check_prediction_LHO_{0}{1}.png'.format(rms_toggle, direction))
    plt.clf()
    
    threshold_file_H1 = open('/home/eric.coughlin/gitrepo/seismon/RfPrediction/data/threshhold_data_{0}{1}.txt'.format(rms_toggle, direction), 'w')
    num_of_lockloss = len(locklosslistY)
    total_lockstatus = num_of_lockloss + len(locklosslistN)
    total_lockstatus_all = num_of_lockloss + len(locklosslistN) + len(locklosslistZ)
    total_percent_lockloss = num_of_lockloss / total_lockstatus
    threshold_file_H1.write('The percentage of total locklosses is {0}% \n'.format(total_percent_lockloss * 100))
    threshold_file_H1.write('The total number of earthquakes is {0}. \n'.format(total_lockstatus_all))

    eqcount_50 = 0
    eqcount_75 = 0
    eqcount_90 = 0
    eqcount_95 = 0
    for item, thing in zip(num_lock_prob_cumsum, YN_peak_list):
        if item >= .5:
            eqcount_50 = eqcount_50 + 1
        if item >= .75:
            eqcount_75 = eqcount_75 + 1
        if item >= .9:
            eqcount_90 = eqcount_90 + 1
        if item >= .95:
            eqcount_95 = eqcount_95 + 1
    threshold_file_H1.write('The number of earthquakes above 50 percent is {0}. \n'.format(eqcount_50))
    threshold_file_H1.write('The number of earthquakes above 75 percent is {0}. \n'.format(eqcount_75))
    threshold_file_H1.write('The number of earthquakes above 90 percent is {0}. \n'.format(eqcount_90))
    threshold_file_H1.write('The number of earthquakes above 95 percent is {0}. \n'.format(eqcount_95))

    probs = [0.5, 0.75, 0.9, 0.95]
    num_lock_prob_cumsum_sort = np.unique(num_lock_prob_cumsum)
    YN_peak_list_sort = np.unique(YN_peak_list)
    num_lock_prob_cumsum_sort, YN_peak_list_sort = zip(*sorted(zip(num_lock_prob_cumsum_sort, YN_peak_list_sort)))
    thresholdsf = interp1d(num_lock_prob_cumsum_sort,YN_peak_list_sort, bounds_error=False)
    for item in probs:
        threshold = thresholdsf(item)
        threshold_file_H1.write('The threshhold at {0}% is {1}(m/s) \n'.format(item * 100, threshold))
    threshold_file_H1.write('The number of times of locklosses is {0}. \n'.format(len(locklosslistY)))
    threshold_file_H1.write('The number of times of no locklosses is {0}. \n'.format(len(locklosslistN)))
    threshold_file_H1.write('The number of times of not locked is {0}. \n'.format(len(locklosslistZ)))
    threshold_file_H1.close()
    plt.figure(4)
    plt.plot(YN_peak_list_sort, num_lock_prob_cumsum_sort, 'kx', label='probability of lockloss')
    plt.title('H1 Lockloss Probability')
    plt.xscale('log')
    plt.grid(True)
    plt.xlabel('peak ground velocity (m/s)')
    plt.ylabel('Lockloss Probablity')
    plt.legend(loc='best')
    #plt.savefig('/home/eric.coughlin/public_html/lockloss_threshold_plots/LHO/lockloss_probablity_LHO_{0}{1}.png'.format(rms_toggle, direction))
    plt.savefig('/home/eric.coughlin/gitrepo/seismon/RfPrediction/plots/lockloss_probablity_LHO_{0}{1}.png'.format(rms_toggle, direction))
    plt.clf()


## LLO
os.system('mkdir -p /home/eric.coughlin/L1O1/')
os.system('mkdir -p /home/eric.coughlin/public_html/lockloss_threshold_plots/LLO/')
for direction in ['Z','X','Y']:
    if rms_toggle == "":
        channel = 'L1:ISI-GND_STS_HAM2_{0}_DQ'.format(direction)
    elif rms_toggle == "RMS_":
        channel = 'L1:ISI-GND_STS_HAM5_{0}_BLRMS_30M_100M'.format(direction)
    L1_lock_time_list = []
    L1_lockloss_time_list = []
    options = parse_commandline()
    predicted_peak_ground_velocity_list = []
    H1_peak_ground_velocity_list =[]
    datafileL1 = open('/home/eric.coughlin/gitrepo/seismon/RfPrediction/data/LLO_O1_{0}{1}.txt'.format(rms_toggle, direction), 'r')
    resultfileL1 = open('/home/eric.coughlin/gitrepo/seismon/RfPrediction/data/LLO_lockstatus_{0}{1}.txt'.format(rms_toggle, direction), 'w')
    L1_channel_lockstatus_data = open('/home/eric.coughlin/gitrepo/seismon/RfPrediction/data/segs_Locked_L_1126569617_1136649617.txt', 'r')
    for item in (line.strip().split() for line in L1_channel_lockstatus_data):
        L1_lock_time = item[0]
        L1_lockloss_time = item[1]
        L1_lock_time_list.append(float(L1_lock_time))
        L1_lockloss_time_list.append(float(L1_lockloss_time))
    #resultfileL1.write('{0:^20} {1:^20} {2:^20} {3:^20} \n'.format('eq arrival time','pw arrival time','peak ground velocity','lockloss'))
    for column in ( line.strip().split() for line in datafileL1):
            eq_time = column[0]
            eq_mag = column[1]
            pw_arrival_time = column[2]
            sw_arrival_time = column[3]
            eq_distance = column[12]
            eq_depth = column[13]
            peak_acceleration = column[17]
            peak_displacement = column[19]
            rw_arrival_time = column[5]
            peak_ground_velocity = column[15]
            predicted_peak_ground_velocity = column[7]
            predicted_peak_ground_velocity_list.append(float(predicted_peak_ground_velocity))
            #L1_lock_time = next((item for item in L1_lock_time_list if item <= float(pw_arrival_time)),[None])
            #L1_lockloss_time = next((item for item in L1_lockloss_time_list if item <= float(float(pw_arrival_time) + float(options.time_after_p_wave))),[None])
            L1_lock_time = min(L1_lock_time_list, key=lambda x:abs(x-float(pw_arrival_time)))
            L1_lockloss_time = min(L1_lockloss_time_list, key=lambda x:abs(x-float(float(pw_arrival_time) + float(options.time_after_p_wave))))
            lockloss = ""
            if (L1_lock_time <= float(pw_arrival_time) and L1_lockloss_time <= float(float(pw_arrival_time) + float(options.time_after_p_wave))):
                lockloss = "Y"
                resultfileL1.write('{0:^20} {1:^20} {2:^20} {3:^20} {4:^20} {5:^20} {6:^20} {7:^20} {8:^20} \n'.format(eq_time,pw_arrival_time,peak_ground_velocity,peak_acceleration,peak_displacement,eq_mag,eq_distance,eq_depth,lockloss))
            elif (L1_lock_time <= float(pw_arrival_time) and L1_lockloss_time > float(float(pw_arrival_time) + float(options.time_after_p_wave))):
                lockloss = "N"
                resultfileL1.write('{0:^20} {1:^20} {2:^20} {3:^20} {4:^20} {5:^20} {6:^20} {7:^20} {8:^20} \n'.format(eq_time,pw_arrival_time,peak_ground_velocity,peak_acceleration,peak_displacement,eq_mag,eq_distance,eq_depth,lockloss))
            elif (L1_lock_time > float(pw_arrival_time)):
                lockloss = "Z"
                resultfileL1.write('{0:^20} {1:^20} {2:^20} {3:^20} {4:^20} {5:^20} {6:^20} {7:^20} {8:^20} \n'.format(eq_time,pw_arrival_time,peak_ground_velocity,peak_acceleration,peak_displacement,eq_mag,eq_distance,eq_depth,lockloss))
    datafileL1.close()
    resultfileL1.close()
    L1_channel_lockstatus_data.close()
    eq_time_list = []
    locklosslist = []
    pw_arrival_list = []
    peak_acceleration_list =[]
    peak_displacement_list = []
    eq_mag_list = []
    eq_distance_list = []
    eq_depth_list = []
    resultfileplotL1 = open('/home/eric.coughlin/gitrepo/seismon/RfPrediction/data/LLO_lockstatus_{0}{1}.txt'.format(rms_toggle, direction), 'r')
    for item in (line.strip().split() for line in resultfileplotL1):
        eq_time = item[0]
        pw_arrival = item[1]
        peakgroundvelocity = item[2]
        peak_acceleration = item[3]
        peak_displacement = item[4]
        eq_mag = item[5]
        eq_distance = item[6]
        eq_depth = item[7]
        lockloss = item[8]
        H1_peak_ground_velocity_list.append(float(peakgroundvelocity))
        locklosslist.append(lockloss)
        eq_time_list.append(eq_time)
        pw_arrival_list.append(pw_arrival)
        peak_acceleration_list.append(peak_acceleration)
        peak_displacement_list.append(peak_displacement)
        eq_mag_list.append(eq_mag)
        eq_distance_list.append(eq_distance)
        eq_depth_list.append(eq_depth)

    L1_binary_file = open('/home/eric.coughlin/gitrepo/seismon/RfPrediction/data/LLO_O1_binary_{0}{1}.txt'.format(rms_toggle, direction), 'w')
    for eq_time, pw_arrival, peakgroundvelocity, peak_acceleration,peak_displacement,eq_mag,eq_distance,eq_depth, lockloss in zip(eq_time_list, pw_arrival_list, H1_peak_ground_velocity_list, peak_acceleration_list,peak_displacement_list,eq_mag_list,eq_distance_list,eq_depth_list, locklosslist):
        if lockloss == "Y":
            lockloss_binary = '1'
            L1_binary_file.write('{0:^20} {1:^20} {2:^20} {3:^20} {4:^20} {5:^20} {6:^20} {7:^20} {8:^20} \n'.format(eq_time,pw_arrival,peakgroundvelocity,peak_acceleration,peak_displacement,eq_mag,eq_distance,eq_depth,lockloss_binary))
        elif lockloss == "N":
            lockloss_binary = '0'
            L1_binary_file.write('{0:^20} {1:^20} {2:^20} {3:^20} {4:^20} {5:^20} {6:^20} {7:^20} {8:^20} \n'.format(eq_time,pw_arrival,peakgroundvelocity,peak_acceleration,peak_displacement,eq_mag,eq_distance,eq_depth,lockloss_binary))
        else:
            pass
    L1_binary_file.close()

    locklosslistZ = []
    locklosslistY = []
    locklosslistN = []
    eq_time_list_Z = []
    eq_time_list_N = []
    eq_time_list_Y = []
    H1_peak_ground_velocity_list_Z = []
    H1_peak_ground_velocity_list_N = []
    H1_peak_ground_velocity_list_Y = []
    peak_ground_acceleration_list_Z = []
    peak_ground_acceleration_list_N = []
    peak_ground_acceleration_list_Y = []
    H1_peak_ground_velocity_sorted_list, locklosssortedlist, predicted_peak_ground_velocity_sorted_list = (list(t) for t in zip(*sorted(zip(H1_peak_ground_velocity_list, locklosslist, predicted_peak_ground_velocity_list))))
    num_lock_list = []
    YN_peak_list = []
    for sortedpeak, sortedlockloss in zip(H1_peak_ground_velocity_sorted_list, locklosssortedlist):
        if sortedlockloss == "Y":
            YN_peak_list.append(sortedpeak)
            num_lock_list.append(1)
        elif sortedlockloss == "N":
            YN_peak_list.append(sortedpeak)
            num_lock_list.append(0)
    num_lock_prob_cumsum = np.cumsum(num_lock_list) / np.cumsum(np.ones(len(num_lock_list)))

    plt.figure(8)
    for t,time,peak,peak_acc,lockloss in zip(range(len(eq_time_list)),eq_time_list,H1_peak_ground_velocity_list,peak_acceleration_list,locklosslist):
            if lockloss == "Z":
                eq_time_list_Z.append(t)
                H1_peak_ground_velocity_list_Z.append(peak)
                locklosslistZ.append(lockloss)
                peak_ground_acceleration_list_Z.append(peak_acc)
            elif lockloss == "N":
                eq_time_list_N.append(t)
                H1_peak_ground_velocity_list_N.append(peak)
                locklosslistN.append(lockloss)
                peak_ground_acceleration_list_N.append(peak_acc)
            elif lockloss == "Y":
                eq_time_list_Y.append(t)
                H1_peak_ground_velocity_list_Y.append(peak)
                locklosslistY.append(lockloss)
                peak_ground_acceleration_list_Y.append(peak_acc)
    plt.plot(eq_time_list_N, H1_peak_ground_velocity_list_N, 'go', label='locked at earthquake(eq)')
    plt.plot(eq_time_list_Y, H1_peak_ground_velocity_list_Y, 'ro', label='lockloss at earthquake(eq)')
    plt.title('L1 Lockstatus Plot')
    plt.yscale('log')
    plt.xlabel('earthquake count(eq)')
    plt.ylabel('peak ground velocity(m/s)')
    plt.legend(loc='best')
    #plt.savefig('/home/eric.coughlin/public_html/lockloss_threshold_plots/LLO/lockstatus_LLO_{0}{1}.png'.format(rms_toggle, direction))
    plt.savefig('/home/eric.coughlin/gitrepo/seismon/RfPrediction/plots/lockstatus_LLO_{0}{1}.png'.format(rms_toggle, direction))
    plt.clf()
    plt.figure(23)
    plt.plot(eq_time_list_N, peak_ground_acceleration_list_N, 'go', label='locked at earthquake(eq)')
    plt.plot(eq_time_list_Y, peak_ground_acceleration_list_Y, 'ro', label='lockloss at earthquake(eq)')
    plt.title('H1 Lockstatus Plot(acceleration)')
    plt.yscale('log')
    plt.xlabel('earthquake count(eq)')
    plt.ylabel('peak ground acceleration(m/s)')
    plt.legend(loc='best')
    plt.savefig('/home/eric.coughlin/public_html/lockstatus_acceleration_LHO_{0}{1}.png'.format(rms_toggle, direction))
    #plt.savefig('/home/eric.coughlin/gitrepo/seismon/RfPrediction/plots/lockstatus_acceleration_LHO_{0}{1}.png'.format(rms_toggle, direction))
    plt.clf()

    plt.figure(9)
    plt.plot(H1_peak_ground_velocity_list, predicted_peak_ground_velocity_list, 'o', label='actual vs predicted')
    plt.title('L1 actual vs predicted ground velocity')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('peak ground velocity(m/s)')
    plt.ylabel('predicted peak ground velocity(m/s)')
    plt.legend(loc='best')
    #plt.savefig('/home/eric.coughlin/public_html/lockloss_threshold_plots/LLO/check_predictionLLO_{0}{1}.png'.format(rms_toggle, direction))
    plt.savefig('/home/eric.coughlin/gitrepo/seismon/RfPrediction/plots/check_predictionLLO_{0}{1}.png'.format(rms_toggle, direction))
    plt.clf()

    threshold_file_L1 = open('/home/eric.coughlin/gitrepo/seismon/RfPrediction/data/threshhold_data_{0}{1}.txt'.format(rms_toggle, direction), 'w')
    num_of_lockloss = len(locklosslistY)
    total_lockstatus = num_of_lockloss + len(locklosslistN)
    total_lockstatus_all = num_of_lockloss + len(locklosslistN) + len(locklosslistZ)
    total_percent_lockloss = num_of_lockloss / total_lockstatus
    threshold_file_L1.write('The percentage of total locklosses is {0}% \n'.format(total_percent_lockloss * 100))
    threshold_file_L1.write('The total number of earthquakes is {0}. \n'.format(total_lockstatus_all))

    eqcount_50 = 0
    eqcount_75 = 0
    eqcount_90 = 0
    eqcount_95 = 0
    for item, thing in zip(num_lock_prob_cumsum, YN_peak_list):
        if item >= .5:
            eqcount_50 = eqcount_50 + 1
        if item >= .75:
            eqcount_75 = eqcount_75 + 1
        if item >= .9:
            eqcount_90 = eqcount_90 + 1
        if item >= .95:
            eqcount_95 = eqcount_95 + 1
    threshold_file_L1.write('The number of earthquakes above 50 percent is {0}. \n'.format(eqcount_50))
    threshold_file_L1.write('The number of earthquakes above 75 percent is {0}. \n'.format(eqcount_75))
    threshold_file_L1.write('The number of earthquakes above 90 percent is {0}. \n'.format(eqcount_90))
    threshold_file_L1.write('The number of earthquakes above 95 percent is {0}. \n'.format(eqcount_95))

    probs = [0.5, 0.75, 0.9, 0.95]
    num_lock_prob_cumsum_sort = np.unique(num_lock_prob_cumsum)
    YN_peak_list_sort = np.unique(YN_peak_list)
    num_lock_prob_cumsum_sort, YN_peak_list_sort = zip(*sorted(zip(num_lock_prob_cumsum_sort, YN_peak_list_sort)))
    thresholds = []
    thresholdsf = interp1d(num_lock_prob_cumsum_sort,YN_peak_list_sort,bounds_error=False)
    for item in probs:
        threshold = thresholdsf(item)
        threshold_file_L1.write('The threshhold at {0}% is {1}(m/s) \n'.format(item * 100, threshold))
    threshold_file_L1.write('The number of times of locklosses is {0}. \n'.format(len(locklosslistY)))
    threshold_file_L1.write('The number of times of no locklosses is {0}. \n'.format(len(locklosslistN)))
    threshold_file_L1.write('The number of times of not locked is {0}. \n'.format(len(locklosslistZ)))
    threshold_file_L1.close()
    
    plt.figure(10)
    plt.plot(YN_peak_list_sort, num_lock_prob_cumsum_sort, 'kx', label='probability of lockloss')
    plt.title('L1 Lockloss Probability')
    plt.xscale('log')
    plt.grid(True)
    plt.xlabel('peak ground velocity (m/s)')
    plt.ylabel('Lockloss Probablity')
    plt.legend(loc='best')
    #plt.savefig('/home/eric.coughlin/public_html/lockloss_threshold_plots/LLO/lockloss_probablity_LLO_{0}{1}.png'.format(rms_toggle, direction))
    plt.savefig('/home/eric.coughlin/gitrepo/seismon/RfPrediction/plots/lockloss_probablity_LLO_{0}{1}.png'.format(rms_toggle, direction))
    plt.clf()
