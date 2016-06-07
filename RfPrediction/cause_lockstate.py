#!/usr/bin/python

import os, sys, glob, optparse, warnings, time, json
import numpy as np
import subprocess
from subprocess import Popen
from lxml import etree

import lal.gpstime

from gwpy.timeseries import TimeSeries

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

options = parse_commandline()
datafileH1 = open('/home/eric.coughlin/H101/earthquakes.txt', 'r')
resultfileH1 = open('/home/eric.coughlin/H101/organized_data.txt', 'w')
H1_channel_lockstatus_data = open('/home/eric.coughlin/gitrepo/seismon/RfPrediction/segs_Locked_H_1126569617_1136649617.txt', 'r')
resultfileH1.write('{0:^20} {1:^20} {2:^20} {3:^20} \n'.format('eq arrival time','pw arrival time','peak ground velocity','lockloss'))
for column in ( line.strip().split() for line in datafileH1):
    for item in (line.strip().split() for line in H1_channel_lockstatus_data):
            eq_time = column[0]
            pw_arrival_time = column[2]
            rw_arrival_time = column[5]
            H1_lock_time = item[0]
            H1_lockloss_time = item[1]
            peak_ground_velocity = column[14]
            lockloss = ""
    
            if float(H1_lock_time) <= float(pw_arrival_time) and float(H1_lockloss_time) <= float(options.time_after_p_wave):
                lockloss = "Y"
            elif float(H1_lock_time) <= float(pw_arrival_time) and float(H1_lockloss_time) > float(options.time_after_p_wave):
                lockloss = "N"
            else:
                lockloss = "Z"
            resultfileH1.write('{0:^20} {1:^20} {2:^20} {3:^20} \n'.format(eq_time,pw_arrival_time,peak_ground_velocity,lockloss))
            break
datafileH1.close()
resultfileH1.close() 
H1_channel_lockstatus_data.close()
