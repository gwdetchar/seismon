#!/usr/bin/env python
#
# code to read the output from seismon_info and write
# the data as epics channels to an EPICS IOC via
# channel access
#
# Data file consists on several lines, one per LIGO site.
# each line is space delimited and has the format
# Earthquake-GPS Magnitude P-wave-arrival S-wave-arrival R3.5-wave-arrival R3.5-wave-arrival R5.0-wave-arrival velocity depth latitude longitude distance ifo
#
# only the line ending with IFO is read
#
# D.Barker LHO 17nov2016
# K. Thorne - use stdenv, allow parameter for data directory

# Setup up CDS standard environment
import sys
sys.path.append('/ligo/cdscfg')
import stdenv as cds
cds.INIT_ENV()

print "Run this for site " + cds.SITE + " ifo " + cds.IFO
# Extract command-line parameters
# 'data directory'
datadir = sys.argv[1:1]
if len(datadir) == 0:
    datadir = "/seisapps/seisdata/all/" + cds.IFO + "O2"

print "seismon data directory is " + datadir

import os
if not os.path.isdir(datadir):
  sys.exit("data directory does not exist")
 
# Make sure it exists

# Setup GPS time
import glob

from gpstime import gpstime
import time
from datetime import datetime
from dateutil import tz

# PYEPICS
from epics import caput

DATADIRPREFIX = "us" # data file names start with this string

# run every two seconds
CYCLE_TIME = 2

counter = 0
uptime = 0

olddatafile = []

while True:
    # find latest file in datadir
  searchpath = datadir + "/*"
  print "searchpath %s" % searchpath
  sys.stdout.flush()
  globstring = glob.iglob(searchpath)
#    print "datadir globstring %s" % globstring
  newest_datadir = max(glob.iglob(searchpath), key=os.path.getctime)
  print "newest_datadir %s" % newest_datadir

# get name os usxxxxxxxx directory under the earthquakes subdirectory
  searchpath = newest_datadir + "/earthquakes/" + DATADIRPREFIX + "*"
  print "searchpath %s" % searchpath
  sys.stdout.flush()

#  First, make sure this is not empty.  If it is, add to uptime, continue
  testglob = glob.glob(searchpath)
  if testglob:

# if OK, get iterator
    globstring = glob.iglob(searchpath)

#    print "datadir globstring %s" % globstring
    newest_usdatadir = max(glob.iglob(searchpath), key=os.path.getctime)
 
    print "newest_usdatadir %s" % newest_usdatadir
    
    datafile = newest_usdatadir + "/earthquakes.txt"

    try:
        df = open(datafile,'r') 
    except IOError:
        print 'cannot open', datafile
    else:
        print "open datafile %s" % datafile
        olddatafile = datafile
        found_ifo_line = False
        for line in df:
            line = line.rstrip()
            if line.endswith(cds.IFO):
	        found_ifo_line = True
	        # decode the IFO line
	        #(eq_gps, eq_mag, p_arr, s_arr, r20_arr, r35_arr, r50_arr, rvel, depth, gps0, gps1, lat, long, dist, ifo) = line.split()
	        (eq_gps, eq_mag, p_arr, s_arr, r20_arr, r35_arr, r50_arr, rvel, gps0, gps1, lat, long, dist, ifo) = line.split()
	        eq_gps_f = float(eq_gps)
	        eq_mag_f = float(eq_mag)
	        p_arr_f = float(p_arr)
	        p_arr_i = int(p_arr_f)
	        s_arr_f = float(s_arr)
	        s_arr_i = int(s_arr_f)
	        r20_arr_f = float(r20_arr)
	        r20_arr_i = int(r20_arr_f)
	        r35_arr_f = float(r35_arr)
	        r35_arr_i = int(r35_arr_f)
	        r50_arr_f = float(r50_arr)
	        r50_arr_i = int(r50_arr_f)
	        rvel_f = float(rvel)
	        #depth_f = float(depth)
	        depth_f = 0.0
	        gps0_d = int(gps0)
	        gps1_d = int(gps1)
	        lat_f = float(lat)
	        long_f = float(long)
	        dist_f = float(dist)
	        #print "EQ_GPS %.1f, EQ_MAG %.1f EQ_LAT %.1f EQ_LONG %.1f EQ_DIST %.1f EQ_DEPTH %.1f PWAVE_TIME %.1f SWAVE_TIME %.1f R2.0_TIME %.1f R3.5_TIME %.1f R5.0_TIME %.1f RVEL %.1f" % \
	        (eq_gps_f, eq_mag_f, lat_f, long_f, dist_f, depth_f, p_arr_f, s_arr_f, r20_arr_f, r35_arr_f, r50_arr_f, rvel_f)
	        
	        # Write data to EPICS Channels, remember ezca appends H1: to channel name
	        caput(cds.IFO + ':CDS-SEISMON_EQ_TIME_GPS', eq_gps_f)
                # get GPS time in UTC, local
                eq_utc = gpstime.fromgps(eq_gps_f)
                eq_utc_str = eq_utc.strftime('%Y-%m-%d %H:%M:%S %Z')
	        caput(cds.IFO + ':CDS-SEISMON_EQ_TIME_UTC', eq_utc_str)               
                eq_ltz = eq_utc.astimezone(tz.tzlocal())
                eq_ltz_str = eq_ltz.strftime('%Y-%m-%d %H:%M:%S %Z') 
	        caput(cds.IFO + ':CDS-SEISMON_EQ_TIME_LTZ', eq_ltz_str)
	        caput(cds.IFO + ':CDS-SEISMON_EQ_MAGNITUDE_MMS', eq_mag_f)
	        caput(cds.IFO + ':CDS-SEISMON_EQ_DEPTH_KM', depth_f)
	        caput(cds.IFO + ':CDS-SEISMON_EQ_LATITUDE_DEG', lat_f)
	        caput(cds.IFO + ':CDS-SEISMON_EQ_LONGITUDE_DEG', long_f)
	        caput(cds.IFO + ':CDS-SEISMON_EQ_' + cds.SITE + '_DISTANCE_M', dist_f)

	        caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_P_ARRIVALTIME_GPS', p_arr_f)   # TESTING
		#p_arr_f = caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_P_ARRIVALTIME_GPS']    # TESTING
	        #p_arr_i = int(p_arr_f)                                 # TESTING
	        caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_S_ARRIVALTIME_GPS', s_arr_f)
		#s_arr_f = caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_S_ARRIVALTIME_GPS']    # TESTING
	        #s_arr_i = int(s_arr_f)                                 # TESTING
	        caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_R20_ARRIVALTIME_GPS', r20_arr_f)
		#r20_arr_f = caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_R20_ARRIVALTIME_GPS']# TESTING
	        #r20_arr_i = int(r20_arr_f)                             # TESTING
	        caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_R35_ARRIVALTIME_GPS', r35_arr_f)
		#r35_arr_f = caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_R35_ARRIVALTIME_GPS']# TESTING
	        #r35_arr_i = int(r35_arr_f)                             # TESTING
	        caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_R50_ARRIVALTIME_GPS', r50_arr_f)
		#r50_arr_f = caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_R50_ARRIVALTIME_GPS']# TESTING
	        #r50_arr_i = int(r50_arr_f)                             # TESTING
	        caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_R20_VELOCITY_MPS', rvel_f)
	        caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_R35_VELOCITY_MPS', rvel_f)
	        caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_R50_VELOCITY_MPS', rvel_f)
    
	        # update program's GPS time
	        gpsnow = int(gpstime.tconvert().gps())
	        caput(cds.IFO + ':CDS-SEISMON_SYSTEM_TIME_GPS', gpsnow)

		# Calculate the time-to-arrive in minutes and seconds
		# P
		p_arrival_time_total_seconds = p_arr_i - gpsnow
		caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_P_ARRIVALTIME_TOTAL_SECS', p_arrival_time_total_seconds)
		if p_arrival_time_total_seconds < 0:
		    multiplier = -1
		    p_arrival_time_total_seconds = p_arrival_time_total_seconds * -1
                else:
		    multiplier = 1
		p_arrival_time_minutes = int(p_arrival_time_total_seconds)/60
		p_arrival_time_seconds = p_arrival_time_total_seconds%60
		caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_P_ARRIVALTIME_MINS', multiplier * p_arrival_time_minutes)
		caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_P_ARRIVALTIME_SECS', multiplier * p_arrival_time_seconds)

		# S
		s_arrival_time_total_seconds = s_arr_i - gpsnow
		caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_S_ARRIVALTIME_TOTAL_SECS', s_arrival_time_total_seconds)
		if s_arrival_time_total_seconds < 0:
		    multiplier = -1
		    s_arrival_time_total_seconds = s_arrival_time_total_seconds * -1
                else:
		    multiplier = 1
		s_arrival_time_minutes = int(s_arrival_time_total_seconds)/60
		s_arrival_time_seconds = s_arrival_time_total_seconds%60
		caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_S_ARRIVALTIME_MINS', multiplier * s_arrival_time_minutes)
		caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_S_ARRIVALTIME_SECS', multiplier * s_arrival_time_seconds)

		# R20
		r20_arrival_time_total_seconds = r20_arr_i - gpsnow
		caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_R20_ARRIVALTIME_TOTAL_SECS', r20_arrival_time_total_seconds)
		if r20_arrival_time_total_seconds < 0:
		    multiplier = -1
		    r20_arrival_time_total_seconds = r20_arrival_time_total_seconds * -1
                else:
		    multiplier = 1
		r20_arrival_time_minutes = int(r20_arrival_time_total_seconds)/60
		r20_arrival_time_seconds = r20_arrival_time_total_seconds%60
		caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_R20_ARRIVALTIME_MINS', multiplier * r20_arrival_time_minutes)
		caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_R20_ARRIVALTIME_SECS', multiplier * r20_arrival_time_seconds)

		# R20
		r20_arrival_time_total_seconds = r20_arr_i - gpsnow
		caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_R20_ARRIVALTIME_TOTAL_SECS', r20_arrival_time_total_seconds)
		if r20_arrival_time_total_seconds < 0:
		    multiplier = -1
		    r20_arrival_time_total_seconds = r20_arrival_time_total_seconds * -1
                else:
		    multiplier = 1
		r20_arrival_time_minutes = int(r20_arrival_time_total_seconds)/60
		r20_arrival_time_seconds = r20_arrival_time_total_seconds%60
		caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_R20_ARRIVALTIME_MINS', multiplier * r20_arrival_time_minutes)
		caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_R20_ARRIVALTIME_SECS', multiplier * r20_arrival_time_seconds)

		# 35R
		r35_arrival_time_total_seconds = r35_arr_i - gpsnow
		caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_R35_ARRIVALTIME_TOTAL_SECS', r35_arrival_time_total_seconds)
		if r35_arrival_time_total_seconds < 0:
		    multiplier = -1
		    r35_arrival_time_total_seconds = r35_arrival_time_total_seconds * -1
                else:
		    multiplier = 1
		r35_arrival_time_minutes = int(r35_arrival_time_total_seconds)/60
		r35_arrival_time_seconds = r35_arrival_time_total_seconds%60
		caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_R35_ARRIVALTIME_MINS', multiplier * r35_arrival_time_minutes)
		caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_R35_ARRIVALTIME_SECS', multiplier * r35_arrival_time_seconds)

		# R50
		r50_arrival_time_total_seconds = r50_arr_i - gpsnow
		caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_R50_ARRIVALTIME_TOTAL_SECS', r50_arrival_time_total_seconds)
		if r50_arrival_time_total_seconds < 0:
		    multiplier = -1
		    r50_arrival_time_total_seconds = r50_arrival_time_total_seconds * -1
                else:
		    multiplier = 1
		r50_arrival_time_minutes = int(r50_arrival_time_total_seconds)/60
		r50_arrival_time_seconds = r50_arrival_time_total_seconds%60
		caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_R50_ARRIVALTIME_MINS', multiplier * r50_arrival_time_minutes)
		caput(cds.IFO + ':CDS-SEISMON_' + cds.SITE + '_R50_ARRIVALTIME_SECS', multiplier * r50_arrival_time_seconds)
                break
 
        df.close()
        if not found_ifo_line:
            print "Error - incomplete file. Does not contain a " + cds.IFO + " line"

  counter = counter +1
  uptime = uptime + CYCLE_TIME
  caput(cds.IFO + ':CDS-SEISMON_SYSTEM_UPTIME_SEC', uptime)
  caput(cds.IFO + ':CDS-SEISMON_SYSTEM_COUNTER', counter)
  time.sleep(CYCLE_TIME)

