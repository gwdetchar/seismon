#!/usr/bin/env python3
#
# code to read the output from 'seismon' using datebase for input IFO 
# then output the data as epics channels to an EPICS IOC via channel access
#
# The 'seismon' database has a table of earthquakes, and a table of per-IFO predictions from the earthquake table.
#
# First, pull all the predictions for the IFO, sorted by P-wave date/time
# For each such, get the earthquake record for it. 
# Pull out the magnitude (earthquake), predicted ground velocity (prediction)
# Save these in arrays
# Now loop through predictions in the stored arrays (still in P-wave date/time order)
# Select the first five with magnitude above cutoff and ground velocity above cutoff
#
#  Input parameters
#  -I <ifo>    (default LLO)
#  -C <dbconfig.file>  (default input/config.yaml)
#
# 2021-01-08 - K. Thorne
# 2021-02-02 - K. Thorne - change to get prediction first, then earthquake
# 2022-05-11 - K. Thorne - change prediction table column names

from datetime import datetime, date
import simplejson as json
import enum
import os
import time
import copy
import configparser
import seismon
import os.path
import numpy as np
import csv
from seismon.eqmon import distance_latlon

from astropy import table
from astropy import coordinates
from astropy import units as u
from astropy.time import Time, TimeDelta

from sqlalchemy import Table, create_engine, MetaData, asc, desc
from sqlalchemy.orm import create_session
from sqlalchemy.ext.declarative import declarative_base


# get path to seismon dir
seismonpath = os.path.dirname(seismon.__file__)
inputpath = os.path.join(seismonpath,'input')

# set up engine to connect to database
def engine_db(user, database, password=None, host=None, port=None):
    url = 'postgresql://{}:{}@{}:{}/{}'
    url = url.format(user, password or '', host or '', port or '', database)

    engine = create_engine(url, client_encoding='utf8')
    return engine

# timezones
from dateutil import tz

# PYEPICS
from epics import caput
import epics
epics.ca.context_create()

def write_epics(SITE,IFO,eqidx,eq_t,magn,lat,lng,depth,eqdist,rvel,p_arr,s_arr,r20_arr,r35_arr,r50_arr,
                c_lat, c_long, c_names):
# here we use the LIGO meaning of IFO and SITE
#  SITE is the observatory, IFO is the GWIC interferometer prefix

# convert the times to GPS for output
    eq_gps = float(eq_t.gps)
    p_arr_gps = float(p_arr.gps)
    s_arr_gps = float(s_arr.gps)
    r20_arr_gps = float(r20_arr.gps)
    r35_arr_gps = float(r35_arr.gps)
    r50_arr_gps = float(r50_arr.gps)

# Write data to EPICS Channels
    caput(IFO + ':SEI-SEISMON_EQ_TIME_GPS_%d'%eqidx, eq_gps)
    eq_utc = eq_t.to_datetime(timezone=tz.tzutc())
    eq_utc_str = str(eq_utc.strftime('%Y-%m-%d %H:%M:%S %Z'))
    caput(IFO + ':SEI-SEISMON_EQ_TIME_UTC_%d'%eqidx, eq_utc_str)          
    eq_ltz = eq_t.to_datetime(timezone=tz.tzlocal())
    eq_ltz_str = str(eq_ltz.strftime('%Y-%m-%d %H:%M:%S %Z'))
    caput(IFO + ':SEI-SEISMON_EQ_TIME_LTZ_%d'%eqidx, eq_ltz_str)
    caput(IFO + ':SEI-SEISMON_EQ_MAGNITUDE_MMS_%d'%eqidx, magn)
    caput(IFO + ':SEI-SEISMON_EQ_DEPTH_KM_%d'%eqidx, depth)
    caput(IFO + ':SEI-SEISMON_EQ_LATITUDE_DEG_%d'%eqidx, lat)
    caput(IFO + ':SEI-SEISMON_EQ_LONGITUDE_DEG_%d'%eqidx, lng)
    caput(IFO + ':SEI-SEISMON_EQ_' + SITE + '_DISTANCE_M_%d'%eqidx, eqdist)
    caput(IFO + ':SEI-SEISMON_' + SITE + '_P_ARRIVALTIME_GPS_%d'%eqidx, p_arr_gps)  
    caput(IFO + ':SEI-SEISMON_' + SITE + '_S_ARRIVALTIME_GPS_%d'%eqidx, s_arr_gps)
    caput(IFO + ':SEI-SEISMON_' + SITE + '_R20_ARRIVALTIME_GPS_%d'%eqidx, r20_arr_gps)
    caput(IFO + ':SEI-SEISMON_' + SITE + '_R35_ARRIVALTIME_GPS_%d'%eqidx, r35_arr_gps)
    caput(IFO + ':SEI-SEISMON_' + SITE + '_R50_ARRIVALTIME_GPS_%d'%eqidx, r50_arr_gps)
    caput(IFO + ':SEI-SEISMON_' + SITE + '_R20_VELOCITY_MPS_%d'%eqidx, rvel)
    caput(IFO + ':SEI-SEISMON_' + SITE + '_R35_VELOCITY_MPS_%d'%eqidx, rvel)
    caput(IFO + ':SEI-SEISMON_' + SITE + '_R50_VELOCITY_MPS_%d'%eqidx, rvel)

    # update program's GPS time
    now_t = Time.now()
    gpsnow = now_t.gps
    caput(IFO + ':SEI-SEISMON_SYSTEM_TIME_GPS_V2', gpsnow)

    # Calculate the time-to-arrive in minutes and seconds
    # All this folderol with multipliers is because mod arithmetic is funny
    # with negative numbers, so we strip the sign off and then put it back on
    p_arrival_time_delta = p_arr - now_t
    if p_arrival_time_delta.sec < 0:
        p_sign = -1
    else:
        p_sign = 1
    p_arrival_time_total_seconds = abs(p_arrival_time_delta.sec) 
    p_arrival_time_minutes = int(p_arrival_time_total_seconds)/60
    p_arrival_time_seconds = p_arrival_time_total_seconds%60
    # S
    s_arrival_time_delta = s_arr - now_t
    if s_arrival_time_delta.sec < 0:
        s_sign = -1
    else:
        s_sign = 1
    s_arrival_time_total_seconds = abs(s_arrival_time_delta.sec)
    s_arrival_time_minutes = int(s_arrival_time_total_seconds)/60
    s_arrival_time_seconds = s_arrival_time_total_seconds%60
    # R20
    r20_arrival_time_delta = r20_arr - now_t
    if r20_arrival_time_delta.sec < 0:
        r20_sign = -1
    else:
        r20_sign = 1
    r20_arrival_time_total_seconds = abs(r20_arrival_time_delta.sec)
    r20_arrival_time_minutes = int(r20_arrival_time_total_seconds)/60
    r20_arrival_time_seconds = r20_arrival_time_total_seconds%60

    # 35R
    r35_arrival_time_delta = r35_arr - now_t
    if r35_arrival_time_delta.sec < 0:
        r35_sign = -1
    else:
        r35_sign = 1
    r35_arrival_time_total_seconds = abs(r35_arrival_time_delta.sec)
    r35_arrival_time_minutes = int(r35_arrival_time_total_seconds)/60
    r35_arrival_time_seconds = r35_arrival_time_total_seconds%60

    # R50
    r50_arrival_time_delta = r50_arr - now_t
    if r50_arrival_time_delta.sec < 0:
        r50_sign = -1
    else:
        r50_sign = 1
    r50_arrival_time_total_seconds = abs(r50_arrival_time_delta.sec)
    r50_arrival_time_minutes = int(r50_arrival_time_total_seconds)/60
    r50_arrival_time_seconds = r50_arrival_time_total_seconds%60

    # P
    caput(IFO + ':SEI-SEISMON_' + SITE + '_P_ARRIVALTIME_TOTAL_SECS_%d'%eqidx, p_sign * p_arrival_time_total_seconds)
    caput(IFO + ':SEI-SEISMON_' + SITE + '_P_ARRIVALTIME_MINS_%d'%eqidx, p_sign * p_arrival_time_minutes)
    caput(IFO + ':SEI-SEISMON_' + SITE + '_P_ARRIVALTIME_SECS_%d'%eqidx, p_sign * p_arrival_time_seconds)

    # S
    caput(IFO + ':SEI-SEISMON_' + SITE + '_S_ARRIVALTIME_TOTAL_SECS_%d'%eqidx, s_sign * s_arrival_time_total_seconds)
    caput(IFO + ':SEI-SEISMON_' + SITE + '_S_ARRIVALTIME_MINS_%d'%eqidx, s_sign * s_arrival_time_minutes)
    caput(IFO + ':SEI-SEISMON_' + SITE + '_S_ARRIVALTIME_SECS_%d'%eqidx, s_sign * s_arrival_time_seconds)

    # R20
    caput(IFO + ':SEI-SEISMON_' + SITE + '_R20_ARRIVALTIME_TOTAL_SECS_%d'%eqidx, r20_sign * r20_arrival_time_total_seconds)
    caput(IFO + ':SEI-SEISMON_' + SITE + '_R20_ARRIVALTIME_MINS_%d'%eqidx, r20_sign * r20_arrival_time_minutes)
    caput(IFO + ':SEI-SEISMON_' + SITE + '_R20_ARRIVALTIME_SECS_%d'%eqidx, r20_sign * r20_arrival_time_seconds)

    # R35
    caput(IFO + ':SEI-SEISMON_' + SITE + '_R35_ARRIVALTIME_TOTAL_SECS_%d'%eqidx, r35_sign * r35_arrival_time_total_seconds)
    caput(IFO + ':SEI-SEISMON_' + SITE + '_R35_ARRIVALTIME_MINS_%d'%eqidx, r35_sign * r35_arrival_time_minutes)
    caput(IFO + ':SEI-SEISMON_' + SITE + '_R35_ARRIVALTIME_SECS_%d'%eqidx, r35_sign * r35_arrival_time_seconds)

    # R50
    caput(IFO + ':SEI-SEISMON_' + SITE + '_R50_ARRIVALTIME_TOTAL_SECS_%d'%eqidx, r50_sign * r50_arrival_time_total_seconds)
    caput(IFO + ':SEI-SEISMON_' + SITE + '_R50_ARRIVALTIME_MINS_%d'%eqidx, r50_sign * r50_arrival_time_minutes)
    caput(IFO + ':SEI-SEISMON_' + SITE + '_R50_ARRIVALTIME_SECS_%d'%eqidx, r50_sign * r50_arrival_time_seconds)

    # nearest country
    # just shortest distance to some point in the country, so not super accurate
    distances = distance_latlon(lat, lng, c_lat, c_long)
    idx = np.argmin(distances)
    country = c_names[idx]
    locationstr = f"{country}"
    truncloc = locationstr[:39]
    caput(f"{IFO}:SEI-SEISMON_EQ_LOCATION_{eqidx}", truncloc)

# run every two seconds
CYCLE_TIME = int(4)

# Set minimum magniture
MAGN_MIN = float(4.5)

# Set minimum velocity
RVEL_MIN = float(8.0e-8) 

# set number of entries in EPICS
MAX_COUNT = int(6)

# main routine
if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser()

    site = 'LLO'
    ifo = None

    # but check environment
    if 'SITE' in os.environ:
        site = os.environ['SITE']

    if 'IFO' in os.environ:
        ifo = os.environ['IFO']

    parser.add_argument('-I', '-s', '--site', default=site)
    parser.add_argument('-i', '--ifo', default=ifo)
    parser.add_argument('-C', '--config', default='input/config.yaml')

    args = parser.parse_args()

# read config file
    if not os.path.isfile(args.config):
        print('Missing config file: %s' % args.config)
        exit(1)

    config = configparser.ConfigParser()
    config.read(args.config)

# get ifo and identify site

    site = args.site
    ifo = args.ifo

    if ifo is None:
        if (site == 'LLO'):
            ifo = 'L1'
        elif (site == 'LHO'):
            ifo = 'H1'
        elif (site == 'VIRGO'):
            ifo = 'V1'
        elif (site == 'KAGRA'):
            ifo = 'K1'
        elif (site == 'GEO'):
            ifo = 'G1'
        elif (site == 'TST'):
            ifo = 'X2'
        else:
            ifo = "_1"

# initialize counters
    counter = 0
    uptime = 0

# connect to the database
    engine = engine_db(config['database']['user'],
                   config['database']['database'],
                   password=config['database']['password'],
                   host=config['database']['host'],
                   port=config['database']['port'])

# get tables from metadata
    metadata = MetaData(bind=engine)

#Reflect each database table we need to use, using metadata
    Base = declarative_base()
    class Earthquake(Base):
        __table__ = Table('earthquakes', metadata, autoload=True)

    class Prediction(Base):
        __table__ = Table('predictions', metadata, autoload=True)
#Create a session to use the tables    
    session = create_session(bind=engine)

    # open up and read in countries file
    countries_file = os.path.join(inputpath, "countries.csv")
    cntry_abbrev, cntry_latitudes, cntry_longitudes, cntry_names = [], [], [], []
    with open(countries_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for abb, lat, lon, country in reader:
            cntry_abbrev.append(abb)
            cntry_latitudes.append(float(lat))
            cntry_longitudes.append(float(lon))
            cntry_names.append(country)
    cntry_latitudes, cntry_longitudes = np.array(cntry_latitudes), np.array(cntry_longitudes)

# continuous loop
    while True:
 
# we will be selecting earthquakes based on limits on magniture and predictions based on limits on ground velocity
# so clear arrays of them
        rvels = []
        preds = []
        quakes = []
        mags = []
 
# get list of predictions for this IFO sorted by P.wave time
        prlist = session.query(Prediction).filter_by(ifo = site).order_by(desc(Prediction.p)).limit(200)

# loop over the predictions
        for pr in prlist:

# get earthquake for this prediction
            eq = session.query(Earthquake).filter_by(event_id = pr.event_id).first()

# Only keep things if not empty
            if eq is None:
                print('Prediction {} finds no matching earthquake record'.format(pr))
            else:
# update list of quakes, predictions, velocities
                quakes.append(eq)
                preds.append(pr)
                mags.append(float(eq.magnitude))
                rvels.append(float(pr.rfamp)*1.0e-6)

# -- we are delaying record processsing to minimize database changes between the earthquake and prediction query 

        num_eq = 0
        idx = -1
# loop through list
        for eq in quakes:

# first make sure we have minimum magnitude and minimum ground velocity
            idx = idx + 1
            this_magn = mags[idx]
            this_rvel = rvels[idx]
            # print(f"{idx} {Time(eq.date,format='datetime')}")
            if this_magn > MAGN_MIN or this_rvel > RVEL_MIN:
                pr = preds[idx]

# extract database quantities to variables
                eq_t = Time(eq.date,format='datetime')
                magn = float(eq.magnitude)
                lat = float(eq.lat)
                lng = float(eq.lon)
                depth = float(eq.depth)
                eqdist = float(pr.d)
                rvel = float(pr.rfamp)
                p_arr = Time(pr.p,format='datetime')
                s_arr = Time(pr.s,format='datetime')
                r20_arr = Time(pr.r2p0,format='datetime')
                r35_arr  = Time(pr.r3p5,format='datetime')
                r50_arr = Time(pr.r5p0,format='datetime')
                # print(f"time={eq_t} mag={magn} lat={lat} long={lng}")

# update EPICS channels from them
                write_epics(site,ifo,num_eq,eq_t,magn,lat,lng,depth,eqdist,rvel,p_arr,s_arr,r20_arr,r35_arr,r50_arr,
                            cntry_latitudes, cntry_longitudes, cntry_names)


# stop once we hit the limit
                num_eq = num_eq + 1
                if (num_eq >= MAX_COUNT ):
                    break 

        counter = counter + 1
        uptime = uptime + CYCLE_TIME
        caput(ifo + ':SEI-SEISMON_SYSTEM_UPTIME_SEC_V2', uptime)
        caput(ifo + ':SEI-SEISMON_SYSTEM_COUNTER_V2', counter)



        time.sleep(CYCLE_TIME)
