#!/usr/bin/env python

# EPICS IOC for seismon data
# K. Thorne - use stdenv, allow parameter for data directory
# E. von Reis - use environment instead of stdenv for IFO and SITE values

# Setup up CDS standard environment
from collections import namedtuple
import os

# create a named tuple to handle site info
# replaces cdscfg object no longer used.
# IFO and SITE come straight from the environment.
CDS = namedtuple("CDS", "SITE IFO")
cds = CDS(os.environ['SITE'], os.environ['IFO'])

print ("Run this for site " + cds.SITE + " ifo " + cds.IFO)

# create server
from pcaspy import Driver, SimpleServer
import time

prefix = cds.IFO+':CDS-SEISMON_'
pvdb = {
    'EQ_TIME_GPS' : {
        'prec' : 3,
    },
    'EQ_TIME_UTC' : {
        'type' : 'string',
    },
    'EQ_TIME_LTZ' : {
        'type' : 'string',
    },
    'EQ_MAGNITUDE_MMS' : {
        'prec' : 3,
    },
    'EQ_LATITUDE_DEG' : {
        'prec' : 3,
    },
    'EQ_LONGITUDE_DEG' : {
        'prec' : 3,
    },
    'EQ_DEPTH_KM' : {
        'prec' : 3,
    },
    'EQ_' + cds.SITE + '_DISTANCE_M' : {
        'prec' : 3,
    },
    cds.SITE + '_P_ARRIVALTIME_GPS' : {
        'prec' : 3,
    },
    cds.SITE + '_S_ARRIVALTIME_GPS' : {
        'prec' : 3,
    },
    cds.SITE + '_R20_ARRIVALTIME_GPS' : {
        'prec' : 3,
    },
    cds.SITE + '_R35_ARRIVALTIME_GPS' : {
        'prec' : 3,
    },
    cds.SITE + '_R50_ARRIVALTIME_GPS' : {
        'prec' : 3,
    },
    cds.SITE + '_P_VELOCITY_MPS' : {
        'prec' : 3,
    },
    cds.SITE + '_S_VELOCITY_MPS' : {
        'prec' : 3,
    },
    cds.SITE + '_R20_VELOCITY_MPS' : {
        'prec' : 3,
    },
    cds.SITE + '_R35_VELOCITY_MPS' : {
        'prec' : 3,
    },
    cds.SITE + '_R50_VELOCITY_MPS' : {
        'prec' : 3,
    },
    cds.SITE + '_P_ARRIVALTIME_MINS' : {
        'prec' : 0,
    },
    cds.SITE + '_P_ARRIVALTIME_SECS' : {
        'prec' : 0,
    },
    cds.SITE + '_P_ARRIVALTIME_TOTAL_SECS' : {
        'prec' : 0,
    },
    cds.SITE + '_S_ARRIVALTIME_MINS' : {
        'prec' : 0,
    },
    cds.SITE + '_S_ARRIVALTIME_SECS' : {
        'prec' : 0,
    },
    cds.SITE + '_S_ARRIVALTIME_TOTAL_SECS' : {
        'prec' : 0,
    },
    cds.SITE + '_R20_ARRIVALTIME_MINS' : {
        'prec' : 0,
    },
    cds.SITE + '_R20_ARRIVALTIME_SECS' : {
        'prec' : 0,
    },
    cds.SITE + '_R20_ARRIVALTIME_TOTAL_SECS' : {
        'prec' : 0,
    },
    cds.SITE + '_R35_ARRIVALTIME_MINS' : {
        'prec' : 0,
    },
    cds.SITE + '_R35_ARRIVALTIME_SECS' : {
        'prec' : 0,
    },
    cds.SITE + '_R35_ARRIVALTIME_TOTAL_SECS' : {
        'prec' : 0,
    },
    cds.SITE + '_R50_ARRIVALTIME_MINS' : {
        'prec' : 0,
    },
    cds.SITE + '_R50_ARRIVALTIME_SECS' : {
        'prec' : 0,
    },
    cds.SITE + '_R50_ARRIVALTIME_TOTAL_SECS' : {
        'prec' : 0,
    },
    'SYSTEM_TIME_GPS' : {
        'prec' : 0,
    },
    'SYSTEM_UPTIME_SEC' : {
        'prec' : 0,
    },
    'SYSTEM_COUNTER' : {
        'prec' : 0,
    },
}

ioc_uptime = 0

class myDriver(Driver):
    def  __init__(self):
        super(myDriver, self).__init__()

if __name__ == '__main__':
    iocuptime = 0
    server = SimpleServer()
    server.createPV(prefix, pvdb)
    driver = myDriver()

    # process CA transactions
    while True:
        server.process(0.1)
        ioc_uptime = ioc_uptime + 1

