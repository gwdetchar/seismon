#!/usr/bin/env python3
#
# code to create an EPICS IOC for 'seismon' earthquake predictions for a site 
#  Input parameters
#  -I <ifo>    (default LLO)
#
#  Keith Thorne 2020-12-23 - Python3 version
#  Keith Thorne 2022-05-11 - Add notes on setup

# add environment variables to restrict EPICS to LAN port
# i.e. if LAN port in 10.110.10.149
# export EPICS_CAS_INTF_ADDR_LIST=10.110.10.149
# export EPICS_CAS_BEACON_ADDR_LIST=10.110.10.255
#


# create server
from pcaspy import Driver, SimpleServer
import os

def init_pvdb(SITE):

    idxs = [0,1,2,3,4,5]
    pvdb =  {'SYSTEM_TIME_GPS_V2' : {
            'prec' : 0,
            'type' : 'float', 
            'unit' : 'sec', 
            'lolim' : 1000000000, 
            'hilim' : 2000000000,
            },
            'SYSTEM_UPTIME_SEC_V2' : {
            'prec' : 0,
            'type' : 'float', 
            'unit' : 'sec', 
            'lolim' : 0, 
            'hilim' : 5000000,
            },
            'SYSTEM_COUNTER_V2' : {
            'prec' : 0,
            'type' : 'float', 
            'unit' : 'none', 
            },
    }

    for idx in idxs:
        pvdb_temp = {
            'EQ_TIME_GPS_%d'%idx : {
            'prec' : 3,
            'type' : 'float', 
            'unit' : 'sec', 
            'lolim' : 1000000000, 
            'hilim' : 2000000000,
            },
            'EQ_TIME_UTC_%d'%idx : {
                'type' : 'string',
            },
            'EQ_TIME_LTZ_%d'%idx : {
                'type' : 'string',
            },
            'EQ_MAGNITUDE_MMS_%d'%idx : {
            'prec' : 3,
            'type' : 'float', 
            'unit' : 'mms', 
            'lolim' : 0, 
            'hilim' : 15,
            },
            'EQ_LATITUDE_DEG_%d'%idx : {
            'prec' : 3,
            'type' : 'float', 
            'unit' : 'deg', 
            'lolim' : -90, 
            'hilim' : 90,
            },
            'EQ_LONGITUDE_DEG_%d'%idx : {
            'prec' : 3,
            'type' : 'float', 
            'unit' : 'deg', 
            'lolim' : -180, 
            'hilim' : 180,
            },
            'EQ_DEPTH_KM_%d'%idx : {
            'prec' : 3,
            'type' : 'float', 
            'unit' : 'km', 
            'lolim' : 0, 
            'hilim' : 6000,
            },
            'EQ_LOCATION_%d'%idx : {
                'type' : 'string',
            },
            'EQ_' + SITE + '_DISTANCE_M_%d'%idx : {
            'prec' : 3,
            'type' : 'float', 
            'unit' : 'meters', 
            'lolim' : 0, 
            'hilim' : 450000000,
            },
            SITE + '_P_ARRIVALTIME_GPS_%d'%idx : {
            'prec' : 3,
            'type' : 'float', 
            'unit' : 'sec', 
            'lolim' : 1000000000, 
            'hilim' : 2000000000,
            },
            SITE + '_S_ARRIVALTIME_GPS_%d'%idx : {
            'prec' : 3,
            'type' : 'float', 
            'unit' : 'sec', 
            'lolim' : 1000000000, 
            'hilim' : 2000000000,
            },
            SITE + '_R20_ARRIVALTIME_GPS_%d'%idx : {
            'prec' : 3,
            'type' : 'float', 
            'unit' : 'sec', 
            'lolim' : 1000000000, 
            'hilim' : 2000000000,
            },
            SITE + '_R35_ARRIVALTIME_GPS_%d'%idx : {
            'prec' : 3,
            'type' : 'float', 
            'unit' : 'sec', 
            'lolim' : 1000000000, 
            'hilim' : 2000000000,
            },
            SITE + '_R50_ARRIVALTIME_GPS_%d'%idx : {
            'prec' : 3,
            'type' : 'float', 
            'unit' : 'sec', 
            'lolim' : 1000000000, 
            'hilim' : 2000000000,
            },
            SITE + '_P_VELOCITY_MPS_%d'%idx : {
            'prec' : 4,
            'type' : 'float', 
            'unit' : 'mps', 
            'lolim' : 0, 
            'hilim' : 10000,
            },
            SITE + '_S_VELOCITY_MPS_%d'%idx : {
            'prec' : 4,
            'type' : 'float', 
            'unit' : 'm/s', 
            'lolim' : 0, 
            'hilim' : 10000,
            },
            SITE + '_R20_VELOCITY_MPS_%d'%idx : {
            'prec' : 4,
            'type' : 'float', 
            'unit' : 'm/s', 
            'lolim' : 0, 
            'hilim' : 10000,
            },
            SITE + '_R35_VELOCITY_MPS_%d'%idx : {
            'prec' : 4,
            'type' : 'float', 
            'unit' : 'm/s', 
            'lolim' : 0, 
            'hilim' : 10000,
            },
            SITE + '_R50_VELOCITY_MPS_%d'%idx : {
            'prec' : 4,
            'type' : 'float', 
            'unit' : 'm/s', 
            'lolim' : 0, 
            'hilim' : 10000,
            },
            SITE + '_P_ARRIVALTIME_MINS_%d'%idx : {
            'prec' : 0,
            'type' : 'float', 
            'unit' : 'min', 
            'lolim' : -2400, 
            'hilim' : 600,
            },
            SITE + '_P_ARRIVALTIME_SECS_%d'%idx : {
            'prec' : 0,
            'type' : 'float', 
            'unit' : 'sec', 
            'lolim' : -60, 
            'hilim' : 60,
            },
            SITE + '_P_ARRIVALTIME_TOTAL_SECS_%d'%idx : {
            'prec' : 0,
            'type' : 'float', 
            'unit' : 'sec', 
            'lolim' : -14400, 
            'hilim' : 36000,
            },
            SITE + '_S_ARRIVALTIME_MINS_%d'%idx : {
            'prec' : 0,
            'type' : 'float', 
            'unit' : 'min', 
            'lolim' : -2400, 
            'hilim' : 600,
            },
            SITE + '_S_ARRIVALTIME_SECS_%d'%idx : {
            'prec' : 0,
            'type' : 'float', 
            'unit' : 'sec', 
            'lolim' : -60, 
            'hilim' : 60,
            },
            SITE + '_S_ARRIVALTIME_TOTAL_SECS_%d'%idx : {
            'prec' : 0,
            'type' : 'float', 
            'unit' : 'sec', 
            'lolim' : -14400, 
            'hilim' : 36000,
            },
            SITE + '_R20_ARRIVALTIME_MINS_%d'%idx : {
            'prec' : 0,
            'type' : 'float', 
            'unit' : 'min', 
            'lolim' : -2400, 
            'hilim' : 600,
            },
            SITE + '_R20_ARRIVALTIME_SECS_%d'%idx : {
            'prec' : 0,
            'type' : 'float', 
            'unit' : 'sec', 
            'lolim' : -60, 
            'hilim' : 60,
            },
            SITE + '_R20_ARRIVALTIME_TOTAL_SECS_%d'%idx : {
            'prec' : 0,
            'type' : 'float', 
            'unit' : 'sec', 
            'lolim' : -14400, 
            'hilim' : 36000,
            },
            SITE + '_R35_ARRIVALTIME_MINS_%d'%idx : {
            'prec' : 0,
            'type' : 'float', 
            'unit' : 'min', 
            'lolim' : -2400, 
            'hilim' : 600,
            },
            SITE + '_R35_ARRIVALTIME_SECS_%d'%idx : {
            'prec' : 0,
            'type' : 'float', 
            'unit' : 'sec', 
            'lolim' : -60, 
            'hilim' : 60,
            },
            SITE + '_R35_ARRIVALTIME_TOTAL_SECS_%d'%idx : {
            'prec' : 0,
            'type' : 'float', 
            'unit' : 'sec', 
            'lolim' : -14400, 
            'hilim' : 36000,
            },
            SITE + '_R50_ARRIVALTIME_MINS_%d'%idx : {
            'prec' : 0,
            'type' : 'float', 
            'unit' : 'sec', 
            'lolim' : -60, 
            'hilim' : 60,
            },
            SITE + '_R50_ARRIVALTIME_SECS_%d'%idx : {
            'prec' : 0,
            'type' : 'float', 
            'unit' : 'sec', 
            'lolim' : -60, 
            'hilim' : 60,
            },
            SITE + '_R50_ARRIVALTIME_TOTAL_SECS_%d'%idx : {
            'prec' : 0,
            'type' : 'float', 
            'unit' : 'sec', 
            'lolim' : -14400, 
            'hilim' : 36000,
            },
        }
        pvdb.update(pvdb_temp)

    return pvdb    
  
class myDriver(Driver):
    def  __init__(self):
        super(myDriver, self).__init__()

if __name__ == '__main__':
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

    args = parser.parse_args()

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


    prefix = ifo +':SEI-SEISMON_'

    print(f"Starting IOC for SITE={site} and IFO={ifo}")

    ioc_uptime = 0
    server = SimpleServer()
    pvdb = init_pvdb(site)
    server.createPV(prefix, pvdb)
    driver = myDriver()

    # process CA transactions
    while True:
        server.process(0.1)
        ioc_uptime = ioc_uptime + 1
