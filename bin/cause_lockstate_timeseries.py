#!/usr/bin/python

import sys
from gwpy.timeseries import TimeSeries

sys.stdout.write("starting first time series \n")
    #H1_channel_lockstatus_data_1 = TimeSeries.get('H1:DMT-DC_READOUT_LOCKED',float(pw_arrival_time),float(float(pw_arrival_time) + float(options.time_after_p_wave)))
    H1_channel_lockstatus_data_1 = TimeSeries.get('H1:DMT-DC_READOUT_LOCKED',float(pw_arrival_time)-10,float(rw_arrival_time))

    print >> sys.stdout.write("data gathered")
    firstcheck = H1_channel_lockstatus_data_1[0]
    if firstcheck == 1:
        for dataoutput in H1_channel_lockstatus_data_1[1:]:
            if dataoutput == 0:
                locklosscheck1 = "Y"
                break
            else:
                locklosscheck1 = "N"
                break
    else:
        locklosscheck1 = "Z"
    print >> sys.stdout.write("first timeseries complete \n")
    #H1_channel_lockstatus_data_2 = TimeSeries.get('H1:LSC-POP_A_LF_OUT_DQ',float(pw_arrival_time),float(float(pw_arrival_time) + float(options.time_after_p_wave)))
    H1_channel_lockstatus_data_2 = TimeSeries.get('H1:LSC-POP_A_LF_OUT_DQ',float(pw_arrival_time)-10,float(rw_arrival_time))

    secondcheck = H1_channel_lockstatus_data_2[0]
    if secondcheck > 10000:
        for dataoutput in H1_channel_lockstatus_data_2:
            if dataoutput < 10:
                locklosscheck2 = "Y"
                break
            else:
                 locklosscheck2 = "N"
                 break
    else:
        locklosscheck2 = "Z"
    print >> sys.stdout.write("completed second timeseries \n")
    if locklosscheck1 == "Y" or locklosscheck2 == "Y":
        lockloss = "Y"
    elif locklosscheck1 == "N" or locklosscheck2 == "N":
        lockloss = "N"
    else:
        lockloss = "Z"
