#!/usr/bin/python

import numpy as np
from gwpy.timeseries import TimeSeries


O1_start = 1126569617
#O1_end = 1136649617
O1_end = O1_start + 2
for ifo in ['H1', 'L1']:
    channels = ['{0}:ISI-GND_STS_HAM2_Z_DQ'.format(ifo),'{0}:ISI-GND_STS_HAM2_X_DQ'.format(ifo),'{0}:ISI-GND_STS_HAM2_Y_DQ'.format(ifo),'{0}:ISI-GND_STS_HAM5_Z_BLRMS_30M_100M'.format(ifo),'{0}:ISI-GND_STS_HAM5_X_BLRMS_30M_100M'.format(ifo),'{0}:ISI-GND_STS_HAM5_Y_BLRMS_30M_100M'.format(ifo)]
    for channel in channels:
        print('Getting time series')
        velocities = TimeSeries.get(channel,O1_start,O1_end)
        print('Time series done')
        acceleration = np.diff(velocities)
        print(acceleration.value)
