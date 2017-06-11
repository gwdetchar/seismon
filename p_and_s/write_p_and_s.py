
import os, sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (13, 8) if False else (10, 6)

from obspy.taup.taup import getTravelTimes
from obspy.core.util.geodetics import gps2DistAzimuth
from obspy.taup import TauPyModel

from seismon.eqmon import ampRf, shoot

degrees = np.linspace(1,180,180)
distances = degrees*(np.pi/180)*6370000
depths = np.linspace(1,100,100)

model = TauPyModel(model="iasp91")
#model = TauPyModel(model="1066a")

fwd = 0
back = 0

eqlat, eqlon = 35.6895, 139.6917

GPS = 0
magnitude = 6.0
depth = 20.0
Rf0 = 76.44
Rfs = 1.37
cd = 440.68
rs = 1.57

Rfamp = ampRf(magnitude,distances/1000.0,depth,Rf0,Rfs,cd,rs)

lats = []
lons = []
Ptimes = []
Stimes = []
#Rtimes = []
Rtwotimes = []
RthreePointFivetimes = []
Rfivetimes = []
Rfamps = []

fid1 = open('p.dat','w')
fid2 = open('s.dat','w')

for distance, degree in zip(distances, degrees):
    lon, lat, baz = shoot(eqlon, eqlat, fwd, distance/1000)
    lats.append(lat)
    lons.append(lon)

    for depth in depths:    

        print "Calculating arrival for %.5f ..."%distance
        #arrivals = model.get_travel_times(source_depth_in_km=depth,distance_in_degree=degree,phase_list=('P','S'))
        arrivals = model.get_travel_times(source_depth_in_km=depth,distance_in_degree=degree)

        Ptime = -1
        Stime = -1
        Rtime = -1
        for phase in arrivals:
            if Ptime == -1 and phase.name.lower()[0] == "p":
                Ptime = GPS+phase.time
            if Stime == -1 and phase.name.lower()[-1] == "s":
                Stime = GPS+phase.time
            Rtwotime = GPS+distance/2000.0
            RthreePointFivetime = GPS+distance/3500.0
            Rfivetime = GPS+distance/5000.0

        fid1.write('%.5f '%Ptime)
        fid2.write('%.5f '%Stime)

    fid1.write("\n")
    fid2.write("\n")

fid1.close()
fid2.close()

