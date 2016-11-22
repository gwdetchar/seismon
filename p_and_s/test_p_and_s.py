
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

model = TauPyModel(model="iasp91")
#model = TauPyModel(model="1066a")

fwd = 0
back = 0

eqlat, eqlon = 35.6895, 139.6917

GPS = 1000000000
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

parrivals = np.loadtxt('p.dat')
sarrivals = np.loadtxt('s.dat')
depths = np.linspace(1,100,100)
index = np.argmin(np.abs(depths-depth))
parrivals = parrivals[:,index]
sarrivals = sarrivals[:,index]

for distance, degree, parrival, sarrival in zip(distances, degrees,parrivals,sarrivals):
    lon, lat, baz = shoot(eqlon, eqlat, fwd, distance/1000)
    lats.append(lat)
    lons.append(lon)    

    print "Calculating arrival for %.5f ..."%distance
    #arrivals = model.get_travel_times(source_depth_in_km=depth,distance_in_degree=degree,phase_list=('P','S'))
    arrivals = model.get_travel_times(source_depth_in_km=depth,distance_in_degree=degree)

    Ptime = -1
    Stime = -1
    Rtime = -1
    for phase in arrivals:
        if Ptime == -1 and phase.name.lower()[0] == "p":
            Ptime = GPS+phase.time
        if Stime == -1 and phase.name.lower()[0] == "s":
            Stime = GPS+phase.time
    Ptime = GPS+parrival
    Stime = GPS+sarrival
    Rtwotime = GPS+distance/2000.0
    RthreePointFivetime = GPS+distance/3500.0
    Rfivetime = GPS+distance/5000.0

    Ptimes.append(Ptime)
    Stimes.append(Stime)
    Rtwotimes.append(Rtwotime)
    RthreePointFivetimes.append(RthreePointFivetime)
    Rfivetimes.append(Rfivetime)

    print Ptime - parrival
    #print Ptime, Stime, Rtwotime, RthreePointFivetime, Rfivetime


plotDir = '.'
plt.figure()
plt.plot(degrees,Ptimes,'kx')
plt.plot(degrees,Stimes,'kx')
plotName = os.path.join(plotDir,'times.png')
plt.savefig(plotName)
plotName = os.path.join(plotDir,'times.eps')
plt.savefig(plotName)
plotName = os.path.join(plotDir,'times.pdf')
plt.savefig(plotName)
plt.close()


