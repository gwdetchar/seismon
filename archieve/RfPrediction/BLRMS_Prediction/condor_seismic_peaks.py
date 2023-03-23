
import os, sys
import glob
import optparse

import tables
import pandas as pd
import numpy as np
import h5py

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option('-i','--ifos', type=str,  default='LHO,LLO', help='GW Observatories: LLO,LHO...')

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

condorDir = './'
logDir = os.path.join(condorDir,'logs')
if not os.path.isdir(logDir):
    os.makedirs(logDir)

condordag = os.path.join(condorDir,'condor.dag')
fid = open(condordag,'w')
condorsh = os.path.join(condorDir,'condor.sh')
fid1 = open(condorsh,'w')

job_number = 0

ifos = opts.ifos.split(",")
for ifo in ifos:
    x = np.genfromtxt('./masterlists/{}.dat'.format(ifo))
    for ii,row in enumerate(x):
        fid1.write('python fetch_seismic_peaks.py -i %s -ID %d -blrmsBand 30M_100M -saveResult 1 -saveImage 0\n'%(ifo,ii))

        fid.write('JOB %d condor.sub\n'%(job_number))
        fid.write('RETRY %d 3\n'%(job_number))
        fid.write('VARS %d jobNumber="%d" ifo="%s" id="%d"\n'%(job_number,job_number, ifo, ii))
        fid.write('\n\n')
        job_number = job_number + 1

fid1.close()
fid.close()

fid = open(os.path.join(condorDir,'condor.sub'),'w')
fid.write('executable = ./fetch_seismic_peaks.py\n')
fid.write('output = logs/out.$(jobNumber)\n');
fid.write('error = logs/err.$(jobNumber)\n');
fid.write('arguments = -IFO $(ifo) -ID $(id) -blrmsBand 30M_100M -saveResult 1 -saveImage 0\n')
fid.write('requirements = OpSys == "LINUX"\n');
fid.write('request_memory = 8192\n');
fid.write('request_cpus = 1\n');
fid.write('accounting_group = ligo.dev.o2.burst.allsky.stamp\n');
fid.write('notification = never\n');
fid.write('getenv = true\n');
fid.write('log = /usr1/mcoughlin/seismon.log\n')
fid.write('+MaxHours = 24\n');
fid.write('universe = vanilla\n');
fid.write('queue 1\n');
fid.close()

