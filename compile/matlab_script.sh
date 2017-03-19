#! /usr/bin/env bash

# MATLAB_ROOT must be set differently for atlas than for Caltech.
h=`hostname -d`

if [[ $h =~ "atlas" ]]; then
    MATLAB_ROOT="/opt/matlab/2016a"
elif [[ $h =~ "caltech" ]] || [[ $h =~ "cit" ]]; then
    MATLAB_ROOT="/ldcg/matlab_r2016a"
    eval `/ligotools/bin/use_ligotools`
elif [[ $h =~ "MIT" ]]; then
    MATLAB_ROOT="/usr/local/MATLAB/R2016b"
fi

export MATLAB_ROOT
ARCH=glnxa64
export ARCH

LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MATLAB_ROOT}/sys/opengl/lib/glnxa64
export LD_LIBRARY_PATH
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MATLAB_ROOT}/sys/java/jre/glnxa64/jre/lib/amd64
export LD_LIBRARY_PATH
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MATLAB_ROOT}/sys/java/jre/glnxa64/jre/lib/amd64/server
export LD_LIBRARY_PATH
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MATLAB_ROOT}/sys/java/jre/glnxa64/jre/lib/amd64/native_threads
export LD_LIBRARY_PATH
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MATLAB_ROOT}/bin/glnxa64
export LD_LIBRARY_PATH
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MATLAB_ROOT}/sys/os/glnxa64
export LD_LIBRARY_PATH
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MATLAB_ROOT}/runtime/glnxa64
export LD_LIBRARY_PATH
XAPPLRESDIR=${MATLAB_ROOT}/X11/app-defaults
export XAPPLRESDIR

# EOF
