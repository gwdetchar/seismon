#!/bin/bash

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

rm 0
rm *.txt *.log *.ctf
rm run_seismon_lockloss_prediction.sh
rm seismon_lockloss_prediction

${MATLAB_ROOT}/bin/matlab <<EOF
 mcc -R -nojvm -R -nodisplay -R -singleCompThread -m seismon_lockloss_prediction
EOF

rm *.txt *.log *.ctf
rm run_seismon_lockloss_prediction.sh
rm -rf seismon_lockloss_prediction_mcr/

source matlab_script.sh
./seismon_lockloss_prediction 0 0 0 0 0 0 0

rm 0
mv seismon_lockloss_prediction ../bin/
