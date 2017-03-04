#!/bin/bash

rm 0
rm *.txt *.log *.ctf
rm run_seismon_lockloss_prediction.sh
rm seismon_lockloss_prediction

/ldcg/matlab_r2016a/bin/matlab <<EOF
 mcc -R -nojvm -R -nodisplay -R -singleCompThread -m seismon_lockloss_prediction
EOF

rm *.txt *.log *.ctf
rm run_seismon_lockloss_prediction.sh
rm -rf seismon_lockloss_prediction_mcr/

source ${HOME}/matlab_scripts/matlab_script_2016a.sh
./seismon_lockloss_prediction 0 0 0 0 0 0 0

rm 0

