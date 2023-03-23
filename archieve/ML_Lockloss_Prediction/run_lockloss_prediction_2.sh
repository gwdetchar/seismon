
mag=$1
vel=$2
distance=$3
depth=$4
azimuth=$5
ifo=$6
outfile=$7

MATLAB=/ldcg/matlab_r2016a/bin/matlab

#${MATLAB} -nodesktop -nosplash -nojvm -r " FeatureSet = load('FeatureSet.dat'); [label, score] = predictLOCKLOSS('llo',FeatureSet ); csvwrite('label.dat',label); csvwrite('lockloss_probability.dat',score(:,2)); quit;"

${MATLAB} -nodesktop -nosplash -nojvm -r "predictLOCKLOSS_2(${mag},${vel},${distance},${depth},${azimuth},'${ifo}','${outfile}'); quit;"



