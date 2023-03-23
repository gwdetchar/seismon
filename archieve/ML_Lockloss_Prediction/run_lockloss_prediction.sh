
MATLAB=/ldcg/matlab_r2016a/bin/matlab

${MATLAB} -nodesktop -nosplash -nojvm -r " FeatureSet = load('FeatureSet.dat'); [label, score] = predictLOCKLOSS('llo',FeatureSet ); csvwrite('label.dat',label); csvwrite('lockloss_probability.dat',score(:,2)); quit;"


