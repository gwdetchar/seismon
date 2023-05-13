
##################################################################################
# Instructions to install the MATLAB Compiler Runtime
#
# Create a folder for MATLAB Runtime (modify the default folder name )
MATLAB_RUNTIME_R2016B="/usr/local/seismon/MATLAB_RUNTIME"

mkdir $MATLAB_RUNTIME_R2016B
cd $MATLAB_RUNTIME_R2016B

# Get the MATLAB Compiler Runtime from mathworks.com (No license required)
wget http://ssd.mathworks.com/supportfiles/downloads/R2016b/deployment_files/R2016b/installers/glnxa64/MCR_R2016b_glnxa64_installer.zip

# Unzip 
unzip MCR_R2016b_glnxa64_installer.zip 

# Install the compiler
./install -mode silent -agreeToLicense yes -destinationFolder $MATLAB_RUNTIME_R2016B

# Add LD_LIBRARY_PATH
#echo 'export LD_LIBRARY_PATH='$MATLAB_RUNTIME_R2016B/v91/runtime/glnxa64:$MATLAB_RUNTIME_R2016B/v91/bin/glnxa64:$MATLAB_RUNTIME_R2016B/v91/sys/os/glnxa64:' ' >> ~/.bashrc

#source ~/.bashrc
##################################################################################    

