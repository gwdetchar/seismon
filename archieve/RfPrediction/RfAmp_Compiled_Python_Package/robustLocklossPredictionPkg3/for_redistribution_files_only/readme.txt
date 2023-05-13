Packaging and Deploying robustLocklossPredictionPkg3

1. Prerequisites for Deployment 

A. If MATLAB Runtime version 9.1 (R2016b) has not been installed, install it in one of 
   these ways:

i. Run the package installer, which will also install the MATLAB Runtime.

ii. Download the Linux 64-bit version of the MATLAB Runtime for R2016b from:

    http://www.mathworks.com/products/compiler/mcr/index.html
    
##################################################################################
# Instructions to install the MATLAB Compiler Runtime
#
# Create a folder for MATLAB Runtime (modify the default folder name )
MATLAB_RUNTIME_R2016B="/home/nikhil.mukund/MATLAB_RUNTIME"

mkdir $MATLAB_RUNTIME_R2016B
cd $MATLAB_RUNTIME_R2016B

# Get the MATLAB Compiler Runtime from mathworks.com (No license required)
wget http://ssd.mathworks.com/supportfiles/downloads/R2016b/deployment_files/R2016b/installers/glnxa64/MCR_R2016b_glnxa64_installer.zip

# Unzip 
unzip MCR_R2016b_glnxa64_installer.zip 

# Install the compiler
./install -mode silent -agreeToLicense yes -destinationFolder $MATLAB_RUNTIME_R2016B

# Add LD_LIBRARY_PATH
echo 'export LD_LIBRARY_PATH='$MATLAB_RUNTIME_R2016B/v91/runtime/glnxa64:$MATLAB_RUNTIME_R2016B/v91/bin/glnxa64:$MATLAB_RUNTIME_R2016B/v91/sys/os/glnxa64:' ' >> ~/.bashrc

source ~/.bashrc
##################################################################################    



iii. Run the MATLAB Runtime installer provided with MATLAB.

B. Verify that a Linux 64-bit version of Python 2.7, 3.3, and/or 3.4 is installed.

2. Installing the robustLocklossPredictionPkg3 Package

A. Go to the directory that contains the file setup.py and the subdirectory 
   robustLocklossPredictionPkg3. If you do not have write permissions, copy all its 
   contents to a temporary location and go there.

B. Execute the command:

    python setup.py install [options]
    
If you have full administrator privileges, and install to the default location, you do 
   not need to specify any options. Otherwise, use --user to install to your home folder, 
   or --prefix="installdir" to install to "installdir". In the latter case, add 
   "installdir" to the PYTHONPATH environment variable. For details, refer to:

    https://docs.python.org/2/install/index.html

C. Copy the following to a text editor:

setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:<MCR_ROOT>/v90/runtime/glnxa64:<MCR_ROOT>/v90/bin/glnxa64:<MCR_ROOT>/v90/sys/os/glnxa64:<MCR_ROOT>/v90/sys/opengl/lib/glnxa64
setenv XAPPLRESDIR <MCR_ROOT>/v90/X11/app-defaults

Make the following changes:
- If LD_LIBRARY_PATH is not yet defined, remove the string "${LD_LIBRARY_PATH}:". 
- Replace "<MCR_ROOT>" with the directory where the MATLAB Runtime is installed.
- If your shell does not support setenv, use a different command to set the environment 
   variables.

Finally, execute the commands or add them to your shell initialization file.

3. Using the robustLocklossPredictionPkg3 Package

The robustLocklossPredictionPkg3 package is on your Python path. To import it into a 
   Python script or session, execute:

    import robustLocklossPredictionPkg3

If a namespace must be specified for the package, modify the import statement accordingly.


######################################################################
## SEISMON RfAmp Prediction  Code
## 
##
## Uses PYTHON package robustLocklossPredictionPkg3 & MATLAB 2016b shared libraries,
## Make sure to run the script set_shared_library_paths.sh prior to running this script. 
## To re-install the package go through readme.txt 
##
## Input Parameters : ifo, earthquake mag, latitude,longitude,distance, depth, azimuth
## Output file : predicted amplitude, lockloss_prediction(value btw 1&2 --> no lockloss to lockloss)
##
##Example:
##   python makePredictions.py -ifo 'H1' -mag 7.5 -lat -6.2 -lon 130.6 -dist 10690548.79 -depth 126.5 -azi 42.9
##
## To embed the same functionality in another code as a function use the commented lines of code at the end
##    Rfamp,LocklossTag = makePredictions('H1',5.1,-18.2,-174.9,1.048178e+07,197.7,59.4)
##
## Nikhil Mukund Menon (Last Edited : 14/4/2018)
## nikhil@iucaa.in, nikhil.mukund@LIGO.ORG
######################################################
