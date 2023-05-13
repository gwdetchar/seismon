Packaging and Deploying robustLocklossPredictionPkg

1. Prerequisites for Deployment 

A. If MATLAB Runtime version 9.1 (R2016b) has not been installed, install it in one of 
   these ways:

i. Run the package installer, which will also install the MATLAB Runtime.

ii. Download the Linux 64-bit version of the MATLAB Runtime for R2016b from:

    http://www.mathworks.com/products/compiler/mcr/index.html
   
iii. Run the MATLAB Runtime installer provided with MATLAB.

B. Verify that a Linux 64-bit version of Python 2.7, 3.3, and/or 3.4 is installed.

2. Installing the robustLocklossPredictionPkg Package

A. Go to the directory that contains the file setup.py and the subdirectory 
   robustLocklossPredictionPkg. If you do not have write permissions, copy all its 
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

3. Using the robustLocklossPredictionPkg Package

The robustLocklossPredictionPkg package is on your Python path. To import it into a 
   Python script or session, execute:

    import robustLocklossPredictionPkg

If a namespace must be specified for the package, modify the import statement accordingly.

Refer example script makePredictions.py 
