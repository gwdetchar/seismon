===========================
Getting Earthquake Analysis
===========================

Preface
-------
This is a document designed to get the user started on generating data from earthquakes using SeisMon. This particular example is designed to grab and organize data from 6 channels during O1. These channels are two seismic motion channels that operate with three degrees of freedom X, Y, and Z. **Important Note:** SeisMon is designed to run from the CIT servers on LIGO so the user should log onto there in order to make sure that things will even begin to work properly.

Getting Started
---------------

In order to get started, we want to look and see where the files we need are located within SeisMon's directory. SeisMon's directory should be located here.

.. code:: bash

   ~/gitrepo/seismon

Many of the files that we need to run are located within the folders of the seismon directory itself, usually located within the seismon/bin directory. Run the ls command inside of the seismon directory like this

.. code:: bash

   ls

to make note of the folders and files. For this example, the user will want to take note of both the bin directory and the input directory. The next step is to make sure gwpy is sourced before running any of the scripts mentioned in this file. The user can do this by running 

.. code:: bash

   pip install --user gwpy

Once gwpy is ready to go, then we can move onto the next step of generating the list of xml files needed to do the analysis.

First Stage: seismon_traveltimes
--------------------------------

In order to get seismon_traveltimes running we have to go to our home directory and make a directory called eventfiles and then inside eventfiles create a directory called iris.

.. code:: bash

   cd ~
   mkdir eventfiles
   cd eventfiles/
   mkdir iris
   cd ~

The next step after this is to cd into the input directory of seismon. Open up the file seismon_params_traveltimes.txt and inside you should find. 
.. code:: bash
   
   dataLocation /home/mcoughlin/Seismon/ProductClient/data/receiver_storage/origin
   publicdataLocation /home/mcoughlin/Seismon/publicdata
   databasedataLocation /home/mcoughlin/Seismon/databasedata
   **eventfilesLocation /home/eric.coughlin/eventfiles**

If you look at the fourth line, which I bolded for clarity, you'll want to change this parameter to your own home directory as long as you followed the above steps correctly.
The next step is to cd back to bin then.

.. code:: bash 

  screen
  python seismon_traveltimes -p /home/$USER/gitrepo/seismon/input/seismon_params_traveltimes.txt -s 1126569617 -e 1136649617 --minMagnitude 4.0 --doIRIS

Screen is a program designed to use multiple windows within one terminal session. These screen will continue to operate even if you use disconnect from the session. In order to get back to your regular session, just detach from the process by clicking ctrl + a and then d. If you want to reatach just use the following commands.

.. code:: bash 

  screen -ls
  screen -r [Whatever process you want to reatach]

Just copy and paste whichever screen you want to go to from the output of screen -ls after the screen -r command.

This process will take quite a bit of time to complete, think days instead of hours. This is why using screen is a must.

Second Stage: seismon_run_run_H1O1 and seismon_run_run_L1O1
-----------------------------------------------------------

After completing the first stage, the next step is to run both H1O1 and L1O1.

The first thing to do in order to run both of these scripts is to 

.. code:: bash

   cd ~/gitrepo/seismon/input
   vi seismon_params_H1O1.txt

Inside you'll find a file that looks like this.

.. code:: bash

   ifo H1
   frameType H1_R
   runName H1O1
   user eric.coughlin
   dirPath /home/eric.coughlin/gitrepo
   publicPath /home/eric.coughlin/public_html
   codePath /home/eric.coughlin/gitrepo
   executableDir /home/eric.coughlin/gitrepo/seismon/bin
   eventfilesLocation /home/eric.coughlin/eventfiles
   #eventfilesLocation /home/mcoughlin/Seismon/eventfiles/database
   velocitymapsLocation /home/mcoughlin/Seismon/velocity_maps

You'll want to change the user directory to your own.

A nice way to do that is to use within vim 

.. code:: bash 

   :%s/eric.coughlin/$USER/gc

It'll ask you to confirm each change made.

Now do the same steps with L1O1.

Once you are done, you should use screen again to run both seismon_run_run_H1O1 and seismon_run_run_L1O1.

.. code:: bash 

   cd ~/gitrepo/seismon/bin
   screen
   python seismon_run_run_H1O1
   ctrl-a then d
   screen
   python seismon_run_run_H1O1
   

This will also take some time.

Third Stage: seismon_run_prediction_vs_actual_ec
---------------------------------------------

.. code:: bash 

   vi seismon_run_prediction_vs_actual_ec
   :%s/eric.coughlin/$USER/gc
   :wq
   screen
   python seismon_run_prediction_vs_actual_ec
   ctrl-a then d


