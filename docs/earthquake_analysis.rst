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

.. include:: seismon/input/seismon_params_traveltimes.txt
   start-line:3

If you look at the fourth line, you'll want to change this parameter to your own home directory as long as you followed the above steps correctly.
