***************
Getting started
***************
.. replace:: 

==============
Importing Seismon
==============

Seismon is a small package with a number sub-packages, so importing the root package via::

    >>> import seismon

isn't going to be very useful. Instead, it is best to import the desired sub-package as::

    >>> from seismon import NLNM

===========================
Using Seismon
===========================

Seismon has one main function designed to perform all of the basic tasks one needs when performing detector characterization (with an emphasis on seismic uses).

Seismon requires a params file to run, which contains information like:

.. code-block:: python

 ifo H1
 frameType R
 runName H1S6
 user mcoughlin
 codePath /home/mcoughlin/Seismon
 dirPath /home/mcoughlin/Seismon
 eventfilesLocation /home/mcoughlin/Seismon/eventfiles
 velocitymapsLocation /home/mcoughlin/Seismon/velocity_maps

These provide the inputs Seismon will use in its various tasks.

Generating PSDs:

.. code-block:: bash

 python seismon_run -p /home/mcoughlin/Seismon/seismon/input/seismon_params_H1S6.txt -s 955187679 -e 955188191 -c H1:LSC-DARM_ERR,H1:LSC-MICH_CTRL,H1:LSC-PRC_CTRL --doPlots --fftDuration 1 --fmin 40 --fmax 256 --doPSD


Generating Coherence:

.. code-block:: bash

 python seismon_run -p /home/mcoughlin/Seismon/seismon/input/seismon_params_H1S6.txt -s 955187679 -e 955188191 -c H1:LSC-DARM_ERR,H1:LSC-MICH_CTRL,H1:LSC-PRC_CTRL --doPlots --fftDuration 1 --fmin 40 --fmax 256 --doCoherence

Generating a Wiener filter:

.. code-block:: bash

 python seismon_run -p /home/mcoughlin/Seismon/seismon/input/seismon_params_H1S6.txt -s 955187679 -e 955188191 -c H1:LSC-DARM_ERR,H1:LSC-MICH_CTRL,H1:LSC-PRC_CTRL --doPlots --fftDuration 1 --fmin 40 --fmax 256 --doWiener -N 10

