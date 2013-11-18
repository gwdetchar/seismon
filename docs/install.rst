***************
Installing Seismon
***************

===================
Installing from git
===================

The source code for Seismon is under ``git`` version control, hosted by http://github.com.

You can install the package by first cloning the repository

.. code-block:: bash

    git clone https://github.com/gwpy/seismon.git

and then running the ``setup.py`` script as follows:

.. code-block:: bash

    cd seismon
    python setup.py install --user

The ``--user`` option tells the installer to copy codes into the standard user library paths, on linux machines this is

.. code-block:: bash

    ~/.local/lib

while on Mac OS this is

.. code-block:: bash

    ~/Library/Python/X.Y/lib

where ``X.Y`` is the python major and minor version numbers, e.g. ``2.7``. In either case, python will autmatically know about these directories, so you don't have to fiddle with any environment variables.

============
Dependencies
============

**Build dependencies**

The Seismon package has the following build-time dependencies (i.e. required for installation):

* `astropy <http://astropy.org>`_
* `NumPy <http://www.numpy.org>`_ >= 1.7.1 (inherited dependency from astropy)
* `GWpy <https://github.com/gwpy/gwpy>`_ 

**Runtime dependencies**

Additionally, in order for much of the code to import and run properly, users are required to have the following packages:

* `matplotlib <http://matplotlib.org>`_
* `glue <https://www.lsc-group.phys.uwm.edu/daswg/projects/glue.html>`_
* `lal <https://www.lsc-group.phys.uwm.edu/daswg/projects/lalsuite.html>`_ and `lalframe <https://www.lsc-group.phys.uwm.edu/daswg/projects/lalsuite.html>`_ (same URL)
* `NDS2 <https://www.lsc-group.phys.uwm.edu/daswg/projects/nds-client.html>`_ (including SWIG-wrappings for python)

