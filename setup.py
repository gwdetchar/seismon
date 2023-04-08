#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2016)
#
# This file is part of seismon.
#
# seismon is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# seismon is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with seismon.  If not, see <http://www.gnu.org/licenses/>.

"""Setup the seismon package
"""

# ignore all invalid names (pylint isn't good at looking at executables)
# pylint: disable=invalid-name

from __future__ import print_function

import os, sys
from distutils.version import LooseVersion

from setuptools import (setup, find_packages,
                        __version__ as setuptools_version)

def get_scripts(scripts_dir='bin'):
    """Get relative file paths for all files under the ``scripts_dir``
    """ 
    scripts = []
    for (dirname, _, filenames) in os.walk(scripts_dir):
        scripts.extend([os.path.join(dirname, fn) for fn in filenames])
    return scripts

import versioneer
#from setup_utils import (CMDCLASS, get_setup_requires, get_scripts)
__version__ = versioneer.get_version()
CMDCLASS=versioneer.get_cmdclass()

# -- dependencies -------------------------------------------------------------

# build dependencies
#setup_requires = get_setup_requires()

# package dependencies
install_requires = [
    'arrow',
    'astropy',
    'flask_caching',
    'flask_login',
    'flask_sqlalchemy',
    'flask_wtf',
    'gwpy',
    'lxml',
    'matplotlib>=2.2.0',
    'numpy>=1.7.1',
    'obspy',
    'passlib',
    'psycopg2',
    'redis',
    'scipy>=0.12.1',
    'simplejson',
    'sqlalchemy',
]

# test dependencies
tests_require = [
    'pytest>=3.1',
    'freezegun',
    'sqlparse',
    'bs4',
]
if sys.version < '3':
    tests_require.append('mock')

# -- run setup ----------------------------------------------------------------

setup(
    # metadata
    name='seismon',
    provides=['seismon'],
    version=__version__,
    description="A python package for mitigating the effects of earthquakes on GW detectors",
    long_description=("seismon is a python package for mitigating the effects of earthquakes on GW detectors"),
    author='Michael Coughlin',
    author_email='michael.coughlin@ligo.org',
    license='GPL-2.0-or-later',
    url='https://github.com/gwdetchar/seismon/',

    # package content
    packages=find_packages(),
    scripts=get_scripts(),
    include_package_data=True,
    package_data={'seismon': ['input/*','robustLocklossPredictionPkg/*']},
 
    # dependencies
    cmdclass=CMDCLASS,
    install_requires=install_requires,
    tests_require=tests_require,

    # classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Intended Audience :: Science/Research',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    ],
    extras_require={
        "frontend": [
            'WTForms',
            'WTForms-Alchemy',
            'WTForms-Components'
        ],
    },
)
