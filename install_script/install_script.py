
import os, sys, glob, optparse, shutil, warnings
import urllib
import zipfile

__author__ = "Michael Coughlin <michael.coughlin@ligo.org>"
__version__ = 0.1
__date__    = "10/25/2016"

def parse_commandline():
    """@Parse the options given on the command-line.
    """
    parser = optparse.OptionParser(usage=__doc__,version=__version__)

    parser.add_option("-o", "--outputDir", help="Seismon installation directory.",
                      default ="/Users/mcoughlin/Code/LIGO/seismon/install_script/install")

    parser.add_option("-v", "--verbose", action="store_true", default=False,
                      help="Run verbosely. (Default: False)")

    opts, args = parser.parse_args()

    # show parameters
    if opts.verbose:
        print >> sys.stderr, ""
        print >> sys.stderr, "running pylal_seismon_run..."
        print >> sys.stderr, "version: %s"%__version__
        print >> sys.stderr, ""
        print >> sys.stderr, "***************** PARAMETERS *****************    ***"
        for o in opts.__dict__.items():
          print >> sys.stderr, o[0]+":"
          print >> sys.stderr, o[1]
        print >> sys.stderr, ""
    return opts

warnings.filterwarnings("ignore")
# Parse command line
opts = parse_commandline()

outputDir = opts.outputDir
if not os.path.isdir(outputDir):
    os.mkdir(outputDir)

productclient = "http://ehppdl1.cr.usgs.gov/ProductClient.zip"
productclient_zip = "%s/ProductClient.zip"%outputDir
productclient_output = "%s/ProductClient"%outputDir

seismon = "https://github.com/ligovirgo/seismon"
seismon_output = "%s/seismon"%outputDir

if not os.path.isfile(productclient_zip):
    urllib.urlretrieve(productclient, filename=productclient_zip)

zip_ref = zipfile.ZipFile(productclient_zip, 'r')
zip_ref.extractall(outputDir)
zip_ref.close()

if not os.path.isdir(seismon_output):
    os.system("cd %s; git clone %s"%(outputDir,seismon))
    os.system("cd %s; python setup.py install --user"%seismon_output)

configexample = "%s/input/config.ini"%seismon_output
configini = "%s/config.ini"%productclient_output

file = open(configexample)
contents = file.read()
replaced_contents = contents.replace('XXX_PRODUCTCLIENT', productclient_output)

text_file = open(configini, "w")
text_file.write(replaced_contents)
text_file.close()

initfile = "%s/init.sh"%productclient_output
os.system("chmod +rwx %s"%initfile)

