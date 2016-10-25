
import os, sys
import urllib
import zipfile

outputDir = "/Users/mcoughlin/Code/LIGO/seismon/install_script/install"
if not os.path.isdir(outputDir):
    os.mkdir(outputDir)

productclient = "http://ehppdl1.cr.usgs.gov/ProductClient.zip"
productclient_zip = "%s/ProductClient.zip"%outputDir
productclient_output = "%s/ProductClient"%outputDir

seismon = "https://github.com/ligovirgo/seismon"
seismon_output = "%s/seismon"%outputDir

if not os.path.isfile(productclient_zip):
    urllib.urlretrieve(productclient, filename=productclient_output)

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

