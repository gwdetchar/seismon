
import os,glob

outputDir = "/home/mcoughlin/Seismon/L1/L1WDStudy"

plotsDir = "plots/"

folders = glob.glob(os.path.join(outputDir,"*-*"))

for folder in folders:
    plotName = "%s/L1_ISI-GND_STS_HAM2_Z_DQ/timeseries.png"%(folder)
    folderSplit = folder.split("/")
    plotNameOut = "%s/%s.png"%(plotsDir,folderSplit[-1])

    cp_command = "cp %s %s"%(plotName,plotNameOut)
    os.system(cp_command)


