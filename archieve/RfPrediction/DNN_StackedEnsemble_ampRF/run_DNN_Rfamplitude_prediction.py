
import os, sys, glob

def get_command(earthquakesFile, outputDirectory, runType):
    system_command = "python DNN_Rfamplitude_prediction.py -f %s -o %s -r %s --doPlots --Nepoch 10"%(earthquakesFile, outputDirectory, runType)
    return system_command

runTypes = ["original", "lowlatency", "cmt"]

for runType in runTypes:

    # H1
    earthquakesFile = "/home/mcoughlin/Seismon/Predictions/H1O1O2_CMT/earthquakes.txt"
    outputDirectory = "/home/mcoughlin/Seismon/MLA/H1O1O2/"
    system_command = get_command(earthquakesFile, outputDirectory, runType)
    os.system(system_command)

    # L1
    earthquakesFile = "/home/mcoughlin/Seismon/Predictions/L1O1O2_CMT/earthquakes.txt"
    outputDirectory = "/home/mcoughlin/Seismon/MLA/L1O1O2/"
    system_command = get_command(earthquakesFile, outputDirectory, runType)
    os.system(system_command)

    # V1
    earthquakesFile = "/home/mcoughlin/Seismon/Predictions/V1O1O2_CMT/earthquakes.txt"
    outputDirectory = "/home/mcoughlin/Seismon/MLA/V1O1O2/"
    system_command = get_command(earthquakesFile, outputDirectory, runType)
    os.system(system_command)

    # IRIS
    earthquakeFiles = glob.glob("/home/mcoughlin/Seismon/USArray/Text_Files_EQ/CMT/*.txt")
    for earthquakeFile in earthquakeFiles:
        channelName = earthquakeFile.split("/")[-1].replace(".txt","")    
        outputDirectory = "/home/mcoughlin/Seismon/MLA/%s"%channelName
        system_command = get_command(earthquakesFile, outputDirectory, runType)
        os.system(system_command)

    print stop
