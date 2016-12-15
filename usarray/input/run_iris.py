
import obspy.iris, obspy
#client = obspy.iris.Client()
client = obspy.fdsn.client.Client("IRIS")

channel = "BH*"
channel = "BHZ"
network = "*"
network = "IU"
location = "*"
#location = "00"
station = "*"

file = "LIST.txt"
lines = [line.strip() for line in open(file)]

#aval = client.availability(network=network, station=station, location=location, channel=channel, starttime=obspy.UTCDateTime(2012, 1, 1, 0, 0, 0, 0), endtime=obspy.UTCDateTime(2013, 1, 1, 0, 0, 0, 0))

starttime=obspy.UTCDateTime(2012, 1, 1, 0, 0, 0, 0)
endtime=obspy.UTCDateTime(2013, 1, 1, 0, 0, 0, 0)

locations = ["","00"] 

f = open("channels.txt","w+")

for line in lines:
    lineSplit = line.split("\t")

    #network = lineSplit[1]
    #station = lineSplit[2]

    network = lineSplit[0].replace(" ","")
    station = lineSplit[1].replace(" ","")
    starttime = lineSplit[2].replace(" ","")
    endtime = lineSplit[3].replace(" ","")

    starttime=obspy.UTCDateTime(starttime)
    endtime=obspy.UTCDateTime(endtime)

    for location in locations:
        channel = "%s:%s:%s:BHZ"%(network,station,location)
    
        channelSplit = channel.split(":")
 
        try:
            response = client.get_stations(network=channelSplit[0], station = channelSplit[1], location = channelSplit[2], channel = channelSplit[3],starttime=starttime,endtime=endtime,level="resp")
            channel_response = response.get_response(channel.replace(":",".").replace("--",""),starttime)
            calibration = channel_response.instrument_sensitivity.value
   
            response = client.get_stations(network=channelSplit[0], station = channelSplit[1], location = channelSplit[2], channel = channelSplit[3],starttime=starttime,endtime=endtime,level='channel')

            latitude = float(response[0].stations[0].channels[0].latitude)
            longitude = float(response[0].stations[0].channels[0].longitude)
            samplef = float(response[0].stations[0].channels[0].sample_rate)

            f.write("%s:%s:%s:%s %.0f %.0f %.5f %.5f %s %s\n"%(channelSplit[0],channelSplit[1],channelSplit[2],channelSplit[3],samplef,calibration,latitude,longitude,starttime,endtime))
        except:
            continue

f.close()

