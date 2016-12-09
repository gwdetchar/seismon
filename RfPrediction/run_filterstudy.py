
import os, sys

# 1165097029.44 6.5 5.2812 96.1076
startGPS = 1165097900
endGPS = 1165099700

# 1165095758.36 5.9 11.0693 -60.9048
startGPS = 1165096300
endGPS = 1165097200

# 1164667243.23 6.3 -15.3207 -70.8229
startGPS = 1164667900
endGPS = 1164669000

# 1136777150.64 6.7 41.9723 142.781
startGPS = 1136777700 
endGPS = 1136778700

# 1133509823.6 7.2 38.2 72.8
startGPS = 1133510600
endGPS = 1133515100

# 1164792232.0 6.0 52.2 174.2
startGPS = 1164792600
endGPS = 1164794600

# 1165097029.4 6.5 5.3 96.1
#startGPS = 1165097900
#endGPS = 1165103700

# 1164935601.8 6.3 -7.3 123.4
#startGPS = 1164936400
#endGPS = 1164942000

# http://earthquake.usgs.gov/earthquakes/eventpage/us10007e55#executive
# 1164935601.85 6.3 -7.3236 123.4039
# LHO 1164936476 1164936514 1164941924 1164939215 1164938131 8.85252e-07 12645165
# LLO 1164936596 1164936633 1164943420 1164940069 1164938729 2.31259e-06 15636275
ifo = "H1"
startGPS = 1164936400
endGPS = 1164938200
#os.system("python filterstudy.py -i %s -s %d -e %d"%(ifo,startGPS,endGPS))
ifo = "L1"
startGPS = 1164936500
endGPS = 1164938800
#os.system("python filterstudy.py -i %s -s %d -e %d"%(ifo,startGPS,endGPS))

# http://earthquake.usgs.gov/earthquakes/eventpage/us10007ggp#executive
# 1165095758.36 5.9 11.0693 -60.9048
# LHO 1165096366 1165096381 1165099133 1165097687 1165097108 3.05155e-06 6748639
# LLO 1165096157 1165096172 1165097637 1165096832 1165096510 7.75074e-06 3757848
ifo = "H1"
startGPS = 1165096300
endGPS = 1165097200
#os.system("python filterstudy.py -i %s -s %d -e %d"%(ifo,startGPS,endGPS))
ifo = "L1"
startGPS = 1165096100
endGPS = 1165096600
#os.system("python filterstudy.py -i %s -s %d -e %d"%(ifo,startGPS,endGPS))

# 1164705118.82 4.9 -9.6095 117.4358
# LHO 1164706020 1164706052 1164711754 1164708911 1164707773 8.82514e-08 13270925
# LLO 1164706141 1164706172 1164713267 1164709775 1164708378 2.07672e-07 16295405
ifo = "H1"
startGPS = 1164706000
endGPS = 1164707800
#os.system("python filterstudy.py -i %s -s %d -e %d"%(ifo,startGPS,endGPS))
ifo = "L1"
startGPS = 1164706100
endGPS = 1164708400
#os.system("python filterstudy.py -i %s -s %d -e %d"%(ifo,startGPS,endGPS))

# http://earthquake.usgs.gov/earthquakes/eventpage/us20007z2r#executive
# 1165209321.56 5.9 43.8159 86.3035
# LHO 1165210086 1165210092 1165214154 1165212083 1165211254 1.85754e-06 9664646
# LLO 1165210172 1165210178 1165215206 1165212684 1165211676 2.72753e-06 11769731
ifo = "H1"
startGPS = 1165210000
endGPS = 1165211300
#os.system("python filterstudy.py -i %s -s %d -e %d"%(ifo,startGPS,endGPS))
ifo = "L1"
startGPS = 1165210100
endGPS = 1165211700
#os.system("python filterstudy.py -i %s -s %d -e %d"%(ifo,startGPS,endGPS))

# http://earthquake.usgs.gov/earthquakes/eventpage/us20007z6r#executive
# 1165243803.73 6.5 40.4753 -126.1528
# LHO 1165243917 1165243922 1165244237 1165244051 1165243977 2.38581e-04 865749
# LLO 1165244175 1165244180 1165245490 1165244767 1165244478 2.62987e-05 3371604
ifo = "H1"
startGPS = 1165243900
endGPS = 1165244000
#os.system("python filterstudy.py -i %s -s %d -e %d"%(ifo,startGPS,endGPS))
ifo = "L1"
startGPS = 1165244100
endGPS = 1165244500
#os.system("python filterstudy.py -i %s -s %d -e %d"%(ifo,startGPS,endGPS))

# http://earthquake.usgs.gov/earthquakes/eventpage/us20007z80#executive
# 1165253943.38 7.8 -10.676 161.3298
# LHO 1165254720 1165254737 1165258966 1165256813 1165255952 4.76459e-05 10044280
# LLO 1165254812 1165254829 1165260105 1165257464 1165256408 7.50933e-05 12322382
ifo = "H1"
startGPS = 1165254700
endGPS = 1165256000
#os.system("python filterstudy.py -i %s -s %d -e %d"%(ifo,startGPS,endGPS))
ifo = "L1"
startGPS = 1165254800
endGPS = 1165256500
#os.system("python filterstudy.py -i %s -s %d -e %d"%(ifo,startGPS,endGPS))

# http://earthquake.usgs.gov/earthquakes/eventpage/us20007z8m#executive
# 1165255402.15 5.6 -10.3758 161.3233
# LHO 1165256178 1165256195 1165260412 1165258265 1165257406 8.95940e-07 10020283
# LLO 1165256270 1165256288 1165261556 1165258918 1165257864 1.40158e-06 12307054
ifo = "H1"
startGPS = 1165256100
endGPS = 1165257500
#os.system("python filterstudy.py -i %s -s %d -e %d"%(ifo,startGPS,endGPS))
ifo = "L1"
startGPS = 1165256200
endGPS = 1165257900
#os.system("python filterstudy.py -i %s -s %d -e %d"%(ifo,startGPS,endGPS))

# http://earthquake.usgs.gov/earthquakes/eventpage/us20007z2r#executive
# 1165209321.56 5.9 43.8159 86.3035
# LHO 1165210086 1165210092 1165214154 1165212083 1165211254 1.85754e-06 9664646
# LLO 1165210172 1165210178 1165215206 1165212684 1165211676 2.72753e-06 11769731
ifo = "H1"
startGPS = 1165210000
endGPS = 1165211300
os.system("python filterstudy.py -i %s -s %d -e %d"%(ifo,startGPS,endGPS))
ifo = "L1"
startGPS = 1165210100
endGPS = 1165211700
os.system("python filterstudy.py -i %s -s %d -e %d"%(ifo,startGPS,endGPS))

