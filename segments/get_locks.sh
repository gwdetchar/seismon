
gpsStart=1126051217
gpsEnd=1187888978

ligolw_segment_query_dqsegdb --segment-url=https://segments.ligo.org --query-segments --include-segments H1:ODC-MASTER_GRD_IFO_LOCKED:1 --gps-start-time $gpsStart --gps-end-time $gpsEnd | ligolw_print -t segment:table -c start_time -c end_time -d " " > H1-LOCK.txt

ligolw_segment_query_dqsegdb --segment-url=https://segments.ligo.org --query-segments --include-segments L1:ODC-MASTER_GRD_IFO_LOCKED:1 --gps-start-time $gpsStart --gps-end-time $gpsEnd | ligolw_print -t segment:table -c start_time -c end_time -d " " > L1-LOCK.txt

