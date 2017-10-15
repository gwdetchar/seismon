ligolw_segment_query_dqsegdb --segment-url https://segments.ligo.org --query-segments --include-segments H1:ODC-MASTER_GRD_IFO_LOCKED:1 --gps-start-time 1126569617 --gps-end-time 1136649617 | ligolw_print -t segment -c start_time -c end_time -d ' ' > segs_Locked_H_1126569617_1136649617.txt

ligolw_segment_query_dqsegdb --segment-url https://segments.ligo.org --query-segments --include-segments L1:ODC-MASTER_GRD_IFO_LOCKED:1 --gps-start-time 1126569617 --gps-end-time 1136649617 | ligolw_print -t segment -c start_time -c end_time -d ' ' > segs_Locked_L_1126569617_1136649617.txt
