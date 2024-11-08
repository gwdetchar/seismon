# Sample script to upload new entries to Carleton Catalogue Database


from sshtunnel import SSHTunnelForwarder
from sqlalchemy import create_engine  
import pandas as pd
from argparse import ArgumentParser
import datetime
import numpy
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_rows',100)

parser = ArgumentParser()

'''
parser.add_argument( '--event_id', default='us20003j2v', type=str,help='unique ID assigned by USGS')
parser.add_argument( '--time', default='12-Sep-2015 20:32:26', type=str,help='EQ event time [UTC]')
parser.add_argument( '--created_at', default='Mon Jun 07 22:56:39  2021',type=str, help='created at [UTC]')
parser.add_argument( '--modified',   default='Mon Jun 07 22:56:39  2021',type=str, help='modified [UTC]')
parser.add_argument( '--place',   default='153km SSE of L\'Esperance Rock, New Zealand',type=str, help='EQ event location')
parser.add_argument( '--latitude', default=-32.6066, type=float,help='EQ latitude')
parser.add_argument( '--longitude', default=-178.0287, type=float, help='EQ longitude')
parser.add_argument( '--mag', default=5.9, type=float, help='EQ magnitude')
parser.add_argument( '--depth', default=8, type=float, help='EQ depthh [km]')
parser.add_argument( '--SNR', default=19.9, type=float, help='SNR of PeakAmplitude estimation')
parser.add_argument( '--peak_data_um_mean_subtracted', default=0.33, type=float, help='EQ estimated peak amplitude [um/s]')
parser.add_argument( '--db_catalogue_name', default='llo_catalogues', type=str,help="Specify catalogue to write to {'llo_catalogues','lho_catalogues','virgo_catalogues'}")
parser.add_argument( '--uname', default='nmukund', type=str,help='username')
parser.add_argument( '--pubkey', default='~/.ssh/id_rsa.pub', type=str,help='ssh public key')
'''


parser.add_argument( '--event_id', default='us7000fvy7', type=str,help='unique ID assigned by USGS')
parser.add_argument( '--time', default='21-Nov-2021 11:50:59', type=str,help='EQ event time [UTC]')
parser.add_argument( '--created_at', default='Mon Jun 07 22:56:39  2021',type=str, help='created at [UTC]')
parser.add_argument( '--modified',   default='Mon Jun 07 22:56:39  2021',type=str, help='modified [UTC]')
parser.add_argument( '--place',   default='153km SSE of L\'Esperance Rock, New Zealand',type=str, help='EQ event location')
parser.add_argument( '--latitude', default=20.0000, type=float,help='EQ latitude')
parser.add_argument( '--longitude', default=-110.0000, type=float, help='EQ longitude')
parser.add_argument( '--mag', default=5.5, type=float, help='EQ magnitude')
parser.add_argument( '--depth', default=10, type=float, help='EQ depthh [km]')
parser.add_argument( '--SNR', default=3.6, type=float, help='SNR of PeakAmplitude estimation')
parser.add_argument( '--peak_data_um_mean_subtracted', default=0.33, type=float, help='EQ estimated peak amplitude [um/s]')
parser.add_argument( '--db_catalogue_name', default='lho_catalogues', type=str,help="Specify catalogue to write to {'llo_catalogues','lho_catalogues','virgo_catalogues'}")
parser.add_argument( '--uname', default='nmukund', type=str,help='username')
parser.add_argument( '--pubkey', default='~/.ssh/id_rsa.pub', type=str,help='ssh public key')



args = parser.parse_args()


#-------------------------------------
# if event already exists then 
if_exists_then = 'append' #{'append','replace'}


server = SSHTunnelForwarder(
    ('virgo.physics.carleton.edu', 22),
    ssh_username=args.uname,
    ssh_pkey=args.pubkey,
    ssh_private_key_password="",
    remote_bind_address=('127.0.0.1', 5432))

server.start()
engine = create_engine(f'postgresql+psycopg2://seismon:seismon@{server.local_bind_address[0]}:{server.local_bind_address[1]}/seismon')


# Connect DataBase
conn = engine.connect()
print('Remote connection to Carleton database successful')


data_dict = {
    'event_id':[args.event_id],
    'time':[args.time],
    'created_at':[args.created_at],
    'modified':[args.modified],
    'place':[args.place],
    'latitude':[args.latitude],
    'longitude':[args.longitude],
    'mag':[args.mag],
    'depth':[args.depth],
    'SNR':[args.SNR],
    'peak_data_um_mean_subtracted':[args.peak_data_um_mean_subtracted]
    }
data_df_filtered = pd.DataFrame(data_dict)


# upload dataframe remotely to database
print('Remotely updating {} processed catalog'.format(args.db_catalogue_name))
data_df_filtered.to_sql('{}'.format(args.db_catalogue_name), con=engine,  if_exists=if_exists_then, index=False)



#-------------------------------------
# Code snippet to add back measurements to Predictions table (for comparison)
# Added on 05/10/2021 by NM
#get PREDICTIONS DATABASE Table
predictions_db = pd.read_sql_query('select * from public.predictions',con=engine)
# get CATALOG DATABASE Tables
llo_processed_catalogue_db = pd.read_sql_query('select * from public.llo_catalogues',con=engine)
lho_processed_catalogue_db = pd.read_sql_query('select * from public.lho_catalogues',con=engine)


# get event_id
event_id=args.event_id


#get IFO name from the catalog 
if args.db_catalogue_name=='llo_catalogues':
    ifo_name = 'LLO'
    processed_catalogue_db = llo_processed_catalogue_db
elif args.db_catalogue_name=='lho_catalogues':
    ifo_name = 'LHO'
    processed_catalogue_db = lho_processed_catalogue_db

# get corresponding id from processed_catalogue_db
miD = processed_catalogue_db['event_id'] == event_id 
# get corresponding rfamp measured value from processed_catalog
try:
    rfamp_measured = processed_catalogue_db.loc[miD]['peak_data_um_mean_subtracted'].values[0]
except:
    rfamp_measured = np.array([])

if rfamp_measured.size != 0:
    # get event id from predictions table
    piD=(predictions_db['event_id']==event_id) & (predictions_db['ifo']==ifo_name)
    #piD = (predictions_db['event_id'].str.contains(event_id)) & (predictions_db['ifo'].str.contains(ifo_name))
    # replace current value for the rfamp_measured (-1)  with the observed value
    current_val  = predictions_db['rfamp_measured'][piD].values[0]
    predictions_db['rfamp_measured'][piD]=predictions_db['rfamp_measured'][piD].replace(current_val,rfamp_measured)
    # Update actual database 'predictions' table 
    if_exists_then='append'
    print('Updating measured amplitude for event_id:{} at ifo:{} in predictions table to {} um/s'.format(event_id,ifo_name,rfamp_measured))

    #predictions_db.reset_index(drop=True).to_sql('{}'.format('predictions'), con=engine,  if_exists=if_exists_then, index=False)
    predictions_db.to_sql('{}'.format('predictions'), con=engine,  if_exists=if_exists_then, index=False)
    #conn.execute('ALTER TABLE `predictions` ADD PRIMARY KEY (`event_id`);')
else:
    print('Event {} not found in {}. Skipping remote update of predictions DataBase table'.format(args.event_id,args.db_catalogue_name))

#-------------------------------------
print('Remote upload successful')
# Check if things worked (load remotely)
print('Attempting to read back from database-table (for verification)...')



kiD = processed_catalogue_db['event_id'] == event_id
jiD = predictions_db['event_id'] == event_id


print('Printing processed catalog')
print(processed_catalogue_db.loc[kiD,:])

print('Printing updated prediction table')
pd.set_option('display.max_rows', None)
print(predictions_db.loc[jiD,['event_id', 'ifo', 'rfamp','rfamp_measured', 'lockloss']])
# close connection
conn.close()


########################################################
## Compare Predictions vs Measurements (both LLO & LHO combined)
############################
print('Comparing Predictions vs Measurements')


# make a new dataframe
testData = predictions_db.loc[:,['rfamp','rfamp_measured']]
# rename columns
testData = testData.rename(columns={"rfamp":"predictions","rfamp_measured":"measurements"})
# keep only measured events
testData  = testData.loc[testData['measurements']!=-1,:]
# Get prediction accuracy for peak_data_mean_subtracted
FAC=5
event_below_upper_lim = np.sum(np.divide(testData.predictions,testData.measurements) <= FAC)
event_above_lower_lim = np.sum(np.divide(testData.predictions,testData.measurements) <= 1/FAC)
total_num_events      = len(testData.predictions)
# percentage captured within a given factor
percentage_captured = 100*(event_below_upper_lim + event_above_lower_lim)/total_num_events
#make scatter plot
plt.rc('font', size=20) #controls default text size
plt.rc('axes', titlesize=20) #fontsize of the title
plt.rc('axes', labelsize=20) #fontsize of the x and y labels
plt.rc('xtick', labelsize=20) #fontsize of the x tick labels
plt.rc('ytick', labelsize=20) #fontsize of the y tick labels
plt.rc('legend', fontsize=20) #fontsize of the legend
fig = plt.figure(figsize=(10,10))
ax = plt.gca()
ax.plot(testData.measurements, testData.predictions, 'o', c='blue', alpha=0.3, markeredgecolor='none',markersize=15)
ax.set_aspect('equal')
ax.set_yscale('log')
ax.set_xscale('log')
ax.grid()
ax.axis([1e-2,1e2,1e-2,1e2]);
ax.set_ylabel('Prediction [um/s]');
ax.set_xlabel('Measurement [um/s]');
#ax.set_title('Predictor variable: {}'.format(predictor),fontsize=15)
"""ax.plot(ax.get_xlim(), ax.get_xlim(),'--k')
ax.plot(ax.get_xlim(), ax.get_xlim(),'--k')"""
LIM = np.linspace(*ax.get_xlim())
ax.plot(LIM, LIM,'--k')
ax.plot(LIM, FAC*LIM,'--k')
ax.plot(LIM, (1/FAC)*LIM,'--k')
plt.text(1.5e-1, 1.5e-2, 'percentage captured within a factor of {} : {:0.2f} %'.format(FAC,percentage_captured), fontsize=15)

# save fig
plt.savefig('/home/nikhil.mukund/public_html/SEISMON/compare_predictions_with_measured.png')
print('Plot saved to: compare_predictions_with_measured.png')
