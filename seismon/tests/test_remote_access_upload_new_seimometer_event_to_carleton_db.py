# Sample script to upload new entries to Carleton Catalogue Database


from sshtunnel import SSHTunnelForwarder
from sqlalchemy import create_engine  
import pandas as pd
from argparse import ArgumentParser
import datetime
from argparse import ArgumentParser
parser = ArgumentParser()

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
    # get corresponding id from processed_catalog
    miD = llo_processed_catalogue_db['event_id'] == event_id 
    # get corresponding rfamp measured value from processed_catalog
    rfamp_measured = llo_processed_catalogue_db.loc[miD]['peak_data_um_mean_subtracted'].values[0]

elif args.db_catalogue_name=='lho_catalogues':
    ifo_name = 'LHO'
    # get corresponding id from processed_catalog
    miD = lho_processed_catalogue_db['event_id'] == event_id 
    # get corresponding rfamp measured value from processed_catalog
    rfamp_measured = lho_processed_catalogue_db.loc[miD]['peak_data_um_mean_subtracted'].values[0]



# get event id from predictions table
piD=(predictions_db['event_id']==event_id) & (predictions_db['ifo']==ifo_name)


# replace current value for the rfamp_measured (-1)  with the observed value
current_val  = predictions_db['rfamp_measured'][piD].values[0]
predictions_db['rfamp_measured'][piD]=predictions_db['rfamp_measured'][piD].replace(current_val,rfamp_measured)


# Update actual database 'predictions' table 
if_exists_then='replace'
predictions_db.to_sql('{}'.format('predictions'), con=engine,  if_exists=if_exists_then, index=False)
#-------------------------------------

print('Remote upload successful')

# Check if things worked (load remotely)
print('Attempting to read back from database-table (for verification)...')
processed_catalogue_db = pd.read_sql_query('select * from public.{}'.format(args.db_catalogue_name),con=engine)
predictions_db = pd.read_sql_query('select * from public.predictions',con=engine)
print('Printing processed catalog')
print(processed_catalogue_db)
print('Printing updated prediction table')
print(predictions_db)
# close connection
conn.close()
