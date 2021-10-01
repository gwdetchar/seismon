# Sample script to access and load data from Carleton Database

from sshtunnel import SSHTunnelForwarder
from sqlalchemy import create_engine  
import pandas as pd

pd.set_option('display.max_rows', None)

server = SSHTunnelForwarder(
    ('virgo.physics.carleton.edu', 22),
    ssh_username="nmukund",
    ssh_pkey="~/.ssh/id_rsa.pub",
    ssh_private_key_password="",
    remote_bind_address=('127.0.0.1', 5432))

server.start()
engine = create_engine(f'postgresql+psycopg2://seismon:seismon@{server.local_bind_address[0]}:{server.local_bind_address[1]}/seismon')

#load files from database to dataframe
earthquakes_db = pd.read_sql_query('select * from public.earthquakes',con=engine)
ifos_db        = pd.read_sql_query('select * from public.ifos',con=engine)
predictions_db = pd.read_sql_query('select * from public.predictions',con=engine)


llo_processed_catalogue_db = pd.read_sql_query('select * from public.predictions',con=engine)
lho_processed_catalogue_db = pd.read_sql_query('select * from public.predictions',con=engine)

print('Printing earthquakes_db')
print(earthquakes_db)
print('Printing ifos_db')
print(ifos_db)
print('Printing seismon ml predictions')
print(predictions_db.loc[:,['event_id', 'ifo', 'rfamp', 'lockloss']])
print('Printing llo_processed_catalogue_dbi (needs to fix since it actually public.predictions)')
print(llo_processed_catalogue_db)

