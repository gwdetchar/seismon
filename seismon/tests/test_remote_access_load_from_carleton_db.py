# Sample script to access and load data from Carleton Database

from sshtunnel import SSHTunnelForwarder
from sqlalchemy import create_engine  
import pandas as pd
import numpy as np


#pd.set_option('display.max_rows', None)

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


llo_processed_catalogue_db = pd.read_sql_query('select * from public.llo_catalogues',con=engine)
lho_processed_catalogue_db = pd.read_sql_query('select * from public.lho_catalogues',con=engine)

print('Printing earthquakes_db')
print(earthquakes_db)
print('Printing ifos_db')
print(ifos_db)
print('Printing seismon ml predictions')
print(predictions_db.loc[:,['event_id', 'ifo', 'rfamp','rfamp_measured', 'lockloss']])



## Compare Predictions vs Measurements (both LLO & LHO combined)
############################
print('Comparing Predictions vs Measurements')
# get predictions &  measurements
pred=predictions_db.loc[:,['rfamp','rfamp_measured']]
# pandas to scatter plot
ax = pred.loc[pred['rfamp_measured']!=-1,:].plot.scatter(x='rfamp_measured',y='rfamp')
ax.grid(color='r', linestyle='-', linewidth=0.3)
#ax.set_aspect ('equal', adjustable='box')
#ax.set_aspect('equal', adjustable='datalim')

# Add x=y line
ax.plot([0,1],[0,1], transform=ax.transAxes,linestyle='--', color='k', lw=1.5)
ax.set_yscale('log')
ax.set_xscale('log')

ax.set_aspect('equal')

minimum = np.min((ax.get_xlim(),ax.get_ylim()))
maximum = np.max((ax.get_xlim(),ax.get_ylim()))

ax.set_xlim(minimum*1.2,maximum*1.2)
ax.set_ylim(minimum*1.2,maximum*1.2)

ax.margins(0.1)


# get figure
fig = ax.get_figure()
# save fig
fig.savefig('compare_predictions_with_measured.png')
print('Plot saved to: compare_predictions_with_measured.png')
