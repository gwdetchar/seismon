# load database table into pandas
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('postgresql+psycopg2://seismon:seismon@localhost:5432/seismon')

#df_from_db = pd.read_sql_query('select * from public.earthquakes',con=engine)
#df_from_db = pd.read_sql_query('select * from public.ifos',con=engine) 

earthquakes_db = pd.read_sql_query('select * from public.earthquakes',con=engine)
ifos_db        = pd.read_sql_query('select * from public.ifos',con=engine)
predictions_db = pd.read_sql_query('select * from public.predictions',con=engine)


llo_processed_catalogue_db = pd.read_sql_query('select * from public.llo_catalogues',con=engine)
lho_processed_catalogue_db = pd.read_sql_query('select * from public.lho_catalogues',con=engine)
virgo_processed_catalogue_db = pd.read_sql_query('select * from public.virgo_catalogues',con=engine)

print(earthquakes_db)
print(ifos_db)
#print(predictions_db)
print(llo_processed_catalogue_db)
print(lho_processed_catalogue_db)

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
print(predictions_db)

