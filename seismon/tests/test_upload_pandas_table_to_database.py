# convert EQ prediction from CSV to  pandas table and insert into the database

from os.path import basename,splitext
from sqlalchemy import create_engine
import pandas as pd
import datetime


# Specify path csv file
csv_file_path='../input/LLO_processed_USGS_global_EQ_catalogue.csv'
db_catalogue_name = 'llo_catalogues'


#-------------------------------
engine = create_engine('postgresql+psycopg2://seismon:seismon@localhost:5432/seismon')

# Connect DataBase
conn = engine.connect()

"""# OLD 
# get just the filename in lower case without path and extension
tbl_name = splitext(basename(csv_file_path))[0].lower()   
data_df = pd.read_csv(csv_file_path)
data_df.to_sql(tbl_name, con=engine,  if_exists='replace')"""


# load to dataframe from CSV file
data_df = pd.read_csv(csv_file_path)

# Select few columns [unique_id, peak_data_um-pers-sec_mean_subtracted]
data_df_filtered = data_df.filter(['id','time','place','latitude','longitude','mag','depth','SNR','peak_data_um_mean_subtracted'],axis=1)
data_df_filtered = data_df_filtered.rename(columns={'id':'event_id'})

# get current UTC time
created_at_value = datetime.datetime.utcnow().strftime("%a %b %d %H:%M:%S %Z %Y")
modified_value = datetime.datetime.utcnow().strftime("%a %b %d %H:%M:%S %Z %Y")

# add created_at &  modified time
data_df_filtered.insert(2,'created_at',created_at_value,True)
data_df_filtered.insert(3,'modified',modified_value,True)

# Only keep initial entries (to speed up the test)
data_df_filtered = data_df_filtered.loc[0:1,:]

# upload dataframe remotely to database
data_df_filtered.to_sql('{}'.format(db_catalogue_name), con=engine,  if_exists='append', index=False)

print('Remote upload successful')

# Check if things worked (load remotely)
print('Attempting to read table remotely...')
processed_catalogue_db = pd.read_sql_query('select * from public.{}'.format(db_catalogue_name),con=engine)
print(processed_catalogue_db)

# close connection
conn.close()