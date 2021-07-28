# convert measured past EQ Rayleight amplitudes from CSV to  pandas table 
# and insert them into the database

from os.path import basename,splitext
from sqlalchemy import create_engine
import pandas as pd
import datetime
import configparser
import seismon
import os
import sys


#-------------------------------
# Set Params

if_exists_then = 'skip'# {'replace' or 'append' or 'skip' }, replace option will swipe clean the entire catalogue-database and fill it again using entries from the csv file. 'skip' will check&skip if the database length matches the number of elements given in the CSV, if not it changes the flag to 'replace' and continues execution.

IFO_list = {'LLO','LHO'}

config_filename = 'config.yaml' # in inputs folder
#-------------------------------
# Options For Debugging/Testing
# Set to 1 to  Drop all entries from the dataFrame (tested with if_exists_then = 'replace') 
toggle_to_initialize_table_without_entries = 0

#-------------------------------
# Set Path
seismonpath = os.path.dirname(seismon.__file__)
csvfilepath = os.path.join(seismonpath,'input')
configpath = os.path.join(seismonpath,'input',config_filename)

#-------------------------------
# Read config
config = configparser.ConfigParser()
config.read(configpath)
password = config['database']['password']
user     = config['database']['user']
host     = config['database']['host']
port     = config['database']['port']
database = config['database']['database']
#-------------------------------
# Create database engine
engine = create_engine('postgresql+psycopg2://{}:{}@{}:{}/{}'.format(user,password,host,port,database))
# Connect DataBase
conn = engine.connect()

#-------------------------------
for IFO in IFO_list:
    # Specify path csv file
    csv_file_path=os.path.join(csvfilepath,'{}_processed_USGS_global_EQ_catalogue.csv'.format(IFO.upper()))
    db_catalogue_name = '{}_catalogues'.format(IFO.lower())



    # load to dataframe from CSV file
    data_df = pd.read_csv(csv_file_path)


   # Check if if_exists_then=='skip' and exit if database is already created
    if if_exists_then=='skip':
        processed_catalogue_db = pd.DataFrame()
        try:
            processed_catalogue_db = pd.read_sql_query('select * from public.{}'.format(db_catalogue_name),con=engine)
        except:
            processed_catalogue_db = pd.DataFrame()
        #check if lengths of data_df and processed_catalogue_db match 
        if len(data_df) == len(processed_catalogue_db):
            print('Database {} already exits and matches in length with the input CSV file, skipping...'.format(db_catalogue_name))
            print('To replace/append existing catalog database, modify if_exists_then variable.')
            sys.exit(0)
        else:
            if_exists_then='replace' 


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
    #data_df_filtered = data_df_filtered.loc[0:1,:]


    # Drop all entries from the dataFrame
    if toggle_to_initialize_table_without_entries:
        print('Dropping all entries from the dataFrame')
        data_df_filtered = data_df_filtered.iloc[0:0]

    # upload dataframe remotely to database
    data_df_filtered.to_sql('{}'.format(db_catalogue_name), con=engine,  if_exists=if_exists_then, index=False)
    print('Upload of {} successful'.format(db_catalogue_name))

    # Check if things worked (load remotely)
    print('Attempting to read {} table for verification...'.format(db_catalogue_name))
    processed_catalogue_db = pd.read_sql_query('select * from public.{}'.format(db_catalogue_name),con=engine)
    print(processed_catalogue_db)

# close connection
conn.close()


