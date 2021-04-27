# convert EQ prediction from CSV to  pandas table and insert into the database

from os.path import basename,splitext
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('postgresql+psycopg2://seismon:seismon@localhost:5432/seismon')

csv_file_path='../input/LLO_processed_USGS_global_EQ_catalogue.csv'


# user inputs ends
# get just the filename in lower case without path and extension
tbl_name = splitext(basename(csv_file_path))[0].lower()   


data_df = pd.read_csv(csv_file_path)
data_df.to_sql(tbl_name, con=engine,  if_exists='replace')
