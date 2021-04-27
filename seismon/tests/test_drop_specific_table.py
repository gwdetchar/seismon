
#https://stackoverflow.com/questions/35918605/how-to-delete-a-table-in-sqlalchemy

import logging
from sqlalchemy import MetaData
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.declarative import declarative_base
from os import system as sys


#table_name_to_drop = 'LHO_processed_USGS_global_EQ_catalogue.csv'
print("Available database tables...")
sys("python test_check_existing_tables.py")
table_name_to_drop = txt = input("Enter the tablename to delete: ")


DATABASE = {
           'drivername': 'postgresql',
           'host': 'localhost',
           'port': '5432',
           'username': 'seismon',
           'password': 'seismon',
           'database': 'seismon'
                          }

def drop_table(table_name):
    engine = create_engine(URL(**DATABASE))
    base = declarative_base()
    metadata = MetaData(engine, reflect=True)
    table = metadata.tables.get(table_name)
    if table is not None:
        logging.info(f'Deleting {table_name} table')
        base.metadata.drop_all(engine, [table], checkfirst=True)


drop_table(table_name_to_drop)
