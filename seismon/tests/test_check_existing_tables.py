#check for available tables
from sqlalchemy.schema import MetaData
from sqlalchemy import create_engine

engine = create_engine('postgresql+psycopg2://seismon:seismon@localhost:5432/seismon')

meta = MetaData()
meta.reflect(bind=engine)
avail_tables = meta.tables.keys()  

print(avail_tables)
