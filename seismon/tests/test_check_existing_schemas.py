#check for available schemas
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy import create_engine

engine = create_engine('postgresql+psycopg2://seismon:seismon@localhost:5432/seismon')
insp = Inspector.from_engine(engine)
avail_schemas = insp.get_schema_names()
print(avail_schemas)



