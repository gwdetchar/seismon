# Update database using sqlalchemy
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.schema import MetaData
from argparse import ArgumentParser


dialect = 'postgresql'
driver  = 'psycopg2'
uname   = 'seismon'
paswd   = 'seismon'
host    = 'localhost'
port    = '5432'
database = 'seismon'

#Create DataBase Engine: dialect+driver://username:password@host:port/database
#engine = create_engine('postgresql+psycopg2://postgres:letmein@localhost:5432/postgres')
engine = create_engine('{}+{}://{}:{}@{}:{}/{}'.format(dialect,driver,uname,paswd,host,port,database))


# Update DataBase
conn = engine.connect()
# Begin transaction
trans = conn.begin()


# Read existings database into Pandas DataFrame
earthquakes_db = pd.read_sql_query('select * from public.earthquakes',con=engine)
ifos_db        = pd.read_sql_query('select * from public.ifos',con=engine)
predictions_db = pd.read_sql_query('select * from public.predictions',con=engine)


# Get prediction table height
current_db_height = predictions_db.shape[0]


# Set Random values to column
parser = ArgumentParser()
parser.add_argument('--id_val',default=current_db_height+int(1),type=int, help="index corres. to db insert/update")
parser.add_argument('--created_at_val',default='2021-01-15 19:53:10.303660',type=str, help="Creation time")
parser.add_argument('--modified_val',default='2021-01-15 19:53:10.303660',type=str, help="Modified time")
parser.add_argument('--event_id_val',default='nn00805321',type=str, help="Unique Event ID")
parser.add_argument('--ifo_val',default='LHO',type=str,  help="IFO")
parser.add_argument('--d_val',default=803.700371, type=float, help="Distance from source[km]")
parser.add_argument('--p_val',default='2021-01-12 07:23:30.157502', type=str, help="Arrival time: P")
parser.add_argument('--s_val',default='2021-01-12 07:23:32.152035', type=str, help="Arrival time: S")
parser.add_argument('--r2p0_val',default='2021-01-12 07:28:25.730186',type=str, help="Arrival time: R2p0")
parser.add_argument('--r3p5_val',default='2021-01-12 07:25:33.508677',type=str, help="Arrival time: R3p5")
parser.add_argument('--r5p0_val',default='2021-01-12 07:24:24.620074',type=str, help="Arrival time: R5p0")
parser.add_argument('--rfamp_val',default=0.216506,type=float, help="Predicted Rf Amplitude [um/s]")
parser.add_argument('--lockloss_val',default=0,type=int, help="lockloss state")
parser.add_argument('--rfamp_measured_val',default=-1,type=float, help="measured Rayleigh wave amplitude at the site [um/s]")
args = parser.parse_args()

# Insert to DataBase
conn.execute('INSERT INTO public.predictions \
(id,\
created_at,\
modified,\
event_id,\
ifo,\
d,\
p,\
s,\
r2p0,\
r3p5,\
r5p0,\
rfamp,\
lockloss,\
rfamp_measured)\
VALUES \
(\
%(id)s,\
%(created_at)s,\
%(modified)s,\
%(event_id)s,\
%(ifo)s,\
%(d)s,\
%(p)s,\
%(s)s,\
%(r2p0)s,\
%(r3p5)s,\
%(r5p0)s,\
%(rfamp)s,\
%(lockloss)s,\
%(rfamp_measured)s\
)',\
{\
"id":                           args.id_val,\
"created_at":                   args.created_at_val,\
"modified":                     args.modified_val,\
"event_id":                     args.event_id_val,\
"ifo":                          args.ifo_val,\
"d":                            args.d_val,\
"p":                            args.p_val,\
"s":                            args.s_val,\
"r2p0":                         args.r2p0_val,\
"r3p5":                         args.r3p5_val,\
"r5p0":                         args.r5p0_val,\
"rfamp":                        args.rfamp_val,\
"lockloss":                     args.lockloss_val,
"rfamp_measured":               args.rfamp_measured_val
})


trans.commit()
# Close connection
conn.close()


