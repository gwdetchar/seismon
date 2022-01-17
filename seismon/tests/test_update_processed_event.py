
from obspy.geodetics.base import gps2dist_azimuth
from obspy.taup import TauPyModel

import numpy as np

import seismon
from seismon import (eqmon, utils)
from seismon.config import app

from flask_login.mixins import UserMixin
from flask_sqlalchemy import SQLAlchemy

from sshtunnel import SSHTunnelForwarder


import sqlalchemy as sa
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import sessionmaker, scoped_session, relationship
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import create_engine  



import pandas as pd
from argparse import ArgumentParser
import datetime
from argparse import ArgumentParser

pd.set_option('display.max_rows',100)

parser = ArgumentParser()
parser.add_argument('--created_at',default='2021-01-15 19:53:10.303660',type=str, help="Creation time")
parser.add_argument('--modified',default='2021-01-15 19:53:10.303660',type=str, help="Modified time")
parser.add_argument( '--time', default='12-Sep-2015 20:32:26', type=str,help='EQ event time [UTC]')
parser.add_argument('--event_id',default='nn00805321',type=str, help="Unique Event ID")
parser.add_argument( '--place',   default='Not Specified',type=str, help='EQ event location')

parser.add_argument( '--latitude', default=-32.6066, type=float,help='EQ latitude')
parser.add_argument( '--longitude', default=-178.0287, type=float, help='EQ longitude')
parser.add_argument( '--mag', default=5.9, type=float, help='EQ magnitude')
parser.add_argument( '--depth', default=8, type=float, help='EQ depthh [km]')
parser.add_argument( '--SNR', default=19.9, type=float, help='SNR of PeakAmplitude estimation')
parser.add_argument( '--peak_data_um_mean_subtracted', default=0.33, type=float, help='EQ estimated peak amplitude [um/s]')


parser.add_argument('--ifo',default='LHO',type=str,  help="IFO")
parser.add_argument('--d',default=803.700371, type=float, help="Distance from source[km]")
parser.add_argument('--p',default='2021-01-12 07:23:30.157502', type=str, help="Arrival time: P")
parser.add_argument('--s',default='2021-01-12 07:23:32.152035', type=str, help="Arrival time: S")
parser.add_argument('--r2p0',default='2021-01-12 07:28:25.730186',type=str, help="Arrival time: R2p0")
parser.add_argument('--r3p5',default='2021-01-12 07:25:33.508677',type=str, help="Arrival time: R3p5")
parser.add_argument('--r5p0',default='2021-01-12 07:24:24.620074',type=str, help="Arrival time: R5p0")
parser.add_argument('--rfamp',default=0.216506,type=float, help="Predicted Rf Amplitude [um/s]")
parser.add_argument('--lockloss',default=0,type=int, help="lockloss state")
parser.add_argument('--rfamp_measured',default=-1,type=float, help="measured Rayleigh wave amplitude at the site [um/s]")
parser.add_argument( '--db_catalogue_name', default='lho_catalogues', type=str,help="Specify catalogue to write to {'llo_catalogues','lho_catalogues','virgo_catalogues'}")


parser.add_argument( '--uname', default='nmukund', type=str,help='username')
parser.add_argument( '--pubkey', default='~/.ssh/id_rsa.pub', type=str,help='ssh public key')

args = parser.parse_args()


#-------------------------------------
# if event already exists then for catalogue tables
if_exists_then = 'append' #{'append','replace'}


server = SSHTunnelForwarder(
    ('virgo.physics.carleton.edu', 22),
    ssh_username=args.uname,
    ssh_pkey=args.pubkey,
    ssh_private_key_password="",
    remote_bind_address=('127.0.0.1', 5432))

server.start()
engine = create_engine(f'postgresql+psycopg2://seismon:seismon@{server.local_bind_address[0]}:{server.local_bind_address[1]}/seismon')


# Connect DataBase
conn = engine.connect()
print('Remote connection to Carleton database successful')



db = SQLAlchemy(app)

DBSession = scoped_session(sessionmaker())
EXECUTEMANY_PAGESIZE = 50000
utcnow = func.timezone('UTC', func.current_timestamp())



class BaseMixin(object):
    query = DBSession.query_property()
    id = sa.Column(sa.Integer, primary_key=True)
    created_at = sa.Column(sa.DateTime, nullable=False, default=utcnow)
    modified = sa.Column(sa.DateTime, default=utcnow, onupdate=utcnow,
                         nullable=False)

    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower() + 's'

    __mapper_args__ = {'confirm_deleted_rows': False}

    def __str__(self):
        return to_json(self)

    def __repr__(self):
        attr_list = [f"{c.name}={getattr(self, c.name)}"
                     for c in self.__table__.columns]
        return f"<{type(self).__name__}({', '.join(attr_list)})>"

    def to_dict(self):
        if sa.inspection.inspect(self).expired:
            DBSession().refresh(self)
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def get_if_owned_by(cls, ident, user, options=[]):
        obj = cls.query.options(options).get(ident)

        if obj is not None and not obj.is_owned_by(user):
            raise AccessError('Insufficient permissions.')

        return obj

    def is_owned_by(self, user):
        raise NotImplementedError("Ownership logic is application-specific")

    @classmethod
    def create_or_get(cls, id):
        obj = cls.query.get(id)
        if obj is not None:
            return obj
        else:
            return cls(id=id)


Base = declarative_base(cls=BaseMixin)


DBSession.configure(bind=conn)
Base.metadata.bind = conn

class Prediction(Base):
    """Prediction information"""

    id = sa.Column(sa.Integer, primary_key=True)

    event_id = sa.Column(
        sa.String,
        nullable=False,
        comment='Earthquake ID')

    ifo = sa.Column(
        sa.String,
        nullable=False,
        comment='Detector name')

    d = sa.Column(
        sa.Float,
        nullable=False,
        comment='Distance [km]')

    p = sa.Column(
        sa.DateTime,
        nullable=False,
        comment='P-wave time')

    s = sa.Column(
        sa.DateTime,
        nullable=False,
        comment='S-wave time')

    r2p0 = sa.Column(
           sa.DateTime,
           nullable=False,
           comment='R-2.0 km/s-wave time')

    r3p5 = sa.Column(
           sa.DateTime,
           nullable=False,
           comment='R-3.5 km/s-wave time')

    r5p0 = sa.Column(
           sa.DateTime,
           nullable=False,
           comment='R-5.0 km/s-wave time')

    rfamp = sa.Column(
            sa.Float,
            nullable=False,
            comment='Earthquake amplitude predictions [m/s]')

    #(added by NM on 04/10/21)
    rfamp_measured = sa.Column(
            sa.Float,
            nullable=False,
            comment='Earthquake amplitude measured [m/s]')

    lockloss = sa.Column(
               sa.INT,
               nullable=False,
               comment='Earthquake lockloss prediction')


data_dict = {
    'event_id':[args.event_id],
    'time':[args.time],
    'created_at':[args.created_at],
    'modified':[args.modified],
    'place':[args.place],
    'latitude':[args.latitude],
    'longitude':[args.longitude],
    'mag':[args.mag],
    'depth':[args.depth],
    'SNR':[args.SNR],
    'peak_data_um_mean_subtracted':[args.peak_data_um_mean_subtracted]
    }
data_df_filtered = pd.DataFrame(data_dict)


# upload dataframe remotely to database
print('Remotely updating {} processed catalog',args.db_catalogue_name)
data_df_filtered.to_sql('{}'.format(args.db_catalogue_name), con=engine,  if_exists=if_exists_then, index=False)

#-------------------------------------
# Code snippet to add back measurements to Predictions table (for comparison)
# Added on 05/10/2021 by NM
#get PREDICTIONS DATABASE Table
predictions_db = pd.read_sql_query('select * from public.predictions',con=engine)
# get CATALOG DATABASE Tables
llo_processed_catalogue_db = pd.read_sql_query('select * from public.llo_catalogues',con=engine)
lho_processed_catalogue_db = pd.read_sql_query('select * from public.lho_catalogues',con=engine)


# get event_id
event_id=args.event_id


#get IFO name from the catalog 
if args.db_catalogue_name=='llo_catalogues':
    ifo_name = 'LLO'
    processed_catalogue_db = llo_processed_catalogue_db
elif args.db_catalogue_name=='lho_catalogues':
    ifo_name = 'LHO'
    processed_catalogue_db = lho_processed_catalogue_db

# get corresponding id from processed_catalogue_db
miD = processed_catalogue_db['event_id'] == event_id 
# get corresponding rfamp measured value from processed_catalog
try:
    rfamp_measured = processed_catalogue_db.loc[miD]['peak_data_um_mean_subtracted'].values[0]
except:
    rfamp_measured = np.array([])    

if np.size(rfamp_measured)!= 0:
    # get event id from predictions table
    piD=(predictions_db['event_id']==event_id) & (predictions_db['ifo']==ifo_name)
    # replace current value for the rfamp_measured (-1)  with the observed value
    current_val  = predictions_db['rfamp_measured'][piD].values[0]

    #predictions_db['rfamp_measured'][piD]=predictions_db['rfamp_measured'][piD].replace(current_val,rfamp_measured,ifo_name)    
    # Update actual database 'predictions' table 
    #if_exists_then='append'
    #print('Event {}  found in {}. Remotely updating predictions DataBase table for {}'.format(args.event_id,args.db_catalogue_name))
    #predictions_db.to_sql('{}'.format('predictions'), con=engine,  if_exists=if_exists_then, index=False)


    DBSession().merge(Prediction(event_id=str(predictions_db[piD].loc[0]['event_id']) ,
                                    ifo=str(predictions_db[piD].loc[0]['ifo']) ,
                                    d=predictions_db[piD].loc[0]['d'] ,
                                    p=predictions_db[piD].loc[0]['p'] ,
                                    s=predictions_db[piD].loc[0]['s'] ,
                                    r2p0=predictions_db[piD].loc[0]['r2p0'] ,
                                    r3p5=predictions_db[piD].loc[0]['r3p5'] ,
                                    r5p0=predictions_db[piD].loc[0]['r5p0'] ,
                                    rfamp=predictions_db[piD].loc[0]['rfamp'] ,
                                    rfamp_measured=rfamp_measured,
                                    lockloss= int(predictions_db[piD].loc[0]['lockloss'])   ))
    DBSession().commit()



else:
    print('Event {} not found in {}. Skipping remote update of predictions DataBase table for {}'.format(args.event_id,args.db_catalogue_name))





# close connection
conn.close()