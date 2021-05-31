# Sample script to upload to Carleton Database


from sshtunnel import SSHTunnelForwarder
from sqlalchemy import create_engine  
import pandas as pd
from argparse import ArgumentParser
import random
import string


server = SSHTunnelForwarder(
    ('virgo.physics.carleton.edu', 22),
    ssh_username="nmukund",
    ssh_pkey="~/.ssh/id_rsa.pub",
    ssh_private_key_password="",
    remote_bind_address=('127.0.0.1', 5432))

server.start()
engine = create_engine(f'postgresql+psycopg2://seismon:seismon@{server.local_bind_address[0]}:{server.local_bind_address[1]}/seismon')


# Update DataBase
conn = engine.connect()
# Begin transaction
trans = conn.begin()



# generate random string uniqueID
rand_unique_ID = 'test_' + ''.join(random.choices(string.ascii_lowercase +
                             string.digits, k = 4))


lho_processed_catalogue_db = pd.read_sql_query('select * from public.predictions',con=engine)

# Get current table height
current_db_height = predictions_db.shape[0]

# Set Random values to column
parser = ArgumentParser()
parser.add_argument('--id_val',default=current_db_height+int(1),type=int, help="index corres. to db insert/update")
parser.add_argument('--created_at_val',default='2021-01-15 19:53:10.303660',type=str, help="Creation time")
