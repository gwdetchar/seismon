import datetime
import json
import os
import io
import urllib.parse
import math
import re
import requests
import shutil
import tempfile
from itertools import chain

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import time
import astropy.units as u
from astropy.table import Table
import pandas as pd
import matplotlib.style
import pkg_resources
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import cartopy.crs as ccrs

from flask import (
    abort, flash, jsonify, make_response, redirect, render_template, request,
    Response, url_for)
from flask_caching import Cache
from flask_login import (
    current_user, login_required, login_user, logout_user, LoginManager)
from wtforms import (
    BooleanField, FloatField, RadioField, TextField, IntegerField, StringField, PasswordField, SubmitField)
from wtforms_components.fields import (
    DateTimeField, DecimalSliderField, SelectField)
from wtforms import validators
from wtforms.validators import (
    DataRequired,
    Email,
    EqualTo,
    Length,
    Optional
)
from wtforms_alchemy.fields import PhoneNumberField
from passlib.apache import HtpasswdFile
from sqlalchemy.orm.exc import MultipleResultsFound, NoResultFound

from seismon.config import app
from seismon import models

#
#
# From http://wtforms-alchemy.readthedocs.io/en/latest/advanced.html#using-wtforms-alchemy-with-flask-wtf  # noqa: E501
from flask_wtf import FlaskForm
from wtforms_alchemy import model_form_factory
# The variable db here is a SQLAlchemy object instance from
# Flask-SQLAlchemy package


BaseModelForm = model_form_factory(FlaskForm)



class ModelForm(BaseModelForm):
    @classmethod
    def get_session(cls):
        return models.db.session
#
#
#

# Server-side cache for rendered view functions.
cache = Cache(app, config={
    'CACHE_DEFAULT_TIMEOUT': 86400,
    'CACHE_REDIS_HOST': 'redis',
    'CACHE_TYPE': 'redis'})

def one_or_404(query):
    # FIXME: https://github.com/mitsuhiko/flask-sqlalchemy/pull/527
    rv = query.one_or_none()
    if rv is None:
        abort(404)
    else:
        return rv


def human_time(*args, **kwargs):
    secs = float(datetime.timedelta(*args, **kwargs).total_seconds())
    units = [("day", 86400), ("hour", 3600), ("minute", 60), ("second", 1)]
    parts = []
    for unit, mul in units:
        if secs / mul >= 1 or mul == 1:
            if mul > 1:
                n = int(math.floor(secs / mul))
                secs -= n * mul
            else:
                n = secs if secs != int(secs) else int(secs)
            parts.append("%s %s%s" % (n, unit, "" if n == 1 else "s"))
    return parts[0]


@app.route('/')
def index():

    earthquakes = models.db.session.query(models.Earthquake).order_by(models.Earthquake.date.desc()).all()

    return render_template(
        'index.html',
        earthquakes=earthquakes)

@app.route('/earthquake/<event_id>/')
def earthquake(event_id):

    query = models.db.session.query(models.Earthquake).filter_by(event_id=event_id)

    try:
        earthquake = query.first()
    except NoResultFound:
        abort(404)
    predictions = models.db.session.query(models.Prediction).filter_by(event_id=event_id).all()

    return render_template(
            'earthquake.html',
            earthquake=earthquake,
            predictions=predictions)

@app.route('/earthquake/<event_id>/globemap.png')
def globemap(event_id):

    query = models.db.session.query(models.Earthquake).filter_by(event_id=event_id)

    try:
        earthquake = query.first()
    except NoResultFound:
        abort(404)

    ifos = models.db.session.query(models.Ifo).all()

    fig = Figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

    # make the map global rather than have it zoom in to
    # the extents of any plotted data
    ax.set_global()
    ax.stock_img()
    ax.coastlines()

    ax.plot(earthquake.lon, earthquake.lat, '*',
            markersize=15, color='k',
            transform=ccrs.PlateCarree())
    for ifo in ifos:
        ax.plot(ifo.lon, ifo.lat, 'o',
                markersize=6, color='r',
                transform=ccrs.PlateCarree())
        ax.text(ifo.lon+2, ifo.lat+2, ifo.ifo,
                transform=ccrs.PlateCarree())        

    plt.show()

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

