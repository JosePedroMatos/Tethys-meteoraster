'''
Created on 10/02/2023

@author: Jose Pedro Matos
'''

import sys
import types
import matplotlib
import pickle
import math
import copy
import gc
import os
import zipfile
import warnings
import numpy as np
import xarray as xr
import datetime as dt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xml.etree.ElementTree as et
from pathlib import Path
from osgeo import gdal, ogr
from dateutil import relativedelta
from collections.abc import Iterable
import scipy.interpolate as interp
import cartopy.crs as ccrs #conda install -c conda-forge cartopy
import cartopy.feature as cf
import cartopy.io.shapereader as shpreader
from collections.abc import Iterable
from tethys_raster_file import TethysRasterFile

def readNMME(variable, date):
    '''
    Reads forecast data from all the available NMME data for a certain date (month).
    '''
    
    '''
    pr - total precipitation
    tas - temperature
    
    Return a dict:
        data: 5-D numpy array [productionDate, ensembleMember, leadtime, latitude, longitude]
        production dates: 1-D numpy array
        leadtimes: 1-D numpy array
        latitudes: 1-D numpy array
        longitudes: 1-D numpy array
    '''
    
    variableTranslation = {'t2m': 'tmp2m',
                           'p3059': 'prate'}
    
    
    path = Path(NMMEpATH) / date.strftime('%Y%m0800')
    files = [i for i in path.glob(variableTranslation[variable] + '.*.fcst.*.grb')]
    
    dataMatrix = None
    data = {}
    for f0 in files:
        product = f0.name.split('.')[2]
        print('Collecting %s (%s)' % (product, date.strftime('%m-%Y')))

        with xr.open_dataset(str(f0)) as ds:
            latitudes = ds.latitude.data
            longitudes = ds.longitude.data
            productionDates = ds.time.data
            tmp = ds[variable][:, :, :].data
            values = np.expand_dims(tmp, [0, 1])
            leadtimes = ds.step.data
        print(leadtimes)
        print(values.shape)


        if product in NMMEensemble:
            productIdx = NMMEensemble.index(product)
    
            if isinstance(dataMatrix, type(None)):
                dataMatrix = np.empty([1, len(NMMEensemble), 12] + list(values.shape[-2:]) ) * np.NaN
                data['leadtimes'] = leadtimes
                data['latitudes'] = ds.latitude.data
                data['longitudes'] = ds.longitude.data
                data['productionDates'] = np.array([pd.to_datetime(ds.time.data) + pd.DateOffset(days=-7)]).astype('datetime64[ns]')

            else:
                if data['leadtimes'].size < leadtimes.size:
                    data['leadtimes'] = leadtimes
    
    
            dataMatrix[:, productIdx, :values.shape[2], :, :] = values
        
    dataMatrix = dataMatrix[:, :, :leadtimes.size, :, :]
    data['data'] = dataMatrix
    data['leadtimes'] = np.array(data['leadtimes'])[:dataMatrix.shape[2]]

    return data

def readSEAS5(variable, date):
    '''
    
    ''' 
    
    with xr.open_dataset(SEAS5pATH[variable] % date.strftime('%Y.%m'), engine='cfgrib', indexpath='') as ds:
        latitudes = ds.latitude.data
        longitudes = ds.longitude.data
        time = ds.time.data
        step = ds.step.data
        data = ds[variable][:, :, :, :].data
        dates = time + step


    return (data, dates, time, latitudes, longitudes)

def readIberia01(variable):
    '''
    pr - total precipitation
    tas - temperature
    
    Return a dict:
        data: 5-D numpy array [productionDate, ensembleMember, leadtime, latitude, longitude]
        production dates: 1-D numpy array
        leadtimes: 1-D numpy array
        latitudes: 1-D numpy array
        longitudes: 1-D numpy array
    '''
    
    data = {}
    with xr.open_dataset(IBERIA01pATH % variable) as ds:
        data['latitudes'] = ds.lat.data
        data['longitudes'] = ds.lon.data
        data['productionDates'] = ds.time.data
        tmp = ds[variable][:, :, :].data
        data['data'] = np.expand_dims(tmp, [1, 2])
        data['leadtimes'] = np.array([pd.DateOffset(days=0)])
    
    return data

def readERA5_monthly(file, variable, **kwargs):
    '''
    Return a dict:
        data: 5-D numpy array [productionDate, ensembleMember, leadtime, latitude, longitude]
        production dates: 1-D numpy array
        leadtimes: 1-D numpy array
        latitudes: 1-D numpy array
        longitudes: 1-D numpy array
    '''
    
    data = {}
    with xr.open_dataset(file, engine='cfgrib', indexpath='', **kwargs) as ds:
        data['latitudes'] = ds.latitude.data
        data['longitudes'] = ds.longitude.data
        data['productionDates'] = pd.to_datetime(ds.time.data)
        tmp = ds[variable][:, :, :].data
        data['data'] = np.expand_dims(tmp, [1, 2])
        data['leadtimes'] = np.array([pd.DateOffset(days=0)])

    
    if variable == 'tp':
        data['data'] *= 1000 #to mm/day
        days_in_month = data['productionDates'].days_in_month        
        days_in_month = np.expand_dims(days_in_month, (1, 2, 3, 4))
        days_in_month = np.tile(days_in_month, [1] + list(data['data'].shape[1:]))
        data['data'] *= days_in_month
        units = 'mm/month'
    elif variable == 't2m':
        data['data'] -= 273.15
        units = 'C'
    elif variable == 'ro':
        data['data'] *= 1000
        units = 'mm/month'
    elif variable == 'e':
        data['data'] *= 1000
        units = 'mm/month'
    
    tmp = MeteoRaster(data, variable=variable, units=units)
    return tmp
    
def readERA5Land_monthly(file, variable, **kwargs):
    '''
    Return a dict:
        data: 5-D numpy array [productionDate, ensembleMember, leadtime, latitude, longitude]
        production dates: 1-D numpy array
        leadtimes: 1-D numpy array
        latitudes: 1-D numpy array
        longitudes: 1-D numpy array
    '''
    
    data = {}
    with xr.open_dataset(file, engine='cfgrib', indexpath='', **kwargs) as ds:
        data['latitudes'] = ds.latitude.data
        data['longitudes'] = ds.longitude.data
        data['productionDates'] = pd.to_datetime(ds.time.data)
        tmp = ds[variable][:, :, :].data
        data['data'] = np.expand_dims(tmp, [1, 2])
        data['leadtimes'] = np.array([pd.DateOffset(days=0)])

    
    if variable == 'tp':
        data['data'] *= 1000 #to mm/day
        days_in_month = data['productionDates'].days_in_month        
        days_in_month = np.expand_dims(days_in_month, (1, 2, 3, 4))
        days_in_month = np.tile(days_in_month, [1] + list(data['data'].shape[1:]))
        data['data'] *= days_in_month
        units = 'mm/month'
    elif variable == 't2m':
        data['data'] -= 273.15
        units = 'C'
    elif variable == 'ro':
        data['data'] *= 1000
        units = 'mm/month'
    elif variable == 'e':
        data['data'] *= 1000
        units = 'mm/month'
    
    tmp = MeteoRaster(data, variable=variable, units=units)
    return tmp

def read_ERA5Land_hourly(file, variable, file_next_year=None):
    '''
    Returns a MeteoRaster object with the ERA5 Land data
    '''
    
    cumulative = {'tp': True, 'ssr': True, 'u10': False, 'v10': False, 'sd': False, 't2m': False}
    
    def _read_ERA5Land_helper(file, variable):
        '''
        Just reads the grib file
        '''
        
        data = {}
        with xr.open_dataset(file, engine='cfgrib', indexpath='') as ds:
            data['latitudes'] = ds.latitude.data
            data['longitudes'] = ds.longitude.data
            data['productionDates'] = ds.time.data
            if isinstance(data['productionDates'], np.datetime64):
                data['productionDates'] = np.array([data['productionDates']])
            data['data'] = ds[variable][:, :, :].data
            data['steps'] = ds.step.data
            
            return data
        
    data = _read_ERA5Land_helper(file, variable)
    
    if cumulative[variable]:
        first_timestamp = pd.Timestamp(data['productionDates'][0])
        last_timestamp = pd.Timestamp(data['productionDates'][-1])
        if first_timestamp==pd.Timestamp(first_timestamp.year, 12, 31) and np.isnan(data['data'][0, -2, :, :]).min():
            # Last day of the previous year with data on the last hour only... To remove.
            data['data'] = data['data'][1:, :, :, :]
            data['productionDates'] = data['productionDates'][1:]
            warnings.warn('Last hour of the previous year dropped...')

        if last_timestamp==pd.Timestamp(last_timestamp.year, 12, 31) and np.isnan(data['data'][-1, -1, :, :]).max():
            if file_next_year is not None:
                # Last day of the previous year with data missing on the last hour... To lead next year and take that value
                data_last = _read_ERA5Land_helper(file_next_year, variable)
                data['data'][-1, -1, :, :] = data_last['data'][0, -1, :, :]
                warnings.warn('First hour of the following year added...')
            
        if len(data['data'].shape)==4:
            # (date, hour, ...) > (timestamp, ---)
            data['data'] = np.diff(data['data'], n=1, axis=1, prepend=0)
        else:
            data['data'] = np.diff(data['data'], n=1, axis=0, prepend=0)
    
    if variable == 'tp':
        data['data'] *= 1000
        units = 'mm/hr'
    elif variable == 't2m':
        data['data'] -= 273.15
        units = 'C'
    elif variable == 'sd':
        #=======================================================================
        # raise Exception('Cumulative?')
        #=======================================================================
        data['data'] *= 1000
        units = 'mm'
    elif variable == 'ssr':
        data['data'] /= 3600
        units = 'W/m2'
    elif variable == 'u10' or variable == 'v10':
        units = 'm/s'

        #=======================================================================
        # if variable == 'tp' or variable == 't2m':
        #     tmp = np.reshape(tmp, (tmp.shape[0]*tmp.shape[1], tmp.shape[2], tmp.shape[3]))
        #     times = np.tile(data['productionDates'], (24, 1)).transpose() + np.tile(ds.step.data, (data['productionDates'].shape[0], 1))
        #     data['productionDates'] = times.ravel()
        #=======================================================================
        
    if not isinstance(data['steps'], Iterable):
        data['steps'] = [data['steps']]
        
    if variable !='sd':
        data['data'] = np.reshape(data['data'], (data['data'].shape[0]*data['data'].shape[1], data['data'].shape[-2], data['data'].shape[-1]))
    
        times = np.tile(data['productionDates'], (24, 1)).transpose() + np.tile(data['steps'] - data['steps'][0], (data['productionDates'].shape[0], 1))
        data['productionDates'] = times.ravel()
    
    data['data'] = np.expand_dims(data['data'], [1, 2])
    data['leadtimes'] = np.array([pd.DateOffset(days=0)])
            
    tmp = MeteoRaster(data, units=units, variable=variable)
    tmp.trim()
    
    return tmp

def read_GFS(path, timestamp, leadtime, variable, crop=None):
    '''
        TMP
        PRATE
        WEASD
        DPT
        RH
        SNOD
        SUNSD
        UGRD
        VGRD
    '''
    
    GFS_info = dict(TMP=dict(units='C', variable='t2m', transform=lambda x: x),
                    PRATE=dict(units='mm/3h', variable='tp', transform=lambda x: x*3600*3),  # kg/(m^2 s) > mm/3h
                    )
    
    lead_hours = leadtime // np.timedelta64(1, 'h')
    
    path_ = Path(path) / timestamp.strftime('%Y/%m') / '00' / ('%03u' % lead_hours)
    file = path_ / 'gfs_4_{date:s}_00_{lead:03d}.nc'.format(date=timestamp.strftime('%Y.%m.%d'), lead=lead_hours)
    
    data = {}
    with xr.open_dataset(file) as ds:
        data['latitudes'] = ds.lat.data
        data['longitudes'] = ds.lon.data
        data['productionDates'] = ds.productionDatetimes.data
        data['leadtimes'] = np.array([pd.Timedelta(hours=int(ds.leadtimes.data/1000000000/3600))])
        tmp = ds[variable][:, :, :, :].data
        data['data'] = np.expand_dims(tmp, [1])
        
        if GFS_info[variable]['transform']:
            data['data'] = GFS_info[variable]['transform'](data['data'])
    
    meteoraster = MeteoRaster(data, units=GFS_info[variable]['units'], variable=GFS_info[variable]['variable'])
    if crop:
        meteoraster = meteoraster.getCropped(**crop)
    
    return meteoraster

def read_C3S(file, variable, engine='cfgrib', **kwargs):
    '''
    Reads c3s forecasts (seasonal forecast monthly statistics on single levels)
    https://cds.climate.copernicus.eu/datasets/seasonal-monthly-single-levels?tab=overview
    
    v0.1
    2024.10.04
    
    Return a dict:
    data: 5-D numpy array [productionDate, ensembleMember, leadtime, latitude, longitude]
    production dates: 1-D numpy array
    leadtimes: 1-D numpy array
    latitudes: 1-D numpy array
    longitudes: 1-D numpy array
    '''

    data = {}
    with xr.open_dataset(file, engine=engine, indexpath='', **kwargs) as ds:
        data['latitudes'] = ds.latitude.data
        data['longitudes'] = ds.longitude.data
        data['productionDates'] = ds.time.data
        data_ = ds[variable][:, :, :, :].data
        steps_ = pd.to_timedelta([pd.Timedelta(f'{d}d') for d in (ds.step.data/86400/1000000000).astype(int)])
        
    data_ = np.rollaxis(data_, 1, 0)
        
    months_ = np.round(steps_.days/30).astype(int)
    months_unique = np.unique(months_)
    data['leadtimes'] = [pd.DateOffset(months=m-1) for m in months_unique]
    
    data_shape = list(data_.shape)
    data_shape[2] = len(months_unique)
    data['data'] = np.empty(data_shape)*np.nan  
    for i0, m0 in enumerate(months_unique):
        idxs = months_==m0
        data['data'][:, :, i0, :, :] = np.nanmean(data_[:, :, idxs, :, :], axis=2)
        
    if variable == 'tprate':
        data['data'] *= 1000 # to mm/s
        
        idx = pd.MultiIndex.from_product([data['productionDates'], data['leadtimes']],
                                         names=['productionDates', 'leadtimes']).to_frame()
        days_in_month = idx.sum(axis=1).dt.days_in_month.unstack('leadtimes')
        seconds_in_month = days_in_month*86400
        
        multiplier = np.expand_dims(seconds_in_month.values, (1, -2, -1))
        tmp = data_shape.copy()
        tmp[0] = 1
        tmp[2] = 1
        multiplier = np.tile(multiplier, tmp)
        data['data'] *= multiplier
        
        units = 'mm/month'
        variable='tp'
    elif variable == 't2m':
        data['data'] -= 273.15
        units = 'C'
    
    tmp = MeteoRaster(data, units=units, variable=variable)
    return(tmp)
