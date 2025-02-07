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

class MeteoRaster(object):
    '''
    Custom class to handle meteorological raster files, including ensembles and forecasts
    
    v1.04:
        added a new function to extract values from KML as quantiles
        added a new function to convert .tys files
        updatated the load function. It now is recursive when including dummy classes
    '''
    
    VERSION = '1.04'
    VERBOSE = True
    ENSEMBLEMEMBERpOSITION = 1
    
    def __init__(self, data, latitudes=None, longitudes=None, productionDates=None, leadtimes=None, units='unknown', variable='unknown'):
        '''
        data: 5-D numpy array [productionDate, ensembleMember, leadtime, latitude, longitude]
        production dates: 1-D numpy array
        ensemble members: 1-D numpy array
        leadtimes: 1-D numpy array
        latitudes: 1-D numpy array
        longitudes: 1-D numpy array
        
        alternatively, data can be a dict with the previous fields
        '''
        
        self.units=units
        self.variable=variable
        
        if isinstance(data, dict):
            self.data = data['data']
            self.latitudes = data['latitudes']
            self.longitudes = data['longitudes']
            self.productionDates = data['productionDates']
            self.leadtimes = data['leadtimes']
        else:
            self.data = data

        if not isinstance(latitudes, type(None)):
            self.latitudes = latitudes
        if not isinstance(longitudes, type(None)):
            self.longitudes = longitudes
        if not isinstance(productionDates, type(None)):
            self.productionDates = productionDates
        if not isinstance(leadtimes, type(None)):
            self.leadtimes = leadtimes

        
        if self.longitudes.ndim==1:
            if max(self.longitudes)>180:
                self._fixStartAtGreenwich()
                
        # Process geometry so that lats and lons are stored in 2D
        if self.longitudes.ndim==1 and self.latitudes.ndim==1:
            self.longitudes, self.latitudes = np.meshgrid(self.longitudes, self.latitudes)
    
        # Flip latitudes (if required)
        if self.latitudes[0, 0]<self.latitudes[1, 0]:
            self._flipLatitude()
    
    def plotMean(self, *args, **kwargs):
        self.plot_mean(*args, **kwargs)
    
    def create_plot(self, central_longitude, projection_kwargs={}, figsize=None, projection_fun=ccrs.PlateCarree, crs=ccrs.PlateCarree(), extent=None):
        '''
        extent [west, east, south, north]
        '''
        
        plt.figure(figsize=figsize)
        projection = projection_fun(central_longitude=central_longitude, **projection_kwargs)
        ax = plt.axes(projection=projection)
        if not isinstance(extent, type(None)):
            ax.set_extent(extent, crs=crs)
        return ax

    def add_shapefile(self, ax, shapefile_path, crs=ccrs.PlateCarree(), **kwargs):
        '''
        
        '''

        ax.add_geometries(shpreader.Reader(shapefile_path).geometries(), crs=crs, **kwargs)
                      
    def plot_mean(self, ax=None, xarray=None, block=False, multiplier=1, coastline=False, borders=False, colorbar=True,
                  colorbar_label=None, cmap='viridis', central_longitude=None, central_latitude=None, *args, **kwargs):
        '''
        Plots the mean behavior for the full time series
        '''
          
        if isinstance(ax, type(None)):
            ax = self.create_plot(central_longitude, central_latitude)
        
        if isinstance(xarray, type(None)):
            xarray = self.to_xarray()
            xarray = xarray.mean(dim=['production date', 'ensemble member', 'leadtime'])*multiplier
            
            
        plot = xarray.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), x="lon", y="lat", add_colorbar=False, cmap=cmap, *args, **kwargs)
        if colorbar:
            if colorbar_label is None:
                colorbar_label = self.units
            cbar = plt.colorbar(plot, orientation='vertical', label=colorbar_label)
        else:
            cbar = None
            
        if coastline:
            ax.add_feature(cf.COASTLINE)
        if borders:
            ax.add_feature(cf.BORDERS)
            
        plt.show(block=block)
    
        return ax, cbar
    
    def add_shape(self, ax, path, color='black', facecolor='none', linewidth=1, crs=ccrs.PlateCarree()):
        '''
        
        '''
        reader = shpreader.Reader(path)

        for record in reader.records():
            geometry = record.geometry
            ax.add_geometries([geometry], crs=crs, edgecolor=color, facecolor=facecolor, linewidth=linewidth)
        plt.show(block=False)
    
    def plotMeanProjected(self, ax=None, block=False, multiplier=1, *args, **kwargs):
        '''
        Plots the mean behavior for the full time series using the true coordinates
        '''

        sns.color_palette('viridis')
    
        tmp = np.nanmean(self.data, axis=(0,1,2))*multiplier
    
        if isinstance(ax, type(None)):
            plt.figure()
    
        plt.scatter(self.longitudes.ravel(), self.latitudes.ravel(), c=tmp.ravel(), *args, **kwargs)
        plt.colorbar()
        plt.show(block=block)

    def plotCoordinates(self, ax=None, block=False, *args, **kwargs):
        '''
        Plots the available coordinates in the lat and lon matrixes
        '''

        if isinstance(ax, type(None)):
            plt.figure()
    
        plt.plot(self.longitudes.ravel(), self.latitudes.ravel(), 'k.', *args, **kwargs)
        plt.show(block=block)

    def plotSeasonal(self, lat, lon, window=31, leadtime=None, ax=None, block=False, multiplier=1, *args, **kwargs):
        
        data = self.getDataFromLatLon(lat, lon)
        if isinstance(leadtime, type(None)):
            leadtime = self.leadtimes[0]

        data = data.loc[(slice(None), leadtime), :]
        data.index = data.index.get_level_values(0)
        
        tmp = data.rolling(window=window, center=True, axis=0).mean().dropna().stack().to_frame(name='values').reset_index()
        tmp.loc[:, 'Day of year'] = tmp.loc[:, 'Production dates'].dt.day_of_year

        if isinstance(ax, type(None)):
            plt.figure(figsize=(20,10))
        ax = sns.lineplot(x='Day of year', y='values', data=tmp, errorbar=('pi', 95), ax=ax, *args, **kwargs)
        
        return ax

    def getValuesFromKML(self, kml, nameField=None, coverage_info=None, getCoverageInfo=False, elementwise=False):
        '''
        Returns a pandas table with averaged values        
        '''
        
        if isinstance(coverage_info, type(None)):
            cvr, columns, centroids = self.__coverageMatrixFromKML(kml, nameField)
            coverage_info = (cvr, columns, centroids)
        else:
            cvr, columns, centroids = coverage_info
        
        agg = self.__groupByMatrix(cvr, columns, elementwise=elementwise)
        
        if getCoverageInfo:
            return agg, centroids, coverage_info
        else:
            return agg, centroids
    
    def getQuantilesFromKML(self, kml, nameField=None, coverage_info=None, getCoverageInfo=False, resampling=None, precision=0.1, quantiles=[0.01, 0.1, 0.5, 0.9, 0.99]):
        '''
        Returns a pandas table with averaged values      
        '''
        
        if isinstance(coverage_info, type(None)):
            cvr, columns, centroids = self.__coverageMatrixFromKML(kml, nameField, normalize_by_area=False)
            coverage_info = (cvr, columns, centroids)
        else:
            cvr, columns, centroids = coverage_info
        
        agg = self.__groupByQuantile(cvr, columns, resampling, quantiles, precision)
        
        if getCoverageInfo:
            return agg, centroids, coverage_info
        else:
            return agg, centroids
    
    def getCropped(self, from_prod_date=dt.datetime(1900, 1, 1), to_prod_date=dt.datetime(2199, 12, 31),
                   from_lat=-90, to_lat=90, from_lon=-180, to_lon=180,
                   from_leadtime=None, to_leadtime=None):
        '''
        returns a MeteoRaster cropped in time and space
        '''
        
        if self.VERBOSE:
            print('Cropping meteorology...')
        
        date_idxs = np.arange(self.productionDates.shape[0])[(self.productionDates>=np.datetime64(from_prod_date)) & (self.productionDates<=np.datetime64(to_prod_date))]
        
        inside = np.zeros_like(self.latitudes).astype(bool)
        inside[(np.round(self.longitudes,6)>=np.round(from_lon,6)) & (np.round(self.longitudes,6)<=np.round(to_lon,6)) & (np.round(self.latitudes,6)>=np.round(from_lat,6)) & (np.round(self.latitudes,6)<=np.round(to_lat,6))] = True
        tmp = np.where(inside)
        lat_idxs = np.arange(tmp[0].min(), tmp[0].max()+1)
        lon_idxs = np.arange(tmp[1].min(), tmp[1].max()+1)
        
        #=======================================================================
        # lat_idxs = np.arange(self.latitudes.shape[0])[(self.latitudes>=from_lat) & (self.latitudes<=to_lat)]
        # lon_idxs = np.arange(self.longitudes.shape[0])[(self.longitudes>=from_lon) & (self.longitudes<=to_lon)]
        #=======================================================================
        
        productionDates = self.productionDates[date_idxs]
        latitudes = np.round(self.latitudes[lat_idxs, :],6)
        latitudes = np.round(latitudes[:, lon_idxs],6)
        longitudes = np.round(self.longitudes[lat_idxs, :],6)
        longitudes = np.round(longitudes[:, lon_idxs],6)
        data = self.data[date_idxs, :, :, :, :]
        data = data[:, :, :, lat_idxs, :]
        data = data[:, :, :, :, lon_idxs]
        
        cropped = copy.deepcopy(self)
        cropped.productionDates = productionDates
        cropped.latitudes = latitudes
        cropped.longitudes = longitudes
        
        if from_leadtime:
            idxs = cropped.leadtimes>=from_leadtime
            cropped.leadtimes = cropped.leadtimes[idxs]
            data = data[:, :, idxs, :, :]
        
        if to_leadtime:
            idxs = cropped.leadtimes<=to_leadtime
            cropped.leadtimes = cropped.leadtimes[idxs]
            data = data[:, :, idxs, :, :]
                    
        cropped.data = data
        
        if self.VERBOSE:
            print('    Done.')
        
        return cropped
    
    def trim(self):
        '''
        
        '''
        tmp = np.where(np.isfinite(np.nanmean(self.data, axis=(1, 2, 3, 4))))[0]
        #=======================================================================
        # tmp = np.where(np.isfinite(self.data.mean(axis=(1, 2, 3, 4))))[0]
        #=======================================================================
        start = tmp[0]
        end = tmp[-1]+1
        
        self.data = self.data[start:end, :, : ,:, :]
        self.productionDates = self.productionDates[start:end]
    
    def join_(self, to_join, trim=False, strickt=True):
        '''
        
        '''
        
        if not isinstance(to_join, Iterable):
            to_join = [to_join]
    
        # check variable
        variable = self.variable
        for data in to_join:
            if data.variable != variable:
                raise Exception('Variables do not match')
            
        # check units
        units = self.units
        for data in to_join:
            if data.units != units:
                raise Exception('Units do not match')            
    
        # check latitudes
        latitudes = self.latitudes
        for data in to_join:
            if (data.latitudes != latitudes).any():
                raise Exception('Latitudes do not match')          
        
        # check longitudes
        longitudes = self.longitudes
        for data in to_join:
            if (data.longitudes != longitudes).any():
                raise Exception('Longitudes do not match')       
        
        # check ensemble size
        ensemble_size = self.data.shape[1]
        max_ensemble_size = ensemble_size
        for data in to_join:
            tmp = data.data.shape[1]
            max_ensemble_size = max((max_ensemble_size, tmp))
            if strickt and tmp != ensemble_size:
                raise Exception('Ensemble sizes do not match in strickt mode')   
    
        # check duplicates
        productionDates, leadtimes = np.meshgrid(self.productionDates, self.leadtimes)
        base_info = pd.DataFrame({'Production dates': productionDates.ravel(),
                             'Leadtime': leadtimes.ravel(),
                             })
        base_info.loc[:, 'Dataset'] = 'Base'
        all_info = [base_info]
        
        for i0, data in enumerate(to_join):
            productionDates, leadtimes = np.meshgrid(data.productionDates, data.leadtimes)
            tmp = pd.DataFrame({'Production dates': productionDates.ravel(),
                             'Leadtime': leadtimes.ravel(),
                             })
            tmp.loc[:, 'Dataset'] = i0 
            all_info.append(tmp)
        all_info = pd.concat(all_info, axis=0)
        
        productionDates = all_info.loc[:, 'Production dates'].unique()
        leadtimes = all_info.loc[:, 'Leadtime'].unique()
        new_data = np.zeros([productionDates.shape[0],
                             max_ensemble_size,
                             leadtimes.shape[0]] + list(self.data.shape[-2:]), dtype=self.data.dtype)
        
        idx0a = np.isin(productionDates, self.productionDates)
        idx0b = np.isin(self.productionDates, productionDates)
        idx1a = np.zeros((max_ensemble_size), dtype=bool)
        idx1a[:self.data.shape[1]] = True
        idx2a = np.isin(leadtimes, self.leadtimes.astype('timedelta64[ns]'))
        idx2b = np.isin(self.leadtimes.astype('timedelta64[ns]'), leadtimes)
        
        if len(idx0a)==1:
            idx0a = slice(None)
        if len(idx1a)==1:
            idx1a = slice(None)
        if len(idx2a)==1:
            idx2a = slice(None)
        if len(idx0b)==1:
            idx0b = slice(None)
        if len(idx2b)==1:
            idx2b = slice(None)
        new_data[idx0a, idx1a, idx2a, :, :] = self.data[idx0b, :, idx2b, :, :]
        
        for i0, data in enumerate(to_join):
            idx0a = np.isin(productionDates, data.productionDates)
            idx0b = np.isin(data.productionDates, productionDates)
            idx1a = np.zeros((max_ensemble_size), dtype=bool)
            idx1a[:data.data.shape[1]] = True
            idx2a = np.isin(leadtimes, data.leadtimes.astype('timedelta64[ns]'))
            idx2b = np.isin(data.leadtimes.astype('timedelta64[ns]'), leadtimes)
            
            if len(idx0a)==1:
                idx0a = slice(None)
            if len(idx1a)==1:
                idx1a = slice(None)
            if len(idx2a)==1:
                idx2a = slice(None)
            if len(idx0b)==1:
                idx0b = slice(None)
            if len(idx2b)==1:
                idx2b = slice(None)
            new_data[idx0a, idx1a, idx2a, :, :] = data.data[idx0b, :, idx2b, :, :]
    
        self.productionDates = productionDates
        self.leadtimes = pd.to_timedelta(leadtimes)
        self.data = new_data
    
    def join(self, meteoRaster, strickt=False, trim=False):
        '''
        joins a new meteoraster to self based on production dates
        keeps a minimum number of ensembles.
        keeps only matching leadtimes
        '''
        
        if trim:
            self.trim()
            meteoRaster.trim()
        
        newData = meteoRaster.data
        newProductionDates = meteoRaster.productionDates
        newLeadtimes = meteoRaster.leadtimes
        newLatitudes = meteoRaster.latitudes
        newLongitudes = meteoRaster.longitudes
        
            
        # handle spatial coverage
        if (np.abs(self.latitudes-newLatitudes)>1E-10).any() or self.data.shape[-2]!=newData.shape[-2]:
            raise(Exception('Latitudes do not match.'))
        
        if (np.abs(self.longitudes-newLongitudes)>1E-10).any() or self.data.shape[-1]!=newData.shape[-1]:
            raise(Exception('Longitudes do not match.'))
        
        
        # handle ensembles
        e0 = self.data.shape[1]
        e1 = newData.shape[1]
        warningMessage = 'The ensemble numbers do not match (base: %d, joint: %d). Filled missing with NaN.' % (e0, e1)
        errorMessage = 'The ensemble numbers do not match (base: %d, joint: %d). Error issued in strickt mode.' % (e0, e1)
        if e0>e1:
            if strickt:
                raise(Exception(errorMessage))
            else:
                warnings.warn(warningMessage)
            shape = newData.shape
            data = np.empty([shape[0], e0] + list(shape[2:])) * np.NaN
            data[:, :e1, :, :, :] = newData
            newData = data
        elif e0<e1:
            if strickt:
                raise(Exception(errorMessage))
            else:
                warnings.warn(warningMessage)
            shape = self.data.shape
            data = np.empty([shape[0], e1] + list(shape[2:])) * np.NaN
            data[:, :e0, :, :, :] = self.data
            self.data = data

        
        # handle leadtimes
        v0 = np.where(np.isin(self.leadtimes, newLeadtimes))[0]    
        v1 = np.where(np.isin(newLeadtimes, self.leadtimes))[0]
    
        if v0.size!=self.leadtimes.size or v1.size!=newLeadtimes.size:
            if strickt:
                raise(Exception('Leadtimes do not all match (base: %d, joint: %d). Error issued in strickt mode.' % (self.leadtimes.size, newLeadtimes.size)))
            else:
                warnings.warn('Leadtimes do not all match (base: %d, joint: %d). Keeping only matching values.' % (self.leadtimes.size, newLeadtimes.size)) 
                self.leadtimes = self.leadtimes[v0]
                self.data = self.data[:, :, v0, :, :]
                newData = newData[:, :, v1, :, :]
    
    
        # join (earliest above)
        if self.productionDates[-1]<newProductionDates[0]:
            # self if first
            self.productionDates = np.concatenate((self.productionDates, newProductionDates))
            self.data = np.concatenate((self.data, newData), axis=0)
            
        elif self.productionDates[0]>newProductionDates[-1]: 
            # joint is first
            self.productionDates = np.concatenate((newProductionDates, self.productionDates))
            self.data = np.concatenate((newData, self.data), axis=0)
        elif (self.productionDates == newProductionDates).all() and (self.leadtimes != newLeadtimes).all():
            # difference of leadtimes
            raise(Exception('Not implemented'))
            pass
            
        else:
            # mixed production dates
                # remove duplicates
            raise(Exception('Must be debugged'))
                
            d1 = np.where(np.logical_not(np.isin(newProductionDates, self.productionDates)))[0]
            if d1.size==0:
                warnings.warn('No new values. Are the objects identical?')
            else:
                newProductionDates = newProductionDates[d1]
                newData = newData[d1, :, :, :, :]
                
                # preliminary sort (to retrieve only useful indexes)
                dates = np.concatenate((self.productionDates, newProductionDates))
                sortedIdxs = np.argsort(dates)
                fileIdxs = np.concatenate((np.zeros_like(self.productionDates, dtype=int), np.ones_like(newProductionDates, dtype=int)))[sortedIdxs]
                idxsFrom0 = sortedIdxs[fileIdxs==0]
                idxsFrom1 = sortedIdxs[fileIdxs==1]-self.productionDates.size
                
                # join
                self.productionDates = np.concatenate((self.productionDates[idxsFrom0], newProductionDates[idxsFrom1]))
                self.data = np.concatenate((self.data[idxsFrom1, :, :, :, :], newData[idxsFrom0, :, :, :, :]), axis=0)

                # sort again
                sortedIdxs = np.argsort(self.productionDates)
                self.productionDates = self.productionDates[sortedIdxs]
                self.data = self.data[sortedIdxs, :, :, :, :]

    def adjustLeadtimes(self, period='months'):
        '''
        Adjust leadtimes to relative time steps (useful for monthly)
        '''
        self.__verbosePrint('Adjusting lead times...')
        
        periodDefinition = {
            'months': pd.Series([pd.DateOffset(months=i) for i in range(1, math.ceil(pd.Series(self.leadtimes[-1]).dt.days.values[0]/30)+1)]),
            }
        
        if period not in periodDefinition.keys():
            raise(Exception('%s is not a valid period.' % period))
        
        warnings.warn('the function adjustLeadTimes() is experimental.')
    
        times = self.productionDates[0] + self.leadtimes
        validTimes = pd.Timestamp(self.productionDates[0]) + periodDefinition[period]
    
        idxs = self.closestIdx(times, validTimes)
        validPeriods =  periodDefinition[period][:idxs.max()+1]
        
        tmp = list(self.data.shape)
        data = np.empty(tmp[:2] + [validPeriods.size] + tmp[-2:]) * np.NaN
        
        for i0 in idxs:
            dataForLeadtime = self.data[:, :, idxs==i0, :, :]
            finite = np.isfinite(dataForLeadtime)
            count = np.sum(finite, axis=2)
            if count.max()>1:
                warnings.warn('Overlapping information detected when adjusting lead times (%s).' % str(validPeriods[i0]))
            
            tmp = np.nanmean(dataForLeadtime, axis=2)
            tmp[count==0] = np.NaN
    
            if np.isnan(tmp).all():
                warnings.warn('Unused Lead time (%s)' % str(validPeriods[i0]))
            
            data[:, :, i0, :, :] = tmp
    
        self.data = data
        if period=='months':
            self.leadtimes = np.array([pd.DateOffset(months=i0.months-1) for i0 in validPeriods])
            
    
        self.__verbosePrint('    Done.')
    
    def getMissing(self):
        '''
        Provides a diagnostic of missing data
        '''
        
        missingFraction = np.isnan(self.data).mean(axis=(-2, -1))
        missingFraction = pd.DataFrame(missingFraction.reshape((missingFraction.shape[0], np.prod(missingFraction.shape[1:]))), index=self.productionDates)
        missingFraction.index.name = 'Production dates'
        missingFraction.columns = pd.MultiIndex.from_product((self.leadtimes, np.arange(self.data.shape[1])), names=['Leadtime', 'Ensemble member'])
        
        return missingFraction.transpose()

    def getDataFromLatLon(self, lat, lon):
        '''
        Returns a dataframe with data from the pixel closest to the desired point
        '''

        idxLat, idxLon = self._getClosestPixels(lat, lon)

        if idxLat.size>1 or idxLon.size>1:
            warnings.warn('Several pixels are "closest". Averaging results.')
        tmp = []
        for iLat, iLon in zip(idxLat, idxLon):
            tmp.append(np.expand_dims(self.data[:, :, :, iLat, iLon], axis=0))
        tmp = np.concatenate(tmp, axis=0)
        tmp = np.nanmean(tmp, axis=0) # [production dates, ensemble, leadtimes]
        
        data = pd.DataFrame(tmp.reshape(tmp.shape[0]*tmp.shape[1], tmp.shape[2]))
        data.index = pd.MultiIndex.from_product((self.productionDates.ravel(), np.arange(self.data.shape[1])), names=['Production dates', 'Ensemble member'])
        data.columns = self.leadtimes
        data.columns.name = 'Leadtime'

        data = data.unstack('Ensemble member')

        return data

    @staticmethod
    def dataFromLatLon_eventDate(production_date_dataframe):
        '''
        !!!!! Change the index name to Event dates
        '''
        
        ensembles = production_date_dataframe.columns.get_level_values('Ensemble member').unique()
        full_data_ = []
        for ensemble in ensembles:
            tmp = production_date_dataframe.loc[:, production_date_dataframe.columns.get_level_values('Ensemble member')==ensemble]
            data_ = []
            for i0 in range(tmp.shape[1]):
                tmp_ = tmp.iloc[:, [i0]]
                tmp_.index = tmp_.index + tmp_.columns.get_level_values('Leadtime')[0]
                data_.append(tmp_)
            full_data_.append(pd.concat(data_, axis=1))
        
        return pd.concat(full_data_, axis=1)

    def plotAvailability(self, missing=None, individualMembers=False):
        '''
        Plots the availability of the data
        '''
        
        if isinstance(missing, type(None)):
            missing = self.getMissing()

        available = missing < 0.95
        
        if individualMembers:
            tmp = available.stack().to_frame()
        else:
            virtualLevels = pd.to_datetime(self.productionDates[0]) + self.leadtimes
            available.index = available.index.set_levels(virtualLevels, level=0)
            tmp = available.groupby(axis=0, level=0).sum().transpose().stack().to_frame()
            tmp.index = tmp.index.set_levels(self.leadtimes, level=1)

        
        tmp.columns = ['Available']
        tmp = tmp.loc[tmp['Available']>0,:].reset_index()
        tmp.loc[:, 'Event dates'] = tmp.loc[:, 'Production dates'] + tmp.loc[:, 'Leadtime']
        
        sns.set_theme(style="whitegrid")
        if individualMembers:
            g = sns.relplot(
                data=tmp,
                x='Event dates', y='Production dates', hue='Ensemble member', 
                palette='plasma',
                )
        else:
            g = sns.relplot(
                data=tmp,
                x='Event dates', y='Production dates', hue='Available', size='Available',
                palette='viridis',
                sizes={s: s*10 for s in np.arange(1,max((11, tmp.loc[:, 'Available'].max()+1)))},
                #===================================================================
                # size_order=np.arange(1, max((11, tmp.loc[:, 'Available'].max()+1)))
                #===================================================================
                )
        g.ax.xaxis.grid(True, "minor", linewidth=.25)
        g.ax.yaxis.grid(True, "minor", linewidth=.25)
        
        plt.show(block=False)

    def resampleTimeStep(self, rule, fun=np.mean):
        '''
        Resamples data according to pandas conventions
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html
        
        rule='MS' resamples to the start of the month
        '''
        
        warnings.warn('the function resampleTimeStep() is experimental.')

        self.__verbosePrint('    Resampling time step (rule: %s, %s)...' % (rule, str(fun)))        

        # prepare a template dataframe
        tmp = pd.DataFrame(np.arange(self.productionDates.shape[0]), index=self.productionDates)
        idxMax = tmp.resample(rule=rule).max().astype(int)+1
        
        resampled = np.empty([idxMax.shape[0]] + list(self.data.shape[1:])) * np.NaN
        idx0 = 0
        for i0, i1 in enumerate(idxMax.values.ravel()):
            resampled[i0, :, :, :, :] = fun(self.data[idx0:i1, :, :, :, :], axis=0, keepdims=True)
            idx0 = i1
    
        self.data = resampled
        self.productionDates = idxMax.index
    
        self.__verbosePrint('        Done (%s to %s).' % (self.productionDates[0].strftime('%Y-%m-%d %H:%M:%S'),
                                                          self.productionDates[-1].strftime('%Y-%m-%d %H:%M:%S')))
    
    def save(self, file):
        '''
        Saves a MeteoRaster file (.mr)
        '''
        
        if not isinstance(file, Path):
            file = Path(file)
          
        self.__verbosePrint('Saving %s...' % file)
        
        with open(file, 'wb') as f0:
            pickle.dump(self, f0)
        
        self.__verbosePrint('    Saved.')
    
    def to_netCDF(self, file, variable, units, title='', institution='', references='', description='', history='', compress=True):
    
        if self.data.shape[1]!=1 or self.data.shape[2]!=1:
            raise Exception('Sorry. Ensemble data or forecasts not supported at this time...')
    
        if not isinstance(file, Path):
            file = Path(file)
          
        self.__verbosePrint('Saving %s...' % file)
        
        data_array = xr.DataArray(
            data=self.data.squeeze(),
            dims=['time', 'y', 'x'],
            coords={
                'lat': (['y', 'x'], self.latitudes),
                'lon': (['y', 'x'], self.longitudes)
            }
        )
        
        # Create xarray dataset
        ds = xr.Dataset(
            {
                variable: data_array.squeeze(),
            },
            coords={'time': self.productionDates, 'lat': (['y', 'x'], self.latitudes), 'lon': (['y', 'x'], self.longitudes)},
        )
    
        # Set coordinate attributes
        ds['lat'].attrs['standard_name'] = 'latitude'
        ds['lon'].attrs['standard_name'] = 'longitude'
    
        # Set attributes
        ds[variable].attrs = {
            'units': units,
            'long_name': variable,
        }
    
        # Set global attributes
        ds.attrs = {
            'title': title,
            'institution': institution,
            'description': description,
            'references': references,
            'history': history,
        }
    
        ds.to_netcdf(file)
    
        if compress:
            with zipfile.ZipFile(file.parent / file.name.replace(file.suffix, '.zip'), 'w') as zf:
                zf.write(file, file.name, compress_type=zipfile.ZIP_DEFLATED)
            file.unlink()
    
    def to_xarray(self):
        '''
        
        '''
        
        dims = ['production date', 'ensemble member', 'leadtime', 'y', 'x']
        coords = {'lat':(['y', 'x'], self.latitudes),
                  'lon':(['y', 'x'], self.longitudes),
                  'production date': self.productionDates,
                  'ensemble member': np.arange(self.data.shape[1]),
                  'leadtime': self.leadtimes,
                  'y': np.arange(self.data.shape[3]),
                  'x': np.arange(self.data.shape[4]),
                  }
        array = xr.DataArray(self.data, dims=dims, coords=coords, name=self.variable, attrs=dict(units=self.units))
    
        return array
    
    @classmethod
    def load(cls, file):
        '''
        Loads a MeteoRaster file
        '''
        
        cls.__verbosePrint('Loading %s...' % file)
        
        with open(file, 'rb') as f0:
            done = False
            while not done:
                try:
                    tmp = pickle.load(f0)
                    done = True
                except ModuleNotFoundError as ex:
                    dummy_module = types.ModuleType(ex.name)
                    dummy_module.MeteoRaster = MeteoRaster
                    sys.modules[ex.name] = dummy_module
                    #===========================================================
                    # tmp = pickle.load(f0)
                    #===========================================================
                    print('        Module updated (%s)' % ex.name)
        if not hasattr(tmp, 'units'):
            units = 'unknown'
        else:
            units = tmp.units
                
        meteo = cls(data=tmp.data, latitudes=tmp.latitudes, longitudes=tmp.longitudes, productionDates=tmp.productionDates, leadtimes=tmp.leadtimes, units=units)
        
        cls.__verbosePrint('    Loaded.')    

        return meteo

    def copy(self):
        '''
        Makes a copy of a MeteoRaster file
        '''
        
        meteo = MeteoRaster(data=self.data.copy(), latitudes=self.latitudes.copy(), longitudes=self.longitudes.copy(), productionDates=self.productionDates.copy(), leadtimes=self.leadtimes.copy())
        return meteo

    @classmethod
    def convert_tys(cls, file, units=None):
        '''
        Converts a .tys file (used to store Tethys meteorology)
        '''
        
        class DummyClass:
            pass
        
        timeSeries = types.ModuleType('timeSeries')
        sys.modules['timeSeries'] = timeSeries
        
        timeSeries.rasterFile = types.ModuleType('timeSeries.rasterFile')
        timeSeries.rasterFile.TethysRasterFile = TethysRasterFile
        sys.modules['timeSeries.rasterFile'] = timeSeries.rasterFile
        
        tys = TethysRasterFile(file=file)
        tys.load()
        
        data = np.expand_dims(tys.data, 1)
        production_datetimes = pd.to_datetime(tys.productionDatetimes)
        leadtimes = pd.to_timedelta(tys.leadtimes)
        latitudes = tys.latitudes
        longitudes = tys.longitudes
        
        meteo = cls(data=data, latitudes=latitudes, longitudes=longitudes,
                    productionDates=production_datetimes, leadtimes=leadtimes, units=units)
        
        return meteo

    def _getClosestPixels(self, lat, lon):
        '''
        Returns the indexes of the pixel closer to a point 
        '''
    
        dist = ((self.longitudes-lon)**2+(self.latitudes-lat)**2)**0.5
        idx_lat, idx_lon = np.where(dist==dist.ravel().min())
                    
        return (idx_lat, idx_lon)
    
    @staticmethod
    def closestIdx(X, Y):
        '''
        Returns the list of Y indexes closest the values of X
        Dimension of X.
        Indexes to be used in Y.
        '''
    
        idxs = np.empty_like(X, dtype=int)
        for i0, x0 in enumerate(X):
            idxs[i0] = np.argmin(np.abs(x0-np.array(Y)))
                
        return idxs
  
    def _flipLatitude(self):
        '''
        flips the latitude of the raster
        '''
        
        self.data = np.flip(self.data, axis=3)
        self.latitudes = np.flip(self.latitudes, axis=0)
        self.longitudes = np.flip(self.longitudes, axis=0)
        
        self.__verbosePrint('Flipped latitudes.')
    
    def _fixStartAtGreenwich(self):
        '''
        modifies rasters with longitudes starting at Greenwich to the more conveninent -180E.
        '''
        
        westIndexes = np.arange(len(self.longitudes))[self.longitudes>180]
        eastIndexes = np.arange(len(self.longitudes))[self.longitudes<=180]

        longitudes = np.concatenate((self.longitudes[westIndexes]-360, self.longitudes[eastIndexes]))
        
        westData = self.data[:, :, : ,: ,westIndexes]
        eastData = self.data[:, :, : ,: ,eastIndexes]
        
        data = np.concatenate((westData, eastData), axis=-1)
        
        self.longitudes = longitudes
        self.data = data
        
        self.__verbosePrint('Fixed start at Greenwich.')
    
    def __groupByMatrix(self, coverage, columns, elementwise=False):
        '''
        Groups the data by a coverage matrix (see groupByKML for matrix definition)
        '''
        
        self.__verbosePrint('Executing spatial aggregation...')
        
        #Verification of valid tiles (assumes first coverage zone to be representative (for speed and memory)
        val = np.isfinite(self.data).astype(bool)
        cvr = np.tile(coverage[:, :, :, :, :, 0].astype(np.single), list(val.shape[:(val.ndim-2)]) + [1, 1])
        val = np.nansum(cvr*val, axis=(-2, -1))>0.8
        val = np.tile(np.expand_dims(val,-1), [1]*val.ndim + [coverage.shape[-1]])
        
        if elementwise:
            agg = np.empty(list(self.data.shape[:3]) + [len(columns)]).astype(np.single) * np.NaN
            
            data_single = self.data.astype(np.single)
            coverage_single = coverage.astype(np.single)
            
            for i0 in range(len(columns)):
                agg[:, :, :, i0] = np.nansum(data_single * np.tile(coverage_single[:, :, :, :, :, i0], list(data_single.shape[:(self.data.ndim-2)]) + [1, 1]), axis=(-2, -1))
                self.__verbosePrint('    % 5.1f%%' % ((i0+1)/len(columns)*100))
        else:
            #Creation of matrices for calculation
            dat = np.tile(np.expand_dims(self.data.astype(np.single), axis=-1), [1]*self.data.ndim + [coverage.shape[-1]])
            cvr = np.tile(coverage.astype(np.single), list(dat.shape[:(dat.ndim-3)]) + [1, 1, 1])
            dat *= cvr
            cvr = None
            gc.collect()
            
            #Calculation with garbage collection
            agg = np.nansum(dat, axis=(-3, -2))
            dat = None
            gc.collect()
            
        #Mark invalid tiles as NaN
        agg[~val] = np.NaN 
        
        tmp = np.reshape(agg, (agg.shape[0], np.prod(agg.shape[1:])), order='F')
        agg = pd.DataFrame(tmp, index=self.productionDates.ravel())
        agg.columns = columns
        agg.index.name = 'Production date'

        self.__verbosePrint('    Done.')

        return agg  
    
    def __groupByQuantile(self, coverage, columns, resampling, quantiles, precision=0.1):
        '''
        Groups the data by quantiles in time and space
        
        precision should be between 0 and 1.
        computation time will be proportional to 1/precision
        '''
        
        self.__verbosePrint('Executing spatial aggregation...')

        if not resampling is None:
            resampled_idx = pd.DataFrame(self.productionDates, self.productionDates).resample(resampling).indices
            time_idx = list(resampled_idx.keys())
            time_groups = [i for _, i in resampled_idx.items()]
        else:
            time_idx = self.productionDates
            time_groups = [[i] for i, _ in enumerate(time_idx)]
                    
        #Creation of matrices for calculation
        dat = np.tile(np.expand_dims(self.data.astype(np.single), axis=-1), [1]*self.data.ndim + [coverage.shape[-1]])
        coverage_ = coverage.astype(np.single)
        coverage_[coverage_==0] = np.nan
        gc.collect()
        
        #Calculation with garbage collection
        agg_ = []
        for g0 in time_groups:
            data = dat[g0,:, :, :, :].copy()
            
            # Verification of valid tiles (assumes first coverage zone to be representative (for speed and memory)
            cvr = np.tile(coverage_, list(data.shape[:(dat.ndim-3)]) + [1, 1, 1])
            val = cvr.copy()
            val[val<=0] = np.nan
            val[val>0] = 1

            data_ = data * val
            
            folds = np.round(cvr/precision)
            n_folds = int(np.nanmax(folds))
            data_unfolded = np.tile(np.expand_dims(data, 0), [n_folds] + [1] * data.ndim) * np.nan
            tmp = data_.copy()
            for i1 in range(n_folds):
                tmp[folds<i1+1] = np.nan
                data_unfolded[i1,:,:,:,:,:,:] = tmp 
           
            # Check what pixels are valid (more than 80% values available)
            val = np.nansum(np.isfinite(data_), axis=(0, -3, -2)) / np.nansum(np.isfinite(val), axis=(0, -3, -2)) > 0.8
            val = np.tile(np.expand_dims(val, (0, -1)), [1]*(val.ndim + 1) + [len(quantiles)])
            
            # Calculate the quantiles
            tmp = np.rollaxis(np.nanquantile(np.expand_dims(data_unfolded, 0), quantiles, axis=(1, 2, -3, -2)), 0, len(quantiles)) #[time, lead, ensemble, zones, quantiles]
            tmp[~val] = np.nan 
            agg_.append(tmp)
    
        agg__ = np.r_[agg_] 
                
        tmp = np.reshape(agg__, (agg__.shape[0], np.prod(agg__.shape[1:])), order='F')
        agg = pd.DataFrame(tmp, index=time_idx)
        
        tmp = pd.concat([columns.to_frame()]*len(quantiles))
        tmp.loc[:, 'Quantile'] = np.array([[q]*np.product(agg__.shape[1:-1]) for q in quantiles]).ravel()
        agg.columns = pd.MultiIndex.from_frame(tmp)
        agg.index.name = 'Production date'
        agg.sort_index(axis=1, inplace=True)

        #=======================================================================
        # agg.to_clipboard(excel=True)
        #=======================================================================

        self.__verbosePrint('    Done.')

        return agg  
    
    def __coverageMatrixFromKML(self, kml, nameField=None, buffer=2, normalize_by_area=True):
        '''
        Creates a coverage matrix from polygons contained in a kml file
        '''
        
        self.__verbosePrint('Creating a coverage matrix for spatial aggregation...')
        
        # Correct kml file if required
        self.__correctKMLHeader(kml) 
        
        # Read geometry reference KML file
        ids = []
        coordinates = []
        document = et.parse(kml).getroot().find('Document')
        for i0, placemark in enumerate(document.findall('.//Placemark')):
            #https://docs.python.org/3/library/xml.etree.elementtree.html
            try:
                if not isinstance(nameField, type(None)):
                    id0 = placemark.find(".//SimpleData[@name='%s']" % nameField).text
                else:
                    id0 = str(i0)
                outer = [pd.Series(c.text.replace(' \n', '').replace('\n', '').replace('\t', '').split(' ')).str.split(',', n=2, expand=True) for c in placemark.findall('.//Polygon/outerBoundaryIs//coordinates')]
                outer = [c.astype(float) for c in outer]
                inner = [pd.Series(c.text.replace(' \n', '').replace('\n', '').replace('\t', '').split(' ')).str.split(',', n=2, expand=True) for c in placemark.findall('.//Polygon/innerBoundaryIs//coordinates')]
                inner = [c.astype(float) for c in inner]
                ids.append(id0)
                coordinates.append({'outer': outer, 'inner': inner})
            except Exception as ex:
                print(str(ex)) 
        
        cvr = np.zeros([1, 1, 1] + list(self.data.shape[3:]) + [len(ids)], dtype=np.single) # the last coordinate of the cvr represents the polygon
        
        longitudes = self.longitudes
        latitudes = self.latitudes
        idx_lat_ = np.tile(np.expand_dims(np.arange(longitudes.shape[0]),axis=1),(1, longitudes.shape[1])).ravel() 
        idx_lon_ = np.tile(np.arange(longitudes.shape[1]),(longitudes.shape[0], 1)).ravel()
        lonInterpolator = interp.LinearNDInterpolator(np.array([idx_lat_, idx_lon_]).T, longitudes.ravel()) # f(lat_id, lon_id)
        latInterpolator = interp.LinearNDInterpolator(np.array([idx_lat_, idx_lon_]).T, latitudes.ravel()) # f(lat_id, lon_id)
        
        # Create GDAL geometry
        centroids = {'x':[], 'y':[]}
        for i0, coords in enumerate(coordinates):
            try:
                # Add all "positive" rings
                poly = ogr.Geometry(ogr.wkbPolygon)
                for o1 in coords['outer']:
                    for r0 in o1:
                        polyTmp = ogr.Geometry(ogr.wkbPolygon)
                        ring = ogr.Geometry(ogr.wkbLinearRing)
                        for row in o1.iterrows():
                            ring.AddPoint(row[1][0], row[1][1])
                        _ = polyTmp.AddGeometry(ring)
                        poly = poly.Union(polyTmp)
                
                # Join all "negative rings
                polyHoles = ogr.Geometry(ogr.wkbPolygon)
                for o1 in coords['inner']:
                    for r0 in o1:
                        polyTmp = ogr.Geometry(ogr.wkbPolygon)
                        ring = ogr.Geometry(ogr.wkbLinearRing)
                        for row in o1.iterrows():
                            ring.AddPoint(row[1][0], row[1][1])
                        _ = polyTmp.AddGeometry(ring)
                        polyHoles = polyHoles.Union(polyTmp)
                    
                # Obtain the final polygon
                poly = poly.SymmetricDifference(polyHoles)
    
                # Select pixel subset
                #===============================================================
                # plt.figure(); plt.imshow(longitudes); plt.colorbar()
                # plt.figure(); plt.imshow(latitudes); plt.colorbar()
                # plt.figure(); plt.scatter(longitudes.ravel(), latitudes.ravel(),c=self.data.squeeze().mean(axis=0).ravel()); plt.colorbar()
                # plt.figure(); plt.scatter(longitudes.ravel(), latitudes.ravel(),c=longitudes.ravel()); plt.colorbar()
                # plt.figure(); plt.scatter(longitudes.ravel(), latitudes.ravel(),c=latitudes.ravel()); plt.colorbar()
                #===============================================================
                
                extent = poly.Buffer(buffer).GetEnvelope() #wesn
                inside = np.zeros_like(latitudes).astype(bool)
                inside[(longitudes>=extent[0]) & (longitudes<=extent[1]) & (latitudes>=extent[2]) & (latitudes<=extent[3])] = True
                tmp = np.where(inside)
                lat_idxs = np.arange(tmp[0].min()-1, tmp[0].max()+1)
                lat_idxs = lat_idxs[(lat_idxs>0) & (lat_idxs<latitudes.shape[0]-1)]
                lon_idxs = np.arange(tmp[1].min()-1, tmp[1].max()+1)
                lon_idxs = lon_idxs[(lon_idxs>0) & (lon_idxs<longitudes.shape[1]-1)]
                
                #===============================================================
                # plt.figure(); plt.imshow(inside); plt.colorbar()
                # plt.figure(); plt.scatter(longitudes.ravel(), latitudes.ravel(),c=inside.ravel()); plt.colorbar()
                #===============================================================
                
                #===============================================================
                # extent = [extent[0]-hdLon*2, extent[1]+hdLon*2, extent[2]-hdLat*2, extent[3]+hdLat*2]
                # lon_idxs = np.arange(self.longitudes.shape[0])[(self.longitudes>=extent[0]) & (self.longitudes<=extent[1])]
                # lat_idxs = np.arange(self.latitudes.shape[0])[(self.latitudes>=extent[2]) & (self.latitudes<=extent[3])]
                #===============================================================
    
                x, y, _ = poly.Centroid().GetPoint()
                centroids['x'].append(x)
                centroids['y'].append(y)
    
                for i2 in lon_idxs:
                    for i3 in lat_idxs:
                        #=======================================================
                        # lon = self.longitudes[i3, i2]
                        # lat = self.latitudes[i3, i2]
                        # plt.plot(lon, lat, 'rx')
                        # plt.plot(lonInterpolator(i3-0.5,i2+0.5), latInterpolator(i3-0.5,i2+0.5), '.m')
                        #=======================================================
                        
                        points = []
                        points.append((float(lonInterpolator(i3-0.5, i2+0.5)), float(latInterpolator(i3-0.5, i2+0.5))))
                        points.append((float(lonInterpolator(i3+0.5, i2+0.5)), float(latInterpolator(i3+0.5, i2+0.5))))
                        points.append((float(lonInterpolator(i3+0.5, i2-0.5)), float(latInterpolator(i3+0.5, i2-0.5))))
                        points.append((float(lonInterpolator(i3-0.5, i2-0.5)), float(latInterpolator(i3-0.5, i2-0.5))))
                        points.append((float(lonInterpolator(i3-0.5, i2+0.5)), float(latInterpolator(i3-0.5, i2+0.5))))
                        
                        ring = ogr.Geometry(ogr.wkbLinearRing)
                        for p in points:
                            ring.AddPoint(*p)
                            #===================================================
                            # plt.plot(*p,'.b')
                            #===================================================
                        
                        pixel = ogr.Geometry(ogr.wkbPolygon)
                        _ = pixel.AddGeometry(ring)
    
                        intersection = pixel.Intersection(poly)
                        area = intersection.Area()
    
                        if area>0:
                            cvr[0, 0, 0, i3, i2, i0] = area
            except Exception as ex:
                raise(ex)

        # Define columns (drops redundant levels (except zone))
        columns = pd.MultiIndex.from_product((ids, self.leadtimes, 1+np.arange(self.data.shape[self.ENSEMBLEMEMBERpOSITION])), names=('Zone', 'Leadtime', 'Ensemble member'))
        drop = []
        for i0, (n0, l0) in enumerate(zip(columns.names, list(columns.levels))):
            if l0.shape[0]==1 and n0!='Zone':
                self.__verbosePrint('    Level %s is redundant and will be dropped.' % n0)
                drop.append(i0)
        for i0 in drop[::-1]:
            columns = columns.droplevel(i0)
                
        # Normalize
        if normalize_by_area:
            areas = cvr.sum(axis=(0, 1, 2, 3, 4))
            cvr /= np.tile(areas, list(cvr.shape[:-1]) + [1])
        else:
            max_pixel_area = cvr.max(axis=tuple(np.arange(cvr.ndim-1)))
            cvr /= np.tile(max_pixel_area, list(cvr.shape[:-1]) + [1])
        
        # Prepare centroids for export
        centroids = pd.DataFrame(centroids, index=ids)
        
        self.__verbosePrint('    Done.')
        
        return cvr, columns, centroids
        
    def __correctKMLHeader(self, kml):
        '''
        Corrects the header of a KML file to that it can be read for model data preparation
        '''
        
        modificationRequired = False
        line = ''
        with open(kml, 'r') as file:
            for line in file:
                if line.strip().startswith('<kml'):
                    if line.strip() != '<kml>':
                        modificationRequired = True
                    break
                
        if modificationRequired:
            if not isinstance(kml, str):
                kml = str(kml)
                
            tmpKML = kml.replace('.kml', '__tmp.kml')
            with open(kml, 'r') as file_in:
                with open(tmpKML, 'w') as file_out:
                    for line in file_in:
                        if line.strip().startswith('<kml'):
                            if line.strip() != '<kml>':
                                line = '<kml>\n'
                        __ = file_out.write(line)
            os.remove(kml)
            os.rename(tmpKML, kml)

    @classmethod
    def __verbosePrint(cls, message):
        '''
        Prints only if VERBOSE=True
        '''
        
        if cls.VERBOSE:
            print('    %s' % message)

class Resample(object):
    '''
    Resamples meteorological data for use in hydrological and forecasting models
    handles pandas dataframes [production timestamps x leadtime timedeltas]
    Does not need initialization as the main method (resample) is a class method.
    
    
    v1.01:
        imported to meteoraster from tethys
    '''
    
    VERSION = '1.01'
    
    @classmethod
    def resample(cls, data, timestep_frequency, resamplingType, date_from=None, date_to=None, print_func=None):
        '''
        Main method of the class.
        Takes the following inputs:
            data: pandas dataframe [production timestamps x leadtime timedeltas]
            timestep_frequency: pandas pd.Timedelta or equivalent string (e.g., '1D', '3H', '30min', '30T', etc...).
            resamplingType: 'sum', 'mean', or 'linear. 'sum' and 'mean' are very similar. There is a difference only when the timestep_frequency is larger than the production frequency (1D case) or the leadtime frequency (2D case).
            date_from: first datetime that is displayed (rows)
            date_to: last datetime that is displayed (rows) - when upsampling additional values may be added.
            print_func: a function handle to print the evolution of the calculations
            
            
            ### THE VALIDITY CHECK SHOULD BE REVIEWED
        '''
        
        headers = list(data.columns.names)
        headers.remove('leadtime')
        
        # Default values for the date
        if isinstance(date_from, type(None)):
            date_from = data.index[0]
        if isinstance(date_to, type(None)):
            date_to = data.index[-1]
        
        if isinstance(timestep_frequency, dict):
            timestep_frequency = pd.Timedelta(**timestep_frequency)
        elif isinstance(timestep_frequency, str):
            timestep_frequency = pd.Timedelta(timestep_frequency)
            
        idx_map = None
        resampled = []
        for g0_, mis0_ in data.groupby(headers, axis=1):
            # Loop - separate handling of each header combination (indexes are calculated on the first pass)
            g0_ = [mis0_.columns.get_level_values(h)[0] for h in headers]
            mis0_ = mis0_.droplevel(headers, axis=1)
            
            # Progress message
            if not isinstance(print_func, type(None)):
                print_func(g0_[-1])
            
            # Infer frequencies
                # Production (rows)
            prod_frequency = pd.infer_freq(mis0_.index.values[0:3])
            if not prod_frequency:
                prod_frequency = pd.infer_freq(mis0_.index.values[3:6])
            if not prod_frequency[0].isdigit():
                prod_frequency = '1' + prod_frequency
            prod_frequency = pd.Timedelta(prod_frequency)
            
            date_from_ = date_from - timestep_frequency * math.ceil(prod_frequency / timestep_frequency)
            
                # Leadtime (columns)
            if mis0_.shape[1]>2:
                lead_frequency = pd.infer_freq(mis0_.columns.get_level_values('leadtime'))
                if not lead_frequency[0].isdigit():
                    lead_frequency = '1' + lead_frequency
                lead_frequency = pd.Timedelta(lead_frequency)
            elif mis0_.shape[1]==2:    
                tmp = mis0_.columns.get_level_values('leadtime')
                lead_frequency = tmp[1]-tmp[0]
            else:
                lead_frequency = prod_frequency
            
            # Check frequency ratio
            fr_ratio = prod_frequency/lead_frequency
            if mis0_.shape[1]!=1 and mis0_.shape[1]<fr_ratio:
                raise Exception('Problem with the leadtime [%u (multiple of %u) "%s"] . Does it agree with the production step ["%s"]?' % (mis0_.shape[1], fr_ratio, lead_frequency, prod_frequency))
                
            # Check timestep ratio
            ts_ratio = timestep_frequency/lead_frequency
            if mis0_.shape[1]!=1 and mis0_.shape[1]/ts_ratio!=int(mis0_.shape[1]/ts_ratio):
                additional_mis0_steps = int((int(mis0_.shape[1]/ts_ratio)+1) * ts_ratio)-mis0_.shape[1]
                additional_cols = pd.timedelta_range(start=mis0_.columns[-1], periods=additional_mis0_steps+1, freq=lead_frequency, closed='right')
                additional_mis0 = pd.DataFrame(np.nan, index=mis0_.index, columns=additional_cols)
                mis0_ = pd.concat((mis0_, additional_mis0), axis=1)
                #===============================================================
                # valid_mis0_steps = int(int(mis0_.shape[1]/ts_ratio) * ts_ratio)
                # mis0_ = mis0_.iloc[:, :valid_mis0_steps]
                #===============================================================
                
                print('            Issue with the leadtime [%u (multiple of %u) "%s"]. Does it agree with the time step ["%s"]? Augmented %u steps.' % (mis0_.shape[1], ts_ratio, lead_frequency, prod_frequency, additional_mis0_steps))
                #===============================================================
                # raise Exception('Problem with the leadtime [%u (multiple of %u) "%s"]. Does it agree with the time step ["%s"]?' % (mis0_.shape[1], ts_ratio, lead_frequency, prod_frequency))
                #===============================================================
            
            
            calculation_frequency = min((prod_frequency, pd.Timedelta('1H'), timestep_frequency, lead_frequency))
            if mis0_.shape[1]==1:
                resampled_ = cls._resample_aux_1D(mis0_, prod_frequency, calculation_frequency, timestep_frequency, resamplingType, date_from_, date_to)
            else:
                resampled_, idx_map = cls._resample_aux_2D(mis0_, prod_frequency, lead_frequency, calculation_frequency, timestep_frequency, resamplingType, date_from_, date_to, idx_map)
            
            #===================================================================
            # mis0_.to_clipboard(excel=True)
            # resampled_.reindex(production_range, axis=0).to_clipboard(excel=True)
            #===================================================================
            resampled_.columns = pd.MultiIndex.from_product([[i] for i in g0_] + [resampled_.columns.tolist()], names=headers + ['delay'])
            resampled.append(resampled_)
     
        # Aggregate to the desired time step
        resampled = pd.concat(resampled, axis=1)
        production_range = pd.date_range(start=date_from, end=date_to, freq=timestep_frequency)
    
        return resampled.reindex(production_range, axis=0)

    @classmethod
    def _resample_aux_1D(cls, data, prod_frequency, calculation_frequency, timestep_frequency, resamplingType, date_from, date_to):
        '''
        
        prod_frequency assumed to be compatible with timestep_frequency
        calculation frequency is at least hourly to account for timezone shifts
        '''
        
        # Calculate reference step windows    
        production_calculation_window = int(prod_frequency/calculation_frequency)
        timestep_calculation_window = int(timestep_frequency/calculation_frequency)    
        
        calculation_range = pd.date_range(start=date_from, end=date_to + prod_frequency, freq=calculation_frequency, closed='left')
        resampled_ = data.reindex(calculation_range, axis=0)
        
        if resamplingType=='sum' or resamplingType=='mean':
            if production_calculation_window>1:
                resampled_ = resampled_.ffill(limit=production_calculation_window-1) / production_calculation_window # value per calculation step
            resampled_ = resampled_.resample(rule=timestep_frequency, axis=0).sum(min_count=timestep_calculation_window) # sum per time step
            
            production_ratio = int(production_calculation_window / timestep_calculation_window)
            if  production_ratio > 1:
                if production_ratio != production_calculation_window / timestep_calculation_window:
                    raise Exception('Production step not multiple of calculation time step.')
                
                tmp = []
                for i0 in range(production_ratio):
                    tmp.append(resampled_.shift(-i0))
                tmp = pd.concat(tmp, axis=1)
                tmp.columns = pd.timedelta_range('0D', end=prod_frequency, freq=timestep_frequency, name='leadtimes', closed='left')
                resampled_ = tmp.sum(axis=1, min_count=production_ratio).to_frame()
                if resamplingType=='mean':
                    resampled_ /= production_ratio
                resampled_.columns = [pd.Timedelta('0D')]
                
            else:
                if resamplingType=='mean':
                    resampled_ *= production_calculation_window / timestep_calculation_window
        elif resamplingType=='linear':
            if production_calculation_window>1:
                resampled_ = resampled_.interpolate(axis=0, limit=production_calculation_window-1, method='linear', limit_area='inside')
        elif resamplingType=='max':
            if production_calculation_window>1:
                resampled_ = resampled_.ffill(limit=production_calculation_window-1) # value per calculation step
            resampled_ = resampled_.resample(rule=timestep_frequency, axis=0).max(min_count=timestep_calculation_window) # sum per time step
        else:
            raise Exception('Resampling type not implemented: %s.' % resamplingType) 
        
        return resampled_
        
    @classmethod
    def _resample_aux_2D(cls, data, prod_frequency, leadtime_frequency, calculation_frequency, timestep_frequency, resamplingType, date_from, date_to, idx_map):
        '''
        
        '''
        
        # Reindex to calculation delta when required
        if isinstance(idx_map, type(None)):
            idx_map = cls._resample_aux_2D_idx(data, prod_frequency, leadtime_frequency, calculation_frequency, timestep_frequency, resamplingType, date_from, date_to, idx_map)
        
        # Calculate reference step windows    
        leadtime_calculation_window = int(leadtime_frequency/calculation_frequency) 
        timestep_calculation_window = int(timestep_frequency/calculation_frequency)
         
        resampled_ = idx_map.copy()*np.nan
        tmp0 = idx_map.dropna().astype(int).values.ravel()
        tmp1 = np.isfinite(idx_map).astype(bool).values.ravel()
        resampled_.loc[tmp1,:] = data.values.ravel()[tmp0]
        resampled_ = resampled_.unstack('leadtime')
        resampled_ = resampled_.droplevel(0, axis=1)
         
        # Aggregate to desired time step (columns)      
        if timestep_calculation_window>1 or leadtime_calculation_window>1:  
            if resamplingType=='sum' or resamplingType=='mean':
                resampled_ = resampled_.resample(rule=timestep_frequency, axis=1).sum(min_count=timestep_calculation_window) / timestep_calculation_window
                if resamplingType=='sum':
                    resampled_ *= timestep_calculation_window / leadtime_calculation_window
            elif resamplingType=='linear':
                if leadtime_calculation_window>1:
                    resampled_ = resampled_.interpolate(axis=1, limit=leadtime_calculation_window-1, method='linear')
                leadtimes_ = pd.timedelta_range(start=max((pd.Timedelta(hours=0), resampled_.columns[0])), end=resampled_.columns[-1], freq=timestep_frequency)
                resampled_ = resampled_.reindex(leadtimes_, axis=1)
            elif resamplingType=='max':
                resampled_ = resampled_.resample(rule=timestep_frequency, axis=1).max(min_count=timestep_calculation_window)
            else:
                raise Exception('Resampling type not implemented: %s.' % resamplingType)
         
        return (resampled_, idx_map)
    
    @classmethod
    def _resample_aux_2D_idx(cls, data, prod_frequency, leadtime_frequency, calculation_frequency, timestep_frequency, resamplingType, date_from, date_to, idx_map):
        '''
        
        '''
        # Calculate reference step windows    
        production_calculation_window = int(prod_frequency/calculation_frequency)
        leadtime_calculation_window = int(leadtime_frequency/calculation_frequency)  
        
        # Reindex to calculation delta when required
        if isinstance(idx_map, type(None)):
            idx_map = data.copy()
            idx_map.loc[:,:] = np.arange(idx_map.size).reshape(idx_map.shape, order='C')
            idx_map = idx_map
            leadtimes = data.columns
             
            # Handle columns (leadtimes)
            if leadtime_calculation_window>1:
                if resamplingType=='linear':
                    leadtime_index = pd.timedelta_range(start=leadtimes[0] - leadtime_frequency + calculation_frequency, end=leadtimes[-1] + leadtime_frequency, freq=calculation_frequency, closed='left')
                else:
                    leadtime_index = pd.timedelta_range(start=leadtimes[0], end=leadtimes[-1]+leadtime_frequency, freq=calculation_frequency, closed='left')
                if resamplingType=='sum' or resamplingType=='mean' or resamplingType=='max':
                    idx_map = idx_map.reindex(leadtime_index, axis=1).ffill(limit=leadtime_calculation_window-1, axis=1)
                elif resamplingType=='linear':
                    idx_map = idx_map.reindex(leadtime_index, axis=1)
                else:
                    raise Exception('Resampling type not implemented: %s.' % resamplingType)
            idx_map.columns.name = 'leadtime'
             
            # Handle rows (production times)
            if production_calculation_window>1:
                calculation_index = pd.date_range(start=date_from, end=date_to+leadtime_frequency, freq=calculation_frequency, closed='left')
                 
                idx_map = idx_map.reindex(calculation_index, axis=0)
                valid = idx_map.iloc[0, :]*np.NaN
                shift = 1
                for r0 in range(0, idx_map.shape[0]):
                    if np.isfinite(idx_map.iloc[r0, :]).any():
                        valid = idx_map.iloc[r0, :]
                        shift = 1
                    else:
                        idx_map.iloc[r0, :] = valid.shift(-shift)
                        shift += 1
             
            idx_map = idx_map.stack('leadtime', dropna=False).to_frame()
        
        return idx_map

    @classmethod
    def production_to_event(cls, data, prod_frequency):
        '''
        '''
        
        raise(Exception('Not implemented'))


if __name__=='__main__':
    
    window = dict(from_lon=-7.95, to_lon=-6.95, from_lat=41.05, to_lat=41.85)
    path = Path(r'C:\Users\zepedro\Desktop\ERA5_t2m_IP')
    mr = None
    for f0 in path.glob('*.mr'):
        if mr is None:
            mr = MeteoRaster.load(f0).getCropped(**window)
        else:
            tmp = MeteoRaster.load(f0)
            if f0.name == '_ERA5_Land_tp_2022pls.mr' or f0.name == '_ERA5_Land_t2m_2022pls.mr':
                tmp = tmp.getCropped(from_prod_date=mr.productionDates[-1] + pd.Timedelta('1h'), **window)
            else:
                tmp = tmp.getCropped(**window)
            mr.join(tmp)
    mr.leadtimes = [pd.Timedelta('0d')]
    mr.save(Path(r'C:\Users\zepedro\Dropbox\01.WorkInProgress\Tethys\Hidroerg\Feasibility\data\ERA5_t2m.mr'))
     
    data = mr.getDataFromLatLon(41.7, -7.8)
    
    
    file = Path(r'C:\Users\zepedro\Dropbox\01.WorkInProgress\Tethys\Hidroerg\Feasibility\data\ecmwf_t2m_2022.mr')
    mr = MeteoRaster.load(file)
    #===========================================================================
    # mr.units = 'C'
    # mr.leadtimes = pd.to_timedelta(mr.leadtimes)
    # mr.productionDates = pd.to_datetime(mr.productionDates)
    # mr.data = mr.data.swapaxes(2,1)
    # mr.save(file)
    #===========================================================================
    
    data = mr.getDataFromLatLon(41.7, -7.8)
    for i in range(50):
        data.loc['2022-01-01 00:00:00', (slice(None), i)].plot()
    
    file = Path(r'C:\Users\zepedro\Dropbox\01.WorkInProgress\Tethys\Hidroerg\Feasibility\data\ecmwf_tp_2022.mr')
    mr = MeteoRaster.load(file)
    #===========================================================================
    # mr.units = 'mm/3h'
    # mr.leadtimes = pd.to_timedelta(mr.leadtimes)
    # mr.productionDates = pd.to_datetime(mr.productionDates)
    # mr.data = mr.data.swapaxes(2,1)
    # mr.save(file)
    #===========================================================================
    
    data = mr.getDataFromLatLon(41.7, -7.8)
    for i in range(50):
        data.loc['2022-01-01 00:00:00', (slice(None), i)].cumsum().plot()
    
 #==============================================================================
 #    path = Path(r'C:\Users\zepedro\Desktop\Covas do Barroso (GFS)\gfspcppeninsulaiberica')
 #    units = 'mm/3hr'
 # 
 #    path = Path(r'C:\Users\zepedro\Desktop\Covas do Barroso (GFS)\gfstmppeninsulaiberica')
 #    units = 'C'
 #    
 #    tys_files = path.rglob('*.tys')
 #    window = dict(from_lon=-8, to_lon=-7, from_lat=41, to_lat=42)
 #    mr = None
 #    for f0 in tys_files:
 #        print(f0)
 #        if mr is None:
 #            mr = MeteoRaster.convert_tys(f0, units=units).getCropped(**window)
 #        else:
 #            mr.join(MeteoRaster.convert_tys(f0, units=units).getCropped(**window))
 #    
 #    mr.save(Path(r'C:\Users\zepedro\Dropbox\01.WorkInProgress\Tethys\Hidroerg\Feasibility\data\GFS_t2m.mr'))
 #==============================================================================
    
    file = Path(r'C:\Users\zepedro\Dropbox\12.Temporary\Georgia\ERA5land_Caucasus_tp_2023_24.mr')
    kml = Path(r'C:\Users\zepedro\Dropbox\01.WorkInProgress\Tethys\Enguri\GIS\enguri_areas.kml')
   
    mr = MeteoRaster.load(file)
    mr = mr.getCropped(from_lon=41.5, to_lon=43.5, from_lat=42, to_lat=43.5, from_prod_date=mr.productionDates[1])
    #===========================================================================
    # mr.data = np.ones_like(mr.data)
    # mr.data = np.concatenate([mr.data, mr.data+1], axis=1)
    # mr.data = np.concatenate([mr.data, mr.data+10, mr.data+100, mr.data+1000], axis=2)   
    # mr.leadtimes = pd.timedelta_range('0d', '3d', freq='d')
    #===========================================================================
    agg, centroids = mr.getQuantilesFromKML(kml, nameField='iWatershed', resampling='1d')
    
    a = 1
    pass
    
#===============================================================================
#     matplotlib.use('TkAgg')
#     
#     path = r'D:\raster download\NOAA_GFS_0.25'
#     crop = dict(from_lat=36, to_lat=42, from_lon=65, to_lon=78)
# 
#     timestamps = pd.date_range('2021-04-16', '2024-01-01', freq='1D')
#     leadtimes = pd.timedelta_range('0H', '239H', freq='3H')
#     
#     data = []
#     for timestamp in timestamps:
#         for leadtime in leadtimes:
#             try:
#                 data.append(read_GFS(path, timestamp, leadtime, 'PRATE', crop=crop))
#             except Exception:
#                 pass
#         print(timestamp)
#      
#     meteoraster = data.pop(0)
#     meteoraster.join_(data)
#      
#     meteoraster.save('GFS_tp.mr')
#     
#     
#     data = []
#     for timestamp in timestamps:
#         for leadtime in leadtimes:
#             try:
#                 data.append(read_GFS(path, timestamp, leadtime, 'TMP', crop=crop))
#             except Exception:
#                 pass
#         print(timestamp)
#     
#     meteoraster = data.pop(0)
#     meteoraster.join_(data)
#     
#     meteoraster.save('GFS_t2m.mr')
#===============================================================================
    
#===============================================================================
# #===============================================================================
# # SEAS5pATH = {'t2m': 'D:/Data/SEAS5_TMP/SEAS5_2m_temperature_%s.grib',
# #              'tp': 'D:/Data/SEAS5_PCP/SEAS5_total_precipitation_%s.grib'}
# # NMMEpATH = 'E:/realtime_anom/ENSMEAN'
# # ERA5mONTHLYpATH = 'X:/raster download/ERA5_Monthly/ERA5_monthly_PIberica_%s.grib'
# # IBERIA01pATH = 'X:/raster download/Iberia01/Iberia01_v1.0_DD_010reg_aa3d_%s.nc'
# #===============================================================================
# 
# SEAS5pATH = {'t2m': 'D:/raster download/SEAS5_TMP/SEAS5_2m_temperature_%s.grib',
#              'tp': 'D:/raster download/SEAS5_PCP/SEAS5_total_precipitation_%s.grib'}
# NMMEpATH = 'G:/NOAA seasonal forecasts/ENSMEAN'
# NMMEensemble = ['CFSv1', 'CFSv2', 'CMC1', 'CMC2', 'ECHAMA', 'ECHAMF', 'GFDL', 'GFDL_FLOR', 'GFDL_FLORa06', 'GFDL_FLORb01', 'NASA', 'NCAR', 'NCAR_CCSM4', 'NCAR_CESM']
# ERA5mONTHLYpATH = 'D:/raster download/ERA5_monthly/ERA5_monthly_PIberica_%s.grib'
# IBERIA01pATH = 'D:/raster download/Iberia01/Iberia01_v1.0_DD_010reg_aa3d_%s.nc'
# KMLpATH = 'W:/IST/Projectos/EDP 2023/SIG/CatchmentsPI.kml'
#===============================================================================

    