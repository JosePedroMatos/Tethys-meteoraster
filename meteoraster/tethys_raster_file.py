'''
Created on 10/02/2023

@author: Jose Pedro Matos
'''

import os
import gc
import time
import gzip
import pickle
import json
import random
import warnings
import numpy as np
import pandas as pd
import datetime as dt
from netCDF4 import Dataset, num2date #@UnresolvedImport
from osgeo import ogr
import xml.etree.ElementTree as et

class TethysRasterFile(object):
    '''
    Class used to store and work with raster files within Tethys.
    
    Should not be used with information from unreliable sources as it used pickle.
    '''

    def __init__(self, file, precision=np.double):
        
        
        if os.path.splitext(file)[-1]!='.tys':
            raise Exception('File extension not recognized. Please use a .tys.')
        
        self.file = file
        self.precision = precision
        self.format = 'data [productionDatetime, leadtime, latitude, longitude], latitudes and longitudes [WGS84 coordinates], leatimes [timedelta64[ns]], productionDatetimes [datetime64[ns]]'
        
        self.data = None
        self.productionDatetimes = None
        self.leadtimes = None
        self.latitudes = None
        self.longitudes = None
        self.missing = None
        self.complete = None
        self.description = None
        self.history = None

    def __str__(self):
        '''
        Overloads the default string function
        '''

        try:
            self.__prepared()
            return ('%s (%06.2f%% missing)' % (self.file, 100*np.mean(self.missing.ravel())))
        except Exception:
            return ('%s (without data)' % (self.file,))

    def load(self):
        '''
        Loads the file from disk
        '''

        self.__waitForLock()

        f = gzip.GzipFile(self.file, 'rb', 1)
        tmp = pickle.load(f)
        f.close()
        
        ignore = ['file']
        
        for k0 in self.__dict__.keys():
            if k0 in tmp.__dict__.keys() and k0 not in ignore:
                self.__dict__[k0] = tmp.__dict__[k0]
        
    def save(self):
        '''
        Saves the file to disk
        '''
        
        self.__prepared('save')
        
        self.completeness()
        
        lockFile = self.__waitForLock(lock=True)
        
        f = gzip.GzipFile(self.file, 'wb', 1)
        pickle.dump(self, f, -1)
        f.close()
    
        os.remove(lockFile)
    
    def create(self, productionDatetimes, leadtimes, latitudes, longitudes, description, data=None):
        '''
        Creates the base information
        '''
        
        self.clear()

        if not self.__isIterable(productionDatetimes):
            productionDatetimes = np.array(productionDatetimes)
        if not self.__isIterable(leadtimes):
            leadtimes = np.array(leadtimes)
        if not self.__isIterable(latitudes):
            latitudes = np.array(latitudes)
        if not self.__isIterable(longitudes):
            longitudes = np.array(longitudes)

        self.productionDatetimes = productionDatetimes.astype('datetime64[ns]')
        self.leadtimes = leadtimes.astype('timedelta64[ns]')
        self.latitudes = latitudes.astype(np.double)
        self.longitudes = longitudes.astype(np.double)
        self.description = description
        self.history = 'Created the %s' % dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if isinstance(data, type(None)):
            self.data = np.empty((self.productionDatetimes.shape[0],
                                  self.leadtimes.shape[0],
                                  self.latitudes.shape[0],
                                  self.longitudes.shape[0],
                                  ), dtype=self.precision) * np.NaN
        else:
            if not isinstance(data, np.array):
                raise Exception('Data must be a numpy array.')
            if len(data.shape)!=4:
                raise Exception('The Data must have four dimensions [productionDatetime, leadtime, latitude, longitude].')
            
            self.data = data
        
        self.completeness()

    def clear(self):
        '''
        Clears stored contents
        '''
        
        self.__init__(self.file, self.precision)
    
    def update(self, data, productionDatetimes=None, leadtimes=None, latitudes=None, longitudes=None):
        '''
        Updates the data
        '''
    
        self.__prepared('update')

        productionDatetimeIdxs, leadtimeIdxs, latitudeIdxs, longitudeIdxs = self.__argsToIdxs(productionDatetimes, leadtimes, latitudes, longitudes)

        self.data[productionDatetimeIdxs[0]:productionDatetimeIdxs[1],
                  leadtimeIdxs[0]:leadtimeIdxs[1],
                  latitudeIdxs[0]:latitudeIdxs[1],
                  longitudeIdxs[0]:longitudeIdxs[1],
                  ] = data

        self.completeness()
    
    def subset(self, productionDatetimes=None, leadtimes=None, latitudes=None, longitudes=None, ):
        '''
        Returns a subset of the data
        '''
    
        self.__prepared('subset')

        productionDatetimeIdxs, leadtimeIdxs, latitudeIdxs, longitudeIdxs = self.__argsToIdxs(productionDatetimes, leadtimes, latitudes, longitudes)
        
        raise Exception('Requires revision! Perhaps 1 should be -1 below.')
        
        return self.data[productionDatetimeIdxs[0]:productionDatetimeIdxs[1],
                  leadtimeIdxs[0]:leadtimeIdxs[1],
                  latitudeIdxs[0]:latitudeIdxs[1],
                  longitudeIdxs[0]:longitudeIdxs[1],
                  ].copy()

    def completeness(self, force=True):
        '''
        Updates the completion status
        '''
        
        self.__prepared('check for completeness')
    
        if not self.complete or force:
            tmp = np.min(np.min(np.isnan(self.data), axis=3), axis=2).astype(bool)
            self.complete = not np.max(tmp)
            self.missing = pd.DataFrame(tmp, index=self.productionDatetimes, columns=self.leadtimes)
        
    def loadNetCDF(self, file_NetCDF):
        '''
        Loads a NetCDF file created by and older version of Tethys 
        '''
        
        rootgrp = Dataset(file_NetCDF, 'r', format="NETCDF4", keepweakref=True)

        self.latitudes = self.__fillMaskedArray(rootgrp.variables['lat'][:])
        self.longitudes = self.__fillMaskedArray(rootgrp.variables['lon'][:])
        self.productionDatetimes = self.__num2date(rootgrp.variables['productionDatetime']).astype('datetime64[ns]')
        self.leadtimes = pd.Series(self.__fillMaskedArray(rootgrp.variables['leadtime'][:])).astype('timedelta64[ns]')
        self.data = self.__fillMaskedArray(rootgrp.variables['data'][:, :, :, :]).astype(self.precision)
        self.description = rootgrp.description
        self.history = rootgrp.history

        rootgrp.close()

        self.missing = bool((self.productionDatetimes.shape[0], self.leadtimes.shape[0]))
        self.completeness()
        
    def observationDatetimes(self, productionDatetimes=None, leadtimes=None):
        '''
        Returns observation datetimes for the whole record or a subset
        '''

        self.__prepared('obtain observation datetimes')

        datetimes = np.empty(self.data.shape[:2])*np.NaN
        datetimes = pd.DataFrame(datetimes, index=self.productionDatetimes, columns=self.leadtimes)
        datetimes = datetimes.apply(lambda x: x.index + x.name).astype('datetime64[ns]')
        datetimes.index.name = 'Production datetime'
        datetimes.columns.name = 'Leadtimes'

        if not isinstance(productionDatetimes, type(None)):
            datetimes = datetimes.loc[productionDatetimes, :]
        
        if not isinstance(leadtimes, type(None)):
            datetimes = datetimes.loc[:, leadtimes]

        return datetimes
        
    def mean(self, productionDatetimes=None, leadtimes=None):
        '''
        Returns means for the whole record or a subset
        '''
        
        self.__prepared('obtain means')

        means = np.empty(self.data.shape[:2])*np.NaN
        means = pd.DataFrame(means, index=self.productionDatetimes, columns=self.leadtimes)
        means.loc[:, :] = np.nanmean(np.nanmean(self.data, axis=3), axis=2)
        means.index.name = 'Production datetime'
        means.columns.name = 'Leadtimes'

        if not isinstance(productionDatetimes, type(None)):
            means = means.loc[productionDatetimes, :]
        
        if not isinstance(leadtimes, type(None)):
            means = means.loc[:, leadtimes]

        return means
        
    def sum(self, productionDatetimes=None, leadtimes=None):
        '''
        Returns sums for the whole record or a subset
        '''
        
        self.__prepared('obtain sums')

        sums = np.empty(self.data.shape[:2])*np.NaN
        sums = pd.DataFrame(sums, index=self.productionDatetimes, columns=self.leadtimes)
        sums.loc[:, :] = np.nansum(np.nansum(self.data, axis=3), axis=2)
        sums.index.name = 'Production datetime'
        sums.columns.name = 'Leadtimes'

        if not isinstance(productionDatetimes, type(None)):
            sums = sums.loc[productionDatetimes, :]
        
        if not isinstance(leadtimes, type(None)):
            sums = sums.loc[:, leadtimes]

        return sums
        
    def coverage(self, productionDatetimes=None, leadtimes=None):
        '''
        Returns coverage for the whole record or a subset
        Should only be equal to 1 if the associated geometry is square matching the full latitude x longitude box
        '''
        
        self.__prepared('obtain coverage')

        coverage = np.empty(self.data.shape[:2])*np.NaN
        coverage = pd.DataFrame(coverage, index=self.productionDatetimes, columns=self.leadtimes)
        coverage.loc[:, :] = np.mean(np.mean(np.isfinite(self.data), axis=3), axis=2)
        coverage.index.name = 'Production datetime'
        coverage.columns.name = 'Leadtimes'

        if not isinstance(productionDatetimes, type(None)):
            coverage = coverage.loc[productionDatetimes, :]
        
        if not isinstance(leadtimes, type(None)):
            coverage = coverage.loc[:, leadtimes]

        return coverage

    def coverageMatrixFromKML(self, kml, nameField):
        '''
        Creates a coverage matrix from polygons contained in a kml file
        '''
        
        # Correct kml file if required
        self.__correctKMLHeader(kml) 
        
        # Read geometry reference KML file
        ids = []
        coordinates = []
        document = et.parse(kml).getroot().find('Document')
        for placemark in document.findall('.//Placemark'):
            #https://docs.python.org/3/library/xml.etree.elementtree.html
            try:
                id0 = placemark.find(".//SimpleData[@name='%s']" % nameField).text
                coords = [pd.Series(c.text.replace(' \n', '').replace('\n', '').replace('\t', '').split(' ')).str.split(',', n=2, expand=True) for c in placemark.findall('.//Polygon/outerBoundaryIs//coordinates')]
                coords = [c.astype(float) for c in coords]
                ids.append(id0)
                coordinates.append(coords)
            except Exception as ex:
                print(str(ex)) 

        # Process geometry
        hdLon = abs(self.longitudes[1]-self.longitudes[0])/2*0.99999 
        hdLat = abs(self.latitudes[1]-self.latitudes[0])/2*0.99999
        
        cvr = np.zeros([1, 1] + list(self.data.shape[2:]) + [len(ids)], dtype=np.single)
        centroids = {'x':[], 'y':[]}
        for i0, coords in enumerate(coordinates):
             
            # Create GDAL geometry
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for row in coords[0].iterrows():
                ring.AddPoint(row[1][0], row[1][1])
            poly = ogr.Geometry(ogr.wkbPolygon)
            _ = poly.AddGeometry(ring)
        
            # Select pixel subset
            extent = poly.GetEnvelope() #wesn 
            extent = [extent[0]-hdLon*2, extent[1]+hdLon*2, extent[2]-hdLat*2, extent[3]+hdLat*2]
            lon_idxs = np.arange(self.longitudes.shape[0])[(self.longitudes>=extent[0]) & (self.longitudes<=extent[1])]
            lat_idxs = np.arange(self.latitudes.shape[0])[(self.latitudes>=extent[2]) & (self.latitudes<=extent[3])]

            x, y, _ = poly.Centroid().GetPoint()
            centroids['x'].append(x)
            centroids['y'].append(y)

            for i2 in lon_idxs:
                for i3 in lat_idxs:
                    lon = self.longitudes[i2]
                    lat = self.latitudes[i3]
                    
                    ring = ogr.Geometry(ogr.wkbLinearRing)
                    ring.AddPoint(lon-hdLon, lat+hdLat)
                    ring.AddPoint(lon+hdLon, lat+hdLat)
                    ring.AddPoint(lon+hdLon, lat-hdLat)
                    ring.AddPoint(lon-hdLon, lat-hdLat)
                    ring.AddPoint(lon-hdLon, lat+hdLat)
                    
                    pixel = ogr.Geometry(ogr.wkbPolygon)
                    _ = pixel.AddGeometry(ring)

                    intersection = pixel.Intersection(poly)
                    area = intersection.Area()
                    if area>0:
                        cvr[0, 0, i3, i2, i0] = area
        
        columns = pd.MultiIndex.from_product((ids, self.leadtimes), names=('Zones', 'Leadtimes'))
        areas = cvr.sum(axis=(0, 1, 2, 3))
        cvr /= np.tile(areas, list(cvr.shape[:-1]) + [1])
        centroids = pd.DataFrame(centroids, index=ids)
        
        #=======================================================================
        # dbg_ = pd.DataFrame((cvr.squeeze()!=0).sum(axis=-1), index=self.latitudes)
        # dbg_.index.name = 'latitudes'
        # dbg_.columns = self.longitudes
        # dbg_.columns.name = 'longitudes'
        # dbg_.to_clipboard(excel=True, sep=',')
        #=======================================================================
        
        return cvr, columns, centroids
    
    def groupByMatrix(self, coverage, columns):
        '''
        Groups the data by a coverage matrix (see groupByKML for matrix definition)
        '''
        
        astype = np.double
        
        #Verification of valid tiles (assumes first coverage zone to be representative (for speed and memory)
        val = np.isfinite(self.data).astype(bool)
        cvr = np.tile(coverage[:,:,:,:,0].astype(astype), list(val.shape[:2]) + [1, 1])
        val = np.nansum(cvr*val, axis=(2, 3))>0.8
        val = np.tile(val, (1, coverage.shape[-1]))
        
        #Creation of matrices for calculation
        dat = np.tile(np.expand_dims(self.data.astype(astype), 4), [1, 1, 1, 1, coverage.shape[-1]])
        cvr = np.tile(coverage.astype(astype), list(dat.shape[:2]) + [1, 1, 1]) # OK
        dat *= cvr
        cvr = None
        gc.collect()
        
        #Calculation with garbage collection
        agg = np.nansum(dat, axis=(2, 3))
        dat = None
        gc.collect()

        agg = pd.DataFrame(np.reshape(agg, (agg.shape[0], agg.shape[1]*agg.shape[2]), order='F'), index=self.productionDatetimes)
        agg.columns = columns
        #=======================================================================
        # agg.loc[agg.index[5], (slice(None), agg.columns.get_level_values('Leadtimes').max())]
        #=======================================================================

        agg.values[~val] = np.NaN 
        
        #=======================================================================
        # dbg_ = agg.dropna(how='all').loc[:, (slice(None), agg.columns.get_level_values('Leadtimes').min())]
        # dbg_.to_clipboard(excel=True, sep=',')
        #=======================================================================
        
        return agg  
    
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
            
    def __intersection(self, coordinates):
        '''
        Returns a matrix [Latitude, Longitude].
        0 > the pixel is outside the boundary
        1 > the pixel is inside the boundary
        '''
        # TODO: extend to cases where the border goes beyond [-180, 180]
        
        pointsX = self.longitudes
        pointsY = self.latitudes
        
        borderX = [c.iloc[:, 0].tolist() for c in coordinates]
        borderY = [c.iloc[:, 1].tolist() for c in coordinates]
        
        pixels = len(pointsX) * len(pointsY)
        segments = sum([len(i)-1 for i in borderX])

        tmpY = [i for j in borderY for i in j]
        tmpX = [i for j in borderX for i in j]
        maxBorderY = max(tmpY)
        maxBorderX = max(tmpX)
        minBorderY = min(tmpY)
        minBorderX = min(tmpX)

        # Defining matrices for calculation
        pointsX = np.expand_dims(pointsX, axis=1)
        pointsY = np.expand_dims(pointsY, axis=0)

        pixelBaseX = np.repeat(pointsX, pointsY.shape[1], axis=1).ravel()
        pixelBaseY = np.repeat(pointsY, pointsX.shape[0], axis=0).ravel()

        validBoolX = np.logical_and(pixelBaseX<=maxBorderX, pixelBaseX>=minBorderX)
        validBoolY = np.logical_and(pixelBaseY<=maxBorderY, pixelBaseY>=minBorderY)
        validBool = np.logical_and(validBoolX, validBoolY)

        L = np.zeros(pixels, dtype=int)
        pixelRedX = pixelBaseX[validBool]
        pixelRedY = pixelBaseY[validBool]
        x2 = 9999
        y2 = 9999

        # Calculate constants
        a = (pixelRedX*y2-pixelRedY*x2)
        d = (pixelRedX-x2)
        e = (pixelRedY-y2)

        # Loop each segment within each polygon
        ctr = 0
        for x, y in zip(borderX, borderY):
            for i1 in range(len(x)-1):
                ctr += 1
                if np.mod(ctr, 1000)==0:
                    progress = ctr / segments
                    self._print(progress)

                x3 = x[i1]
                y3 = y[i1]
                x4 = x[i1+1]
                y4 = y[i1+1]

                # Computing intersection coordinates
                b = (x3*y4-y3*x4)
                c = d*(y3-y4)-e*(x3-x4)

                px = (a*(x3-x4)-d*b)/c
                py = (a*(y3-y4)-e*b)/c

                # Bounding intersections to the real lines
                lx = np.logical_and(
                                    px>=pixelRedX,
                                    np.logical_or(
                                                  np.logical_and(px<=x3+1E-6, px>=x4-1E-6),
                                                  np.logical_and(px<=x4+1E-6, px>=x3-1E-6)))
                ly = np.logical_and(
                                    py>=pixelRedY,
                                    np.logical_or(
                                                  np.logical_and(py<=y3+1E-6, py>=y4-1E-6),
                                                  np.logical_and(py<=y4+1E-6, py>=y3-1E-6)))
                L[validBool] += np.logical_and(lx, ly)
        L = np.mod(L, 2)==1
        L = np.reshape(L, (pointsY.shape[1], pointsX.shape[0]), order='F')
    
        return L % 2
    
    def __prepared(self, info='execute'):
        '''
        Raises an exception if the object is not prepared
        '''
        
        conditions = [self.data,
                      self.productionDatetimes,
                      self.leadtimes,
                      self.latitudes,
                      self.longitudes,
                      self.description,
                      self.history,
                      ]
        
        if any([isinstance(c, type(None)) for c in conditions]):
            raise Exception('Cannot %s without previously loading or creating contents.' % info)
    
    def __idxsAreContinuous(self, array):
        '''
        Checks whether the array contains a continuous sequence of indexes
        '''
        
        if array.shape[0]>0:
            if any(np.diff(array, n=1) != 1):
                raise Exception('The arguments must be continuous.')
    
    def __argsToIdxs(self, productionDatetimes, leadtimes, latitudes, longitudes):
        '''
        Transforms arguments into indexes of the data matrix
        '''
            
        productionDatetimeIdxs = np.arange(self.productionDatetimes.shape[0])
        if not isinstance(productionDatetimes, type(None)):
            if not self.__isIterable(productionDatetimes):
                productionDatetimes = np.array(productionDatetimes, dtype='datetime64[ns]')
            productionDatetimeIdxs = productionDatetimeIdxs[np.isin(self.productionDatetimes, productionDatetimes)]
            self.__idxsAreContinuous(productionDatetimeIdxs)
        productionDatetimeIdxs = (productionDatetimeIdxs[0], productionDatetimeIdxs[-1]+1)
        
        leadtimeIdxs = np.arange(self.leadtimes.shape[0])
        if not isinstance(leadtimes, type(None)):
            if not self.__isIterable(leadtimes):
                leadtimes = pd.Series(leadtimes).values.astype('timedelta64[ns]') #np.array(leadtimes, dtype='timedelta64[ns]')
            leadtimeIdxs = leadtimeIdxs[np.isin(self.leadtimes, leadtimes)]
            self.__idxsAreContinuous(leadtimeIdxs)
        leadtimeIdxs = (leadtimeIdxs[0], leadtimeIdxs[-1]+1)
    
        latitudeIdxs = np.arange(self.latitudes.shape[0])
        if not isinstance(latitudes, type(None)):
            if not self.__isIterable(latitudes):
                latitudes = np.array(latitudes)
            latitudeIdxs = latitudeIdxs[np.isin(self.latitudes, latitudes)]
            self.__idxsAreContinuous(latitudeIdxs)
        latitudeIdxs = (latitudeIdxs[0], latitudeIdxs[-1]+1)
        
        longitudeIdxs = np.arange(self.longitudes.shape[0])
        if not isinstance(longitudes, type(None)):
            if not self.__isIterable(longitudes):
                longitudes = np.array(longitudes)
            longitudeIdxs = longitudeIdxs[np.isin(self.longitudes, longitudes)]
            self.__idxsAreContinuous(longitudeIdxs)
        longitudeIdxs = (longitudeIdxs[0], longitudeIdxs[-1]+1)
    
        return productionDatetimeIdxs, leadtimeIdxs, latitudeIdxs, longitudeIdxs
    
    def __waitForLock(self, lock=False):
        '''
        Waits until the lock is removed from the file before saving
        '''
        
        lockFile = str(self.file) + '.lock'
        requestTime = dt.datetime.now()

        # Wait for the lock to disappear or throw an error if the lock remains in place for a long time. Erase very old locks
        while os.path.exists(lockFile):
            try:
                lockAge = (dt.datetime.now()-dt.datetime.fromtimestamp(os.stat(lockFile).st_ctime)).total_seconds()
                if lockAge>60:
                    warnings.warn('Lock older than 1 minute (%s). Attempting to remove...' % lockFile)
                    os.remove(lockFile)
                
                if (dt.datetime.now()-requestTime).total_seconds() > 30:
                    raise Exception('Access time exceeded...')
            except Exception as ex:
                warnings.warn(str(ex))
            
            time.sleep(0.01)
    
        # Create new lock
        if lock:
            with open(lockFile, 'w'): pass
    
        return lockFile
    
    @staticmethod
    def __isIterable(obj):
        try:
            iter(obj)
        except Exception:
            return False
        else:
            return True
    
    @staticmethod
    def __fillMaskedArray(array):
        '''
        Fills a masked array when needed
        '''
    
        if np.ma.core.isMaskedArray(array):
            return array.filled()
        else:
            return array
    
    @staticmethod
    def __num2date(times):
        '''
        Corrects the native netCDF4 num2date function's results, which are wrong in the microsecond range
        '''
    
        dates = num2date(times[:], units=times.units, calendar=times.calendar)
        corrected = np.array([d - dt.timedelta(microseconds=d.microsecond) for d in dates], dtype='datetime64[ns]')
    
        return corrected

