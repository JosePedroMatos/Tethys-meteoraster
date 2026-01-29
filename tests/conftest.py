from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import shutil
import tempfile
import os
import numpy as np
import pandas as pd
import pytest
from osgeo import ogr, osr

from meteoraster import MeteoRaster

@pytest.fixture
def sample_meteoraster():
    # Create a 5D data array
    # Dimensions: [production_datetime, ensemble_member, leadtime, y, x]
    # Shape: [2, 3, 2, 4, 5]
    n_prod = 2
    n_ens = 3
    n_lead = 2
    n_lat = 4
    n_lon = 5
    
    data = np.random.rand(n_prod, n_ens, n_lead, n_lat, n_lon)
    
    # Latitudes and longitudes
    lats = np.linspace(40, 30, n_lat)
    lons = np.linspace(-10, 10, n_lon)
    latitudes, longitudes = np.meshgrid(lats, lons)
    latitudes = latitudes.T
    longitudes = longitudes.T

    # Production dates
    production_datetime = pd.to_datetime(["2023-01-01", "2023-01-02"]).values
    
    # Leadtimes
    leadtimes = np.array([pd.Timedelta(days=1), pd.Timedelta(days=2)])

    mr = MeteoRaster(
        data=data,
        latitudes=lats,
        longitudes=lons,
        production_datetime=production_datetime,
        leadtimes=leadtimes,
        units="mm",
        variable="precip",
        verbose=False
    )
    return mr

@pytest.fixture
def sample_kml():
    # Remove XML namespace to avoid ElementTree finding quirks in the code under test
    kml_content = """<?xml version="1.0" encoding="UTF-8"?>
<kml>
  <Document>
    <Placemark>
      <ExtendedData>
        <SchemaData schemaUrl="#test">
            <SimpleData name="zone_id">ZoneA</SimpleData>
        </SchemaData>
      </ExtendedData>
      <Polygon>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>-2.0,32.0,0 2.0,32.0,0 2.0,38.0,0 -2.0,38.0,0 -2.0,32.0,0</coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>
    <Placemark>
      <ExtendedData>
        <SchemaData schemaUrl="#test">
            <SimpleData name="zone_id">ZoneB</SimpleData>
        </SchemaData>
      </ExtendedData>
      <Polygon>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>-1.0,33.0,0 1.0,33.0,0 1.0,37.0,0 -1.0,37.0,0 -1.0,33.0,0</coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>
  </Document>
</kml>
"""
    # Create temp file
    with tempfile.NamedTemporaryFile(suffix=".kml", delete=False, mode='w') as tmp:
        tmp.write(kml_content)
        tmp_path = Path(tmp.name)
    
    yield tmp_path
    
    if tmp_path.exists():
        os.remove(tmp_path)

@pytest.fixture
def sample_shapefile():
    # Create a temporary directory for shapefile components
    tmp_dir = tempfile.mkdtemp()
    shp_path = os.path.join(tmp_dir, "test_shape.shp")

    # Create driver
    driver = ogr.GetDriverByName("ESRI Shapefile")
    data_source = driver.CreateDataSource(shp_path)

    # Create spatial reference (WGS84)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    # Create layer
    layer = data_source.CreateLayer("test_shape", srs, ogr.wkbPolygon)

    # Add a field
    field_defn = ogr.FieldDefn("name", ogr.OFTString)
    field_defn.SetWidth(24)
    layer.CreateField(field_defn)

    # Create feature
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetField("name", "TestPolygon")

    # Create geometry
    # A simple square
    wkt = "POLYGON ((-1 33, 1 33, 1 37, -1 37, -1 33))"
    poly = ogr.CreateGeometryFromWkt(wkt)
    feature.SetGeometry(poly)

    # Create the feature in the layer
    layer.CreateFeature(feature)

    # cleanup
    feature = None
    data_source = None
    
    yield Path(shp_path)
    
    # Cleanup directory
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
