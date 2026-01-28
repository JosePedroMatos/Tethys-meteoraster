from __future__ import annotations

import numpy as np
import pytest

meteoraster = pytest.importorskip("meteoraster")
MeteoRaster = meteoraster.MeteoRaster

def test_to_xarray_dims_and_coords():
    data = np.ones((1, 1, 1, 2, 2), dtype=float)
    latitudes = np.array([20.0, 10.0])
    longitudes = np.array([-5.0, 5.0])
    production_dates = np.array([np.datetime64("2022-06-01")])
    leadtimes = np.array([np.timedelta64(1, "D")])

    mr = MeteoRaster(
        data=data,
        latitudes=latitudes,
        longitudes=longitudes,
        productionDates=production_dates,
        leadtimes=leadtimes,
        units="mm",
        variable="tp",
        verbose=False,
    )

    da = mr.to_xarray()

    assert da.dims == ("production_datetime", "leadtime", "ensemble_idx", "y", "x")
    assert da.shape == (1, 1, 1, 2, 2)
    assert da.attrs["units"] == "mm"
    assert da.attrs["variable"] == "tp"
    assert da.coords["lat"].shape == (2, 2)
    assert da.coords["lon"].shape == (2, 2)
