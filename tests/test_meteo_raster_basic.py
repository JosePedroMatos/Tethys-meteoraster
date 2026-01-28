from __future__ import annotations

import numpy as np
import pytest

meteoraster = pytest.importorskip("meteoraster")
MeteoRaster = meteoraster.MeteoRaster


def test_init_fix_greenwich_and_flip_latitudes():
    data = np.arange(1 * 1 * 1 * 2 * 3, dtype=float).reshape(1, 1, 1, 2, 3)
    latitudes = np.array([10.0, 20.0])
    longitudes = np.array([170.0, 190.0, 200.0])
    production_dates = np.array([np.datetime64("2020-01-01")])
    leadtimes = np.array([np.timedelta64(0, "D")])

    mr = MeteoRaster(
        data=data,
        latitudes=latitudes,
        longitudes=longitudes,
        productionDates=production_dates,
        leadtimes=leadtimes,
        units="unit",
        variable="var",
        verbose=False,
    )

    assert mr.longitudes.ndim == 2
    assert mr.latitudes.ndim == 2
    assert np.min(mr.longitudes) < 0
    assert mr.latitudes[0, 0] > mr.latitudes[-1, 0]


def test_closest_idx():
    idxs = MeteoRaster.closestIdx(np.array([0.1, 1.9, 3.1]), [0, 1, 2, 3])
    assert np.array_equal(idxs, np.array([0, 2, 3]))
