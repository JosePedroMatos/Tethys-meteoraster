from __future__ import annotations

import numpy as np
import pandas as pd

from meteoraster.resample import Resample


def test_resample_sum_daily_identity():
    index = pd.date_range("2020-01-01", periods=3, freq="1D")
    columns = pd.MultiIndex.from_product(
        [["site-1"], [pd.Timedelta("0D")]],
        names=["site", "leadtime"],
    )
    data = pd.DataFrame([[1.0], [2.0], [3.0]], index=index, columns=columns)

    result = Resample.resample(data, "1D", "sum")

    assert result.index.equals(index)
    assert result.columns.names == ["site", "delay"]
    assert np.allclose(result.values.ravel(), np.array([1.0, 2.0, 3.0]))