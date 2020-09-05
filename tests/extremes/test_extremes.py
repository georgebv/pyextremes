import pandas as pd
import pytest

from pyextremes.extremes import get_extremes


def test_get_extremes():
    with pytest.raises(ValueError, match=r"invalid value.*method.*argument"):
        get_extremes(ts=pd.Series([1, 2, 3]), method="BAD METHOD", extremes_type="high")
