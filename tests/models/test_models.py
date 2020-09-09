import pandas as pd
import pytest

from pyextremes.models import get_model


def test_get_model():
    with pytest.raises(ValueError, match=r"invalid value.*'model' argument"):
        get_model(
            model="BAD MODEL", extremes=pd.Series([1, 2, 3]), distribution="genextreme"
        )
