import numpy as np
import pandas as pd
import pytest

from pyextremes.extremes import ExtremesTransformer


@pytest.mark.parametrize("extremes_type", ["high", "low"])
def test_extremes_transformer(extremes_type):
    extremes_hl = {
        "high": pd.Series([1, 1, 1, 2, 2, 3]),
        "low": pd.Series([1, 2, 2, 3, 3, 3]),
    }

    # Test transforms
    extremes = extremes_hl[extremes_type]
    transformer = ExtremesTransformer(extremes=extremes, extremes_type=extremes_type)

    # Ensure transform works as expected
    if extremes_type == "high":
        assert np.all(transformer.transformed_extremes == extremes)
    else:
        assert np.all(transformer.transformed_extremes == pd.Series([5, 4, 4, 3, 3, 3]))

    # Ensure transform works both ways
    for value in [-10, 0, 50]:
        assert transformer.transform(value=transformer.transform(value=value)) == value

    # Test null-transform
    assert transformer.transform(value=None) is None
