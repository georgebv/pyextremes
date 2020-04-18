# pyextremes, Extreme Value Analysis in Python
# Copyright (C), 2020 Georgii Bocharov
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import os
import pathlib

import numpy as np
import pandas as pd
import pytest

from pyextremes.extremes import get_extremes

test_data_folder = pathlib.Path(os.path.realpath(__file__)).parent.parent / 'data'
test_data = pd.read_csv(test_data_folder/'battery_wl.csv', index_col=0, parse_dates=True, squeeze=True)


def test_invalid_arguments():
    # Test wrong extremes_type value
    with pytest.raises(ValueError):
        get_extremes(
            ts=test_data,
            method='BM',
            extremes_type='BAD TYPE',
            block_size='1Y',
            errors='coerce'
        )

    # Test wrong block_size type
    with pytest.raises(TypeError):
        get_extremes(
            ts=test_data,
            method='BM',
            extremes_type='high',
            block_size=1,
            errors='coerce'
        )

    # Test wrong errors value
    with pytest.raises(ValueError):
        get_extremes(
            ts=test_data,
            method='BM',
            extremes_type='high',
            block_size='1Y',
            errors='BAD ERRORS'
        )


def test_extreme_value_extraction():
    # Test errors raise
    with pytest.raises(ValueError):
        get_extremes(
            ts=test_data,
            method='BM',
            extremes_type='high',
            block_size='1Y',
            errors='raise'
        )

    # Test extremes_type=(high|low) and errors=(ignore|coerce)
    for extremes_type in ['high', 'low']:
        extremes_ignored = get_extremes(
            ts=test_data,
            method='BM',
            extremes_type=extremes_type,
            block_size='1Y',
            errors='ignore'
        )
        assert len(extremes_ignored) == 96

        extremes_coerced = get_extremes(
            ts=test_data,
            method='BM',
            extremes_type=extremes_type,
            block_size='1Y',
            errors='coerce'
        )
        assert len(extremes_coerced) == 100

        if extremes_type == 'high':
            assert np.isclose(extremes_ignored.max(), extremes_coerced.max())
        else:
            assert np.isclose(extremes_ignored.min(), extremes_coerced.min())
        assert np.isclose(extremes_ignored.mean(), extremes_coerced.mean())
