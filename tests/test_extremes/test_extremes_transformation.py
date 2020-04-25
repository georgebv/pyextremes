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

from pyextremes.extremes import ExtremesTransformer

test_data_folder = pathlib.Path(os.path.realpath(__file__)).parent.parent / 'data'
test_data = pd.read_csv(test_data_folder/'battery_wl.csv', index_col=0, parse_dates=True, squeeze=True)


def test_extremes_transformer():
    extremes = pd.Series([1, 2, 3])

    # Test bad extremes_method
    with pytest.raises(ValueError):
        ExtremesTransformer(
            extremes=extremes,
            extremes_method='BAD METHOD',
            extremes_type='high'
        )

    # Test bad extremes_type
    with pytest.raises(ValueError):
        ExtremesTransformer(
            extremes=extremes,
            extremes_method='POT',
            extremes_type='BAD TYPE'
        )

    # Test transforms for all input combinations
    for extremes_method in ['BM', 'POT']:
        for extremes_type in ['high', 'low']:
            extremes = pd.read_csv(
                test_data_folder/f'extremes_{extremes_method.lower()}_{extremes_type}.csv',
                index_col=0, parse_dates=True, squeeze=True
            )
            transformer = ExtremesTransformer(
                extremes=extremes,
                extremes_method=extremes_method,
                extremes_type=extremes_type
            )

            # Ensure that transformed extremes have lower probabilities for higher values
            hist, bins = np.histogram(
                a=transformer.transformed_extremes,
                bins=np.histogram_bin_edges(transformer.transformed_extremes, bins='sturges')
            )
            assert sum(hist[:len(hist)//2]) > sum(hist[len(hist)//2:])

            # Ensure forward and inverse transforms are consistent
            for value in [-10, 0, 50]:
                assert transformer.inverse_transform(value=transformer.forward_transform(value=value)) == value

            # Ensure series starts at 0 for extremes_method=POT
            if extremes_method == 'POT':
                assert transformer.transformed_extremes.min() == 0
