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

import numpy as np
import pandas as pd

from pyextremes.extremes import ExtremesTransformer


def test_extremes_transformer():
    extremes_hl = {
        'high': pd.Series([1, 1, 1, 2, 2, 3]),
        'low': pd.Series([1, 2, 2, 3, 3, 3])
    }

    # Test transforms for all input combinations
    for extremes_type in ['high', 'low']:
        extremes = extremes_hl[extremes_type]
        transformer = ExtremesTransformer(
            extremes=extremes,
            extremes_type=extremes_type
        )

        # Ensure transform works as expected
        if extremes_type == 'high':
            assert np.all(transformer.transformed_extremes == extremes)
        else:
            assert np.all(transformer.transformed_extremes == pd.Series([5, 4, 4, 3, 3, 3]))

        # Ensure transform works both ways
        for value in [-10, 0, 50]:
            assert transformer.transform(value=transformer.transform(value=value)) == value
