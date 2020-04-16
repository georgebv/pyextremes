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

import logging
import os

import numpy as np
import pandas as pd
import pytest

from pyextremes.extremes import get_extremes

# Set up logging
logger = logging.getLogger('pyextremes.extremes')
logger.setLevel(logging.CRITICAL)

test_data = pd.read_csv(
    os.sep.join([*os.path.realpath(__file__).split(os.sep)[:-2], 'data', 'battery_wl.csv']),
    index_col=0, parse_dates=True, squeeze=True
)


def test_get_extremes():
    # Test POT with wrong extremes_type value
    with pytest.raises(ValueError):
        get_extremes(
            method='POT',
            ts=test_data,
            extremes_type='BAD TYPE',
            threshold=2,
            r='24H'
        )

    # Test POT with wrong r type
    with pytest.raises(TypeError):
        get_extremes(
            method='POT',
            ts=test_data,
            extremes_type='high',
            threshold=2,
            r=1
        )

    # Test POT with extremes_type=high
    extremes_bm_high = get_extremes(
        method='BM',
        ts=test_data,
        block_size='1Y',
        extremes_type='high',
        errors='coerce'
    )
    extremes_pot_high = get_extremes(
        method='POT',
        ts=test_data,
        extremes_type='high',
        threshold=1.35,
        r='24H'
    )
    assert len(extremes_pot_high) == 127
    assert np.isclose(extremes_pot_high.max(), extremes_bm_high.max())

    # Test POT with extremes_type=low
    extremes_bm_low = get_extremes(
        method='BM',
        ts=test_data,
        block_size='1Y',
        extremes_type='low',
        errors='coerce'
    )
    extremes_pot_low = get_extremes(
        method='POT',
        ts=test_data,
        extremes_type='low',
        threshold=-1.65,
        r='24H'
    )
    assert len(extremes_pot_low) == 109
    assert np.isclose(extremes_pot_low.min(), extremes_bm_low.min())