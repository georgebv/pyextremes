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

import pandas as pd

from pyextremes.models.emcee.distribution_base import AbstractEmceeDistributionBaseClass
from pyextremes.models.emcee.genextreme import Genextreme
from pyextremes.models.emcee.genpareto import Genpareto

logger = logging.getLogger(__name__)

distributions = {
    'genextreme': Genextreme,
    'genpareto': Genpareto
}


def get_distribution(
        distribution: str,
        extremes: pd.Series,
) -> AbstractEmceeDistributionBaseClass:
    """
    Get a distribution object to be used within the Emcee fitting model.

    Parameters
    ----------
    distribution : str
        Distribution name compatible with scipy.stats.
    extremes : pandas.Series
        Time series of extreme events.

    Returns
    -------
    emcee-compatible distribution object
    """

    logger.info(f'fetching \'{distribution}\' distribution')
    try:
        return distributions[distribution](extremes=extremes)
    except KeyError:
        raise NotImplementedError(
            f'\'{distribution}\' distribution is not implemented for the \'Emcee\' model\n'
            f'Available \'Emcee\' distributions: {", ".join(distributions.keys())}'
        )
