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

from pyextremes.models.mcmc import MCMC
from pyextremes.models.mle import MLE

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def get_fitting_model(
        model: str
) -> type:
    """
    Get an extreme value model.

    Parameters
    ----------
    model : str
        Fitting model name.
        Supported names:
            MLE - Maximum Likelihood Estimate model
            MCMC - Markov Chanin Monte Carlo model

    Returns
    -------
    fitting_model : class
        An extreme value model.
    """

    logger.debug(f'calling get_fitting_model with model={model}')
    if model == 'MLE':
        return MLE
    elif model == 'MCMC':
        return MCMC
    else:
        raise ValueError(f'{model} is not a valid model value')
