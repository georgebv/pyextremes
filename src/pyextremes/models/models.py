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
import typing

import pandas as pd
import scipy.stats

from pyextremes.models.emcee import Emcee
from pyextremes.models.mle import MLE

logger = logging.getLogger(__name__)

models = {
    'MLE': MLE,
    'Emcee': Emcee
}


def get_model(
        model: str,
        extremes: pd.Series,
        distribution: typing.Union[str, scipy.stats.rv_continuous],
        distribution_kwargs: dict = None,
        **kwargs
) -> typing.Union[MLE, Emcee]:
    """
    Get an extreme value model and fit it to the given extreme values.

    Parameters
    ----------
    model : str
        Name of an extreme value distribution fitting model.
        Supported names:
            MLE - Maximum Likelihood Estimate (MLE) model, based on scipy (scipy.stats.rv_continuous.fit)
            Emcee - Markov Chain Monte Carlo (MCMC) model, based on the emcee package by Daniel Foreman-Mackey
    extremes : pandas.Series
        Time series of extreme events.
    distribution : str or scipy.stats.rv_continuous
        Distribution name compatible with scipy.stats or a subclass of scipy.stats.rv_continuous
        See https://docs.scipy.org/doc/scipy/reference/stats.html for a list of continuous distributions
    distribution_kwargs : dict, optional
        Dictionary with special keyword arguments, passsed to the .fit method of the continuous distribution.
        These keyword arguments represent parameters to be held fixed and must be shape, scale, or location
        parameter names with sufix 'f', e.g. 'fc', 'floc', or 'fscale'. By default no parameters are fixed.
        See documentation of a specific scipy.stats distribution for names of available parameters.
        Examples:
            dict(fc=0) holds shape parameter 'c' at 0 essentially eliminating it as an independent parameter
                of the distribution, reducting its degree of freedom (number of free parameters) by one.
            dict(floc=0) hold the location parameter 'loc' at 0
            dict(fc=0, floc=10) holds shape and location parameters fixed at 0 and 10 respectively
    kwargs
        Keyword arguments passed to a model .fit method.
        MLE model:
            MLE model takes no additional arguments.
        Emcee model:
            n_walkers : int, optional
                The number of walkers in the ensemble (default=100).
            n_samples : int, optional
                The number of steps to run (default=500).
            progress : bool or str, optional
                If True, a progress bar will be shown as the sampler progresses.
                If a string, will select a specific tqdm progress bar - most notable is
                'notebook', which shows a progress bar suitable for Jupyter notebooks.
                If False, no progress bar will be shown (default=False).
                This progress bar is a part of the emcee package.

    Returns
    -------
    model : Emcee or MLE instance
        An extreme value model fitted to the extreme values.
    """

    logger.info(f'fetching \'{model}\' model')
    try:
        model = models[model]
    except KeyError:
        raise ValueError(f'\'{model}\' is not a valid \'model\' value, available models: {", ".join(models.keys())}')

    logger.info(f'creating {model.name} object')
    return model(
        extremes=extremes,
        distribution=distribution,
        distribution_kwargs=distribution_kwargs,
        **kwargs
    )
