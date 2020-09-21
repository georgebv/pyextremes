import typing

import pandas as pd
import scipy.stats

from pyextremes.models.model_emcee import Emcee
from pyextremes.models.model_mle import MLE


def get_model(
    model: str,
    extremes: pd.Series,
    distribution: typing.Union[str, scipy.stats.rv_continuous],
    distribution_kwargs: typing.Optional[dict] = None,
    **kwargs,
) -> typing.Union[MLE, Emcee]:
    """
    Get distribution fitting model and fit it to given extreme values.

    Parameters
    ----------
    model : str
        Name of model.
        Supported models:
            MLE - Maximum Likelihood Estimate (MLE) model.
                Based on 'scipy' package (scipy.stats.rv_continuous.fit).
            Emcee - Markov Chain Monte Carlo (MCMC) model.
                Based on 'emcee' package by Daniel Foreman-Mackey.
    extremes : pandas.Series
        Time series of extreme events.
    distribution : str or scipy.stats.rv_continuous
        Distribution name compatible with scipy.stats
        or a subclass of scipy.stats.rv_continuous.
        See https://docs.scipy.org/doc/scipy/reference/stats.html
    distribution_kwargs : dict, optional
        Special keyword arguments, passsed to the `.fit` method of the distribution.
        These keyword arguments represent parameters to be held fixed.
        Names of parameters to be fixed must have 'f' prefixes. Valid parameters:
            - shape(s): 'fc', e.g. fc=0
            - location: 'floc', e.g. floc=0
            - scale: 'fscale', e.g. fscale=1
        By default, no parameters are fixed.
        See documentation of a specific scipy.stats distribution
        for names of available parameters.
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
                If a string, will select a specific tqdm progress bar.
                Most notable is 'notebook', which shows a progress bar
                suitable for Jupyter notebooks.
                If False (default), no progress bar will be shown.
                This progress bar is a part of the `emcee` package.

    Returns
    -------
    model : MLE or Emcee
        Distribution fitting model fitted to the `extremes`.

    """
    distribution_model_kwargs = {
        "extremes": extremes,
        "distribution": distribution,
        "distribution_kwargs": distribution_kwargs,
        **kwargs,
    }

    if model == "MLE":
        return MLE(**distribution_model_kwargs)
    elif model == "Emcee":
        return Emcee(**distribution_model_kwargs)
    else:
        raise ValueError(
            f"invalid value in '{model}' for the 'model' argument, "
            f"available model: 'MLE', 'Emcee'"
        )
