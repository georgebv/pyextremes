import abc
import logging
import typing

import numpy as np
import pandas as pd
import scipy.stats

from pyextremes.models.distribution import Distribution

logger = logging.getLogger(__name__)


class AbstractModelBaseClass(abc.ABC):
    def __init__(
        self,
        extremes: pd.Series,
        distribution: typing.Union[str, scipy.stats.rv_continuous],
        distribution_kwargs: typing.Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the model.

        Parameters
        ----------
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

        """
        self.extremes = extremes

        # Declare extreme value distribution
        distribution_kwargs = distribution_kwargs or {}
        self.distribution = Distribution(
            extremes=self.extremes, distribution=distribution, **distribution_kwargs
        )

        # Fit the distribution to extremes
        self._fit_parameters: typing.Optional[dict] = None
        self._trace: typing.Optional[np.ndarray] = None
        self.fit(**kwargs)

        # Initialize 'return_value_cache'
        self.return_value_cache: typing.Dict[tuple, tuple] = {}

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return model name."""
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, **kwargs) -> None:
        """
        Set values for self.fit_parameters and self.trace.

        self.trace is set only for MCMC-like models.
        self.fit_parameters is a dictionary with {parameter_name: value},
        e.g. {'c': 0.1, 'loc': -7, 'scale': 0.3}
        self.trace is a numpy.ndarray with shape of
        (n_walkers, n_samples, n_free_parameters)

        """
        raise NotImplementedError

    @property
    def fit_parameters(self) -> typing.Dict[str, float]:
        if self._fit_parameters is None:
            raise AssertionError
        else:
            return self._fit_parameters

    @property
    def trace(self) -> np.ndarray:
        if self._trace is None:
            raise TypeError(f"trace property is not applicable for '{self.name}' model")
        else:
            return self._trace

    @property
    def loglikelihood(self) -> float:
        return np.sum(self.logpdf(x=self.extremes.values))

    @property
    def AIC(self) -> float:
        """
        Return corrected Akaike Information Criterion (AIC) of the model.

        Smaller AIC value corresponds to better model.
        This formula scales well for small sample sizes.
        See https://en.wikipedia.org/wiki/Akaike_information_criterion

        """
        k = self.distribution.number_of_parameters
        n = len(self.extremes)
        aic = 2 * k - 2 * self.loglikelihood
        correction = (2 * k ** 2 + 2 * k) / (n - k - 1)
        return aic + correction

    @abc.abstractmethod
    def get_return_value(
        self,
        exceedance_probability: float,
        alpha: typing.Optional[float] = None,
        **kwargs,
    ) -> tuple:
        """
        Calculate return value and confidence interval bounds.

        Parameters
        ----------
        exceedance_probability : array-like
            Exceedance probability or 1D array of exceedance probabilities.
            Each exceedance probability must be in the [0, 1) range.
        alpha : float, optional
            Width of confidence interval (0, 1).
            If None (default), return None
            for upper and lower confidence interval bounds.
        kwargs
            Model-specific keyword arguments.
            If alpha is None, keyword arguments are ignored
            (error still raised for unrecognized arguments).
            MLE model:
                n_samples : int, optional
                    Number of bootstrap samples used to estimate
                    confidence interval bounds (default=100).
            Emcee model:
                burn_in : int
                    Burn-in value (number of first steps to discard for each walker).

        Returns
        -------
        return_value : array-like
            Return values.
        ci_lower : array-like
            Lower confidence interval bounds.
        ci_upper : array-like
            Upper confidence interval bounds.

        """
        raise NotImplementedError

    def _get_prop(self, prop: str, x):
        return self.distribution.get_prop(
            prop=prop, x=x, free_parameters=self.fit_parameters
        )

    def pdf(self, x):
        return self._get_prop(prop="pdf", x=x)

    def logpdf(self, x):
        return self._get_prop(prop="logpdf", x=x)

    def cdf(self, x):
        return self._get_prop(prop="cdf", x=x)

    def ppf(self, x):
        return self._get_prop(prop="ppf", x=x)

    def isf(self, x):
        return self._get_prop(prop="isf", x=x)
