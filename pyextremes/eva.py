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

import calendar
import logging
import typing
import warnings

import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from pyextremes.extremes import (
    ExtremesTransformer,
    get_extremes,
    get_return_periods
)
from pyextremes.models import get_model, Distribution
from pyextremes.plotting import (
    plot_extremes,
    plot_corner,
    plot_probability,
    plot_return_values,
    plot_trace,
    pyextremes_rc
)

logger = logging.getLogger(__name__)


class EVA:
    """
    Extreme Value Analysis (EVA) class.
    This class brings together most of the tools available in the pyextremes package
    bundled together in a pipeline to perform extreme value analysis.

    A typical workflow using the EVA class would consist of the following:
        - extract extreme values (.get_extremes method)
        - fit a model (.fit_model method)
        - generate outputs (.get_summary method)
        - visualize the model (.plot_diagnostic, .plot_return_values methods)

    Multiple additional graphical and numerical methods are available within this class
    to analyze extracted extreme values, visualize them, assess goodness-of-fit of selected model,
    and to visualize its outputs.

    Parameters
    ----------
    data : pandas.Series
        Time series of the signal.
        Index must be date-time and values must be numeric.
    """

    def __init__(
            self,
            data: pd.Series
    ) -> None:
        logger.info('ensuring that data has correct types')
        if not isinstance(data, pd.Series):
            raise TypeError(f'data must be pandas.Series, {type(data)} was passed')
        if not data.index.is_all_dates:
            raise TypeError('index of data must be a sequence of date-time objects')
        if not np.issubdtype(data.dtype, np.number):
            raise TypeError(f'data must be numeric, {data.dtype} dtype was passed')

        logger.info('ensuring that data is sorted and has no invalid entries')
        self.data = data.copy(deep=True)
        if not data.index.is_monotonic_increasing:
            warnings.warn('data index is not sorted - sorting data by index')
            logger.info('data index is not sorted - sorting data by index')
            self.data = self.data.sort_index(ascending=True)
        if np.any(pd.isna(data)):
            warnings.warn('nan values found in data - removing invalid entries')
            logger.info('nan values found in data - removing invalid entries')
            self.data = self.data.dropna()

        logger.info('initializing attributes related to extreme value extraction')
        self.extremes = None
        self.extremes_method = None
        self.extremes_type = None
        self.extremes_kwargs = None
        self.extremes_transformer = None

        logger.info('initializing attributes related to model fitting')
        self.model = None

    def __repr__(self) -> str:
        # repr parameters
        sep = 6
        width = 100

        def center_text(text: str) -> str:
            lwidth = (width - len(text)) // 2
            rwidth = width - lwidth - len(text)
            return ''.join(
                [
                    ' ' * lwidth,
                    text,
                    ' ' * rwidth
                ]
            )

        def align_text(text: str, value: str) -> str:
            value_width = width - (len(text) + 1)
            return f'{text}:{value:>{value_width:d}}'

        def align_pair(text: tuple, value: tuple) -> str:
            lwidth = int((width - sep) / 2)
            rwidth = width - (lwidth + sep)
            ltext = f'{text[0]}:{value[0]:>{lwidth - len(text[0]) - 1:d}}'
            rtext = f'{text[1]}:{value[1]:>{rwidth - len(text[1]) - 1:d}}'
            return ''.join([ltext, ' ' * sep, rtext])

        # summary header
        start_date = f'{calendar.month_name[self.data.index[0].month]} {self.data.index[0].year}'
        end_date = f'{calendar.month_name[self.data.index[-1].month]} {self.data.index[-1].year}'
        summary = [
            center_text('Extreme Value Analysis'),
            '=' * width,
            center_text('Original Data'),
            '-' * width,
            align_pair(
                ('Data label', 'Data range'),
                (str(self.data.name), f'{start_date} to {end_date}')
            ),
            '=' * width,
            center_text('Extreme Values'),
            '-' * width
        ]

        # extremes section
        if self.extremes is None:
            summary.extend(
                [
                    'Extreme values have not been extracted',
                    '=' * width
                ]
            )
        else:
            if self.extremes_method == 'BM':
                ev_parameters = ('Block size', str(self.extremes_kwargs['block_size']))
            elif self.extremes_method == 'POT':
                ev_parameters = ('Threshold', str(self.extremes_kwargs['threshold']))
            else:
                raise RuntimeError(f'this is a bug - wrong extremes method')
            summary.extend(
                [
                    align_pair(
                        ('Number of extreme events', 'Extraction method'),
                        (f'{len(self.extremes):d}', str(self.extremes_method))
                    ),
                    align_pair(
                        ('Type of extreme events', ev_parameters[0]),
                        (str(self.extremes_type), ev_parameters[1])
                    ),
                    '=' * width
                ]
            )

        # model section
        summary.extend(
            [
                center_text('Model'),
                '-' * width,
            ]
        )
        if self.model is None:
            summary.extend(
                [
                    'Model has not been fit to the extremes',
                    '=' * width
                ]
            )
        else:
            summary.append(
                align_pair(
                    ('Model', 'Distribution'),
                    (self.model.name, self.model.distribution.name)
                )
            )
            if self.model.name == 'Emcee':
                summary.append(
                    align_pair(
                        ('Walkers', 'Samples per walker'),
                        (f'{self.model.n_walkers:d}', f'{self.model.n_samples:d}')
                    )
                )
            free_parameters = ', '.join(
                [
                    f'{parameter}={self.model.fit_parameters[parameter]:.3f}'
                    for parameter in self.model.distribution.free_parameters
                ]
            )
            fixed_parameters = ', '.join(
                [
                    f'{key}={value:.3f}' for key, value in self.model.distribution.fixed_parameters.items()
                ]
            )
            if fixed_parameters == '':
                fixed_parameters = 'All parameters are free'
            summary.extend(
                [
                    align_pair(
                        ('Free parameters', 'Fixed parameters'),
                        (free_parameters, fixed_parameters)
                    ),
                    align_pair(
                        ('Log-likelihood', 'AIC'),
                        (f'{self.model.loglikelihood:.3f}', f'{self.model.AIC:.3f}')
                    ),
                    '=' * width
                ]
            )
        return '\n'.join(summary)

    def get_extremes(
            self,
            method: str,
            extremes_type: str = 'high',
            **kwargs
    ) -> None:
        """
        Get extreme events from a signal time series using a specified extreme value extraction method.
        Stores extreme values in the self.extremes attribute.

        Parameters
        ----------
        method : str
            Extreme value extraction method.
            Supported values: BM or POT.
        extremes_type : str, optional
            high (default) - get extreme high values
            low - get extreme low values
        kwargs
            if method is BM:
                block_size : str or pandas.Timedelta, optional
                    Block size (default='1Y').
                errors : str, optional
                    raise (default) - raise an exception when encountering a block with no data
                    ignore - ignore blocks with no data
                    coerce - get extreme values for blocks with no data as mean of all other extreme events
                        in the series with index being the middle point of corresponding interval
            if method is POT:
                threshold : int or float
                    Threshold used to find exceedances.
                r : str or pandas.Timedelta, optional
                    Duration of window used to decluster the exceedances (default='24H').
        """

        logger.info(f'getting extremes for method {method} and extremes_type {extremes_type}')
        self.extremes = get_extremes(method=method, ts=self.data, extremes_type=extremes_type, **kwargs)
        self.extremes_method = method
        self.extremes_type = extremes_type
        self.extremes_kwargs = kwargs.copy()
        if 'block_size' in self.extremes_kwargs:
            if isinstance(self.extremes_kwargs['block_size'], str):
                self.extremes_kwargs['block_size'] = pd.to_timedelta(self.extremes_kwargs['block_size'])

        logger.info('creating extremes transformer')
        self.extremes_transformer = ExtremesTransformer(extremes=self.extremes, extremes_type=self.extremes_type)

        logger.info('removing any previously declared models')
        self.model = None

    def plot_extremes(
            self,
            figsize: tuple = (8, 8/1.618)
    ) -> tuple:
        """
        Plot time series of extreme events.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size in inches (default=(8, 8/1.618)).

        Returns
        -------
        figure : matplotlib.figure.Figure
            Figure object.
        axes : matplotlib.axes.Axes
            Axes object.
        """

        logger.info('plotting extremes')
        return plot_extremes(
            ts=self.data,
            extremes=self.extremes,
            extremes_method=self.extremes_method,
            extremes_type=self.extremes_type,
            block_size=self.extremes_kwargs.get('block_size'),
            figsize=figsize
        )

    def fit_model(
            self,
            model: str,
            distribution: typing.Union[str, scipy.stats.rv_continuous],
            distribution_kwargs: dict = None,
            **kwargs
    ) -> None:
        """
        Fit a model to the extracted extreme values.

        Parameters
        ----------
        model : str
            Name of an extreme value distribution fitting model (not case-sensitive).
            Supported names:
                MLE - Maximum Likelihood Estimate (MLE) model, based on scipy (scipy.stats.rv_continuous.fit)
                Emcee - Markov Chain Monte Carlo (MCMC) model, based on the emcee package by Daniel Foreman-Mackey
        distribution : str or scipy.stats.rv_continuous
            Distribution name compatible with scipy.stats or a subclass of scipy.stats.rv_continuous
            See https://docs.scipy.org/doc/scipy/reference/stats.html for a list of continuous distributions
        distribution_kwargs : dict, optional
            Dictionary with special keyword arguments, passsed to the .fit method of the continuous distribution.
            These keyword arguments represent parameters to be held fixed and must be shape, scale, or location
            parameter names with sufix 'f', e.g. 'fc', 'floc', or 'fscale'. By default no parameters are fixed.
            See documentation of a specific scipy.stats distribution for names of available parameters.
            By default (None) assigns fixed parameters automatically based on selected model and extremes type.
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
        """

        logger.info('making sure extreme values have been extracted')
        if self.extremes is None:
            raise AttributeError('extreme values must be extracted before fitting a model, use .get_extremes method')

        logger.info('checking if distribution is valid for extremes type')
        if distribution in ['genextreme', 'gumbel_r']:
            if self.extremes_method != 'BM':
                warnings.warn(
                    f'{distribution} distribution is only applicable to extremes extracted using the BM model'
                )
        elif distribution in ['genpareto', 'expon']:
            if self.extremes_method != 'POT':
                warnings.warn(
                    f'{distribution} distribution is only applicable to extremes extracted using the POT model'
                )

        if distribution_kwargs is None:
            logger.info('assigning default distribution_kwargs')
            _distribution = Distribution(extremes=self.extremes, distribution=distribution)
            if _distribution.name in ['genpareto', 'expon']:
                logger.info(f'fixing location parameter at 0 (floc=0) for {_distribution.name} distribution')
                distribution_kwargs = {'floc': self.extremes_transformer.transformed_extremes.min()}

        logger.info(f'fitting {model} model with {distribution} distribution')
        self.model = get_model(
            model=model,
            extremes=self.extremes_transformer.transformed_extremes,
            distribution=distribution,
            distribution_kwargs=distribution_kwargs,
            **kwargs
        )

    def plot_trace(
            self,
            labels: tuple = None,
            figsize: tuple = None
    ) -> tuple:
        """
        Plot a trace plot for a given MCMC sampler trace.

        Parameters
        ----------
        labels : tuple, optional
            Tuple with parameter names, used to label axes (default=None).
        figsize : tuple, optional
            Figure size in inches (default=None).
            If None, calculated automatically as 8 by 2 times number of parameters.

        Returns
        -------
        figure : matplotlib.figure.Figure
            Figure object.
        axes : tuple
            Tuple with n_parameters Axes objects.
        """

        logger.info('making sure a model has been fit')
        if self.model is None:
            raise AttributeError('a model must be fit to extracted extremes first, use .fit_model method')

        logger.info('making sure the fitting model has trace')
        if self.model.trace is None:
            raise AttributeError('this method is applicable only for MCMC-like models with .trace attribute')

        parameter_names = {
            'c': r'Shape, $\xi$',
            'loc': r'Location, $\mu$',
            'scale': r'Scale, $\sigma$'
        }
        if labels is None:
            logger.info('assigning distribution parameter labels')
            labels = []
            for parameter in self.model.distribution.free_parameters:
                try:
                    labels.append(parameter_names[parameter])
                except KeyError:
                    labels.append(parameter)

        logger.info('preparing distribution parameters MAP tuple')
        trace_map = tuple(
            self.model.fit_parameters[parameter]
            for parameter in self.model.distribution.free_parameters
        )

        return plot_trace(
            trace=self.model.trace,
            trace_map=trace_map,
            labels=labels,
            figsize=figsize
        )

    def plot_corner(
            self,
            burn_in: int = 0,
            labels: tuple = None,
            figsize: tuple = (8, 8)
    ) -> tuple:
        """
        Plot a corner plot for a given MCMC sampler trace.

        Parameters
        ----------
        labels : tuple, optional
            Tuple with parameter names, used to label axes (default=None).
        burn_in : int, optional
            Burn-in value (number of first steps to discard for each walker) (default=0).
        figsize : tuple, optional
            Figure size in inches (default=(8, 8).

        Returns
        -------
        figure : matplotlib.figure.Figure
            Figure object.
        axes : tuple
            Tuple with n_parameters Axes objects.
        """

        logger.info('making sure a model has been fit')
        if self.model is None:
            raise AttributeError('a model must be fit to extracted extremes first, use .fit_model method')

        logger.info('making sure the fitting model has trace')
        if self.model.trace is None:
            raise AttributeError('this method is applicable only for MCMC-like models with .trace attribute')

        parameter_names = {
            'c': r'Shape, $\xi$',
            'loc': r'Location, $\mu$',
            'scale': r'Scale, $\sigma$'
        }
        if labels is None:
            logger.info('assigning distribution parameter labels')
            labels = []
            for parameter in self.model.distribution.free_parameters:
                try:
                    labels.append(parameter_names[parameter])
                except KeyError:
                    labels.append(parameter)

        logger.info('preparing distribution parameters MAP tuple')
        trace_map = tuple(
            self.model.fit_parameters[parameter]
            for parameter in self.model.distribution.free_parameters
        )

        return plot_corner(
            trace=self.model.trace,
            trace_map=trace_map,
            labels=labels,
            burn_in=burn_in,
            figsize=figsize
        )

    def get_return_value(
            self,
            return_period: typing.Union[float, tuple, list, np.ndarray],
            return_period_size: typing.Union[str, pd.Timedelta] = '1Y',
            alpha: float = None,
            **kwargs
    ) -> tuple:
        """
        Get return value and confidence interval for given return period(s).

        Parameters
        ----------
        return_period : float or array-like
            Return period or array of return periods.
        return_period_size : str or pandas.Timedelta, optional
            Size of return periods (default='1Y').
            If set to '30D', then a return period of 12 would be equivalent to 1 year return period.
        alpha : float, optional
            Width of confidence interval, from 0 to 1 (default=None).
            If None, return None for upper and lower confidence interval bounds.
        kwargs
            Keyword arguments specific to a model.
            If alpha is None, keyword arguments are ignored (error still raised for unrecognized arguments).
            MLE model:
                n_samples : int, optional
                    Number of bootstrap samples used to estimate confidence interval bounds (default=100).
            Emcee model:
                burn_in : int
                    Burn-in value (number of first steps to discard for each walker).

        Returns
        -------
        return_value : float or array-like
            Return value(s).
        lower_ci_bound : float or array-like
            Lower confidence interval bound(s).
        upper_ci_bount : float or array-like
            Upper confidence interval bound(s).
        """

        logger.info('making sure a model has been fit')
        if self.model is None:
            raise AttributeError('a model must be fit to extracted extremes first, use .fit_model method')

        logger.info('calculating rate of extreme events as number of events per return_period_size')
        if self.extremes_method == 'BM':
            extremes_rate = return_period_size / self.extremes_kwargs['block_size']
            logger.debug('calculated extremes_rate for BM method')
        elif self.extremes_method == 'POT':
            n_periods = (self.data.index[-1] - self.data.index[0]) / return_period_size
            extremes_rate = len(self.extremes) / n_periods
            logger.debug('calculated extremes_rate for POT method')
        else:
            raise RuntimeError('this is a bug - invalid extremes method')

        logger.info('calculating exceedance probability')
        if isinstance(return_period, (tuple, list, np.ndarray)):
            logger.info('getting a list of exceedance probabilities')
            exceedance_probability = 1 / np.array(return_period) / extremes_rate
        elif isinstance(return_period, (int, float, np.number)):
            logger.info('getting a single exceedance probability')
            exceedance_probability = 1 / return_period / extremes_rate
        else:
            raise TypeError(f'invalid type in {type(return_period)} for the \'return_period\' argument')

        logger.info('calculating return value using the model')
        rv = self.model.get_return_value(
            exceedance_probability=exceedance_probability,
            alpha=alpha,
            **kwargs
        )
        return tuple(self.extremes_transformer.transform(value) for value in rv)

    def get_summary(
            self,
            return_period: typing.Union[tuple, list, np.ndarray],
            return_period_size: typing.Union[str, pd.Timedelta] = '1Y',
            alpha: float = 0.95,
            **kwargs
    ) -> pd.DataFrame:
        """
        Generate a pandas DataFrame with return values and confidence interval bounds for given return periods.

        Parameters
        ----------
        return_period : array-like
            Return period or array of return periods.
        return_period_size : str or pandas.Timedelta, optional
            Size of return periods (default='1Y').
            If set to '30D', then a return period of 12 would be equivalent to 1 year return period.
        alpha : float, optional
            Width of confidence interval, from 0 to 1 (default=0.95).
        kwargs
            Model-specific keyword arguments.
            MLE model:
                n_samples : int
                    Number of samles used to get confidence interval.
            Emcee model:
                burn_in : int
                    Burn-in value (number of first steps to discard for each walker).

        Returns
        -------
        summary : pandas.DataFrame
            DataFrame with return values and confidence interval bounds.
        """

        logger.info('calculating return values')
        return_values = self.get_return_value(
            return_period=return_period,
            return_period_size=return_period_size,
            alpha=alpha,
            **kwargs
        )

        logger.info('preparing the summary dataframe')
        return pd.DataFrame(
            data=np.transpose(return_values),
            index=pd.Index(data=return_period, name='return period'),
            columns=['return value', 'lower ci', 'upper ci']
        )

    def plot_return_values(
            self,
            return_period: typing.Iterable = None,
            return_period_size: typing.Union[str, pd.Timedelta] = '1Y',
            alpha: float = 0.95,
            plotting_position: str = 'weibull',
            ax=None,
            figsize: tuple = (8, 8 / 1.618),
            **kwargs
    ) -> tuple:
        """
        Plot return values and confidence intervals for given return periods.

        Parameters
        ----------
        return_period : array-like, optional
            Return period or array of return periods.
            If None, calculates as 100 values uniformly spaced within the range
            of return periods of the extracted extreme values.
        return_period_size : str or pandas.Timedelta, optional
            Size of return periods (default='1Y').
            If set to '30D', then a return period of 12 would be equivalent to 1 year return period.
        alpha : float, optional
            Width of confidence interval, from 0 to 1 (default=0.95).
        plotting_position : str, optional
            Plotting position name (default='weibull'), not case-sensitive.
            Supported plotting positions:
                ecdf, hazen, weibull, tukey, blom, median, cunnane, gringorten, beard
        ax : matplotlib.axes.Axes, optional
            Axes onto which the figure is drawn (default=None).
            If None, a new figure and axes is created.
        figsize : tuple, optional
            Figure size in inches (default=(8, 8/1.618)).
        kwargs
            Model-specific keyword arguments.
            MLE model:
                n_samples : int
                    Number of samles used to get confidence interval.
            Emcee model:
                burn_in : int
                    Burn-in value (number of first steps to discard for each walker).

        Returns
        -------
        if ax is None:
            figure : matplotlib.figure.Figure
                Figure object.
        else:
            None
        axes : matplotlib.axes.Axes
            Axes object.
        """

        logger.info('getting observed return values')
        try:
            block_size = self.extremes_kwargs['block_size']
        except KeyError:
            block_size = None
        observed_return_values = get_return_periods(
            ts=self.data,
            extremes=self.extremes,
            extremes_method=self.extremes_method,
            extremes_type=self.extremes_type,
            block_size=block_size,
            return_period_size=return_period_size,
            plotting_position=plotting_position
        )

        if return_period is None:
            logger.info('creating default return_period array')
            return_period = np.linspace(
                observed_return_values.loc[:, 'return period'].min(),
                observed_return_values.loc[:, 'return period'].max(),
                100
            )

        logger.info('getting modeled return values')
        modeled_return_values = self.get_summary(
            return_period=return_period,
            return_period_size=return_period_size,
            alpha=alpha,
            **kwargs
        )

        return plot_return_values(
            observed_return_values=observed_return_values,
            modeled_return_values=modeled_return_values,
            ax=ax,
            figsize=figsize
        )

    def plot_probability(
            self,
            plot_type: str,
            return_period_size: typing.Union[str, pd.Timedelta] = '1Y',
            plotting_position: str = 'weibull',
            ax=None,
            figsize: tuple = (8, 8)
    ) -> tuple:
        """
        Plot a probability plot (QQ or PP).
        Used to assess goodness-of-fit of the model.

        Parameters
        ----------
        plot_type : str
            Probability plot type. Valid values are PP and QQ.
        return_period_size : str or pandas.Timedelta, optional
            Size of return periods (default='1Y').
            If set to '30D', then a return period of 12 would be equivalent to 1 year return period.
        plotting_position : str, optional
            Plotting position name (default='weibull'), not case-sensitive.
            Supported plotting positions:
                ecdf, hazen, weibull, tukey, blom, median, cunnane, gringorten, beard
        ax : matplotlib.axes.Axes, optional
            Axes onto which the figure is drawn (default=None).
            If None, a new figure and axes is created.
        figsize : tuple, optional
            Figure size in inches (default=(8, 8)).

        Returns
        -------
        if ax is None:
            figure : matplotlib.figure.Figure
                Figure object.
        else:
            None
        axes : matplotlib.axes.Axes
            Axes object.
        """

        logger.info('getting observed return values')
        try:
            block_size = self.extremes_kwargs['block_size']
        except KeyError:
            block_size = None
        observed_return_values = get_return_periods(
            ts=self.data,
            extremes=self.extremes,
            extremes_method=self.extremes_method,
            extremes_type=self.extremes_type,
            block_size=block_size,
            return_period_size=return_period_size,
            plotting_position=plotting_position
        )

        logger.info(f'getting observed and theoretical values for plot_type=\'{plot_type}\'')
        if plot_type == 'PP':
            observed = 1 - observed_return_values.loc[:, 'exceedance probability'].values
            theoretical = self.model.cdf(
                self.extremes_transformer.transform(
                    observed_return_values.loc[:, self.extremes.name].values
                )
            )
        elif plot_type == 'QQ':
            observed = observed_return_values.loc[:, self.extremes.name].values
            theoretical = self.extremes_transformer.transform(
                self.model.isf(observed_return_values.loc[:, 'exceedance probability'].values)
            )
        else:
            raise ValueError(f'\'{plot_type}\' is not a valid \'plot_type\' value. Available plot_types: PP, QQ')

        return plot_probability(
            observed=observed,
            theoretical=theoretical,
            ax=ax,
            figsize=figsize
        )

    def plot_diagnostic(
            self,
            return_period: typing.Iterable = None,
            return_period_size: typing.Union[str, pd.Timedelta] = '1Y',
            alpha: float = 0.95,
            plotting_position: str = 'weibull',
            figsize: tuple = (8, 8),
            **kwargs
    ):
        """
        Plot a diagnostic plot. This plot can be used to quickly assess goodness-of-fit of the selected model.
        The diagnostice plot consists of four axes:
            - top left : return values
            - top right : PDF
            - bottom left : Q-Q
            - bottom right : P-P

        Parameters
        ----------
        return_period : array-like, optional
            Return period or array of return periods.
            If None, calculates as 100 values uniformly spaced within the range
            of return periods of the extracted extreme values.
        return_period_size : str or pandas.Timedelta, optional
            Size of return periods (default='1Y').
            If set to '30D', then a return period of 12 would be equivalent to 1 year return period.
        plotting_position : str, optional
            Plotting position name (default='weibull'), not case-sensitive.
            Supported plotting positions:
                ecdf, hazen, weibull, tukey, blom, median, cunnane, gringorten, beard
        alpha : float, optional
            Width of confidence interval, from 0 to 1 (default=0.95).
        plotting_position : str, optional
            Plotting position name (default='weibull'), not case-sensitive.
            Supported plotting positions:
                ecdf, hazen, weibull, tukey, blom, median, cunnane, gringorten, beard
        figsize : tuple, optional
            Figure size in inches (default=(8, 8)).
        kwargs
            Model-specific keyword arguments.
            MLE model:
                n_samples : int
                    Number of samles used to get confidence interval.
            Emcee model:
                burn_in : int
                    Burn-in value (number of first steps to discard for each walker).

        Returns
        -------
        figure : matplotlib.figure.Figure
            Figure object.
        axes : tuple
            Tuple with four Axes objects: return values, pdf, qq, pp
        """

        with plt.rc_context(rc=pyextremes_rc):
            logger.info('creating figure')
            fig = plt.figure(figsize=figsize, dpi=96)

            logger.info('creating gridspec')
            gs = matplotlib.gridspec.GridSpec(
                nrows=2,
                ncols=2,
                wspace=0.3,
                hspace=0.3,
                width_ratios=[1, 1],
                height_ratios=[1, 1]
            )

            logger.info('creating axes')
            ax_rv = fig.add_subplot(gs[0, 0])
            ax_pdf = fig.add_subplot(gs[0, 1])
            ax_qq = fig.add_subplot(gs[1, 0])
            ax_pp = fig.add_subplot(gs[1, 1])

            logger.info('plotting return values')
            self.plot_return_values(
                return_period=return_period,
                return_period_size=return_period_size,
                alpha=alpha,
                plotting_position=plotting_position,
                ax=ax_rv,
                **kwargs
            )
            ax_rv.set_title('Return value plot')
            ax_rv.grid(False, which='both')

            logger.info('plotting pdf')
            pdf_support = np.linspace(self.extremes.min(), self.extremes.max(), 100)
            pdf = self.model.pdf(self.extremes_transformer.transform(pdf_support))
            ax_pdf.grid(False)
            ax_pdf.set_title('Probability density plot')
            ax_pdf.set_ylabel('Probability density')
            ax_pdf.hist(
                self.extremes.values,
                bins=np.histogram_bin_edges(a=self.extremes.values, bins='auto'),
                density=True, rwidth=0.8,
                facecolor='#5199FF', edgecolor='None', lw=0, alpha=0.25, zorder=5
            )
            ax_pdf.hist(
                self.extremes.values,
                bins=np.histogram_bin_edges(a=self.extremes.values, bins='auto'),
                density=True, rwidth=0.8,
                facecolor='None', edgecolor='#5199FF', lw=1, ls='--', zorder=10
            )
            ax_pdf.plot(
                pdf_support, pdf,
                color='#F85C50', lw=2, ls='-', zorder=15
            )
            ax_pdf.scatter(
                self.extremes.values, np.full(shape=len(self.extremes), fill_value=0),
                marker='|', s=40, facecolor='k', edgecolor='None', lw=0.5, zorder=15
            )
            ax_pdf.set_ylim(0, ax_pdf.get_ylim()[1])

            logger.info('plotting Q-Q plot')
            self.plot_probability(
                plot_type='QQ',
                return_period_size=return_period_size,
                plotting_position=plotting_position,
                ax=ax_qq
            )
            ax_qq.set_title('Q-Q plot')

            logger.info('plotting P-P plot')
            self.plot_probability(
                plot_type='PP',
                return_period_size=return_period_size,
                plotting_position=plotting_position,
                ax=ax_pp
            )
            ax_pp.set_title('P-P plot')

            return fig, (ax_rv, ax_pdf, ax_qq, ax_pp)


if __name__ == '__main__':
    import pathlib
    import os
    test_path = pathlib.Path(os.getcwd()) / 'tests' / 'data' / 'battery_wl.csv'
    test_data = pd.read_csv(test_path, index_col=0, parse_dates=True, squeeze=True)
    test_data = (
        test_data
        .sort_index(ascending=True)
        .dropna()
    )
    test_data = test_data.loc[pd.to_datetime('1925'):]
    test_data = test_data - (test_data.index.array - pd.to_datetime('1992')) / pd.to_timedelta('1Y') * 2.87e-3
    self = EVA(data=test_data)
    self.get_extremes(method='BM', extremes_type='high', block_size='1Y', errors='ignore')
    self.fit_model(model='MLE', distribution='genextreme')
    # self.fit_model(model='Emcee', distribution='genextreme', n_walkers=100, n_samples=500, progress=True)

    self.get_extremes(method='POT', extremes_type='high', threshold=1.3)
    self.fit_model(model='MLE', distribution='genpareto')
    self.plot_diagnostic()
