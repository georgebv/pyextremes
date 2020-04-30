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

import numpy as np
import pandas as pd

from pyextremes.extremes import get_extremes, ExtremesTransformer
from pyextremes.models import get_model
from pyextremes.plotting import plot_extremes

logger = logging.getLogger(__name__)


class EVA:

    def __init__(
            self,
            data: pd.Series
    ) -> None:
        logger.info('ensuring data has correct types')
        if not isinstance(data, pd.Series):
            raise TypeError(f'invalid type in {type(data)} for the \'data\' argument')
        if not data.index.is_all_dates:
            raise TypeError('index of data must be a sequence of date-time objects')

        logger.info('ensuring that data is sorted and has no invalid entries')
        self.data = data.copy(deep=True)
        if not data.index.is_monotonic_increasing:
            logger.warning('data index is not sorted - sorting data by index')
            self.data = self.data.sort_index(ascending=True)
        if np.any(pd.isna(data)):
            logger.warning('nan values found in data - removing invalid entries')
            self.data = self.data.dropna()

        # Attributes related to extremes
        self.extremes = None
        self.extremes_method = None
        self.extremes_type = None
        self.extremes_kwargs = None
        self.extremes_transformer = None

        # Attributes related to model
        self.model = None

    def __repr__(self) -> str:
        # Repre parameters
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
            value_width = int((width - sep) / 2 - (len(text) + 1))
            return f'{text}:{value:>{value_width:d}}'

        def align_pair(text: tuple, value: tuple) -> str:
            lwidth = int((width - sep) / 2)
            rwidth = width - (lwidth + sep)
            ltext = f'{text[0]}:{value[0]:>{lwidth - len(text[0]) - 1:d}}'
            rtext = f'{text[1]}:{value[1]:>{rwidth - len(text[1]) - 1:d}}'
            return ''.join([ltext, ' ' * sep, rtext])

        # Summary header
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

        # Extremes section
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
                raise RuntimeError
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

        logger.info('preparing extremes transformer object')
        self.extremes_transformer = ExtremesTransformer(
            extremes=self.extremes,
            extremes_method=method,
            extremes_type=extremes_type
        )

        logger.info('removing any previously declared models')
        self.model = None

    def plot_extremes(
            self,
            figsize: tuple = (8, 5)
    ) -> tuple:
        """
        Plot time series of extreme events.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size in inches (default=(8, 5)).

        Returns
        -------
        figure : matplotlib.figure.Figure
            Figure object.
        axes : matplotlib.axes.Axes
            Axes object.
        """

        logger.info('plotting extremes')
        return plot_extremes(
            ts=self.data, extremes=self.extremes,
            extremes_method=self.extremes_method, extremes_type=self.extremes_type,
            block_size=self.extremes_kwargs.get('block_size', None), figsize=figsize
        )

    def fit_model(
            self,
            model: str,
            distribution: str,
            **kwargs
    ) -> None:
        logger.info('checking if distribution is valid for extremes type')
        if distribution in ['genextreme', 'gumbel_r']:
            if self.extremes_method != 'BM':
                raise ValueError(
                    f'{distribution} distribution is only applicable to extremes extracted using the BM model'
                )
        elif distribution in ['genpareto', 'expon']:
            if self.extremes_method != 'POT':
                raise ValueError(
                    f'{distribution} distribution is only applicable to extremes extracted using the POT model'
                )

        logger.info(f'fitting {model} model with {distribution} distribution')
        self.model = get_model(
            model=model,
            extremes=self.extremes_transformer.transformed_extremes,
            distribution=distribution,
            **kwargs
        )

    def get_return_value(
            self,
            return_period: typing.Union[float, typing.Iterable],
            return_period_size: typing.Union[str, pd.Timedelta] = '1Y',
            alpha: float = None,
            **kwargs
    ) -> tuple:

        logger.info('calculating rate of extreme events as number of events per return_period_size')
        if self.extremes_method == 'BM':
            extremes_rate = return_period_size / self.extremes_kwargs['block_size']
            logger.debug('calculated extremes_rate for BM method')
        elif self.extremes_method == 'POT':
            n_periods = (self.data.index[-1] - self.data.index[0]) / return_period_size
            extremes_rate = len(self.extremes) / n_periods
            logger.debug('calculated extremes_rate for POT method')
        else:
            raise RuntimeError

        logger.info('calculating exceedance probability')
        if hasattr(return_period, '__iter__') and not isinstance(return_period, str):
            logger.info('getting a list of exceedance probabilities')
            exceedance_probability = 1 / np.array(return_period) / extremes_rate
        elif isinstance(return_period, float):
            logger.info('getting a single exceedance probability')
            exceedance_probability = 1 / return_period / extremes_rate
        else:
            raise TypeError(
                f'invalid type in {type(return_period)} for the \'return_period\' argument'
            )

        logger.info('calculating return value using the model')
        return self.model.get_return_value(
            exceedance_probability=exceedance_probability,
            alpha=alpha,
            **kwargs
        )

    def get_summary(
            self,
            return_period: typing.Iterable,
            return_period_size: typing.Union[str, pd.Timedelta] = '1Y',
            alpha: float = 0.95,
            **kwargs
    ) -> pd.DataFrame:
        logger.info('calculating return values')
        rv = self.get_return_value(
            return_period=return_period,
            return_period_size=return_period_size,
            alpha=alpha,
            kwargs=kwargs
        )

        logger.info('preparing a list of return periods to be used as DataFrame index')
        rp = [rp if isinstance(rp, pd.Timedelta) else pd.to_timedelta(rp) for rp in return_period]

        logger.info('preparing the summary dataframe')
        return pd.DataFrame(
            data=np.transpose(rv),
            index=pd.Index(data=rp, name='return period'),
            columns=['return value', 'lower ci', 'upper ci']
        )
