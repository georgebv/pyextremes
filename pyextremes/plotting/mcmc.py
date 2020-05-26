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

import matplotlib.gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from pyextremes.plotting.style import pyextremes_rc

logger = logging.getLogger(__name__)


def plot_trace(
        trace: np.ndarray,
        trace_map: tuple = None,
        labels: tuple = None,
        figsize: tuple = None
) -> tuple:
    """
    Plot a trace plot for a given MCMC sampler trace.

    Parameters
    ----------
    trace : numpy.ndarray
        Array with MCMC sampler trace.
        Has a shape of (n_walkers, n_samples, n_parameters).
    trace_map : tuple, optional
        Tuple with maximum aposteriori estimate of distribution parameters.
        If provided, MAP values are plotted as orange lines on top of the trace (default=None).
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

    logger.info('calculating figure size')
    n_parameters = trace.shape[2]
    if figsize is None:
        figsize = (8, 2 * n_parameters)

    with plt.rc_context(rc=pyextremes_rc):
        logger.info('creating figure')
        fig = plt.figure(figsize=figsize, dpi=96)

        logger.info('creating gridspec')
        gs = matplotlib.gridspec.GridSpec(
            nrows=n_parameters,
            ncols=1,
            wspace=0.1,
            hspace=0.1,
            width_ratios=[1],
            height_ratios=np.full(shape=n_parameters, fill_value=1)
        )

        logger.info('creating and configuring axes')
        axes = [fig.add_subplot(gs[i, 0]) for i in range(n_parameters)]
        for i, ax in enumerate(axes):
            ax.grid(False)
            if labels is None:
                ax.set_ylabel(f'Parameter {i}')
            else:
                ax.set_ylabel(labels[i])
            if i < n_parameters - 1:
                ax.tick_params(axis='x', which='both', labelbottom=False)

        logger.info('plot the trace plots')
        for i, ax in enumerate(axes):
            ax.plot(
                trace[:, :, i].transpose(),
                color='#231F20', lw=0.1, zorder=5
            )
            if trace_map is not None:
                ax.axhline(
                    trace_map[i],
                    color='#F85C50', lw=2, ls='--', zorder=10
                )

        return fig, axes


def plot_corner(
        trace: np.ndarray,
        trace_map: tuple,
        labels: tuple = None,
        burn_in: int = 0,
        figsize: tuple = (8, 8)
) -> tuple:
    """
    Plot a corner plot for a given MCMC sampler trace.

    Parameters
    ----------
    trace : numpy.ndarray
        Array with MCMC sampler trace.
        Has a shape of (n_walkers, n_samples, n_parameters).
    trace_map : tuple, optional
        Tuple with maximum aposteriori estimate of distribution parameters.
        If provided, MAP values are plotted as orange lines on top of the trace (default=None).
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

    if not isinstance(burn_in, int):
        raise TypeError(f'invalid type in {type(burn_in)} for the \'burn_in\' argument')
    if burn_in < 0:
        raise ValueError(f'\'{burn_in}\' is not a valid \'burn_in\' value, it must be a positive integer')
    if burn_in >= trace.shape[1]:
        raise ValueError(
            f'\'burn_in\' value \'{burn_in}\' exceeds number of samples {trace.shape[1]}'
        )

    n_parameters = trace.shape[2]

    with plt.rc_context(rc=pyextremes_rc):
        logger.info('creating figure')
        fig = plt.figure(figsize=figsize, dpi=96)

        logger.info('creating gridspec')
        gs = matplotlib.gridspec.GridSpec(
            nrows=n_parameters,
            ncols=n_parameters,
            wspace=0.1,
            hspace=0.1,
            width_ratios=np.full(shape=n_parameters, fill_value=1),
            height_ratios=np.full(shape=n_parameters, fill_value=1)
        )

        logger.info('creating and configuring axes')
        axes = [[None] * n_parameters] * n_parameters
        for i in range(n_parameters):
            for j in range(n_parameters):
                if i >= j:
                    logger.info(f'creating axis [{i};{j}]')
                    ax = fig.add_subplot(gs[i, j])
                    ax.grid(False)
                    axes[i][j] = ax

                    if i != n_parameters - 1:
                        logger.info('removing x ticks')
                        ax.tick_params(axis='x', which='both', labelbottom=False, length=0)

                    if j != 0 or i == j == 0:
                        logger.info('removing y ticks')
                        ax.tick_params(axis='y', which='both', labelleft=False, length=0)

                    if i == n_parameters - 1:
                        logger.info(f'setting x label for parameter {j}')
                        if labels is None:
                            ax.set_xlabel(f'Parameter {j}')
                        else:
                            ax.set_xlabel(labels[j])

                    if j == 0:
                        logger.info(f'setting y label for parameter {i}')
                        if labels is None:
                            ax.set_ylabel(f'Parameter {i}')
                        else:
                            ax.set_ylabel(labels[i])

                    if i == j:
                        logger.info(f'plotting histogram for parameter {i}')
                        parameter_samples = trace[burn_in:, :, i].flatten()
                        ax.hist(
                            parameter_samples,
                            bins=np.histogram_bin_edges(a=parameter_samples, bins='auto'),
                            density=True, rwidth=0.8, histtype='step',
                            edgecolor='#231F20', lw=0.5, zorder=5
                        )
                        if trace_map is not None:
                            ax.axvline(
                                trace_map[i],
                                color='#F85C50', lw=1, ls='--', zorder=10
                            )

                    if i != j:
                        logger.info(f'plotting KDE distribution for parameters {i} and {j}')
                        parameter_i = trace[burn_in:, :, i].flatten()
                        parameter_j = trace[burn_in:, :, j].flatten()
                        ax.scatter(
                            parameter_j,
                            parameter_i,
                            marker='o', s=2, alpha=0.1, facecolor='#231F20', edgecolor='None', lw=0, zorder=5
                        )
                        if trace_map is not None:
                            ax.axvline(
                                trace_map[j],
                                color='#F85C50', lw=1, ls='--', zorder=10
                            )
                            ax.axhline(
                                trace_map[i],
                                color='#F85C50', lw=1, ls='--', zorder=10
                            )

        logger.info('setting axes limits')
        for i in range(n_parameters):
            for j in range(n_parameters):
                if i >= j:
                    parameter_i = trace[burn_in:, :, i].flatten()
                    parameter_j = trace[burn_in:, :, j].flatten()
                    ylim = (parameter_i.min(), parameter_i.max())
                    xlim = (parameter_j.min(), parameter_j.max())
                    axes[i][j].set_xlim(xlim)
                    if i != j:
                        axes[i][j].set_ylim(ylim)

        return fig, axes
