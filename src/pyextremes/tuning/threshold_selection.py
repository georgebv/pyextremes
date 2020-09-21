import logging
import typing

import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from pyextremes import EVA
from pyextremes.extremes import get_extremes, ExtremesTransformer
from pyextremes.plotting import pyextremes_rc

logger = logging.getLogger(__name__)


def get_default_thresholds(ts: pd.Series, extremes_type: str, num: int) -> np.ndarray:
    """
    Get an array of threshold values for given time series.
    This array is used to find optimal threshold value in other methods.
    Thresholds are generated as an array of equally spaced values between 90th percentile
    and 10th largest value in the series.

    Parameters
    ----------
    ts : pandas.Series
        Time series of the signal.
    extremes_type : str
        high - extreme high values
        low - extreme low values
    num : int
        Number of thresholds to generate.

    Returns
    -------
    thresholds : numpy.ndarray
        Array with threshold values.
    """

    if extremes_type == "high":
        start = np.quantile(ts.values, 0.9)
        stop = ts.sort_values(ascending=False).iloc[9]
    elif extremes_type == "low":
        start = np.quantile(ts.values, 0.1)
        stop = ts.sort_values(ascending=True).iloc[9]
    else:
        raise ValueError(
            f"'{extremes_type}' is not a valid value of the 'extremes_type' argument"
        )

    return np.linspace(start=start, stop=stop, num=num)


def plot_mean_residual_life(
    ts: pd.Series,
    thresholds: typing.Union[list, np.ndarray] = None,
    extremes_type: str = "high",
    alpha: float = 0.95,
    figsize: tuple = (8, 5),
) -> tuple:
    """
    Make a mean residual life plot for given threshold values.
    The mean residual life plot should be approximately linear above a threshold for which
    the GPD model is valid.
    The strategy is to select the smallest (largest for extremes_type='low') threshold value
    immediately above which the plot is approximately linear.

    Parameters
    ----------
    ts : pandas.Series
        Time series of the signal.
    thresholds : array-like, optional
        An array of thresholds for which the mean residual life plot is made (default=None).
        If None, plots mean residual life for 100 equally-spaced thresholds between 90th percentile
        and 10th largest value in the series.
    extremes_type : str, optional
        high (default) - extreme high values
        low - extreme low values
    alpha : float, optional
        Confidence interval withd (default=0.95).
        If None, then it is not drawn.
    figsize : tuple, optional
        Figure size in inches (default=(8, 5)).

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure object.
    axes : matplotlib.axes.Axes
        Axes object.
    """

    if thresholds is None:
        logger.info("calculating default threshold values")
        thresholds = get_default_thresholds(ts=ts, extremes_type=extremes_type, num=100)

    logger.info("calculating mean residual life for each threshold")
    mean_residual_lives, mrl_confidence = [], []
    for threshold in thresholds:
        if extremes_type == "high":
            exceedances = ts.loc[ts > threshold] - threshold
        elif extremes_type == "low":
            exceedances = ts.loc[ts < threshold] - threshold
        else:
            raise ValueError(
                f"'{extremes_type}' is not a valid value of the 'extremes_type' argument"
            )
        mean_residual_lives.append(exceedances.mean())
        if alpha is None:
            pass
        else:
            mrl_confidence.append(
                scipy.stats.norm.interval(
                    alpha=alpha,
                    loc=exceedances.mean(),
                    scale=exceedances.std(ddof=1) / np.sqrt(len(exceedances)),
                )
            )

    with plt.rc_context(rc=pyextremes_rc):
        logger.info("creating figure and axes")
        fig, ax = plt.subplots(figsize=figsize, dpi=96)

        logger.info("plotting mean residual life")
        ax.plot(
            thresholds, mean_residual_lives, color="#F85C50", lw=2, ls="-", zorder=20
        )

        logger.info("plotting confidence intervals")
        for ci in np.transpose(mrl_confidence):
            ax.plot(thresholds, ci, color="#5199FF", lw=1, ls="--", zorder=15)
        ax.fill_between(
            thresholds,
            *np.transpose(mrl_confidence),
            facecolor="#5199FF",
            edgecolor="None",
            alpha=0.25,
            zorder=10,
        )

        logger.info("configuring axes")
        ax.grid(False)
        if extremes_type == "low":
            ax.set_xlim(ax.get_xlim()[::-1])

        logger.info("labeling axes")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Mean excess")

        return fig, ax


def plot_parameter_stability(
    ts: pd.Series,
    thresholds: typing.Union[list, np.ndarray] = None,
    r: typing.Union[str, pd.Timedelta] = "24H",
    extremes_type: str = "high",
    figsize: tuple = (8, 5),
) -> tuple:
    """
    Make a parameter stability plot for given threshold values.
    The parameter stability plot shows shape and modified scale parameters of the GPD distribution.
    Both shape and modified scale parameters should be approximately constant above a threshold
    for which the GPD model is valid.
    The strategy is to select the smallest (largest for extremes_type='low') threshold value
    immediately above which the GPD parameters are approximately constant.

    Parameters
    ----------
    ts : pandas.Series
        Time series of the signal.
    thresholds : array-like, optional
        An array of thresholds for which the mean residual life plot is made (default=None).
        If None, plots mean residual life for 100 equally-spaced thresholds between 90th percentile
        and 10th largest value in the series.
    r : str or pandas.Timedelta, optional
        Duration of window used to decluster the exceedances (default='24H').
    extremes_type : str, optional
        high (default) - extreme high values
        low - extreme low values
    figsize : tuple, optional
        Figure size in inches (default=(8, 5)).

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure object.
    axes : matplotlib.axes.Axes
        Axes object.
    """

    if thresholds is None:
        logger.info("calculating default threshold values")
        thresholds = get_default_thresholds(ts=ts, extremes_type=extremes_type, num=100)

    logger.info("calculating shape and modified scale parameters for each threshold")
    shape_parameters, scale_parameters = [], []
    for threshold in thresholds:
        extremes = get_extremes(
            ts=ts, method="POT", extremes_type=extremes_type, threshold=threshold, r=r
        )
        extremes_transformer = ExtremesTransformer(
            extremes=extremes, extremes_type=extremes_type
        )
        c, loc, scale = scipy.stats.genpareto.fit(
            data=extremes_transformer.transformed_extremes, floc=threshold
        )
        shape_parameters.append(c)
        scale_parameters.append(scale - c * threshold)

    with plt.rc_context(rc=pyextremes_rc):
        logger.info("creating figure")
        fig = plt.figure(figsize=figsize, dpi=96)

        logger.info("creating gridspec")
        gs = matplotlib.gridspec.GridSpec(
            nrows=2,
            ncols=1,
            wspace=0.1,
            hspace=0.1,
            width_ratios=[1],
            height_ratios=[1, 1],
        )

        logger.info("creating and configuring axes")
        ax_shape = fig.add_subplot(gs[0, 0])
        ax_scale = fig.add_subplot(gs[1, 0])

        logger.info("plotting shape and modified scale parameters")
        ax_shape.plot(
            thresholds, shape_parameters, ls="-", color="#F85C50", lw=2, zorder=5
        )
        ax_scale.plot(
            thresholds, scale_parameters, ls="-", color="#F85C50", lw=2, zorder=5
        )

        logger.info("configuring axes")
        ax_shape.tick_params(axis="x", which="both", labelbottom=False, length=0)
        if extremes_type == "low":
            ax_shape.set_xlim(ax_shape.get_xlim()[::-1])
        ax_scale.set_xlim(ax_shape.get_xlim())

        logger.info("labeling axes")
        ax_shape.set_ylabel(r"Shape, $\xi$")
        ax_scale.set_ylabel(r"Modified scale, $\sigma^*$")
        ax_scale.set_xlabel("Threshold")

        return fig, (ax_shape, ax_scale)


def plot_return_value_stability(
    ts: pd.Series,
    return_period: typing.Union[int, float],
    return_period_size: typing.Union[str, pd.Timedelta] = "1Y",
    thresholds: typing.Union[list, np.ndarray] = None,
    r: typing.Union[str, pd.Timedelta] = "24H",
    extremes_type: str = "high",
    figsize: tuple = (8, 5),
) -> tuple:
    """
    Make a return value stability plot for given threshold values.
    The return value stability plot shows return value of given probability (return period) for a given
    array of thresholds.
    The purpose of this plot is to investigate statibility and sensitivity of GPD model to threshold value.
    Threshold value selection should still be guided by the mean residual life plot and the parameter
    stability plot.

    Parameters
    ----------
    ts : pandas.Series
        Time series of the signal.
    return_period
    return_period_size
    thresholds : array-like, optional
        An array of thresholds for which the mean residual life plot is made (default=None).
        If None, plots mean residual life for 100 equally-spaced thresholds between 90th percentile
        and 10th largest value in the series.
    r : str or pandas.Timedelta, optional
        Duration of window used to decluster the exceedances (default='24H').
    extremes_type : str, optional
        high (default) - extreme high values
        low - extreme low values
    figsize : tuple, optional
        Figure size in inches (default=(8, 5)).

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure object.
    axes : matplotlib.axes.Axes
        Axes object.
    """

    if thresholds is None:
        logger.info("calculating default threshold values")
        thresholds = get_default_thresholds(ts=ts, extremes_type=extremes_type, num=100)

    logger.info("calculating return values for each threshold")
    return_values = []
    model = EVA(data=ts)
    for threshold in thresholds:
        model.get_extremes(
            method="POT", extremes_type=extremes_type, threshold=threshold, r=r
        )
        model.fit_model()
        return_values.append(
            model.get_return_value(
                return_period=return_period,
                return_period_size=return_period_size,
                alpha=None,
            )[0]
        )

    with plt.rc_context(rc=pyextremes_rc):
        logger.info("creating figure and axes")
        fig, ax = plt.subplots(figsize=figsize, dpi=96)

        logger.info("plotting return values")
        ax.plot(thresholds, return_values, color="#1771F1", lw=2, ls="-", zorder=20)

        logger.info("configuring axes")
        ax.grid(False)
        if extremes_type == "low":
            ax.set_xlim(ax.get_xlim()[::-1])

        logger.info("labeling axes")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Return value")

        return fig, ax
