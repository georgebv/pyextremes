import itertools
import multiprocessing
import os
import typing

import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from pyextremes import EVA
from pyextremes.extremes import ExtremesTransformer, get_extremes
from pyextremes.plotting import pyextremes_rc


def get_default_thresholds(
    ts,
    extremes_type: str,
    num: int = 100,
) -> np.ndarray:  # pragma: no cover
    """
    Get an array of threshold values for given time series.

    Used to generate an array of thresholds used to find
    optimal threshold values in other methods.
    Thresholds are generated as an array of equally spaced values
    between 90th percentile and 10th largest value in the series for 'extremes_type'
    being 'high' and between 10th smallest value and 10th percentile in the series
    for 'extremes_type' being 'low'.

    Parameters
    ----------
    ts : array-like
        Time series of the signal.
    extremes_type : str
        high - get extreme high values
        low - get extreme low values
    num : int
        Number of threshold values to generate.

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
            f"invalid value in '{extremes_type}' for the 'extremes_type' argument"
        )

    return np.linspace(start=start, stop=stop, num=num)


def plot_mean_residual_life(
    ts: pd.Series,
    thresholds=None,
    extremes_type: str = "high",
    alpha: float = 0.95,
    figsize: tuple = (8, 5),
) -> tuple:  # pragma: no cover
    """
    Plot mean residual life for given threshold values.

    The mean residual life plot should be approximately linear above a threshold
    for which the Generalized Pareto Distribution model is valid.
    The strategy is to select the smallest (largest for extremes_type='low')
    threshold value immediately above (below for extremes_type='low')
    which the plot is approximately linear.

    Parameters
    ----------
    ts : pandas.Series
        Time series of the signal.
    thresholds : array-like, optional
        An array of thresholds for which the mean residual life plot is plotted.
        If None (default), plots mean residual life for 100 equally-spaced thresholds
        between 90th (10th if extremes_type='low') percentile
        and 10th largest (smallest if extremes_type='low') value in the series.
    extremes_type : str, optional
        high (default) - extreme high values
        low - extreme low values
    alpha : float, optional
        Confidence interval width in the range (0, 1), by default it is 0.95.
        If None, then confidence interval is not shown.
    figsize : tuple, optional
        Figure size in inches in format (width, height).
        By default it is (8, 5).

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure object.
    axes : matplotlib.axes._axes.Axes
        Axes object.

    """
    # Get default thresholds
    if thresholds is None:
        thresholds = get_default_thresholds(
            ts=ts,
            extremes_type=extremes_type,
            num=100,
        )

    # Calculate mean residual life for each threshold
    mean_residual_lives, mrl_confidence = [], []
    for threshold in thresholds:
        if extremes_type == "high":
            exceedances = ts.loc[ts > threshold] - threshold
        elif extremes_type == "low":
            exceedances = ts.loc[ts < threshold] - threshold
        else:
            raise ValueError(
                f"invalid value in '{extremes_type}' for the 'extremes_type' argument"
            )

        mean_residual_lives.append(exceedances.mean())
        if alpha is not None:
            mrl_confidence.append(
                scipy.stats.norm.interval(
                    alpha=alpha,
                    loc=exceedances.mean(),
                    scale=exceedances.std(ddof=1) / np.sqrt(len(exceedances)),
                )
            )

    with plt.rc_context(rc=pyextremes_rc):
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize, dpi=96)
        ax.grid(False)

        # Plotting central estimates of mean residual life
        ax.plot(
            thresholds,
            mean_residual_lives,
            color="#F85C50",
            lw=2,
            ls="-",
            zorder=15,
        )

        # Plot confidence intervals
        if alpha is not None:
            for ci in np.transpose(mrl_confidence):
                ax.plot(thresholds, ci, color="#5199FF", lw=1, ls="--", zorder=10)
            ax.fill_between(
                thresholds,
                *np.transpose(mrl_confidence),
                facecolor="#5199FF",
                edgecolor="None",
                alpha=0.25,
                zorder=5,
            )

        # Label axes
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Mean excess")

        return fig, ax


def get_fit_parameters(params) -> typing.List[tuple]:  # pragma: no cover
    n, fit_function, extremes, fixed_parameters, seed = params
    size = len(extremes)
    rng_generator = np.random.default_rng(seed=seed)
    sampler = rng_generator.choice
    return [
        fit_function(
            data=sampler(a=extremes.values, size=size, replace=True),
            **fixed_parameters,
        )
        for _ in range(n)
    ]


def plot_parameter_stability(
    ts: pd.Series,
    thresholds=None,
    r: typing.Union[str, pd.Timedelta] = "24H",
    extremes_type: str = "high",
    alpha: typing.Optional[float] = None,
    n_samples: int = 100,
    figsize: tuple = (8, 5),
) -> tuple:  # pragma: no cover
    """
    Plot parameter stability plot for given threshold values.

    The parameter stability plot shows shape and modified scale parameters
    of the Generalized Pareto Distribution (GPD).
    Both shape and modified scale parameters should be approximately constant above
    a threshold for which the GPD model is valid.
    The strategy is to select the smallest (largest for extremes_type='low')
    threshold value immediately above (below for extremes_type='low')
    which the GPD parameters are approximately constant.

    Parameters
    ----------
    ts : pandas.Series
        Time series of the signal.
    thresholds : array-like, optional
        An array of thresholds for which the mean residual life plot is plotted.
        If None (default), plots mean residual life for 100 equally-spaced thresholds
        between 90th (10th if extremes_type='low') percentile
        and 10th largest (smallest if extremes_type='low') value in the series.
    r : str or pandas.Timedelta, optional
        Duration of window used to decluster the exceedances.
        By default r='24H' (24 hours).
    extremes_type : str, optional
        high (default) - extreme high values
        low - extreme low values
    alpha : float, optional
        Confidence interval width in the range (0, 1).
        If None (default), then confidence interval is not shown.
    n_samples : int, optional
        Number of bootstrap samples used to estimate
        confidence interval bounds (default=100).
        Ignored if `alpha` is None.
    figsize : tuple, optional
        Figure size in inches in format (width, height).
        By default it is (8, 5).

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure object.
    axes : matplotlib.axes._axes.Axes
        Axes object.

    """
    # Get default thresholds
    if thresholds is None:
        thresholds = get_default_thresholds(
            ts=ts,
            extremes_type=extremes_type,
            num=100,
        )

    # Calculate shape and modified scale parameters for each threshold
    shape_parameters: typing.Dict[str, typing.List[float]] = {
        "values": [],
        "ci_lower": [],
        "ci_upper": [],
    }
    scale_parameters: typing.Dict[str, typing.List[float]] = {
        "values": [],
        "ci_lower": [],
        "ci_upper": [],
    }
    distribution = scipy.stats.genpareto
    for threshold in thresholds:
        # Get extremes
        extremes = get_extremes(
            ts=ts,
            method="POT",
            extremes_type=extremes_type,
            threshold=threshold,
            r=r,
        )
        extremes_transformer = ExtremesTransformer(
            extremes=extremes,
            extremes_type=extremes_type,
        )

        # Get central estimates for shape and scale parameters
        c, _, scale = distribution.fit(
            data=extremes_transformer.transformed_extremes,
            floc=threshold,
        )
        shape_parameters["values"].append(c)
        scale_parameters["values"].append(scale - c * threshold)

        # Get confidence bounds
        if alpha is not None:
            # Prepare local variables used by fit parameter calculator
            fit_function = distribution.fit
            fixed_parameters = {"floc": threshold}

            min_samples_per_core = 50
            if n_samples <= min_samples_per_core:
                # Calculate without multiprocessing
                seed = np.random.randint(low=0, high=1e6, size=None)
                fit_parameters = get_fit_parameters(
                    params=(
                        n_samples,
                        fit_function,
                        extremes,
                        fixed_parameters,
                        seed,
                    )
                )
            else:
                # Find number of cores
                n_cores = min(
                    os.cpu_count() or 2,
                    int(np.ceil(n_samples / min_samples_per_core)),
                )

                # Calculate number of samples per core
                min_samples_per_core = int(n_samples / n_cores)
                core_samples = [min_samples_per_core for _ in range(n_cores)]

                # Distribute remaining samples evenly across cores
                for i in range(n_samples - sum(core_samples)):
                    core_samples[i] += 1

                # Get unique random seed for each core
                seeds: typing.List[int] = []
                while len(seeds) < n_cores:
                    seed = np.random.randint(low=0, high=1e6, size=None)
                    if seed not in seeds:
                        seeds.append(seed)

                # Calculate new fit parameters using processor pool
                with multiprocessing.Pool(processes=n_cores) as pool:
                    fit_parameters = list(
                        itertools.chain(
                            *pool.map(
                                get_fit_parameters,
                                zip(
                                    core_samples,
                                    [fit_function for _ in range(n_cores)],
                                    [extremes for _ in range(n_cores)],
                                    [fixed_parameters for _ in range(n_cores)],
                                    seeds,
                                ),
                            )
                        )
                    )

            # Calculate confidence bounds
            shapes = np.transpose(fit_parameters)[0]
            scales = np.transpose(fit_parameters)[0] - shapes * threshold
            cil, ciu = np.quantile(
                a=shapes,
                q=[(1 - alpha) / 2, (1 + alpha) / 2],
            )
            shape_parameters["ci_lower"].append(cil)
            shape_parameters["ci_upper"].append(ciu)
            cil, ciu = np.quantile(
                a=scales,
                q=[(1 - alpha) / 2, (1 + alpha) / 2],
            )
            scale_parameters["ci_lower"].append(cil)
            scale_parameters["ci_upper"].append(ciu)

    with plt.rc_context(rc=pyextremes_rc):
        # Create figure
        fig = plt.figure(figsize=figsize, dpi=96)

        # Create gridspec
        gs = matplotlib.gridspec.GridSpec(
            nrows=2,
            ncols=1,
            wspace=0.1,
            hspace=0.1,
            width_ratios=[1],
            height_ratios=[1, 1],
        )

        # Create and configure axes
        ax_shape = fig.add_subplot(gs[0, 0])
        ax_scale = fig.add_subplot(gs[1, 0])

        # Plot central estimates of shape and modified scale parameters
        ax_shape.plot(
            thresholds,
            shape_parameters,
            ls="-",
            color="#F85C50",
            lw=2,
            zorder=15,
        )
        ax_scale.plot(
            thresholds,
            scale_parameters,
            ls="-",
            color="#F85C50",
            lw=2,
            zorder=15,
        )

        # Plot confidence bounds
        if alpha is not None:
            for ci in [shape_parameters["ci_lower"], shape_parameters["ci_upper"]]:
                ax_shape.plot(
                    thresholds,
                    ci,
                    color="#5199FF",
                    lw=1,
                    ls="--",
                    zorder=10,
                )
            ax_shape.fill_between(
                thresholds,
                shape_parameters["ci_lower"],
                shape_parameters["ci_upper"],
                facecolor="#5199FF",
                edgecolor="None",
                alpha=0.25,
                zorder=5,
            )
            for ci in [scale_parameters["ci_lower"], scale_parameters["ci_upper"]]:
                ax_scale.plot(
                    thresholds,
                    ci,
                    color="#5199FF",
                    lw=1,
                    ls="--",
                    zorder=10,
                )
            ax_scale.fill_between(
                thresholds,
                scale_parameters["ci_lower"],
                scale_parameters["ci_upper"],
                facecolor="#5199FF",
                edgecolor="None",
                alpha=0.25,
                zorder=5,
            )

        # Configure axes
        ax_shape.tick_params(axis="x", which="both", labelbottom=False, length=0)
        ax_scale.set_xlim(ax_shape.get_xlim())

        # Label axes
        ax_shape.set_ylabel(r"Shape, $\xi$")
        ax_scale.set_ylabel(r"Modified scale, $\sigma^*$")
        ax_scale.set_xlabel("Threshold")

        return fig, (ax_shape, ax_scale)


def plot_return_value_stability(
    ts: pd.Series,
    return_period,
    return_period_size: typing.Union[str, pd.Timedelta] = "1Y",
    thresholds=None,
    r: typing.Union[str, pd.Timedelta] = "24H",
    extremes_type: str = "high",
    alpha: typing.Optional[float] = None,
    n_samples: int = 100,
    figsize: tuple = (8, 5),
) -> tuple:  # pragma: no cover
    """
    Plot return value stability plot for given threshold values.

    The return value stability plot shows return values for given return period
    for given thresholds.
    The purpose of this plot is to investigate statibility and sensitivity of the
    Generalized Pareto Distribution model to threshold value.
    Threshold value selection should still be guided by the mean residual life plot
    and the parameter stability plot. This plot should be used as additional check.

    Parameters
    ----------
    ts : pandas.Series
        Time series of the signal.
    return_period : number
        Return period.
        Given as a multiple of `return_period_size`.
    return_period_size : str or pandas.Timedelta, optional
        Size of return period (default='1Y').
        If set to '30D', then a return period of 12
        would be roughly equivalent to a 1 year return period (360 days).
    thresholds : array-like, optional
        An array of thresholds for which the mean residual life plot is plotted.
        If None (default), plots mean residual life for 100 equally-spaced thresholds
        between 90th (10th if extremes_type='low') percentile
        and 10th largest (smallest if extremes_type='low') value in the series.
    r : str or pandas.Timedelta, optional
        Duration of window used to decluster the exceedances.
        By default r='24H' (24 hours).
    extremes_type : str, optional
        high (default) - extreme high values
        low - extreme low values
    alpha : float, optional
        Confidence interval width in the range (0, 1).
        If None (default), then confidence interval is not shown.
    n_samples : int, optional
        Number of bootstrap samples used to estimate
        confidence interval bounds (default=100).
        Ignored if `alpha` is None.
    figsize : tuple, optional
        Figure size in inches in format (width, height).
        By default it is (8, 5).

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure object.
    axes : matplotlib.axes._axes.Axes
        Axes object.

    """
    # Get default thresholds
    if thresholds is None:
        thresholds = get_default_thresholds(
            ts=ts,
            extremes_type=extremes_type,
            num=100,
        )

    # Instantiate model
    model = EVA(data=ts)

    # Calculate return values for each threshold
    return_values = []
    ci_lower = []
    ci_upper = []
    for threshold in thresholds:
        model.get_extremes(
            method="POT",
            extremes_type=extremes_type,
            threshold=threshold,
            r=r,
        )
        model.fit_model(
            model="MLE",
            distribution="genpareto",
            distribution_kwargs={"floc": threshold},
        )
        rv, cil, ciu = model.get_return_value(
            return_period=return_period,
            return_period_size=return_period_size,
            alpha=None,
            n_samples=n_samples,
        )
        return_values.append(rv)
        ci_lower.append(cil)
        ci_upper.append(ciu)

    with plt.rc_context(rc=pyextremes_rc):
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize, dpi=96)
        ax.grid(False)

        # Plot central estimate of return values
        ax.plot(
            thresholds,
            return_values,
            color="#1771F1",
            lw=2,
            ls="-",
            zorder=15,
        )

        # Plot confidence bounds
        if alpha is not None:
            for ci in [ci_lower, ci_upper]:
                ax.plot(
                    thresholds,
                    ci,
                    color="#5199FF",
                    lw=1,
                    ls="--",
                    zorder=10,
                )
            ax.fill_between(
                thresholds,
                ci_lower,
                ci_upper,
                facecolor="#5199FF",
                edgecolor="None",
                alpha=0.25,
                zorder=5,
            )

        # Label axes
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Return value")

        return fig, ax
