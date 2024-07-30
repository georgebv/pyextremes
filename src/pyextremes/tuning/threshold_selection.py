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
) -> np.ndarray:
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
    num : int, optional
        Number of threshold values to generate.
        By default is 100.

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
    ax: typing.Optional[plt.Axes] = None,
    figsize: tuple = (8, 5),
) -> plt.Axes:
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
        between 90th (10th if extremes_type='high') percentile
        and 10th largest (smallest if extremes_type='low') value in the series.
    extremes_type : str, optional
        high (default) - extreme high values
        low - extreme low values
    alpha : float, optional
        Confidence interval width in the range (0, 1), by default it is 0.95.
        If None, then confidence interval is not shown.
    ax : matplotlib.axes._axes.Axes, optional
        If provided, then the plot is drawn on this axes.
        If None (default), new figure and axes are created
    figsize : tuple, optional
        Figure size in inches in format (width, height).
        By default it is (8, 5).

    Returns
    -------
    matplotlib.axes._axes.Axes
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
                    alpha,
                    loc=exceedances.mean(),
                    scale=exceedances.std(ddof=1) / np.sqrt(len(exceedances)),
                )
            )

    with plt.rc_context(rc=pyextremes_rc):
        if ax is None:
            _, ax = plt.subplots(figsize=figsize, dpi=96)
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

        return ax


def _calculate_modified_parameters(
    args: typing.Tuple[
        pd.Series,  # ts (time series)
        str,  # extremes_type
        float,  # threshold
        typing.Union[str, pd.Timedelta],  # r
        typing.Optional[float],  # alpha
        int,  # n_samples
        int,  # seed
    ],
) -> typing.Dict[str, typing.Optional[float]]:
    (
        ts,
        extremes_type,
        threshold,
        r,
        alpha,
        n_samples,
        seed,
    ) = args

    result: typing.Dict[str, typing.Optional[float]] = {"threshold": threshold}

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
    c, _, scale = scipy.stats.genpareto.fit(
        data=extremes_transformer.transformed_extremes,
        floc=threshold,
    )
    result["shape"] = c
    result["scale"] = scale - c * threshold

    # Get confidence bounds
    if alpha is None:
        result["shape_ci_lower"] = None
        result["shape_ci_upper"] = None
        result["scale_ci_lower"] = None
        result["scale_ci_upper"] = None
    if alpha is not None:
        # Get fit parameters
        rng_generator = np.random.default_rng(seed=seed)
        fit_parameters = [
            scipy.stats.genpareto.fit(
                data=rng_generator.choice(
                    a=extremes.values,
                    size=len(extremes),
                    replace=True,
                ),
                floc=threshold,
            )
            for _ in range(n_samples)
        ]

        # Calculate confidence bounds for shape and scale parameters
        result["shape_ci_lower"], result["shape_ci_upper"] = np.quantile(
            a=np.transpose(fit_parameters)[0],
            q=[(1 - alpha) / 2, (1 + alpha) / 2],
        )
        result["scale_ci_lower"], result["scale_ci_upper"] = np.quantile(
            a=np.transpose(fit_parameters)[2]
            - np.transpose(fit_parameters)[0] * threshold,
            q=[(1 - alpha) / 2, (1 + alpha) / 2],
        )

    return result


def plot_parameter_stability(
    ts: pd.Series,
    thresholds=None,
    r: typing.Union[str, pd.Timedelta] = "24h",
    extremes_type: str = "high",
    alpha: typing.Optional[float] = None,
    n_samples: int = 100,
    axes: typing.Optional[typing.Tuple[plt.Axes, plt.Axes]] = None,
    figsize: tuple = (8, 5),
    progress: bool = False,
) -> typing.Tuple[plt.Axes, plt.Axes]:
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
        between 90th (10th if extremes_type='high') percentile
        and 10th largest (smallest if extremes_type='low') value in the series.
    r : pandas.Timedelta or value convertible to timedelta, optional
        Duration of window used to decluster the exceedances.
        By default r='24H' (24 hours).
        See pandas.to_timedelta for more information.
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
    axes : (ax_shape, ax_scale), optional
        Tuple with matplotlib Axes for shape and scale values.
        If None (default), new figure and axes are created.
    figsize : tuple, optional
        Figure size in inches in format (width, height).
        By default it is (8, 5).
    progress : bool, optional
        If True, shows tqdm progress bar.
        By default False.

    Returns
    -------
    ax_shape : matplotlib.axes._axes.Axes
        Axes with shape parameter values.
    ax_scale : matplotlib.axes._axes.Axes
        Axes with scale parameter values.

    """
    try:
        import tqdm  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        if progress:
            raise ImportError(
                "'tqdm' package is required to display a progress bar"
            ) from error

    # Get default thresholds
    if thresholds is None:
        thresholds = get_default_thresholds(
            ts=ts,
            extremes_type=extremes_type,
            num=100,
        )

    # List of unique seeds - ensures same seed is not reused across sub-processes
    seeds: typing.List[int] = []

    def _input_generator() -> (
        typing.Generator[
            typing.Tuple[
                pd.Series,  # ts (time series)
                str,  # extremes_type
                float,  # threshold
                typing.Union[str, pd.Timedelta],  # r
                typing.Optional[float],  # alpha
                int,  # n_samples
                int,  # seed
            ],
            None,
            None,
        ]
    ):
        for threshold in thresholds:
            seed = np.random.randint(low=0, high=1e6, size=None)
            while seed in seeds:
                seed = np.random.randint(low=0, high=1e6, size=None)
            seeds.append(seed)
            yield (ts, extremes_type, threshold, r, alpha, n_samples, seed)

    iterable = (
        tqdm.tqdm(
            _input_generator(),
            desc="calculating stability parameters",
            total=len(thresholds),
            smoothing=0,
        )
        if progress
        else _input_generator()
    )

    cpu_count = os.cpu_count() or 1
    if cpu_count > 1:
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            _results = list(pool.imap(_calculate_modified_parameters, iterable))
    else:
        _results = []
        for args in iterable:
            _results.append(_calculate_modified_parameters(args))
    results = (
        pd.DataFrame(data=_results).set_index("threshold").sort_index(ascending=True)
    )

    with plt.rc_context(rc=pyextremes_rc):
        if axes is None:
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
        else:
            fig = None
            ax_shape, ax_scale = axes

        # Plot central estimates of shape and modified scale parameters
        ax_shape.plot(
            results.index,
            results.loc[:, "shape"],
            ls="-",
            color="#F85C50",
            lw=2,
            zorder=15,
        )
        ax_scale.plot(
            results.index,
            results.loc[:, "scale"],
            ls="-",
            color="#F85C50",
            lw=2,
            zorder=15,
        )

        # Plot confidence bounds
        if alpha is not None:
            for ax, parameter in [(ax_shape, "shape"), (ax_scale, "scale")]:
                for ci in ["lower", "upper"]:
                    ax.plot(
                        results.index,
                        results.loc[:, f"{parameter}_ci_{ci}"],
                        color="#5199FF",
                        lw=1,
                        ls="--",
                        zorder=10,
                    )
                ax.fill_between(
                    results.index,
                    results.loc[:, f"{parameter}_ci_lower"],
                    results.loc[:, f"{parameter}_ci_upper"],
                    facecolor="#5199FF",
                    edgecolor="None",
                    alpha=0.25,
                    zorder=5,
                )

        if fig is not None:
            # Configure axes
            ax_shape.tick_params(axis="x", which="both", labelbottom=False, length=0)
            ax_scale.set_xlim(ax_shape.get_xlim())

        # Label axes
        ax_shape.set_ylabel(r"Shape, $\xi$")
        ax_scale.set_ylabel(r"Modified scale, $\sigma^*$")
        if fig is not None:
            ax_scale.set_xlabel("Threshold")

        return ax_shape, ax_scale


def _calculate_return_value(
    args: typing.Tuple[
        pd.Series,  # ts (time series)
        float,  # return_period
        typing.Union[str, pd.Timedelta],  # return_period_size
        float,  # threshold
        typing.Union[str, pd.Timedelta],  # r
        str,  # extremes_type
        typing.Union[str, scipy.stats.rv_continuous],  # distribution
        str,  # distribution_name
        typing.Optional[float],  # alpha
        int,  # n_samples
    ],
) -> typing.Dict[str, typing.Union[str, typing.Optional[float]]]:
    (
        ts,
        return_period,
        return_period_size,
        threshold,
        r,
        extremes_type,
        distribution,
        distribution_name,
        alpha,
        n_samples,
    ) = args
    model = EVA(data=ts)
    model.get_extremes(
        method="POT",
        extremes_type=extremes_type,
        threshold=threshold,
        r=r,
    )
    model.fit_model(
        model="MLE",
        distribution=distribution,
    )
    # TODO - this is a hack to avoid spawning nested subprocesses
    _n_samples = n_samples % 10
    while _n_samples < n_samples:
        _n_samples += 10
        model.get_return_value(
            return_period=return_period,
            return_period_size=return_period_size,
            alpha=alpha,
            n_samples=_n_samples,
        )
    rv, cil, ciu = model.get_return_value(
        return_period=return_period,
        return_period_size=return_period_size,
        alpha=alpha,
        n_samples=n_samples,
    )
    return {
        "distribution_name": distribution_name,
        "threshold": threshold,
        "rv": rv,
        "cil": cil,
        "ciu": ciu,
    }


def plot_return_value_stability(
    ts: pd.Series,
    return_period: float,
    return_period_size: typing.Union[str, pd.Timedelta] = "365.2425D",
    thresholds=None,
    r: typing.Union[str, pd.Timedelta] = "24h",
    extremes_type: str = "high",
    distributions: typing.Optional[
        typing.List[typing.Union[str, scipy.stats.rv_continuous]]
    ] = None,
    alpha: typing.Optional[float] = None,
    n_samples: int = 100,
    ax: typing.Optional[plt.Axes] = None,
    figsize: tuple = (8, 5),
    progress: bool = False,
) -> plt.Axes:
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
    return_period : float
        Return period.
        Given as a multiple of `return_period_size`.
    return_period_size : str or pandas.Timedelta, optional
        Size of return period (default='365.2425D').
        If set to '30D', then a return period of 12
        would be roughly equivalent to a 1 year return period (360 days).
    thresholds : array-like, optional
        An array of thresholds for which the return value plot is plotted.
        If None (default), plots return values for 100 equally-spaced thresholds
        between 90th (10th if extremes_type='high') percentile
        and 10th largest (smallest if extremes_type='low') value in the series.
    r : pandas.Timedelta or value convertible to timedelta, optional
        Duration of window used to decluster the exceedances.
        By default r='24H' (24 hours).
        See pandas.to_timedelta for more information.
    extremes_type : str, optional
        high (default) - extreme high values
        low - extreme low values
    distributions : list, optional
        List of distributions for which the return value curves are plotted.
        By default these are "genpareto" and "expon".
        A distribution must be either a name of distribution from scipy.stats
        or a subclass of scipy.stats.rv_continuous.
        See https://docs.scipy.org/doc/scipy/reference/stats.html
    alpha : float, optional
        Confidence interval width in the range (0, 1).
        If None (default), then confidence interval is not shown.
    n_samples : int, optional
        Number of bootstrap samples used to estimate
        confidence interval bounds (default=100).
        Ignored if `alpha` is None.
    ax : matplotlib.axes._axes.Axes, optional
        If provided, then the plot is drawn on this axes.
        If None (default), new figure and axes are created
    figsize : tuple, optional
        Figure size in inches in format (width, height).
        By default it is (8, 5).
    progress : bool, optional
        If True, shows tqdm progress bar.
        By default False.

    Returns
    -------
    matplotlib.axes._axes.Axes
        Axes object.

    """
    try:
        import tqdm  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        if progress:
            raise ImportError(
                "'tqdm' package is required to display a progress bar"
            ) from error

    # Get default `thresholds`
    if thresholds is None:
        thresholds = get_default_thresholds(
            ts=ts,
            extremes_type=extremes_type,
            num=100,
        )

    # Get default `distributions`
    if distributions is None:
        distributions = [
            "genpareto",
            "expon",
        ]
    distribution_names: typing.List[str] = []
    for distribution in distributions:
        if isinstance(distribution, str):
            distribution_names.append(distribution)
        else:
            distribution_names.append(distribution.name)

    def _input_generator() -> (
        typing.Generator[
            typing.Tuple[
                pd.Series,  # ts (time series)
                float,  # return_period
                typing.Union[str, pd.Timedelta],  # return_period_size
                float,  # threshold
                typing.Union[str, pd.Timedelta],  # r
                str,  # extremes_type
                typing.Union[str, scipy.stats.rv_continuous],  # distribution
                str,  # distribution_name
                typing.Optional[float],  # alpha
                int,  # n_samples
            ],
            None,
            None,
        ]
    ):
        for distribution, distribution_name in zip(distributions, distribution_names):
            for threshold in thresholds:
                yield (
                    ts,
                    return_period,
                    return_period_size,
                    threshold,
                    r,
                    extremes_type,
                    distribution,
                    distribution_name,
                    alpha,
                    n_samples,
                )

    iterable = (
        tqdm.tqdm(
            _input_generator(),
            desc="calculating return values",
            total=len(distributions) * len(thresholds),
            smoothing=0,
        )
        if progress
        else _input_generator()
    )

    cpu_count = os.cpu_count() or 1
    if cpu_count > 1:
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            _results = list(pool.imap(_calculate_return_value, iterable))
    else:
        _results = []
        for args in iterable:
            _results.append(_calculate_return_value(args))
    results = pd.DataFrame(data=_results).sort_values("threshold", ascending=True)

    with plt.rc_context(rc=pyextremes_rc):
        if ax is None:
            _, ax = plt.subplots(figsize=figsize, dpi=96)
            ax.grid(False)

        for i, (distribution_name, df) in enumerate(
            results.groupby("distribution_name")
        ):
            # Plot central estimate of return values
            color = pyextremes_rc["axes.prop_cycle"].by_key()["color"][i]
            ax.plot(
                df.loc[:, "threshold"],
                df.loc[:, "rv"],
                color=color,
                lw=2,
                ls="-",
                label=distribution_name,
                zorder=(i + 3) * 5,
            )

            # Plot confidence bounds
            if alpha is not None:
                for column in ["cil", "ciu"]:
                    ax.plot(
                        df.loc[:, "threshold"],
                        df.loc[:, column],
                        color=color,
                        lw=1,
                        ls="--",
                        zorder=(i + 2) * 5,
                    )
                ax.fill_between(
                    df.loc[:, "threshold"],
                    df.loc[:, "cil"],
                    df.loc[:, "ciu"],
                    facecolor=color,
                    edgecolor="None",
                    alpha=0.25,
                    zorder=(i + 1) * 5,
                )

        # Plot legend
        ax.legend(frameon=True, framealpha=0.9)

        # Label axes
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Return value")

        return ax


def plot_aic_scores(
    ts: pd.Series,
    thresholds=None,
    r: typing.Union[str, pd.Timedelta] = "24h",
    extremes_type: str = "high",
    distributions: typing.Optional[
        typing.List[typing.Union[str, scipy.stats.rv_continuous]]
    ] = None,
    ax: typing.Optional[plt.Axes] = None,
    figsize: tuple = (8, 5),
) -> plt.Axes:
    """
    Plot AIC scores for each distribution and threshold.

    Used to investigate which distribution better explains data variance for each
    threshold value. Does NOT indicate which threshold value is better because
    it will always have the same shape - logarithmic curve.

    Parameters
    ----------
    ts : pandas.Series
        Time series of the signal.
    thresholds : array-like, optional
        An array of thresholds for which the AIC plot is plotted.
        If None (default), plots AIC for 100 equally-spaced thresholds
        between 90th (10th if extremes_type='high') percentile
        and 10th largest (smallest if extremes_type='low') value in the series.
    r : pandas.Timedelta or value convertible to timedelta, optional
        Duration of window used to decluster the exceedances.
        By default r='24H' (24 hours).
        See pandas.to_timedelta for more information.
    extremes_type : str, optional
        high (default) - extreme high values
        low - extreme low values
    distributions : list, optional
        List of distributions for which the AIC curves are plotted.
        By default these are "genpareto" and "expon".
        A distribution must be either a name of distribution from scipy.stats
        or a subclass of scipy.stats.rv_continuous.
        See https://docs.scipy.org/doc/scipy/reference/stats.html
    ax : matplotlib.axes._axes.Axes, optional
        If provided, then the plot is drawn on this axes.
        If None (default), new figure and axes are created
    figsize : tuple, optional
        Figure size in inches in format (width, height).
        By default it is (8, 5).

    Returns
    -------
    plt.Axes
        Axes object.

    """
    # Get default `thresholds`
    if thresholds is None:
        thresholds = get_default_thresholds(
            ts=ts,
            extremes_type=extremes_type,
            num=100,
        )

    # Get default `distributions`
    if distributions is None:
        distributions = [
            "genpareto",
            "expon",
        ]
    distribution_names: typing.List[str] = []
    for distribution in distributions:
        if isinstance(distribution, str):
            distribution_names.append(distribution)
        else:
            distribution_names.append(distribution.name)

    # Calculate AIC values
    model = EVA(data=ts)
    results = []
    for distribution, distribution_name in zip(distributions, distribution_names):
        for threshold in thresholds:
            model.get_extremes(
                method="POT",
                extremes_type=extremes_type,
                threshold=threshold,
                r=r,
            )
            model.fit_model(model="MLE", distribution=distribution)
            results.append(
                {
                    "distribution_name": distribution_name,
                    "threshold": threshold,
                    "aic": model.AIC,
                }
            )
    results = pd.DataFrame(data=results).sort_values("threshold", ascending=True)

    with plt.rc_context(rc=pyextremes_rc):
        if ax is None:
            _, ax = plt.subplots(figsize=figsize, dpi=96)
            ax.grid(False)

        for i, (distribution_name, df) in enumerate(
            results.groupby("distribution_name")
        ):
            ax.plot(
                df.loc[:, "threshold"],
                df.loc[:, "aic"],
                color=pyextremes_rc["axes.prop_cycle"].by_key()["color"][i],
                lw=2,
                ls="-",
                label=distribution_name,
                zorder=(i + 3) * 5,
            )

        # Plot legend
        ax.legend(frameon=True, framealpha=0.9)

        # Label axes
        ax.set_xlabel("Threshold")
        ax.set_ylabel("AIC Score")

        return ax


def plot_threshold_stability(
    ts: pd.Series,
    return_period: float,
    return_period_size: typing.Union[str, pd.Timedelta] = "365.2425D",
    thresholds=None,
    r: typing.Union[str, pd.Timedelta] = "24h",
    extremes_type: str = "high",
    distributions: typing.Optional[
        typing.List[typing.Union[str, scipy.stats.rv_continuous]]
    ] = None,
    alpha: typing.Optional[float] = None,
    n_samples: int = 100,
    figsize: typing.Tuple[float, float] = (8, 2.5 * 4),
    progress: bool = False,
) -> typing.Tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes]:
    """
    Plot threshold influence on GPD parameters, return values, and AIC scores.

    Used as a utility function which plots multiple metrics in the same figure.

    Parameters
    ----------
    ts : pandas.Series
        Time series of the signal.
    return_period : float
        Return period.
        Given as a multiple of `return_period_size`.
    return_period_size : str or pandas.Timedelta, optional
        Size of return period (default='365.2425D').
        If set to '30D', then a return period of 12
        would be roughly equivalent to a 1 year return period (360 days).
    thresholds : array-like, optional
        An array of thresholds for which the metrics are plotted.
        If None (default), plots matrics for 100 equally-spaced thresholds
        between 90th (10th if extremes_type='high') percentile
        and 10th largest (smallest if extremes_type='low') value in the series.
    r : pandas.Timedelta or value convertible to timedelta, optional
        Duration of window used to decluster the exceedances.
        By default r='24H' (24 hours).
        See pandas.to_timedelta for more information.
    extremes_type : str, optional
        high (default) - extreme high values
        low - extreme low values
    distributions : list, optional
        List of distributions for which the metrics are plotted.
        By default these are "genpareto" and "expon".
        A distribution must be either a name of distribution from scipy.stats
        or a subclass of scipy.stats.rv_continuous.
        See https://docs.scipy.org/doc/scipy/reference/stats.html
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
    progress : bool, optional
        If True, shows tqdm progress bar.
        By default False.

    Returns
    -------
    ax_shape : matplotlib.axes._axes.Axes
    ax_scale : matplotlib.axes._axes.Axes
    ax_rv : matplotlib.axes._axes.Axes
    ax_aic : matplotlib.axes._axes.Axes

    """
    # Get default thresholds
    if thresholds is None:
        thresholds = get_default_thresholds(
            ts=ts,
            extremes_type=extremes_type,
            num=100,
        )

    with plt.rc_context(rc=pyextremes_rc):
        # Create figure
        fig = plt.figure(figsize=figsize, dpi=96)

        # Create gridspec
        gs = matplotlib.gridspec.GridSpec(
            nrows=4,
            ncols=1,
            wspace=0.1,
            hspace=0.1,
            width_ratios=[1],
            height_ratios=[1, 1, 1, 1],
        )

        # Create and configure axes
        ax_shape = fig.add_subplot(gs[0, 0])
        ax_scale = fig.add_subplot(gs[1, 0])
        ax_rv = fig.add_subplot(gs[2, 0])
        ax_aic = fig.add_subplot(gs[3, 0])
        axes = [ax_shape, ax_scale, ax_rv, ax_aic]

        # Produce individual plots
        plot_parameter_stability(
            ts=ts,
            thresholds=thresholds,
            r=r,
            extremes_type=extremes_type,
            alpha=alpha,
            n_samples=n_samples,
            axes=(ax_shape, ax_scale),
            progress=progress,
        )
        plot_return_value_stability(
            ts=ts,
            return_period=return_period,
            return_period_size=return_period_size,
            thresholds=thresholds,
            r=r,
            extremes_type=extremes_type,
            distributions=distributions,
            alpha=alpha,
            n_samples=n_samples,
            ax=ax_rv,
            progress=progress,
        )
        plot_aic_scores(
            ts=ts,
            thresholds=thresholds,
            r=r,
            extremes_type=extremes_type,
            distributions=distributions,
            ax=ax_aic,
        )

        # Format axes
        for ax in axes[:-1]:
            ax.tick_params(axis="x", which="both", labelbottom=False, length=0)
            ax.set_xlim(axes[-1].get_xlim())
            ax.set_xlabel("")
        axes[-1].set_xlabel("Threshold")

        return ax_shape, ax_scale, ax_rv, ax_aic
