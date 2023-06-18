from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import scipy.stats

plotting_positions = {
    "ecdf": (0, 1),
    "hazen": (0.5, 0.5),
    "weibull": (0, 0),
    "tukey": (1 / 3, 1 / 3),
    "blom": (3 / 8, 3 / 8),
    "median": (0.3175, 0.3175),
    "cunnane": (0.4, 0.4),
    "gringorten": (0.44, 0.44),
    "beard": (0.31, 0.31),
}


def get_return_periods(
    ts: pd.Series,
    extremes: pd.Series,
    extremes_method: Literal["BM", "POT"],
    extremes_type: Literal["high", "low"],
    block_size: Optional[Union[str, pd.Timedelta]] = None,
    return_period_size: Union[str, pd.Timedelta] = "365.2425D",
    plotting_position: Literal[
        "ecdf",
        "hazen",
        "weibull",
        "tukey",
        "blom",
        "median",
        "cunnane",
        "gringorten",
        "beard",
    ] = "weibull",
) -> pd.DataFrame:
    """
    Calculate return periods for given extreme values using given plotting position.

    Return periods are multiples of `return_period_size`.
    Plotting positions were taken from
    https://matplotlib.org/mpl-probscale/tutorial/closer_look_at_plot_pos.html

    Parameters
    ----------
    ts : pandas.Series
        Time series of the signal.
    extremes : pandas.Series
        Time series of extreme events.
    extremes_method : str
        Extreme value extraction method.
        Supported values:
            BM - Block Maxima
            POT - Peaks Over Threshold
    extremes_type : str
        high - provided extreme values are extreme high values
        low - provided extreme values are extreme low values
    block_size : str or pandas.Timedelta, optional
        Block size in the 'BM' `extremes_method` (default=None).
        If None, then is calculated as median distance between extreme events.
    return_period_size : str or pandas.Timedelta, optional
        Size of return periods (default='365.2425D').
        If set to '30D', then a return period of 12
        would be roughly equivalent to a 1 year return period (360 days).
    plotting_position : str, optional
        Plotting position name (default='weibull'), not case-sensitive.
        Supported plotting positions:
            ecdf, hazen, weibull, tukey, blom, median, cunnane, gringorten, beard

    Returns
    -------
    extreme_events : pandas.DataFrame
        A DataFrame with extreme values, exceedance probabilities,
        and return periods as multiples of `return_period_size`.

    """
    if extremes_method == "BM":
        # Parse 'block_size' argument
        if block_size is None:
            # Calculate 'block_size' as median distance between extremes
            block_size = pd.to_timedelta(np.quantile(np.diff(extremes.index), 0.5))
        else:
            if not isinstance(block_size, pd.Timedelta):
                if isinstance(block_size, str):
                    block_size = pd.to_timedelta(block_size)
                else:
                    raise TypeError(
                        f"invalid type in {type(block_size)} "
                        f"for the 'block_size' argument"
                    )
    else:
        if block_size is not None:
            raise ValueError(
                f"'block_size' value is used only if 'extremes_method' is 'BM', "
                f"provided 'extremes_method' is {extremes_method}"
            )

    # Parse the 'return_period_size' argument
    if not isinstance(return_period_size, pd.Timedelta):
        if isinstance(return_period_size, str):
            return_period_size = pd.to_timedelta(return_period_size)
        else:
            raise TypeError(
                f"invalid type in {type(return_period_size)} "
                f"for the 'return_period_size' argument"
            )

    # Calculate rate of extreme events as number of events per one return period
    if extremes_method == "BM":
        extremes_rate = return_period_size / block_size
    elif extremes_method == "POT":
        n_periods = (ts.index.max() - ts.index.min()) / return_period_size
        extremes_rate = len(extremes) / n_periods
    else:
        raise ValueError(
            f"invalid value in '{extremes_method}' for the 'extremes_method' argument"
        )

    # Rank extreme values from most extreme (1) to least extreme (len(extremes))
    if extremes_type == "high":
        ranks = (
            len(extremes) + 1 - scipy.stats.rankdata(extremes.values, method="average")
        )
    elif extremes_type == "low":
        ranks = scipy.stats.rankdata(extremes.values, method="average")
    else:
        raise ValueError(
            f"invalid value in '{extremes_type}' for the 'extremes_type' argument"
        )

    # Get plotting position parameters
    try:
        alpha, beta = plotting_positions[plotting_position.lower()]
    except KeyError as _error:
        raise ValueError(
            f"invalid value in '{plotting_position}' "
            f"for the 'plotting_position' argument"
        ) from _error

    # Calculate exceedance probabilities
    exceedance_probability = (ranks - alpha) / (len(extremes) + 1 - alpha - beta)

    # Calculate return periods
    return_periods = 1 / exceedance_probability / extremes_rate

    # Copy `extremes` to make the returned DataFrame independent from the original
    extremes = extremes.copy(deep=True)

    return pd.DataFrame(
        data={
            extremes.name: extremes.values,
            "exceedance probability": exceedance_probability,
            "return period": return_periods,
        },
        index=extremes.index,
        dtype=np.float64,
    )
