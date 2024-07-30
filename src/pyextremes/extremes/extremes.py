from typing import Any, Literal, Optional, Union, overload

import pandas as pd

from pyextremes.extremes.block_maxima import get_extremes_block_maxima
from pyextremes.extremes.peaks_over_threshold import get_extremes_peaks_over_threshold


@overload
def get_extremes(
    ts: pd.Series,
    method: Literal["BM"],
    extremes_type: Literal["high", "low"] = "high",
    *,
    block_size: str = "365.2425D",
    errors: Literal["raise", "ignore", "coerce"] = "raise",
    min_last_block: Optional[float] = None,
) -> pd.Series:
    ...


@overload
def get_extremes(
    ts: pd.Series,
    method: Literal["POT"],
    extremes_type: Literal["high", "low"] = "high",
    *,
    threshold: float,
    r: Union[pd.Timedelta, Any] = "24h",
) -> pd.Series:
    ...


def get_extremes(
    ts: pd.Series,
    method: Literal["BM", "POT"],
    extremes_type: Literal["high", "low"] = "high",
    **kwargs,
) -> pd.Series:
    """
    Get extreme events from time series.

    Parameters
    ----------
    ts : pandas.Series
        Time series of the signal.
    method : str
        Extreme value extraction method.
        Supported values:
            BM - Block Maxima
            POT - Peaks Over Threshold
    extremes_type : str, optional
        high (default) - get extreme high values
        low - get extreme low values
    kwargs
        if method is BM:
            block_size : str or pandas.Timedelta, optional
                Block size (default='365.2425D').
            errors : str, optional
                raise (default) - raise an exception
                    when encountering a block with no data
                ignore - ignore blocks with no data
                coerce - get extreme values for blocks with no data
                    as mean of all other extreme events in the series
                    with index being the middle point of corresponding interval
            min_last_block : float, optional
                Minimum data availability ratio (0 to 1) in the last block
                for it to be used to extract extreme value from.
                This is used to discard last block when it is too short.
                If None (default), last block is always used.
        if method is POT:
            threshold : float
                Threshold used to find exceedances.
            r : pandas.Timedelta or value convertible to timedelta, optional
                Duration of window used to decluster the exceedances.
                By default r='24H' (24 hours).
                See pandas.to_timedelta for more information.

    Returns
    -------
    extremes : pandas.Series
        Time series of extreme events.

    """
    if method == "BM":
        return get_extremes_block_maxima(
            ts=ts,
            extremes_type=extremes_type,
            **kwargs,
        )
    if method == "POT":
        return get_extremes_peaks_over_threshold(
            ts=ts,
            extremes_type=extremes_type,
            **kwargs,
        )
    raise ValueError(
        f"invalid value in '{method}' for the 'method' argument, "
        f"available methods: 'BM', 'POT'"
    )
