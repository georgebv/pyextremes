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
    min_block_count: Optional[int] = None,
) -> pd.Series: ...


@overload
def get_extremes(
    ts: pd.Series,
    method: Literal["POT"],
    extremes_type: Literal["high", "low"] = "high",
    *,
    threshold: float,
    r: Union[pd.Timedelta, Any] = "24h",
) -> pd.Series: ...


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
                ignore - ignore blocks with no data.
                    Unbiased choice: block maxima return periods depend only
                    on block_size, so skipping empty blocks does not bias them
                coerce - get extreme values for blocks with no data
                    as mean of all other extreme events in the series
                    with index being the middle point of corresponding interval.
                    Not recommended for fitting: the filled values shrink the
                    fitted spread and confidence intervals and bias return
                    values low.
            min_last_block : float, optional
                Minimum data availability ratio (0 to 1) in the last block
                for it to be used to extract extreme value from.
                This is used to discard last block when it is too short.
                If None (default), last block is always used.
            min_block_count : int, optional
                Minimum number of observations in a block
                for it to be used to extract extreme value from.
                Blocks with fewer observations are treated as empty,
                so the `errors` handling applies to them and they are counted
                in NoDataBlockWarning. Set it from the known sampling rate,
                e.g. 300 for daily data and annual blocks.
                If None (default), any block with at least one observation is used.
        if method is POT:
            threshold : float
                Threshold used to find exceedances.
            r : pandas.Timedelta or value convertible to timedelta, optional
                Duration of window used to decluster the exceedances.
                By default r='24h' (24 hours).
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
