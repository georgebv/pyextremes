import pandas as pd

from pyextremes.extremes.block_maxima import get_extremes_block_maxima
from pyextremes.extremes.peaks_over_threshold import get_extremes_peaks_over_threshold


def get_extremes(
    ts: pd.Series,
    method: str,
    extremes_type: str = "high",
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
                Block size (default='1Y').
            errors : str, optional
                raise (default) - raise an exception
                    when encountering a block with no data
                ignore - ignore blocks with no data
                coerce - get extreme values for blocks with no data
                    as mean of all other extreme events in the series
                    with index being the middle point of corresponding interval
        if method is POT:
            threshold : float
                Threshold used to find exceedances.
            r : str or pandas.Timedelta, optional
                Duration of window used to decluster the exceedances.
                By default r='24H' (24 hours).

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
    elif method == "POT":
        return get_extremes_peaks_over_threshold(
            ts=ts,
            extremes_type=extremes_type,
            **kwargs,
        )
    else:
        raise ValueError(
            f"invalid value in '{method}' for the 'method' argument, "
            f"available methods: 'BM', 'POT'"
        )
