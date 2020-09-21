import logging
import typing

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_extremes_peaks_over_threshold(
    ts: pd.Series,
    extremes_type: str,
    threshold: float,
    r: typing.Union[str, pd.Timedelta] = "24H",
) -> pd.Series:
    """
    Get extreme events from time series using the Peaks Over Threshold method.

    Parameters
    ----------
    ts : pandas.Series
        Time series of the signal.
    extremes_type : str
        high - get extreme high values
        low - get extreme low values
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
    logger.debug(
        f"collecting peaks-over-threshold extreme events using "
        f"extremes_type={extremes_type}, threshold={threshold}, r={r}"
    )

    # Get extreme value extraction function
    if extremes_type == "high":
        comparison_function = np.greater
    elif extremes_type == "low":
        comparison_function = np.less
    else:
        raise ValueError(
            f"invalid value in '{extremes_type}' for the 'extremes_type' argument"
        )

    # Parse the 'r' argument
    if not isinstance(r, pd.Timedelta):
        if isinstance(r, str):
            r = pd.to_timedelta(r)
        else:
            raise TypeError(f"invalid type in {type(r)} for the 'r' argument")

    # Get exceedances
    exceedances = ts.loc[comparison_function(ts.values, threshold)]

    # Decluster exceedances
    extreme_indices, extreme_values = [exceedances.index[0]], [exceedances.values[0]]
    for index, value in exceedances.iteritems():
        if (index - extreme_indices[-1]) > r:
            # Starting new cluster
            extreme_indices.append(index)
            extreme_values.append(value)
        else:
            if comparison_function(value, extreme_values[-1]):
                # Found new cluster peak
                extreme_indices[-1] = index
                extreme_values[-1] = value

    logger.info(f"successfully collected {len(extreme_values)} extreme events")
    return pd.Series(
        data=extreme_values,
        index=pd.Index(data=extreme_indices, name=ts.index.name or "date-time"),
        dtype=np.float64,
        name=ts.name or "extreme values",
    )
