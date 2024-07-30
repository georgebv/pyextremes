import logging
import warnings

from typing import Any, Generator, Literal, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _generate_clusters(
    exceedances: pd.Series,
    r: Union[pd.Timedelta, Any],
) -> Generator[pd.Series, None, None]:
    if not isinstance(r, pd.Timedelta):
        try:
            r = pd.to_timedelta(r)
        except Exception as error:
            raise ValueError(f"invalid value in {r} for the 'r' argument") from error

    # There can be no clusters if there are no exceedances
    if len(exceedances) == 0:
        return
    # There can be only one cluster if there is only one exceedance
    if len(exceedances) == 1:
        yield exceedances
        return

    # Locate clusters separated by gaps not smaller than `r`
    gap_indices = np.argwhere(
        (exceedances.index[1:] - exceedances.index[:-1]) > r
    ).flatten()
    if len(gap_indices) == 0:
        # All exceedances fall within the same cluster
        yield exceedances
    else:
        for i, gap_index in enumerate(gap_indices):
            if i == 0:
                # First cluster contains all values left from the gap
                yield exceedances.iloc[: gap_index + 1]
            else:
                # Other clusters contain values between previous and current gaps
                yield exceedances.iloc[gap_indices[i - 1] + 1 : gap_index + 1]

        # Last cluster contains all values right from the last gap
        yield exceedances.iloc[gap_indices[-1] + 1 :]


def get_extremes_peaks_over_threshold(
    ts: pd.Series,
    extremes_type: Literal["high", "low"],
    threshold: float,
    r: Union[pd.Timedelta, Any] = "24h",
) -> pd.Series:
    """
    Get extreme events from time series using the Peaks Over Threshold method.

    Parameters
    ----------
    ts : pandas.Series
        Time series of the signal.
    extremes_type : str
        high - get extreme high values (above threshold)
        low - get extreme low values (below threshold)
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
    logger.debug(
        "collecting peaks-over-threshold extreme events using "
        "extremes_type=%s, threshold=%s, r=%s",
        extremes_type,
        threshold,
        r,
    )

    if extremes_type not in ["high", "low"]:
        raise ValueError(
            f"invalid value in '{extremes_type}' for the 'extremes_type' argument"
        )

    # Get exceedances
    if extremes_type == "high":
        exceedances = ts.loc[ts.values > threshold]
    else:
        exceedances = ts.loc[ts.values < threshold]
    logger.debug("found %d exceedances", len(exceedances))

    if len(exceedances) == 0:
        warnings.warn(
            f"Threshold value '{threshold}' is too {extremes_type} "
            f"and results in zero extreme values"
        )

    # Locate clusters separated by gaps not smaller than `r`
    # and select min or max (depending on `extremes_type`) within each cluster
    extreme_indices, extreme_values = [], []
    for cluster in _generate_clusters(exceedances=exceedances, r=r):
        extreme_indices.append(
            cluster.idxmax() if extremes_type == "high" else cluster.idxmin()
        )
        extreme_values.append(cluster.loc[extreme_indices[-1]])

    logger.debug(
        "successfully collected %d extreme events",
        len(extreme_values),
    )
    return pd.Series(
        data=extreme_values,
        index=pd.Index(data=extreme_indices, name=ts.index.name or "date-time"),
        dtype=np.float64,
        name=ts.name or "extreme values",
    )
