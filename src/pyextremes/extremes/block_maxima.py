import logging
import warnings

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NoDataBlockWarning(Warning):
    pass


def get_extremes_block_maxima(
    ts: pd.Series,
    extremes_type: Literal["high", "low"],
    block_size: Union[str, pd.Timedelta] = "365.2425D",
    errors: Literal["raise", "ignore", "coerce"] = "raise",
    min_last_block: Optional[float] = None,
) -> pd.Series:
    """
    Get extreme events from time series using the Block Maxima method.

    Parameters
    ----------
    ts : pandas.Series
        Time series of the signal.
    extremes_type : str
        high - get extreme high values (block maxima)
        low - get extreme low values (block minima)
    block_size : str or pandas.Timedelta, optional
        Block size (default='365.2425D').
        See pandas.to_timedelta for more information.
    errors : str, optional
        raise (default) - raise an exception when encountering a block with no data
        ignore - ignore blocks with no data
        coerce - get extreme values for blocks with no data
            as mean of all other extreme events in the series
            with index being the middle point of corresponding interval
    min_last_block : float, optional
        Minimum data availability ratio (0 to 1) in the last block
        for it to be used to extract extreme value from.
        This is used to discard last block when it is too short.
        If None (default), last block is always used.

    Returns
    -------
    extremes : pandas.Series
        Time series of extreme events.

    """
    logger.debug(
        "collecting block maxima extreme events using extremes_type=%s, "
        "block_size=%s, errors=%s",
        extremes_type,
        block_size,
        errors,
    )

    if extremes_type not in ["high", "low"]:
        raise ValueError(
            f"invalid value in '{extremes_type}' for the 'extremes_type' argument"
        )

    if errors not in ["raise", "ignore", "coerce"]:
        raise ValueError(f"invalid value in '{errors}' for the 'errors' argument")

    # Get extreme value extraction function
    if extremes_type == "high":
        extremes_func = pd.Series.idxmax
    else:
        extremes_func = pd.Series.idxmin

    # Parse the 'block_size' argument
    if not isinstance(block_size, pd.Timedelta):
        if isinstance(block_size, str):
            block_size = pd.to_timedelta(block_size)
        else:
            raise TypeError(
                f"invalid type in {type(block_size).__name__} "
                f"for the 'block_size' argument"
            )

    # Prepare date-time intervals
    periods = int((ts.index.max() - ts.index.min()) / block_size) + 1
    date_time_intervals = pd.interval_range(
        start=ts.index[0],
        freq=block_size,
        periods=periods,
        closed="left",
    )

    # Collect extreme events
    empty_intervals = 0
    extreme_indices, extreme_values = [], []
    for interval in date_time_intervals:
        ts_slice = ts.loc[(ts.index >= interval.left) & (ts.index < interval.right)]
        if len(ts_slice) > 0:
            extreme_indices.append(extremes_func(ts_slice))
            extreme_values.append(ts_slice.loc[extreme_indices[-1]])
        else:
            empty_intervals += 1
            if errors == "coerce":
                logger.debug(
                    "coerced no-data block in [%s ; %s)",
                    interval.left,
                    interval.right,
                )
                extreme_indices.append(interval.mid)
                extreme_values.append(np.nan)
            elif errors == "ignore":
                logger.debug(
                    "ignored no-data block in [%s ; %s)",
                    interval.left,
                    interval.right,
                )
            else:
                raise ValueError(
                    f"no data in block [{interval.left} ; {interval.right}), "
                    f"fill gaps in the data "
                    f"or set the argument 'errors' to 'coerce' or 'ignore'"
                )

    # Check last block duration
    if min_last_block is not None:
        ratio = (ts.index.max() - date_time_intervals[-1].left) / block_size
        if ratio < min_last_block:
            logger.debug(
                "discarded last block with data availability ratio of %s",
                ratio,
            )
            extreme_indices = extreme_indices[:-1]
            extreme_values = extreme_values[:-1]

    if empty_intervals > 0:
        warnings.warn(
            message=f"{empty_intervals} blocks contained no data",
            category=NoDataBlockWarning,
        )

    logger.debug(
        "successfully collected %d extreme events, found %s no-data blocks",
        len(extreme_values),
        empty_intervals,
    )

    return pd.Series(
        data=extreme_values,
        index=pd.Index(data=extreme_indices, name=ts.index.name or "date-time"),
        dtype=np.float64,
        name=ts.name or "extreme values",
    ).fillna(np.nanmean(extreme_values))
