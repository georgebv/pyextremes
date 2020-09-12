import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyextremes.plotting.style import pyextremes_rc


def plot_extremes(
    ts: pd.Series,
    extremes: pd.Series,
    extremes_method: str,
    extremes_type: typing.Optional[str] = None,
    block_size: typing.Optional[typing.Union[str, pd.Timedelta]] = None,
    threshold: typing.Optional[float] = None,
    figsize: tuple = (8, 5),
    ax: typing.Optional[plt.Axes] = None,
) -> typing.Tuple[plt.Figure, plt.Axes]:
    """
    Plot extreme events.

    Parameters
    ----------
    ts : pandas.Series
        Time series from which `extremes` were extracted.
    extremes : pandas.Series
        Time series of extreme events.
    extremes_method : str
        Extreme value extraction method.
        Supported values:
            BM - Block Maxima
            POT - Peaks Over Threshold
    extremes_type : str, optional
        Type of `extremes`, used only if `extremes_method` is 'POT'
        and `threshold` is not provided.
            high - extreme high values
            low - get low values
    block_size : str or pandas.Timedelta, optional
        Block size, used only if `extremes_method` is 'BM'.
        If None (default), then calculated as median distance between extreme events.
    threshold : float, optional
        Threshold, used only if `extremes_method` is 'POT'.
        If None (default), then is inferred from `extremes` as
        minimum if `extremes_type` is 'high' or maximum if `extremes_type` is 'low'.
    figsize : tuple, optional
        Figure size in inches in format (width, height).
        By default it is (8, 5).
    ax : matplotlib.axes._axes.Axes, optional
        Axes onto which extremes plot is drawn.
        If None (default), a new figure and axes objects are created.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure object.
    axes : matplotlib.axes._axes.Axes
        Axes object.

    """
    with plt.rc_context(rc=pyextremes_rc):
        # Create figure
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=96)
        else:
            try:
                fig = ax.figure
            except AttributeError as _error:
                raise TypeError(
                    f"invalid type in {type(ax)} for the 'ax' argument, "
                    f"must be matplotlib Axes object"
                ) from _error

        # Configure axes
        ax.grid(False)

        # Plot signal time series
        ax.plot(ts.index, ts.values, ls="-", color="#5199FF", lw=0.25, zorder=10)

        # Plot extreme events
        ax.scatter(
            extremes.index,
            extremes.values,
            s=20,
            lw=0.5,
            edgecolor="w",
            facecolor="#F85C50",
            zorder=20,
        )

        # Label the axes
        ax.set_xlabel(extremes.index.name or "date-time")
        ax.set_ylabel(extremes.name or "Extreme value")

        if extremes_method == "BM":
            # Infer 'block_size'
            if block_size is None:
                # Calculate 'block_size' as median of distances between extremes
                block_size = pd.to_timedelta(
                    np.quantile(
                        np.diff(extremes.index),
                        0.5,
                    )
                )
            else:
                if not isinstance(block_size, pd.Timedelta):
                    if isinstance(block_size, str):
                        block_size = pd.to_timedelta(block_size)
                    else:
                        raise TypeError(
                            f"invalid type in {type(block_size)} "
                            f"for the 'block_size' argument"
                        )

            # Plot block boundaries
            block_left_boundary = ts.index[0]
            while block_left_boundary < ts.index.max() + block_size:
                ax.axvline(
                    block_left_boundary, ls="--", lw=0.5, color="#D1D3D4", zorder=5
                )
                block_left_boundary += block_size

        elif extremes_method == "POT":
            # Parse 'threshold'
            if threshold is None:
                if extremes_type is None:
                    raise TypeError(
                        "'extremes_type' argument must be provided "
                        "for 'extremes_method' being 'POT' "
                        "when 'threshold' is not provided"
                    )
                elif extremes_type == "high":
                    threshold = extremes.min()
                elif extremes_type == "low":
                    threshold = extremes.max()
                else:
                    raise ValueError(
                        f"invalid value in '{extremes_type}' "
                        f"for the 'extremes_type' argument"
                    )

            # Plot threshold line
            ax.axhline(threshold, ls="--", lw=1, color="#FF756B", zorder=15)

        else:
            raise ValueError(
                f"invalid value in '{extremes_method}' "
                f"for the 'extremes_method argument"
            )

        return fig, ax
