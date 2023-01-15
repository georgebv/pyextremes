from typing import Any, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyextremes.extremes.peaks_over_threshold import _generate_clusters
from pyextremes.plotting.style import pyextremes_rc


def _plot_cluster(ax: plt.Axes, cluster: pd.Series) -> None:
    if len(cluster) >= 2:
        ax.axvspan(
            xmin=cluster.index[0],
            xmax=cluster.index[-1],
            alpha=0.25,
            edgecolor="None",
            facecolor="#D1D3D4",
            zorder=2.5,
        )
        for cluster_boundary in [cluster.index[0], cluster.index[-1]]:
            ax.axvline(
                cluster_boundary,
                ls="--",
                lw=0.5,
                color="#D1D3D4",
                zorder=5,
            )


def plot_extremes(
    ts: pd.Series,
    extremes: pd.Series,
    extremes_method: Literal["BM", "POT"],
    extremes_type: Optional[Literal["high", "low"]] = None,
    block_size: Optional[Union[str, pd.Timedelta]] = None,
    threshold: Optional[float] = None,
    r: Optional[Union[pd.Timedelta, Any]] = None,
    figsize: Tuple[float, float] = (8, 5),
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
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
    r : pandas.Timedelta or value convertible to timedelta, optional
        Duration of window used to decluster the exceedances.
        See pandas.to_timedelta for more information.
        Used to show clusters. If None (default) then clusters are not shown.
        Clusters are shown only if both `threshold` and `r` were provided.
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
    if extremes_method not in ["BM", "POT"]:
        raise ValueError(
            f"invalid value in '{extremes_method}' for the 'extremes_method' argument"
        )

    if extremes_type not in ["high", "low"]:
        raise ValueError(
            f"invalid value in '{extremes_type}' for the 'extremes_type' argument"
        )

    with plt.rc_context(rc=pyextremes_rc):
        # Create figure
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=96)
        else:
            try:
                fig = ax.get_figure()
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
            while block_left_boundary < extremes.index.max() + block_size:
                ax.axvline(
                    block_left_boundary, ls="--", lw=0.5, color="#D1D3D4", zorder=5
                )
                block_left_boundary += block_size

        else:
            if threshold is None:
                if extremes_type is None:
                    raise TypeError(
                        "'extremes_type' argument must be provided "
                        "for 'extremes_method' being 'POT' "
                        "when 'threshold' is not provided"
                    )
                if extremes_type == "high":
                    threshold = extremes.min()
                else:
                    threshold = extremes.max()
            else:
                if r is not None:
                    # Plot clusters (only if both threshold and r are provided)
                    if extremes_type == "high":
                        exceedances = ts.loc[ts.values > threshold]
                    else:
                        exceedances = ts.loc[ts.values < threshold]
                    for cluster in _generate_clusters(exceedances=exceedances, r=r):
                        _plot_cluster(ax=ax, cluster=cluster)

            # Plot threshold line
            ax.axhline(threshold, ls="--", lw=1, color="#FF756B", zorder=15)

        fig.autofmt_xdate()

        return fig, ax
