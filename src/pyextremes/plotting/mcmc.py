from typing import List, Optional, Tuple

import matplotlib.gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import scipy.stats

from matplotlib.collections import LineCollection

from pyextremes.plotting.style import pyextremes_rc


def plot_trace(
    trace: np.ndarray,
    trace_map: Optional[tuple] = None,
    burn_in: int = 0,
    labels: List[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, list]:
    """
    Plot a trace plot for a given MCMC sampler trace.

    Parameters
    ----------
    trace : numpy.ndarray
        Array with MCMC sampler trace.
        Has a shape of (n_walkers, n_samples, n_parameters).
    trace_map : tuple, optional
        Tuple with maximum aposteriori estimate of distribution parameters.
        If provided, MAP values are plotted as orange lines on top of the trace.
        If None (default) then MAP estimates are not plotted.
    burn_in : int, optional
        Burn-in value (number of first steps to discard for each walker).
        By default it is 0 (no values are discarded).
    labels : list of strings, optional
        Sequence of strings with parameter names, used to label axes.
        If None (default), then axes are labeled sequentially.
    figsize : tuple, optional
        Figure size in inches.
        If None (default), then figure size is calculated automatically
        as 8 by 2 times number of parameters (`trace.shape[2]`).

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure object.
    axes : list
        List with `trace.shape[2]` Axes objects.

    """
    # Parse the 'burn_in' argument
    if not isinstance(burn_in, int):
        raise TypeError(
            f"invalid type in {type(burn_in)} for the 'burn_in' argument, "
            f"must be integer"
        )
    if burn_in < 0:
        raise ValueError(
            f"invalid value in {burn_in} for the 'burn_in' argument, "
            f"must be a positive integer"
        )
    if burn_in >= trace.shape[1]:
        raise ValueError(
            f"'burn_in' value of {burn_in} exceeds number of samples {trace.shape[1]}"
        )

    # Calculate figure size
    n_parameters = trace.shape[2]
    figsize = figsize or (8, 2 * n_parameters)

    with plt.rc_context(rc=pyextremes_rc):
        # Create figure
        fig = plt.figure(figsize=figsize, dpi=96)

        # Create gridspec
        gs = matplotlib.gridspec.GridSpec(
            nrows=n_parameters,
            ncols=1,
            wspace=0.1,
            hspace=0.1,
            width_ratios=[1],
            height_ratios=np.full(shape=n_parameters, fill_value=1),
        )

        # Create and configure axes
        axes = [fig.add_subplot(gs[i, 0]) for i in range(n_parameters)]
        if labels is None:
            labels = [f"Parameter {i}" for i in range(n_parameters)]
        for i, ax in enumerate(axes):
            ax.grid(False)
            ax.set_ylabel(labels[i])
            if i == n_parameters - 1:
                ax.set_xlabel("Sample number")
                ax.xaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, _: f"{x:,.0f}"),
                )
            else:
                ax.tick_params(axis="x", which="both", labelbottom=False)

        # Plot the trace plots
        lines = np.full(
            shape=(trace.shape[0], trace.shape[1] - burn_in, 2),
            fill_value=np.nan,
            dtype=np.float64,
        )
        lines[:, :, 0] = np.arange(burn_in + 1, trace.shape[1] + 1, 1)
        for i, ax in enumerate(axes):
            lines[:, :, 1] = trace[:, burn_in:, i]
            line_collection = LineCollection(
                segments=lines.copy(),
                color="#231F20",
                lw=0.1,
                zorder=5,
            )
            ax.add_collection(line_collection)
            if trace_map is not None:
                ax.axhline(trace_map[i], color="#F85C50", lw=2, ls="--", zorder=10)
            ax.autoscale(enable=True, axis="both", tight=False)

        # Align y-labels
        fig.align_ylabels(axs=axes)

        return fig, axes


def plot_corner(
    trace: np.ndarray,
    trace_map: Optional[tuple] = None,
    burn_in: int = 0,
    labels: List[str] = None,
    levels: Optional[int] = None,
    figsize: Tuple[float, float] = (8, 8),
) -> Tuple[plt.Figure, list]:
    """
    Plot a corner plot for a given MCMC sampler trace.

    Parameters
    ----------
    trace : numpy.ndarray
        Array with MCMC sampler trace.
        Has a shape of (n_walkers, n_samples, n_parameters).
    trace_map : tuple, optional
        Tuple with maximum aposteriori estimate of distribution parameters.
        If provided, MAP values are plotted as orange lines.
        If None (default) then MAP estimates are not plotted.
    burn_in : int, optional
        Burn-in value (number of first steps to discard for each walker).
        By default it is 0 (no values are discarded).
    labels : array-like, optional
        Sequence of strings with parameter names, used to label axes.
        If None (default), then axes are labeled sequentially.
    levels : int, optional
        Number of Gaussian KDE contours to plot.
        If None (default), then not shown.
    figsize : tuple, optional
        Figure size in inches. By default it is (8, 8).

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure object.
    axes : list
        2D list with Axes objects of size N by N, where N is `trace.shape[2]`.
        Empty slots are represented by None. Axes are ordered from left to right
        top to bottom.

    """
    # Parse the 'burn_in' argument
    if not isinstance(burn_in, int):
        raise TypeError(
            f"invalid type in {type(burn_in)} for the 'burn_in' argument, "
            f"must be integer"
        )
    if burn_in < 0:
        raise ValueError(
            f"invalid value in {burn_in} for the 'burn_in' argument, "
            f"must be a positive integer"
        )
    if burn_in >= trace.shape[1]:
        raise ValueError(
            f"'burn_in' value of {burn_in} exceeds number of samples {trace.shape[1]}"
        )

    n_parameters = trace.shape[2]

    with plt.rc_context(rc=pyextremes_rc):
        # Create figure
        fig = plt.figure(figsize=figsize, dpi=96)

        # Create gridspec
        gs = matplotlib.gridspec.GridSpec(
            nrows=n_parameters,
            ncols=n_parameters,
            wspace=0.1,
            hspace=0.1,
            width_ratios=[1] * n_parameters,
            height_ratios=[1] * n_parameters,
        )

        # Create and configure axes
        axes = [[None] * n_parameters for _ in range(n_parameters)]
        for i in range(n_parameters):
            for j in range(n_parameters):
                # Create axes only for axes at or left of the main diagonal
                if i >= j:
                    ax = fig.add_subplot(gs[i, j])
                    ax.grid(False)
                    axes[i][j] = ax

                    if i == n_parameters - 1:
                        # Set x-axis labels for axes located in the first row
                        if labels is None:
                            ax.set_xlabel(f"Parameter {j}")
                        else:
                            ax.set_xlabel(labels[j])
                    else:
                        # Remove x-axis ticks for axes located above the bottom row
                        ax.tick_params(
                            axis="x", which="both", labelbottom=False, length=0
                        )

                    if j == 0:
                        # Set y-axis label for axes located in the first column
                        if labels is None:
                            ax.set_ylabel(f"Parameter {i}")
                        else:
                            ax.set_ylabel(labels[i])

                    if j != 0 or i == j == 0:
                        # Remove y-axis ticks for axes located right of the first column
                        # and for the first axes along the main diagonal
                        ax.tick_params(
                            axis="y", which="both", labelleft=False, length=0
                        )

                    if i == j:
                        # Plot histogram
                        parameter_samples = trace[:, burn_in:, i].flatten()
                        ax.hist(
                            parameter_samples,
                            bins=np.histogram_bin_edges(
                                a=parameter_samples, bins="auto"
                            ),
                            density=True,
                            histtype="step",
                            edgecolor="#231F20",
                            lw=0.5,
                            zorder=5,
                        )
                        if trace_map is not None:
                            ax.axvline(
                                trace_map[i], color="#F85C50", lw=1, ls="--", zorder=10
                            )
                    else:
                        # Plot scatter plot
                        parameter_i = trace[:, burn_in:, i].flatten()
                        parameter_j = trace[:, burn_in:, j].flatten()
                        ax.scatter(
                            parameter_j,
                            parameter_i,
                            marker="o",
                            s=2,
                            alpha=0.1,
                            facecolor="#231F20",
                            edgecolor="None",
                            lw=0,
                            zorder=5,
                        )

                        # Plot trace map lines
                        if trace_map is not None:
                            ax.axvline(
                                trace_map[j], color="#F85C50", lw=1, ls="--", zorder=15
                            )
                            ax.axhline(
                                trace_map[i], color="#F85C50", lw=1, ls="--", zorder=15
                            )

                        # Plot Gaussian KDE contour
                        if levels is not None:
                            kernel = scipy.stats.gaussian_kde(
                                np.vstack([parameter_j, parameter_i])
                            )
                            xx, yy = np.meshgrid(
                                np.linspace(parameter_j.min(), parameter_j.max(), 100),
                                np.linspace(parameter_i.min(), parameter_i.max(), 100),
                            )
                            zz = np.reshape(
                                kernel(np.vstack([xx.ravel(), yy.ravel()])).transpose(),
                                xx.shape,
                            )
                            ax.contour(
                                xx,
                                yy,
                                zz,
                                colors="w",
                                levels=levels,
                                linewidths=1,
                                linestyles="-",
                                zorder=10,
                            )

                    # Set axes limits
                    parameter_i = trace[:, burn_in:, i].flatten()
                    parameter_j = trace[:, burn_in:, j].flatten()
                    xlim = (parameter_j.min(), parameter_j.max())
                    ylim = (parameter_i.min(), parameter_i.max())
                    ax.set_xlim(xlim)
                    if i != j:
                        ax.set_ylim(ylim)

        # Align labels
        fig.align_labels()

        return fig, axes
