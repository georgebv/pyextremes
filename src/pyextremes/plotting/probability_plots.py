import typing

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from pyextremes.plotting.style import pyextremes_rc


def plot_probability(
    observed: np.ndarray,
    theoretical: np.ndarray,
    ax: typing.Optional[plt.Axes] = None,
    figsize: tuple = (8, 8),
) -> typing.Tuple[plt.Figure, plt.Axes]:
    """
    Plot a probability plot (QQ or PP).

    Parameters
    ----------
    observed : numpy.ndarray
        Observed values.
    theoretical : numpy.ndarray
        Theoretical values.
    ax : matplotlib.axes._axes.Axes, optional
        Axes onto which the probability plot is drawn.
        If None (default), a new figure and axes objects are created.
    figsize : tuple, optional
        Figure size in inches in format (width, height).
        By default it is (8, 8).

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

        # Plot scatter of observed and theoretical probabilities
        ax.scatter(
            theoretical,
            observed,
            marker="o",
            s=20,
            lw=0.75,
            facecolor="k",
            edgecolor="w",
            zorder=10,
        )

        # Plot a diagonal perfect-fit line
        min_value = min([min(ax.get_xlim()), min(ax.get_ylim())])
        max_value = max([max(ax.get_xlim()), max(ax.get_ylim())])
        ax.plot(
            [min_value, max_value],
            [min_value, max_value],
            color="#5199FF",
            lw=1,
            ls="--",
            zorder=5,
        )

        # Label axes
        ax.set_xlabel("Theoretical")
        ax.set_ylabel("Observed")

        # Calculate Pearson R statistic and show it in the figure
        pearsonr, p_value = scipy.stats.pearsonr(theoretical, observed)
        axes_range = max_value - min_value
        ax.text(
            x=min_value + 0.05 * axes_range,
            y=max_value - 0.05 * axes_range,
            s=f"$R^2={pearsonr:.3f}$\n$p={p_value:.3f}$",
            horizontalalignment="left",
            verticalalignment="top",
        )

        # Set axes limits
        ax.set_xlim(min_value, max_value)
        ax.set_ylim(min_value, max_value)

        return fig, ax
