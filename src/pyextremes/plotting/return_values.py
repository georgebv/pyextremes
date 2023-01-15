from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from pyextremes.plotting.style import pyextremes_rc


def plot_return_values(
    observed_return_values: pd.DataFrame,
    modeled_return_values: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot return values and confidence intervals for given return periods.

    Parameters
    ----------
    observed_return_values : pandas.DataFrame
        DataFrame with observed return values.
        First column must have extreme values.
        Must have 'return period' column.
    modeled_return_values : pandas.DataFrame
        DataFrame with modeled return values.
        Index has return periods.
        Must have the following columns: 'return value', 'lower ci', 'upper ci'.
    ax : matplotlib.axes._axes.Axes, optional
        Axes onto which the return value plot is drawn.
        If None (default), a new figure and axes objects are created.
    figsize : tuple, optional
        Figure size in inches in format (width, height).
        By default it is (8, 5).

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure object.
    axes : matplotlib.axes._axes.Axes
        Axes object.

    """
    # Validate the 'observed_return_values' argument
    if (
        len(observed_return_values.columns) < 2
        or observed_return_values.columns[0] == "return period"
        or "return period" not in observed_return_values.columns
    ):
        raise ValueError(
            f"'observed_return_values' argument "
            f"has invalid columns in {observed_return_values.columns}, "
            f"must have at least two columns and a 'return period' column "
            f"which is not the first column"
        )

    # Validate the 'modeled_return_values' argument
    if len(modeled_return_values.columns) < 3 or any(
        col not in modeled_return_values.columns
        for col in ["return value", "lower ci", "upper ci"]
    ):
        raise ValueError(
            f"'modeled_return_values' argument "
            f"has invalid columns in {modeled_return_values.columns}, "
            f"must have at least three columns and include columns: "
            f"'return value', 'lower ci', 'upper ci'"
        )

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
        ax.semilogx()
        ax.grid(True, which="both")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        # Plot modeled confidence intervals
        for col in ["lower ci", "upper ci"]:
            ax.plot(
                modeled_return_values.index.values,
                modeled_return_values.loc[:, col].values,
                color="#5199FF",
                lw=1,
                ls="--",
                zorder=15,
            )
        ax.fill_between(
            modeled_return_values.index.values,
            modeled_return_values.loc[:, "lower ci"].values,
            modeled_return_values.loc[:, "upper ci"].values,
            facecolor="#5199FF",
            edgecolor="None",
            alpha=0.25,
            zorder=10,
        )

        # Plot observed extreme values
        ax.scatter(
            observed_return_values.loc[:, "return period"].values,
            observed_return_values.loc[:, observed_return_values.columns[0]].values,
            marker="o",
            s=20,
            lw=1,
            facecolor="k",
            edgecolor="w",
            zorder=20,
        )

        # Plot modeled return values
        ax.plot(
            modeled_return_values.index.values,
            modeled_return_values.loc[:, "return value"].values,
            color="#F85C50",
            lw=2,
            ls="-",
            zorder=25,
        )

        # Label axes
        ax.set_xlabel("Return period")
        ax.set_ylabel(observed_return_values.columns[0])

        return fig, ax
