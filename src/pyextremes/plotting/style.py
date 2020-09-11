import matplotlib

theme_color = "#231F20"
pyextremes_rc = {
    # Lines
    "lines.linewidth": 1.0,
    "lines.linestyle": "-",
    # Font
    "font.family": "sans-serif",
    "font.size": 10,
    # Text
    "text.color": theme_color,
    # Axes
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": theme_color,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.labelsize": 10,
    "axes.labelweight": "normal",
    "axes.labelcolor": theme_color,
    "axes.prop_cycle": matplotlib.cycler(
        "color",
        [
            "#1771F1",
            "#F85C50",
            "#35D073",
            "#FFC11E",
        ],
    ),
    # Ticks
    "xtick.major.size": 2,
    "xtick.minor.size": 1,
    "xtick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "xtick.major.top": True,
    "xtick.major.bottom": True,
    "xtick.minor.top": True,
    "xtick.minor.bottom": True,
    "xtick.color": theme_color,
    "ytick.major.size": 2,
    "ytick.minor.size": 1,
    "ytick.major.width": 0.8,
    "ytick.minor.width": 0.6,
    "ytick.color": theme_color,
    "ytick.major.left": True,
    "ytick.major.right": True,
    "ytick.minor.left": True,
    "ytick.minor.right": True,
    # Grid
    "grid.color": theme_color,
    "grid.linestyle": ":",
    "grid.linewidth": 0.4,
    "grid.alpha": 1.0,
    # Legend
    "legend.frameon": False,
    "legend.edgecolor": theme_color,
    # Figure
    "figure.figsize": (8, 5),
    "figure.dpi": 96,
    "figure.facecolor": "#FFFFFF",
    "figure.edgecolor": "#FFFFFF",
}
