"""
SciML Book — Plotting Helpers

Standard colors, line styles, and convenience functions for consistent figures.

Usage:
    from sciml_plots import setup_style, plot_exact, plot_prediction, plot_data, savefig

    setup_style()
    fig, ax = plt.subplots()
    plot_exact(ax, x, u_exact)
    plot_prediction(ax, x, u_pred)
    plot_data(ax, x_data, u_data)
    savefig(fig, 'my-figure')
"""

import matplotlib.pyplot as plt
from pathlib import Path

# Standard colors and styles
PREDICTION = dict(color='tab:blue', linewidth=2.5, zorder=4)
EXACT = dict(color='#333333', linestyle='--', linewidth=2.0, zorder=5)
DATA = dict(color='tab:red', zorder=6, s=80, edgecolors='white', linewidths=0.5)

STYLE_PATH = Path(__file__).parent / 'sciml_style.mplstyle'


def setup_style():
    """Load the SciML book matplotlib style."""
    plt.style.use(str(STYLE_PATH))


def plot_exact(ax, x, u, label='Exact', **kwargs):
    """Plot the exact/reference solution."""
    kw = {**EXACT, **kwargs}
    return ax.plot(x, u, label=label, **kw)


def plot_prediction(ax, x, u, label='PINN', **kwargs):
    """Plot the neural network prediction."""
    kw = {**PREDICTION, **kwargs}
    return ax.plot(x, u, label=label, **kw)


def plot_data(ax, x, u, label='Data', **kwargs):
    """Plot training data as scatter points."""
    kw = {**DATA, **kwargs}
    return ax.scatter(x, u, label=label, **kw)


def savefig(fig, name, path=None):
    """Save figure as PDF (vector).

    Parameters
    ----------
    fig : matplotlib Figure
    name : str
        Filename without extension.
    path : str or Path, optional
        Directory to save in. Defaults to current directory.
    """
    if path is None:
        path = Path('.')
    else:
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / f'{name}.pdf')
