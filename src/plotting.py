"""Shared plotting style: price axes in $k, legends outside the data area."""
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

BLUE, RED, GREEN, GRAY = "#4C72B0", "#C44E52", "#55A868", "#8C8C8C"

usd_k = FuncFormatter(lambda x, _: f"{'-' if x < 0 else ''}${abs(x) / 1000:,.0f}k")


def setup_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"figure.dpi": 100, "savefig.bbox": "tight", "axes.titlesize": 11})


def legend_outside(ax, **kw):
    """Put the legend to the right of the axes so it never covers data."""
    return ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, **kw)
