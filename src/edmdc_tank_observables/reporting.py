"""Figure style and figure output shared by the experiment scripts.

Figures are saved as PDF (vector, for the paper) and PNG (for a quick look).
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


def set_style() -> None:
    """Apply the common matplotlib style of the figures.

    Font sizes readable at the reduced size a figure has in the paper, a light
    grid and no top or right spines.
    """
    mpl.rcParams.update(
        {
            "figure.figsize": (7.0, 4.0),
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path | str, name: str) -> list[Path]:
    """Save a figure as PDF and PNG and close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    out_dir : pathlib.Path or str
        Folder to write in. It is created if it does not exist.
    name : str
        File name without extension.

    Returns
    -------
    list of pathlib.Path
        Paths of the PDF and PNG files written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("pdf", "png"):
        path = out_dir / f"{name}.{extension}"
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    return paths
