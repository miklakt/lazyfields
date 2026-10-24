"""Dataset viewers shared by all browsers; the most recently registered suitable viewer wins."""

from importlib.util import find_spec
from pprint import pformat

_viewers = {}


def register_viewer(name, accepts, show):
    """Offer show(window, value) for datasets where accepts(value) is true.

    window is a fresh Toplevel titled with the dataset path. Registering a name again replaces it.
    """
    _viewers.pop(name, None)
    _viewers[name] = accepts, show


def suitable_viewers(value):
    """Names and show functions of viewers accepting value, most recently registered first."""
    suitable = []
    for name in reversed(_viewers):
        accepts, show = _viewers[name]
        try:
            if accepts(value):
                suitable.append((name, show))
        except Exception:
            pass
    return suitable


def _show_text(window, value):
    from tkinter.scrolledtext import ScrolledText

    text = ScrolledText(window, width=100, height=30, wrap="word")
    text.pack(fill="both", expand=True)
    text.insert("1.0", pformat(value, width=100, sort_dicts=False))
    text.configure(state="disabled")


def _plottable(value):
    import numpy as np

    try:
        array = np.squeeze(np.asarray(value))
    except Exception:
        return None
    return array if array.dtype.kind in "biuf" and array.ndim in (1, 2) else None


def _show_plot(window, value):
    """Line plot for 1D, heatmap with colorbar for 2D; (0, 0) is bottom left."""
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure

    array = _plottable(value)
    figure = Figure(layout="constrained")
    axes = figure.add_subplot()
    if array.ndim == 1:
        axes.plot(array)
    else:
        figure.colorbar(axes.imshow(array, origin="lower", aspect="auto", interpolation="nearest"))
    canvas = FigureCanvasTkAgg(figure, master=window)
    NavigationToolbar2Tk(canvas, window).pack(side="bottom", fill="x")
    canvas.get_tk_widget().pack(fill="both", expand=True)
    canvas.draw()


register_viewer("Text", lambda value: True, _show_text)
if find_spec("matplotlib") is not None:
    register_viewer("Plot", lambda value: _plottable(value) is not None, _show_plot)
