"""Viewers shared by all browsers; the most recently registered suitable viewer wins."""

from importlib.util import find_spec
from pprint import pformat
from reprlib import repr as short_repr

from ..readers import Reader, value_kind

_viewers = {}


def register_viewer(name, accepts, show):
    """Offer show(window, value) for values, such as datasets, where accepts(value) is true.

    window is a fresh Toplevel titled with what is shown. Registering a name again replaces it.
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


def preview(value, width=160):
    """A one-line preview; floats, NumPy ones too, show up to 8 significant digits, such as 1.2345678e-12."""
    if getattr(value, "shape", None) == () and hasattr(value, "item"):  # A NumPy scalar or 0-D array.
        value = value.item()
    if isinstance(value, float):
        return f"{value:.8g}"
    return short_repr(value)[:width]


def type_name(value):
    """A dtype with any shape, such as float64 or float64 76×160; else the Python type, such as float.

    Values from JSON are plain Python values, so they show Python types; nested lists show their first
    elements' type and lengths like an array, such as list[float] 76×160.
    """
    dtype, shape = getattr(value, "dtype", None), getattr(value, "shape", None)
    if dtype is not None:
        return f"{dtype} {'×'.join(map(str, shape))}" if shape else str(dtype)
    if value is None:
        return "None"
    if not isinstance(value, (list, tuple)):
        return type(value).__name__
    lengths, item = [], value
    while isinstance(item, (list, tuple)):
        lengths.append(len(item))
        if not item:
            break
        item = item[0]
    element = "" if isinstance(item, (list, tuple)) else f"[{type_name(item)}]"
    return f"{type(value).__name__}{element} {'×'.join(map(str, lengths))}"


def open_value(master, title, value, show=None):
    """Show value in a new window with show, or with its preferred viewer; report a failing viewer.

    A registered viewer other than Text gets a button above it that toggles between it and the value as
    text; both keep their state, such as a plot's zoom.
    """
    import tkinter as tk
    from tkinter import messagebox, ttk

    window = tk.Toplevel(master)
    window.title(title)
    viewers = suitable_viewers(value)
    show = show or viewers[0][1]
    view = ttk.Frame(window)
    try:
        show(view, value)
    except Exception as exc:
        window.destroy()
        messagebox.showerror(title, f"{type(exc).__name__}: {exc}", parent=master)
        return
    if show is not _show_text and any(viewer is show for _, viewer in viewers):  # Not an embedded figure.
        text = ttk.Frame(window)

        def toggle():
            if not text.winfo_children():
                _show_text(text, value)
            current, other = (view, text) if view.winfo_manager() else (text, view)
            window.geometry(window.geometry())  # Keep the window's size rather than fit the other view.
            current.pack_forget()
            other.pack(fill="both", expand=True)
            button.configure(text="Viewer" if other is text else "Text")

        bar = ttk.Frame(window)
        button = ttk.Button(bar, text="Text", width=7, command=toggle)
        button.pack(side="right")
        bar.pack(side="top", fill="x")
    view.pack(fill="both", expand=True)


def _show_text(window, value):
    from tkinter.scrolledtext import ScrolledText

    text = ScrolledText(window, width=100, height=30, wrap="word")
    text.pack(fill="both", expand=True)
    text.insert("1.0", pformat(value, width=100, sort_dicts=False))
    text.configure(state="disabled")


def _plottable(value):
    import numpy as np

    try:
        array = np.asarray(value)
        if array.dtype.kind == "O":  # Such as a per-row Series of numbers; infer the actual dtype.
            array = np.asarray(array.tolist())
        array = np.squeeze(array)
    except Exception:
        return None
    return array if array.dtype.kind in "biuf" and array.ndim in (1, 2) else None


def embed_figure(window, figure):
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    canvas = FigureCanvasTkAgg(figure, master=window)
    NavigationToolbar2Tk(canvas, window).pack(side="bottom", fill="x")
    canvas.get_tk_widget().pack(fill="both", expand=True)
    canvas.draw()


def _show_plot(window, value):
    """Line plot for 1D, heatmap with colorbar for 2D; (0, 0) is bottom left."""
    from matplotlib.figure import Figure

    array = _plottable(value)
    figure = Figure(layout="constrained")
    axes = figure.add_subplot()
    if array.ndim == 1:
        axes.plot(array)
    else:
        figure.colorbar(axes.imshow(array, origin="lower", aspect="auto", interpolation="nearest"))
    embed_figure(window, figure)


def _show_tree(window, value):
    """Browse a mapping, or a list of mappings, like a filesystem group."""
    from ..filesystem import FileSystem
    from .filesystem import panes

    reader = Reader(suffixes=(), load=lambda _: value)
    panes(window, FileSystem.from_file("<value>", reader))("/")


register_viewer("Text", lambda value: True, _show_text)
if find_spec("matplotlib") is not None:
    register_viewer("Plot", lambda value: _plottable(value) is not None, _show_plot)
register_viewer("Browser", lambda value: value_kind(value) == "group", _show_tree)
