"""Viewer registration, selection, and display."""

from dataclasses import replace
from importlib.util import find_spec
from pprint import pformat

from ..filesystem import FileSystem
from ..readers import as_array, value_info

_viewers = {}


def register_viewer(name, accepts, view):
    """Register view(value, attrs) for nodes accepted by accepts(node).

    accepts receives a NodeInfo. view returns a matplotlib Figure, a function
    filling a Tk frame, or None when it opens its own window.
    Re-registering a name replaces it at highest priority.
    """
    _viewers.pop(name, None)
    _viewers[name] = accepts, view


def suitable_viewers(node):
    """Return accepted viewers, preferred first, then newest registered."""
    suitable = []
    for name, (accepts, view) in reversed(_viewers.items()):
        try:
            if accepts(node):
                suitable.append((name, view))
        except Exception:
            pass
    return sorted(suitable, key=lambda viewer: viewer[0] != node.attrs.get("viewer"))


def open_value(master, title, value, view=None, attrs=None):
    """Show a value using the chosen or preferred viewer, with a text toggle."""
    import tkinter as tk
    from tkinter import messagebox, ttk

    node = replace(value_info(value), attrs=attrs or {})
    view = view or suitable_viewers(node)[0][1]
    window = tk.Toplevel(master)
    window.title(title)
    frame = ttk.Frame(window)
    try:
        shown = view(value, node.attrs)
        if type(shown).__module__ == "matplotlib.figure":
            _embed_figure(frame, shown)
        elif shown is not None:
            shown(frame)
    except Exception as exc:
        window.destroy()
        messagebox.showerror(title, f"{type(exc).__name__}: {exc}", parent=master)
        return
    if shown is None:  # Shown in a window of its own.
        window.destroy()
        return
    if view is not _TEXT:
        text = ttk.Frame(window)

        def toggle():
            if not text.winfo_children():
                _show_text(text, value)
            current, other = (frame, text) if frame.winfo_manager() else (text, frame)
            window.geometry(window.geometry())  # Preserve the window size while switching views.
            current.pack_forget()
            other.pack(fill="both", expand=True)
            button.configure(text="Viewer" if other is text else "Text")

        bar = ttk.Frame(window)
        button = ttk.Button(bar, text="Text", width=7, command=toggle)
        button.pack(side="right")
        bar.pack(side="top", fill="x")
    frame.pack(fill="both", expand=True)


def _tk(fill):
    """A viewer filling the browser's window with Tk widgets through fill(frame, value)."""
    return lambda value, attrs: lambda frame: fill(frame, value)


def _embed_figure(window, figure):
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    canvas = FigureCanvasTkAgg(figure, master=window)
    NavigationToolbar2Tk(canvas, window).pack(side="bottom", fill="x")
    canvas.get_tk_widget().pack(fill="both", expand=True)
    canvas.draw()


def _show_text(window, value):
    from tkinter.scrolledtext import ScrolledText

    if isinstance(value, FileSystem):  # A lazy source, read only once text is actually requested.
        value = value.read()
    text = ScrolledText(window, width=100, height=30, wrap="word")
    text.pack(fill="both", expand=True)
    text.insert("1.0", value if isinstance(value, str) else pformat(value, width=100, sort_dicts=False))
    text.configure(state="disabled")


def _numeric(node, ndims):
    """Whether node is a dataset of numbers with one of ndims dimensions longer than 1."""
    import numpy as np

    return (node.kind == "dataset" and np.dtype(node.dtype).kind in "biuf" and all(node.shape)
            and sum(size != 1 for size in node.shape) in ndims)


def _array_figure(value):
    """value as an array without dimensions of length 1, and a figure with axes to show it."""
    import numpy as np
    from matplotlib.figure import Figure

    figure = Figure(layout="constrained")
    return np.squeeze(as_array(value)), figure, figure.add_subplot()


def _heatmap(axes, array, **kwargs):
    """One cell per value, with (0, 0) at the bottom left."""
    return axes.imshow(array, origin="lower", aspect="auto", interpolation="nearest", **kwargs)


def _plot(value, attrs):
    """Line plot for 1D, heatmap with colorbar for 2D."""
    array, figure, axes = _array_figure(value)
    if array.ndim == 1:
        axes.plot(array)
    else:
        figure.colorbar(_heatmap(axes, array))
    return figure


def _show_slices(window, value):
    """Heatmap of a 3D array's slice along a chosen axis, picked with a slider; colors fixed across slices."""
    import tkinter as tk
    from tkinter import ttk

    import numpy as np

    array, figure, axes = _array_figure(value)
    image = _heatmap(axes, array[0], vmin=array.min(), vmax=array.max())
    figure.colorbar(image)
    axis = tk.IntVar(value=0)

    def show(*_):
        index = round(scale.get())
        image.set_data(np.take(array, index, axis.get()))
        axes.set_title(f"axis {axis.get()}: {index} / {array.shape[axis.get()] - 1}")
        figure.canvas.draw_idle()

    def cut():
        """Slice along the chosen axis from its first index, showing the whole slice."""
        height, width = np.take(array, 0, axis.get()).shape
        image.set_extent((-0.5, width - 0.5, -0.5, height - 0.5))
        axes.set(xlim=(-0.5, width - 0.5), ylim=(-0.5, height - 0.5))
        scale.configure(to=array.shape[axis.get()] - 1)
        scale.set(0)
        show()

    bar = ttk.Frame(window)
    ttk.Label(bar, text="Axis").pack(side="left")
    for number, size in enumerate(array.shape):
        ttk.Radiobutton(bar, text=f"{number} ({size})", value=number, variable=axis,
                        command=cut).pack(side="left")
    scale = ttk.Scale(bar, from_=0, to=len(array) - 1, command=show)
    scale.pack(side="left", fill="x", expand=True, padx=(8, 0))
    bar.pack(side="top", fill="x")
    _embed_figure(window, figure)
    show()


_TEXT = _tk(_show_text)
register_viewer("Text", lambda node: True, _TEXT)
if find_spec("matplotlib") is not None:
    register_viewer("Plot", lambda node: _numeric(node, (1, 2)), _plot)
    register_viewer("Slices", lambda node: _numeric(node, (3,)), _tk(_show_slices))
    register_viewer("Figure", lambda node: node.type == "matplotlib.figure.Figure", _tk(_embed_figure))
