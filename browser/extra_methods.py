"""Extra methods: TABLE, PLOT, SLICES, SCATTER, FIGURE.

Not loaded by lazyfields.browser. Each is a standalone Method (see Method in methods.py) —
import only the ones you want and register them yourself:

    from lazyfields.browser import register_method
    from lazyfields.browser.extra_methods import TABLE
    register_method(TABLE, category="Display/Tables")

PLOT, SLICES, SCATTER and FIGURE need matplotlib (`pip install lazyfields[plot]`); it is only
imported inside their call functions, so importing this module itself never requires it.
"""

from ..readers import as_array
from .methods import Method, _embed_figure, _tk, numeric

# ----- Table -----
# A pandas DataFrame/Series, or a plain 2D/1D array (columns unnamed), as a scrollable table.


def _fill_table(window, value, limit=5000):
    from tkinter import ttk

    from .filesystem import one_line_repr

    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    if type_name in ("pandas.DataFrame", "pandas.Series"):
        import pandas as pd

        if isinstance(value, pd.Series):
            value = value.to_frame(name=value.name if value.name is not None else "value")
        index_name = " / ".join(str(name) for name in value.index.names if name is not None) or "index"
        index, columns = value.index, list(value.columns)
        rows = value.itertuples(index=False, name=None)
    else:
        array = as_array(value)
        array = array.reshape(-1, 1) if array.ndim == 1 else array
        index_name, index, columns, rows = "index", range(len(array)), list(range(array.shape[1])), iter(array)

    ids = [f"c{position}" for position in range(len(columns))]
    table = ttk.Treeview(window, columns=ids, show="tree headings")
    scroll = ttk.Scrollbar(window, orient="vertical", command=table.yview)
    xscroll = ttk.Scrollbar(window, orient="horizontal", command=table.xview)
    table.configure(yscrollcommand=scroll.set, xscrollcommand=xscroll.set)
    table.heading("#0", text=index_name)
    table.column("#0", width=120, stretch=False)
    for key, column in zip(ids, columns):
        table.heading(key, text=str(column))
        table.column(key, width=120, stretch=False)
    for position, row in zip(index[:limit], rows):
        table.insert("", "end", text=one_line_repr(position, 80),
                     values=tuple(one_line_repr(item, 80) for item in row))
    if len(index) > limit:
        ttk.Label(window, text=f"Showing {limit} of {len(index)} rows").pack(side="bottom", anchor="e")
    xscroll.pack(side="bottom", fill="x")
    scroll.pack(side="right", fill="y")
    table.pack(fill="both", expand=True)


TABLE = Method("Table", lambda node: node.type in ("pandas.DataFrame", "pandas.Series") or numeric(node, (1, 2)),
               _tk(_fill_table))


# ----- Plot -----
# Line plot for a 1D numeric array, heatmap with colorbar for 2D.


def _array_figure(value):
    """value as an array without dimensions of length 1, and a figure with axes to show it."""
    import numpy as np
    from matplotlib.figure import Figure

    figure = Figure(layout="constrained")
    return np.squeeze(as_array(value)), figure, figure.add_subplot()


def _heatmap(axes, array, **kwargs):
    """One cell per value, with (0, 0) at the bottom left."""
    return axes.imshow(array, origin="lower", aspect="auto", interpolation="nearest", **kwargs)


def _show_plot(value, attrs):
    array, figure, axes = _array_figure(value)
    if array.ndim == 1:
        axes.plot(array)
    else:
        figure.colorbar(_heatmap(axes, array))
    return figure


PLOT = Method("Plot", lambda node: numeric(node, (1, 2)), _show_plot)


# ----- Slices -----
# A 2D or 3D array's slice along a chosen axis, picked with a slider: a line for 2D, a heatmap
# for 3D — whichever the remaining axes leave, since that count never changes as the chosen
# axis does. Colors (3D) or the y-axis (2D) stay fixed across slices for comparable scale.


def _fill_slices(window, value):
    import tkinter as tk
    from tkinter import ttk

    import numpy as np

    array, figure, axes = _array_figure(value)
    heatmap = array.ndim == 3
    axis = tk.IntVar(value=0)

    if heatmap:
        image = _heatmap(axes, np.take(array, 0, 0), vmin=array.min(), vmax=array.max())
        figure.colorbar(image)
    else:
        line, = axes.plot(np.take(array, 0, 0))
        axes.set_ylim(array.min(), array.max())

    def show(*_):
        index = round(scale.get())
        data = np.take(array, index, axis.get())
        if heatmap:
            image.set_data(data)
        else:
            line.set_ydata(data)
        axes.set_title(f"axis {axis.get()}: {index} / {array.shape[axis.get()] - 1}")
        figure.canvas.draw_idle()

    def cut():
        """Slice along the chosen axis from its first index, showing the whole slice."""
        shape = np.take(array, 0, axis.get()).shape
        if heatmap:
            height, width = shape
            image.set_extent((-0.5, width - 0.5, -0.5, height - 0.5))
            axes.set(xlim=(-0.5, width - 0.5), ylim=(-0.5, height - 0.5))
        else:
            (length,) = shape
            line.set_xdata(np.arange(length))
            axes.set_xlim(-0.5, length - 0.5)
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


SLICES = Method("Slices", lambda node: numeric(node, (2, 3)), _tk(_fill_slices))


# ----- Scatter -----
# A bare (N, 2) or (N, 3) numeric array of point coordinates, one row per point.


def _accepts_scatter(node):
    import numpy as np

    return (node.kind == "dataset" and len(node.shape) == 2 and node.shape[1] in (2, 3)
            and np.dtype(node.dtype).kind in "biuf")


def _show_scatter(value, attrs):
    from matplotlib.figure import Figure

    array = as_array(value)
    figure = Figure(layout="constrained")
    axes = figure.add_subplot(projection="3d" if array.shape[1] == 3 else None)
    axes.scatter(*array.T)
    axes.set_aspect("equal")
    return figure


SCATTER = Method("Scatter", _accepts_scatter, _show_scatter)


# ----- Figure -----
# Embed an already-built matplotlib Figure as-is; nothing to compute, only to display.


FIGURE = Method("Figure", lambda node: node.kind == "dataset" and node.type == "matplotlib.figure.Figure",
                _tk(_embed_figure))
