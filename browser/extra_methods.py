"""Extra methods: viewers (TABLE, PLOT, SCATTER, FIGURE, IMAGE) and array transforms.

Exported by lazyfields.browser. Combine the standalone methods or bundles you need, placing each with
Method.set(category=..., priority=...) as you go:

    from lazyfields.browser import DEFAULT_METHODS, TABLE, browse_filesystem
    methods = DEFAULT_METHODS | TABLE.set(category="Display/Tables")
    browse_filesystem(filesystem, methods=methods)

or include every viewer and array action:

    from lazyfields.browser import ARRAY_ACTIONS, EXTRA_VIEWERS
    methods = DEFAULT_METHODS | EXTRA_VIEWERS | ARRAY_ACTIONS

PLOT, SCATTER, FIGURE and IMAGE need matplotlib (`pip install lazyfields[plot]`); the actions
need only numpy. Every heavy import happens inside a call function, so importing this module itself
never requires either.
"""

from ..readers import as_array
from .methods import Method, Methods, _embed_figure, _tk, numeric

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


def _is_dataframe(node):
    return node.type in ("pandas.DataFrame", "pandas.Series")


# A reconstructed DataFrame still reports its pickle internals too, so it also matches Browser's
# deliberately loose "any group" catch-all; priority 3 makes Table win that tie. A plain numeric array
# is just one option among Plot and the rest, so Table stays at the default rank there.
TABLE = Method("Table", lambda node: _is_dataframe(node) or numeric(node, (1, 2)), _tk(_fill_table),
               priority=lambda node: 3 if _is_dataframe(node) else 0)


# ----- Plot -----
# A numeric array as a line (1D) or a heatmap (2D). A 3D array is a heatmap of one 2D slice, swept
# along a chosen axis with a slider; reorder axes first with a transform to sweep a different one.


def _array_figure(value):
    """value as an array without dimensions of length 1, and a figure with axes to show it."""
    import numpy as np
    from matplotlib.figure import Figure

    figure = Figure(layout="constrained")
    return np.squeeze(as_array(value)), figure, figure.add_subplot()


def _heatmap(axes, array, **kwargs):
    """One cell per value, with (0, 0) at the bottom left."""
    return axes.imshow(array, origin="lower", aspect="auto", interpolation="nearest", **kwargs)


def _fill_plot(window, value):
    array, figure, axes = _array_figure(value)
    if array.ndim == 1:
        axes.plot(array)
    elif array.ndim == 2:
        figure.colorbar(_heatmap(axes, array))
    else:
        return _fill_scrubber(window, array, figure, axes)
    _embed_figure(window, figure)


def _fill_scrubber(window, array, figure, axes):
    """A 3D array as a heatmap of one 2D slice, swept along a chosen axis. Colours stay fixed across
    slices for a comparable scale."""
    import tkinter as tk
    from tkinter import ttk

    import numpy as np

    axis = tk.IntVar(value=0)
    image = _heatmap(axes, np.take(array, 0, 0), vmin=array.min(), vmax=array.max())
    figure.colorbar(image)

    def show(*_):
        index = round(scale.get())
        image.set_data(np.take(array, index, axis.get()))
        axes.set_title(f"axis {axis.get()}: {index} / {array.shape[axis.get()] - 1}")
        figure.canvas.draw_idle()

    def cut():
        """Sweep the chosen axis from its first index, fitting the view to the slice it leaves."""
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
    scale = ttk.Scale(bar, from_=0, to=array.shape[0] - 1, command=show)
    scale.pack(side="left", fill="x", expand=True, padx=(8, 0))
    bar.pack(side="top", fill="x")
    _embed_figure(window, figure)
    show()


PLOT = Method("Plot", lambda node: numeric(node, (1, 2, 3)), _tk(_fill_plot))


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


# ----- RGB Image -----
# An (H, W, 3) or (H, W, 4) numeric array shown as pixels, not plotted: row 0 at the top, square
# pixels, no axes. Takes uint8 (0-255) or float (0-1) channels, as matplotlib's imshow expects.


def _accepts_rgb(node):
    import numpy as np

    return (node.kind == "dataset" and len(node.shape) == 3 and node.shape[2] in (3, 4)
            and np.dtype(node.dtype).kind in "biuf")


def _show_rgb(value, attrs):
    from matplotlib.figure import Figure

    figure = Figure(layout="constrained")
    axes = figure.add_subplot()
    axes.imshow(as_array(value))  # origin="upper" by default, so row 0 sits at the top like an image.
    axes.set_axis_off()
    return figure


# priority 3: RGB Image wins over Plot for an (H, W, 3/4) array, which both accept - a picture by
# default. Pick Plot from the menu when such an array is really a scalar field to scroll through.
IMAGE = Method("RGB Image", _accepts_rgb, _show_rgb, priority=3)


# ----- Viewer bundle -----

# Categories are this bundle's menu layout, not the methods' own - another browser places them
# differently. Table, Figure and RGB Image stay at the menu root; the chart methods group under Charts.
EXTRA_VIEWERS = (Methods()
                 | TABLE
                 | PLOT.set(category="Charts")
                 | SCATTER.set(category="Charts")
                 | FIGURE
                 | IMAGE)


# ----- Actions -----
# Reshape or re-slice an array and hand the result straight back to the browser: each returns plain
# data, not a rendering, so open_value re-dispatches it to whatever method suits the new shape (a
# reshaped array to Plot/Table, Split's dict of slices to Browser), raw Text still a click
# away. numpy only, imported when the action runs; axis arguments are offered in a form first.


def _dims(node):
    """How many axes an array-like dataset node has, 0 for anything that is not one."""
    return len(node.shape) if node.kind == "dataset" and node.shape else 0


def _numpy(name):
    """A no-argument action calling np.<name> on the value's array, numpy imported on use."""
    def call(value, attrs):
        import numpy as np
        return getattr(np, name)(as_array(value))
    return call


# The two that take axis arguments keep real keyword-only parameters, so open_value offers them in a
# form; the rest are one-liners built from _numpy. Each returns plain data for the browser to re-open.
def _split(value, attrs, *, axis: int = 0):
    """An n-D array as a group of its (n-1)-D slices along axis, keyed by index along that axis."""
    import numpy as np
    array = as_array(value)
    return {str(index): np.take(array, index, axis) for index in range(array.shape[axis])}


def _swap_axes(value, attrs, *, axis1: int = 0, axis2: int = 1):
    import numpy as np
    return np.swapaxes(as_array(value), axis1, axis2)


# Actions are offered in the menu but never become a value's default.
ARRAY_ACTIONS = Methods((
    Method("Split", lambda node: _dims(node) >= 1, _split, kind="action"),
    Method("Swap axes", lambda node: _dims(node) >= 2, _swap_axes, kind="action"),
    Method("Transpose", lambda node: _dims(node) >= 2, _numpy("transpose"), kind="action"),
    Method("Flatten", lambda node: _dims(node) >= 2, _numpy("ravel"), kind="action"),
    Method("Squeeze", lambda node: node.kind == "dataset" and 1 in (node.shape or ()),
           _numpy("squeeze"), kind="action"),
))
