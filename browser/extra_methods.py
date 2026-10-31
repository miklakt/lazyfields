"""Extra methods: viewers (TABLE, PLOT, SCATTER, FIGURE, IMAGE) and actions - array transforms
(Split, Swap axes, Transpose, Flatten, Squeeze, in the ACTIONS tuple).

Not loaded by lazyfields.browser. The viewers are standalone Methods you import and register one by
one if you like (see Method in methods.py):

    from lazyfields.browser import register_viewer
    from lazyfields.browser.extra_methods import TABLE
    register_viewer(TABLE, category="Display/Tables")

or register whole bundles at once (all at the bottom of this module): register_viewers() for the
viewers, register_actions() for the actions, or register_extras() for both.

PLOT, SCATTER, FIGURE and IMAGE need matplotlib (`pip install lazyfields[plot]`); the actions
need only numpy. Every heavy import happens inside a call function, so importing this module itself
never requires either.
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


IMAGE = Method("RGB Image", _accepts_rgb, _show_rgb)


# ----- Register all -----

VIEWERS = (TABLE, PLOT, SCATTER, FIGURE, IMAGE)

# Default menu placement: Table, Figure and RGB Image stay top-level (each just displays its value
# as-is); the methods that compute a chart from numbers group under "Charts".
_VIEWER_CATEGORIES = {"Table": None, "Plot": "Charts",
                      "Scatter": "Charts", "Figure": None, "RGB Image": None}

# RGB Image wins over Plot for an (H, W, 3/4) array, which both accept; pick Plot from the menu when
# such an array is really a scalar field to scroll through rather than a picture.
_VIEWER_SPECIFIC = {"RGB Image"}


def register_viewers(**options):
    """Register every built-in viewer in one call, after importing lazyfields.browser.

    Each lands in its default menu category (Table/Figure/RGB Image top-level, the chart methods
    under "Charts") and RGB Image registers as specific so it beats Plot on an image-shaped array.
    Pass category=/specific= to override for all of them, or register individually for per-method
    control. Registering only stores each Method, so this stays cheap and needs no matplotlib; the
    plotting ones import it only when actually used. Other options pass through unchanged.

    The actions are a separate bundle: call register_actions() for those.
    """
    from .methods import register_method
    for method in VIEWERS:
        per = dict(options)
        per.setdefault("category", _VIEWER_CATEGORIES.get(method.name))
        per.setdefault("specific", method.name in _VIEWER_SPECIFIC)
        register_method(method, **per)
    return VIEWERS


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


# view=False: each is offered in the menu but is never a value's default - opening a value shows it.
ACTIONS = (
    Method("Split", lambda node: _dims(node) >= 1, _split, view=False),
    Method("Swap axes", lambda node: _dims(node) >= 2, _swap_axes, view=False),
    Method("Transpose", lambda node: _dims(node) >= 2, _numpy("transpose"), view=False),
    Method("Flatten", lambda node: _dims(node) >= 2, _numpy("ravel"), view=False),
    Method("Squeeze", lambda node: node.kind == "dataset" and 1 in (node.shape or ()),
           _numpy("squeeze"), view=False),
)


def register_actions(*, category=None, **options):
    """Register the built-in actions (Split, Swap axes, Transpose, Flatten, Squeeze) as a bundle.

    Each is view=False (see Method), so it lands in a value window's Action menu - which groups them
    already, hence no category by default - and is never a value's default. Each returns a new array
    (Split a group of slices) that the browser then displays with whatever method suits it. numpy only.
    """
    from .methods import register_method
    for method in ACTIONS:
        register_method(method, category=category, **options)
    return ACTIONS


def register_extras():
    """Register every extra method - viewers and actions - in their default categories.

    The one call the CLI browsers use; prefer register_viewers()/register_actions(), or register
    individual Methods, when you want only some of them or different categories.
    """
    return (*register_viewers(), *register_actions())
