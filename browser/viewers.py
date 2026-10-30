"""Viewer registration, selection, and display.

Only Text registers here, plus the registry machinery itself; Browser (browser/filesystem.py)
registers alongside it. Everything shape-specific (Table, Plot, Slices, Scatter, Figure) lives
in browser/extra_viewers.py as plain Viewer instances you import and register yourself.
"""

from dataclasses import dataclass, replace
from pprint import pformat
from typing import Any, Callable

from ..filesystem import FileSystem
from ..readers import NodeInfo, value_info

_viewers = {}


def show_error(parent, title, exc):
    """A standard error dialog for an exception, so every Tk entry point reports failures the same way."""
    from tkinter import messagebox
    messagebox.showerror(title, f"{type(exc).__name__}: {exc}", parent=parent)


@dataclass(frozen=True)
class Viewer:
    """Shows nodes accepted by accepts(node: NodeInfo) using view(value, attrs).

    view returns a matplotlib Figure, a function filling a Tk frame, or None when it opens its
    own window.
    """

    name: str
    accepts: Callable[[NodeInfo], bool]
    view: Callable[..., Any]


def register_viewer(viewer: Viewer | str, accepts=None, view=None) -> Viewer:
    """Register a Viewer, building one first if given (name, accepts, view) instead.

    Re-registering a name replaces it at highest priority.
    """
    if not isinstance(viewer, Viewer):
        viewer = Viewer(viewer, accepts, view)
    _viewers.pop(viewer.name, None)
    _viewers[viewer.name] = viewer
    return viewer


def suitable_viewers(node):
    """Return accepted viewers, preferred first, then newest registered."""
    suitable = []
    for viewer in reversed(_viewers.values()):
        try:
            if viewer.accepts(node):
                suitable.append(viewer)
        except Exception:
            pass
    return sorted(suitable, key=lambda viewer: viewer.name != node.attrs.get("viewer"))


def open_value(master, title, value, view=None, attrs=None):
    """Show a value using the chosen or preferred viewer's view(), with a text toggle."""
    import tkinter as tk
    from tkinter import ttk

    node = replace(value_info(value), attrs=attrs or {})
    view = view or suitable_viewers(node)[0].view
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
        show_error(master, title, exc)
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


def numeric(node, ndims):
    """Whether node is a dataset of numbers with one of ndims dimensions longer than 1."""
    import numpy as np

    return (node.kind == "dataset" and np.dtype(node.dtype).kind in "biuf" and all(node.shape)
            and sum(size != 1 for size in node.shape) in ndims)


_TEXT = _tk(_show_text)
register_viewer("Text", lambda node: True, _TEXT)
