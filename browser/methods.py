"""Method registration, selection, and display.

Only Text registers here, plus the registry machinery itself; Browser (browser/filesystem.py)
registers alongside it. Everything shape-specific (Table, Plot, Slices, Scatter, Figure) lives
in browser/extra_methods.py as plain Method instances you import and register yourself.
"""

from dataclasses import dataclass, replace
from inspect import Parameter, signature
from pprint import pformat
from typing import Any, Callable

from ..filesystem import FileSystem
from ..readers import NodeInfo, value_info
from .arguments import ask_arguments

_methods = {}
_method_categories: dict[str, str] = {}
_specific_methods: set[str] = set()
_filled: dict[Callable, dict[str, str]] = {}


def show_error(parent, title, exc):
    """A standard error dialog for an exception, so every Tk entry point reports failures the same way."""
    from tkinter import messagebox
    messagebox.showerror(title, f"{type(exc).__name__}: {exc}", parent=parent)


@dataclass(frozen=True)
class Method:
    """Handles nodes accepted by accepts(node: NodeInfo) using call(value, attrs, **kwargs).

    call returns a matplotlib Figure, a function filling a Tk frame, None when it opens its own
    window, or any other value — handled in turn by whichever method suits it, same as a registered
    store method's result. A plain computation needs no Tk or matplotlib: return its data and let
    another method display it.

    """

    name: str
    accepts: Callable[[NodeInfo], bool]
    call: Callable[..., Any]


def register_method(method: Method | str, accepts=None, call=None, *, specific=False,
                    category: str | None = None) -> Method:
    """Register a Method, building one first if given (name, accepts, call) instead.

    Re-registering a name replaces it. category groups it in a filesystem menu; specific makes it
    win over a looser matching method. Neither option becomes part of the Method itself.
    """
    if not isinstance(method, Method):
        method = Method(method, accepts, call)
    _methods.pop(method.name, None)
    _methods[method.name] = method
    if category is None:
        _method_categories.pop(method.name, None)
    else:
        _method_categories[method.name] = category
    if specific:
        _specific_methods.add(method.name)
    else:
        _specific_methods.discard(method.name)
    return method


def _method_category(method: Method) -> str | None:
    """Return menu metadata supplied when this method was registered."""
    return _method_categories.get(method.name)


def suitable_methods(node):
    """Return accepted methods: an explicit per-node preference, then specific ones, Text last."""
    suitable = []
    for method in _methods.values():
        try:
            if method.accepts(node):
                suitable.append(method)
        except Exception:
            pass
    preferred = node.attrs.get("method")
    return sorted(suitable, key=lambda method: (method.name != preferred, method.call is _TEXT,
                                                method.name not in _specific_methods))


def open_value(master, title, value, call=None, attrs=None):
    """Show a value using the chosen or preferred method's call(), with a text toggle.

    A call may take extra keyword-only parameters after (value, attrs); their defaults are then
    offered in a form before showing, same as a registered store method's arguments.
    """
    import tkinter as tk
    from tkinter import ttk

    node = replace(value_info(value), attrs=attrs or {})
    if call is None:
        method = suitable_methods(node)[0]
        call, name = method.call, method.name
    else:
        name = next((method.name for method in _methods.values() if method.call is call), "Method")
    parameters = [parameter for parameter in list(signature(call).parameters.values())[2:]
                  if parameter.kind is Parameter.KEYWORD_ONLY]

    def show(args, kwargs, text):
        try:
            shown = call(value, node.attrs, *args, **kwargs)
        except Exception as exc:
            show_error(master, title, exc)
            return
        if shown is None:  # Shown in a window of its own.
            return
        is_figure = type(shown).__module__ == "matplotlib.figure"
        if not is_figure and not callable(shown):
            # Not a rendering: a plain result, handled by whichever method suits it instead. A group
            # result browses first unless some method specifically claims it instead.
            open_value(master, title, shown)
            return
        window = tk.Toplevel(master)
        window.title(title)
        frame = ttk.Frame(window)
        if is_figure:
            _embed_figure(frame, shown)
        else:
            shown(frame)
        if call is not _TEXT:
            text_frame = ttk.Frame(window)

            def toggle():
                if not text_frame.winfo_children():
                    _show_text(text_frame, value)
                current, other = (frame, text_frame) if frame.winfo_manager() else (text_frame, frame)
                window.geometry(window.geometry())  # Preserve the window size while switching views.
                current.pack_forget()
                other.pack(fill="both", expand=True)
                button.configure(text=name if other is text_frame else "Text")

            bar = ttk.Frame(window)
            button = ttk.Button(bar, text="Text", width=7, command=toggle)
            button.pack(side="right")
            bar.pack(side="top", fill="x")
        frame.pack(fill="both", expand=True)

    if parameters:
        ask_arguments(master, title, parameters, _filled.setdefault(call, {}), show)
    else:
        show([], {}, "")


def _tk(fill):
    """A method filling the browser's window with Tk widgets through fill(frame, value)."""
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
register_method("Text", lambda node: True, _TEXT)
