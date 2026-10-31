"""Method registration, selection, and display.

Only Text registers here, plus the registry machinery itself; Browser (browser/filesystem.py)
registers alongside it. Everything shape-specific (Table, Plot, Scatter, Figure, RGB Image) lives
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

    view is whether the method shows the value as it is (the default). A method that instead derives
    a new value - a transform - sets view=False, so it is offered in the menu but never picked as a
    value's default: opening a value shows it, it never silently runs a transform.
    """

    name: str
    accepts: Callable[[NodeInfo], bool]
    call: Callable[..., Any]
    view: bool = True


def register_method(method: Method | str, accepts=None, call=None, *, specific=False,
                    category: str | None = None, view: bool | None = None) -> Method:
    """Register a Method, building one first if given (name, accepts, call) instead.

    Re-registering a name replaces it. category groups it in a filesystem menu; specific makes it win
    over a looser matching method; neither becomes part of the Method. view is intrinsic to the Method
    (see Method.view) so it travels when a Method is reused; pass view=False here to register a custom
    transform that returns a new value - it is then offered in the menu but never a default.
    """
    if not isinstance(method, Method):
        method = Method(method, accepts, call, view=True if view is None else view)
    elif view is not None:
        method = replace(method, view=view)
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


def register_viewer(method: Method | str, accepts=None, call=None, *, specific=False,
                    category: str | None = None) -> Method:
    """Register a view: a method that shows a value as it is. The common case, named for intent -
    sugar for register_method(..., view=True). Use register_action for one that derives a value."""
    return register_method(method, accepts, call, specific=specific, category=category, view=True)


def register_action(method: Method | str, accepts=None, call=None, *, specific=False,
                    category: str | None = None) -> Method:
    """Register an action: a method that returns a new value from the old (a transform). It lands in a
    value window's Action menu, never chosen as a default. Sugar for register_method(..., view=False)."""
    return register_method(method, accepts, call, specific=specific, category=category, view=False)


def _method_category(method: Method) -> str | None:
    """Return menu metadata supplied when this method was registered."""
    return _method_categories.get(method.name)


def suitable_methods(node):
    """Return accepted methods in default-first order: an explicit per-node preference, then specific
    views, other views, Text, and finally non-views (transforms) - which are never a default view."""
    suitable = []
    for method in _methods.values():
        try:
            if method.accepts(node):
                suitable.append(method)
        except Exception:
            pass
    preferred = node.attrs.get("method")
    return sorted(suitable, key=lambda method: (method.name != preferred, not method.view,
                                                method.call is _TEXT, method.name not in _specific_methods))


def open_value(master, title, value, call=None, attrs=None):
    """Show a value, with a menu bar to re-show it any other way it accepts.

    Two menus list the same methods the tree offers on a node: View (the current one radio-marked)
    re-renders the value in place; Action runs a transform, opening its result in a new window that
    carries its own menu bar - so results chain. Either menu is dropped when it would be empty or
    offer no choice. A call's extra keyword-only parameters after (value, attrs) are offered in a form
    first, same as a registered store method's arguments.
    """
    import tkinter as tk
    from tkinter import ttk

    from .filesystem import build_method_menu

    # A browsed group arrives as its own lazy FileSystem; value_info would misread it as a dataset,
    # so ask the source itself what it is, leaving value_info for actual in-memory values.
    info = value.info() if isinstance(value, FileSystem) else value_info(value)
    node = replace(info, attrs=attrs or {})
    suitable = suitable_methods(node)
    if call is None:
        call = suitable[0].call
    state = {"window": None, "content": None, "name": None, "selected": None}

    def build_window():
        window = state["window"] = tk.Toplevel(master)
        window.title(title)
        menubar = tk.Menu(window, tearoff=False)
        views = [method for method in suitable if method.view]
        actions = [method for method in suitable if not method.view]
        if len(views) > 1:  # A lone view (plain text, say) has nothing to switch to.
            menu = tk.Menu(menubar, tearoff=False)
            state["selected"] = build_method_menu(menu, views, lambda m: render(m.call), selected=state["name"])
            menubar.add_cascade(label="View", menu=menu)
        if actions:
            menu = tk.Menu(menubar, tearoff=False)
            build_method_menu(menu, actions, lambda m: render(m.call))
            menubar.add_cascade(label="Action", menu=menu)
        if menubar.index("end") is not None:
            window.configure(menu=menubar)
        state["content"] = ttk.Frame(window)
        state["content"].pack(fill="both", expand=True)

    def render(call):
        state["name"] = next((method.name for method in _methods.values() if method.call is call), "Method")
        parameters = [parameter for parameter in list(signature(call).parameters.values())[2:]
                      if parameter.kind is Parameter.KEYWORD_ONLY]

        def show(args, kwargs, text=""):
            try:
                shown = call(value, node.attrs, *args, **kwargs)
            except Exception as exc:
                show_error(state["window"] or master, title, exc)
                return
            if shown is None:  # The method opened a window of its own.
                return
            is_figure = type(shown).__module__ == "matplotlib.figure"
            if not is_figure and not callable(shown):
                # Plain data, not a rendering: a transform's result, shown by whichever method suits
                # it in a new window so this one stays. A group result browses unless a method claims it.
                open_value(master, title, shown)
                return
            if state["window"] is None:
                build_window()
            else:
                state["window"].geometry(state["window"].geometry())  # Keep the size across a switch.
            for child in state["content"].winfo_children():
                child.destroy()
            if is_figure:
                _embed_figure(state["content"], shown)
            else:
                shown(state["content"])

        if parameters:
            ask_arguments(state["window"] or master, title, parameters, _filled.setdefault(call, {}), show)
        else:
            show([], {})

    render(call)


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
