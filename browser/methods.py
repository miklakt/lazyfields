"""Method registration, selection, and display.

Only Text is defined here, plus the Method/Methods machinery itself; Browser (browser/filesystem.py)
is added alongside it. Everything shape-specific (Table, Plot, Scatter, Figure, RGB Image) lives
in browser/extra_methods.py and is re-exported by lazyfields.browser.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from inspect import Parameter, getdoc, isroutine
from pprint import pformat
from typing import Any, Callable, Literal

from ..filesystem import FileSystem, Node
from ..readers import NodeInfo
from .arguments import ask_arguments, method_signature

_filled: dict[Callable, dict[str, str]] = {}


def show_error(parent, title, exc):
    """A standard error dialog for an exception, so every Tk entry point reports failures the same way."""
    from tkinter import messagebox
    messagebox.showerror(title, f"{type(exc).__name__}: {exc}", parent=parent)


@dataclass(frozen=True)
class Method:
    """Handles nodes accepted by accepts(info: NodeInfo) using call(node, **kwargs).

    call returns a matplotlib Figure, a function filling a Tk frame, None when it opens its own
    window, or any other value — handled in turn by whichever method suits it, same as a registered
    store method's result. A plain computation needs no Tk or matplotlib: return its data and let
    another method display it.

    kind is "view" when the method shows the value as it is, or "action" when it derives a new value;
    actions are offered in the menu but never picked as a value's default. category nests it in menu
    submenus ("Display/Plots"); omit it to stay at the menu root. priority breaks ties between methods
    that both accept a node - higher wins, order of addition settling the rest - and is a plain int, or
    a callable priority(node) when the rank depends on what matched (Table outranks Browser for a
    DataFrame but not for a plain array). Built-ins stay within -3..3 for readability; custom ones need not.
    """

    name: str
    accepts: Callable[[NodeInfo], bool]
    call: Callable[..., Any]
    kind: Literal["view", "action"] = "view"
    category: str | None = None
    priority: int | Callable[[NodeInfo], int] = 0

    def __post_init__(self):
        if self.kind not in ("view", "action"):
            raise ValueError("Method kind must be 'view' or 'action'.")

    def set(self, **changes) -> Method:
        """A copy with fields replaced - usually category, priority, or kind. The original is untouched,
        so one Method can be placed differently in different browsers."""
        return replace(self, **changes)

    def rank(self, node) -> int:
        """This method's tie-break priority for node: priority itself, or priority(node) when it varies."""
        return self.priority(node) if callable(self.priority) else self.priority


class Methods:
    """An ordered collection of Methods, combined with a Method or another Methods using ``|``.

    Each Method carries its own category and priority (see Method), so building a set is just unioning:
    ``DEFAULT_METHODS | TABLE.set(category="Tables") | Method("Volume", is_volume, show_volume)``. Adding
    a name already present replaces it, keeping it ordered last.
    """

    def __init__(self, methods=()):
        self._entries: dict[str, Method] = {}
        self |= methods

    def matching(self, node):
        """Accepted methods, default first: views before actions, the node's preferred method, then by
        priority (highest first), with order of addition settling any remaining tie."""
        accepted = []
        for method in self:
            try:
                if method.accepts(node):
                    accepted.append(method)
            except Exception:
                pass
        preferred = node.attrs.get("method")
        return sorted(accepted, key=lambda method: (
            method.kind != "view", method.name != preferred, -method.rank(node),
        ))

    def __iter__(self):
        return iter(self._entries.values())

    def __len__(self):
        return len(self._entries)

    def __or__(self, other):
        combined = Methods(self)
        combined |= other
        return combined

    def __ior__(self, other):
        if other is self:
            return self
        for method in (other,) if isinstance(other, Method) else other:
            if not isinstance(method, Method):
                raise TypeError(f"Cannot combine {method!r} into Methods; expected a Method - place "
                                "one with Method.set(category=..., priority=...).")
            self._entries.pop(method.name, None)  # Drop any earlier one of the same name...
            self._entries[method.name] = method   # ...so the latest wins and orders last.
        return self


DEFAULT_METHODS = Methods()


def open_value(master, title, value, call=None, methods=DEFAULT_METHODS):
    """Show a value, with a menu bar to re-show it any other way it accepts.

    Two menus list the same methods the tree offers on a node: View (the current one radio-marked)
    re-renders the value in place; Action runs a transform, opening its result in a new window that
    carries its own menu bar - so results chain. Either menu is dropped when it would be empty or
    offer no choice. A call's keyword-only parameters after node are offered in a form first, same
    as a registered store method's arguments.
    """
    import tkinter as tk
    from tkinter import ttk

    from .filesystem import _browse, build_method_menu

    node = value if isinstance(value, Node) else FileSystem.from_value(value).node()
    suitable = methods.matching(node.info())
    if call is None:
        call = suitable[0].call
    state = {"window": None, "content": None, "name": None, "selected": None}

    def build_window():
        window = state["window"] = tk.Toplevel(master)
        window.title(title)
        menubar = tk.Menu(window, tearoff=False)
        views = [method for method in suitable if method.kind == "view"]
        actions = [method for method in suitable if method.kind == "action"]
        if len(views) > 1:  # A lone view (plain text, say) has nothing to switch to.
            menu = tk.Menu(menubar, tearoff=False)
            state["selected"] = build_method_menu(
                menu, views, lambda m: render(m.call), selected=state["name"],
            )
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
        state["name"] = next((method.name for method in methods if method.call is call), "Method")
        parameters = [parameter for parameter in list(method_signature(call).parameters.values())[1:]
                      if parameter.kind is Parameter.KEYWORD_ONLY]

        def show(args, kwargs, text=""):
            try:
                shown = call(node, methods) if call is _browse else call(node, *args, **kwargs)
            except Exception as exc:
                show_error(state["window"] or master, title, exc)
                return
            if shown is None:  # The method opened a window of its own.
                return
            is_figure = type(shown).__module__ == "matplotlib.figure"
            # Only a function fills a frame: a safe pickle's proxy is callable too, but it is data.
            if not is_figure and not (isroutine(shown) or isinstance(shown, partial)):
                # Plain data, not a rendering: a transform's result, shown by whichever method suits
                # it in a new window so this one stays. A group result browses unless a method claims it.
                open_value(master, title, shown, methods=methods)
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
            ask_arguments(state["window"] or master, title, parameters, _filled.setdefault(call, {}), show,
                          doc=getdoc(call.func if isinstance(call, partial) else call))  # Not partial's own doc.
        else:
            show([], {})

    render(call)


def _tk(fill):
    """A method filling the browser's window with Tk widgets through fill(frame, node[:])."""
    return lambda node: partial(lambda value, frame: fill(frame, value), node[:])


def _embed_figure(window, figure):
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    canvas = FigureCanvasTkAgg(figure, master=window)
    NavigationToolbar2Tk(canvas, window).pack(side="bottom", fill="x")
    canvas.get_tk_widget().pack(fill="both", expand=True)
    canvas.draw()


def _show_text(window, value):
    from tkinter.scrolledtext import ScrolledText

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
DEFAULT_METHODS |= Method("Text", lambda node: True, _TEXT, priority=-3)
