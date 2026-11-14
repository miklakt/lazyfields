"""Read-only Tk browser for reference tables; a selected row opens its group."""

import argparse
import inspect
import sys
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from functools import partial
from pathlib import Path

import pandas as pd

from .. import store  # noqa: F401  Registers the store accessor.
from .arguments import ask_arguments, method_signature
from .filesystem import _add_menu_command, _clear_menu, bind_context_menu, one_line_repr, panes, type_name
from . import view_settings
from .methods import DEFAULT_METHODS, open_value, show_error


def _run_and_show(master, title, call, methods):
    """Show a call's result with its preferred method; None opens no window.

    A Series, one value per selected row, shows as a group of them named by row label.
    """
    result = call()
    if isinstance(result, pd.Series):
        result = None if all(value is None for value in result) else {
            str(label): value for label, value in result.items()}
    if result is not None:
        open_value(master, title, result, methods=methods)


def _cell(value):
    """A cell's text: an array's dtype and shape, a datetime to the second, other dates, times,
    durations and decimals as their own text, anything else a short repr."""
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")
    if isinstance(value, (date, time, timedelta, Decimal)):
        return str(value)
    if getattr(value, "ndim", 0):
        return type_name(value)
    return one_line_repr(value, 80)


def _sorted(df, levels):
    """Sort df by levels, {column: ascending} in order, each within the groups of those before; "#0" is the index."""
    if not levels:
        return df
    keys = pd.concat([df.index.to_frame(index=False) if column == "#0" else df[[column]].reset_index(drop=True)
                      for column in levels], axis=1, ignore_index=True)
    ascending = [up for column, up in levels.items() for _ in range(df.index.nlevels if column == "#0" else 1)]
    sort = partial(keys.sort_values, list(keys), ascending=ascending, kind="stable")
    try:
        return df.iloc[sort().index]
    except (TypeError, ValueError):  # Mixed or incomparable values in an object column; compare as text.
        return df.iloc[sort(key=lambda s: s.astype(str) if s.dtype == object else s).index]


def browse_table(reference_df: pd.DataFrame | str | Path, max_rows: int = 5000,
                 methods=DEFAULT_METHODS, view: dict | None = None):
    """Show a table, or its pickle, with filtering and sorting; browse the selected row's group below.

    The first column is the index. Shift+clicking a heading, or a column in the Columns dialog,
    cycles its sort level ▲, ▼, off; each level sorts within the groups of the levels before it,
    and the first level's groups are shaded alternately.

    Right-clicking runs a registered row method on the selected rows and shows its result; the
    Table menu runs a table method on the current view: the filtered rows in shown order, with
    lf_path and the shown columns.

    :param view: Initial view settings, any of ``filter`` (query expression), ``sort``
        (``[[column, ascending], ...]`` from the first level, column "#0" for the index),
        ``column_order`` and ``hidden`` (column names). Names missing from the table are
        ignored; columns missing from ``column_order`` follow in table order. None restores
        the view last shown for a table over the same filesystem specification, root and
        column names, pickled or not.
    """
    import tkinter as tk
    from tkinter import ttk

    if isinstance(reference_df, (str, Path)):
        reference_df = pd.read_pickle(reference_df)

    window = tk.Tk()
    window.title("lazyfields: reference table")
    # A fixed size; otherwise the window widens to fit every column.
    window.geometry(f"{window.winfo_screenwidth() * 2 // 5}x{window.winfo_screenheight() * 2 // 5}")
    split = ttk.Panedwindow(window, orient="vertical")
    split.pack(fill="both", expand=True)

    top = ttk.Frame(split)
    bar = ttk.Frame(top)
    bar.pack(fill="x")
    ttk.Label(bar, text="Filter:").pack(side="left")
    query = tk.StringVar()
    entry = ttk.Entry(bar, textvariable=query)
    entry.pack(side="left", fill="x", expand=True)
    status = ttk.Label(bar)
    status.pack(side="right", padx=6)

    # Treeview columns are named by the table's columns; its tree column "#0" shows the index.
    columns = ["lf_path", *(column for column in reference_df.columns if column != "lf_path")]
    headings = ["#0", *columns]
    index_label = ", ".join(str(name) for name in reference_df.index.names if name is not None) or "index"
    table = ttk.Treeview(top, columns=columns, show="tree headings")
    scroll = ttk.Scrollbar(top, orient="vertical", command=table.yview)
    xscroll = ttk.Scrollbar(top, orient="horizontal", command=table.xview)
    table.configure(yscrollcommand=scroll.set, xscrollcommand=xscroll.set)
    for heading in headings:
        table.column(heading, stretch=False)  # Stretching columns would never overflow, so nothing to scroll.
    style = ttk.Style(window)
    field, text = (window.winfo_rgb(style.lookup("Treeview", option) or default)
                   for option, default in (("fieldbackground", "white"), ("foreground", "black")))
    # Shaded rows are a tenth of the way from the theme's background to its text colour.
    table.tag_configure("shaded", background="#" + "".join(f"{(9 * a + b) // 10 >> 8:02x}" for a, b in zip(field, text)))
    xscroll.pack(side="bottom", fill="x")
    scroll.pack(side="right", fill="y")
    table.pack(fill="both", expand=True)

    visible, order = set(columns), list(columns)

    def show_columns():
        table.configure(displaycolumns=[column for column in order if column in visible])
        save_view()

    def choose_columns():
        dialog = tk.Toplevel(window)
        dialog.title("Columns")
        dialog.transient(window)
        dialog.geometry("340x420")
        search = tk.StringVar()
        ttk.Label(dialog, text="Search columns").pack(anchor="w", padx=6)
        field = ttk.Entry(dialog, textvariable=search)
        field.pack(fill="x", padx=6, pady=6)
        buttons = ttk.Frame(dialog)
        buttons.pack(side="bottom", fill="x", padx=6, pady=6)
        ttk.Label(dialog, foreground="gray", text="Click: show or hide · Shift+click: sort · Alt+↑/↓: move").pack(
            side="bottom", anchor="w", padx=6)
        choices = tk.Listbox(dialog, selectmode="multiple", exportselection=False, activestyle="dotbox")
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=choices.yview)
        choices.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        choices.pack(fill="both", expand=True)
        shown = []

        def refresh(active=None):
            shown[:] = [column for column in order if search.get().casefold() in str(column).casefold()]
            choices.delete(0, "end")
            for position, column in enumerate(shown):
                choices.insert("end", heading_text(column))
                if column in visible:
                    choices.selection_set(position)
            if active is not None:
                choices.activate(active)
                choices.see(active)
            show_columns()

        def apply_selection(event):
            visible.difference_update(shown)
            visible.update(shown[position] for position in choices.curselection())
            show_columns()

        def move(step):
            """Move the focused column past its neighbour in the list, keeping its place in the order."""
            position = choices.index("active")
            if 0 <= position + step < len(shown):
                column = shown[position]
                order.remove(column)
                order.insert(order.index(shown[position + step]) + (step > 0), column)
                refresh(active=position + step)
            return "break"  # Not also the plain arrow's focus move.

        def cycle_at(position):
            if 0 <= position < len(shown):
                cycle_sort(shown[position])
                refresh(active=position)
            return "break"  # Not also a show or hide.

        def cycle_at_pointer(event):
            position = choices.nearest(event.y)
            _, y, _, height = choices.bbox(position) or (0, 0, 0, 0)
            return cycle_at(position if y <= event.y < y + height else -1)  # Not below the last item.

        def hide_constant():
            """Hide columns holding one value in every filtered row; a missing value counts as another."""
            df = state["df"]
            # As text, since lists such as lf_store_keys cannot be counted directly.
            visible.difference_update(column for column in columns if df[column].astype(str).nunique(dropna=False) <= 1)
            refresh()

        ttk.Button(buttons, text="Show all", command=lambda: (visible.update(columns), refresh())).pack(side="left")
        ttk.Button(buttons, text="Hide all", command=lambda: (visible.clear(), refresh())).pack(side="left")
        ttk.Button(buttons, text="Hide constant", command=hide_constant).pack(side="left")
        ttk.Button(buttons, text="Close", command=dialog.destroy).pack(side="right")
        choices.bind("<<ListboxSelect>>", apply_selection)
        choices.bind("<Alt-Up>", lambda event: move(-1))
        choices.bind("<Alt-Down>", lambda event: move(1))
        choices.bind("<Shift-Button-1>", cycle_at_pointer)
        choices.bind("<Shift-space>", lambda event: cycle_at(choices.index("active")))
        field.bind("<Down>", lambda event: choices.focus_set())
        search.trace_add("write", lambda *_: refresh())
        dialog.bind("<Escape>", lambda event: dialog.destroy())
        dialog.grab_set()
        refresh()
        field.focus_set()

    bottom = ttk.Frame(split)
    filesystem = reference_df.store._filesystem  # Fails first for a table without a specification.
    view_key = view_settings.table_key(reference_df)
    if view is None:
        view = view_settings.load(view_key)
    set_root = panes(bottom, filesystem, navigation_widgets=(table,), methods=methods)
    split.add(top, weight=1)
    split.add(bottom, weight=2)

    state = {"df": reference_df, "sort": {}, "filter": "", "saved": None}

    def save_view():
        """Save the view at every change, so a killed or suspended browser loses nothing.

        Calls that change nothing, such as typing in the column search, write nothing.
        """
        current = {"filter": state["filter"], "sort": [list(level) for level in state["sort"].items()],
                   "column_order": list(order), "hidden": [column for column in columns if column not in visible]}
        if current != state["saved"]:
            view_settings.save(view_key, current)
            state["saved"] = current

    def label(column):
        return index_label if column == "#0" else column

    def heading_text(column):
        """The column's label with its sort direction, numbered by level when there are several."""
        levels = state["sort"]
        if column not in levels:
            return label(column)
        number = list(levels).index(column) + 1 if len(levels) > 1 else ""
        return f"{label(column)} {'▲' if levels[column] else '▼'}{number}"

    def shading(shown):
        """Whether each row is shaded: the groups of the first sort level alternate."""
        if not state["sort"]:
            return [False] * len(shown)
        column = next(iter(state["sort"]))
        # As text, since lists cannot be compared directly.
        values = pd.Series((shown.index.to_flat_index() if column == "#0" else shown[column]).astype(str).to_numpy())
        return ((values != values.shift()).cumsum() % 2 == 1).tolist()

    def render():
        """Show the filtered rows in sort order, keeping the selected and focused rows."""
        paths = state.get("shown", reference_df.head(0))["lf_path"]
        selected, focused = {paths.iloc[int(item)] for item in table.selection()}, table.focus()
        focused = paths.iloc[int(focused)] if focused else None
        df = _sorted(state["df"], state["sort"])
        shown = state["shown"] = df.head(max_rows)
        table.delete(*table.get_children())
        rows = shown[columns].itertuples(index=False, name=None)
        for position, (index, row, shaded) in enumerate(zip(shown.index, rows, shading(shown))):
            table.insert("", "end", iid=str(position), text=_cell(index), tags="shaded" if shaded else "",
                         values=[_cell(value) for value in row])
        items = {path: str(position) for position, path in enumerate(shown["lf_path"])}
        table.selection_set([items[path] for path in selected if path in items])
        if focused in items:
            table.focus(items[focused])
        more = f" (showing {max_rows})" if len(df) > max_rows else ""
        status.configure(text=f"{len(df)} / {len(reference_df)} rows{more}")

    def fit_columns(limit=300):
        """Size each column to its heading and longest shown cell, up to limit pixels."""
        from tkinter import font

        cell, heading_font = font.nametofont("TkDefaultFont"), font.nametofont("TkHeadingFont")
        shown = state["shown"]
        for heading in headings:
            values = shown.index if heading == "#0" else shown[heading]
            longest = max(map(_cell, values), key=len, default="")  # Measuring every cell is slow.
            width = max(heading_font.measure(f"{label(heading)} ▼9"), cell.measure(longest)) + 24  # Padding, tree indent.
            table.column(heading, width=min(width, limit))

    def apply_filter(event=None):
        """Filter by the query; a failing one keeps the previous rows and filter, reporting the error."""
        expression = query.get().strip()
        try:
            state["df"] = reference_df.query(expression) if expression else reference_df
        except Exception as exc:
            status.configure(text=f"Error: {exc}")
            return
        state["filter"] = expression
        render()
        fit_columns()
        save_view()

    def set_sort(levels):
        state["sort"] = levels
        for heading in headings:
            table.heading(heading, text=heading_text(heading))

    def sort(levels):
        set_sort(levels)
        render()
        save_view()

    def cycle_sort(column):
        """Cycle column's sort level ▲, ▼, off; a new level comes last."""
        levels = dict(state["sort"])
        if column not in levels:
            levels[column] = True
        elif levels[column]:
            levels[column] = False
        else:
            del levels[column]
        sort(levels)

    def heading_at(display):
        """The heading at a Treeview display column such as "#2", which counts shown columns only."""
        return display if display == "#0" else [column for column in order if column in visible][int(display[1:]) - 1]

    def click_heading(event):
        if table.identify_region(event.x, event.y) != "heading":
            return None
        if event.state & 0x0001:  # Shift.
            cycle_sort(heading_at(table.identify_column(event.x)))
        return "break"

    def open_row(event=None):
        """Open the focused row's group below, unless it is already open."""
        item = table.focus()
        path = item and "/" + state["shown"].iloc[int(item)]["lf_path"].lstrip("/")
        if path and path != state.get("opened"):
            state["opened"] = path
            set_root(path)

    menu = tk.Menu(table, tearoff=False)
    header_menu = tk.Menu(table, tearoff=False)
    arguments = {}

    def add_methods(target, scope):
        """Add a Run entry per registered method of scope, or a note that there is none."""
        registered = {name: method for name, method in reference_df.store._methods.items()
                      if method["scope"] == scope}
        submenus = {}
        for name, method in registered.items():
            _add_menu_command(target, submenus, method["category"],
                              label=f"Run {name}", command=lambda name=name: run(name))
        if not registered:
            target.add_command(label=f"No {scope} methods", state="disabled")

    def fill_table_menu():
        _clear_menu(table_menu)
        add_methods(table_menu, "table")
        table_menu.add_separator()
        table_menu.add_command(label="Clear sort", command=lambda: sort({}),
                               state="normal" if state["sort"] else "disabled")

    menubar = tk.Menu(window, tearoff=False)
    menubar.add_command(label="Columns", command=choose_columns)
    table_menu = tk.Menu(menubar, tearoff=False, postcommand=fill_table_menu)
    menubar.add_cascade(label="Table", menu=table_menu)
    window.configure(menu=menubar)

    def open_header_menu(display, x, y):
        column = heading_at(display)
        if column == "#0":  # The index always shows.
            return

        def hide():
            visible.discard(column)
            show_columns()

        _clear_menu(header_menu)
        header_menu.add_command(label="Hide", command=hide)
        header_menu.tk_popup(x, y)

    def run(name):
        if reference_df.store._methods[name]["scope"] == "table":  # All filtered rows, shown columns.
            rows = _sorted(state["df"], state["sort"])
            rows = rows[["lf_path", *(column for column in order if column in visible and column != "lf_path")]]
            target = f"{len(rows)} rows"
        else:
            positions = sorted(int(item) for item in table.selection())
            rows = state["shown"].iloc[positions[0] if len(positions) == 1 else positions]
            target = rows["lf_path"] if len(positions) == 1 else f"{len(positions)} rows"
        method = getattr(rows.store, name)
        parameters = list(method_signature(method).parameters.values())[1:]  # Without the row or table.

        def call(args, kwargs, text):
            window.configure(cursor="watch")
            window.update_idletasks()
            try:
                _run_and_show(window, f"{target} · {name}({text})",
                              lambda: method(*args, **kwargs), methods)
            except Exception as exc:
                show_error(window, name, exc)
            finally:
                window.configure(cursor="")

        if parameters:
            ask_arguments(window, name, parameters, arguments.setdefault(name, {}), call,
                          doc=inspect.getdoc(method))
        else:
            call([], {}, "")

    def open_menu(widget, item, x, y):
        if item not in table.selection():
            table.focus(item)
            table.selection_set(item)
        _clear_menu(menu)
        add_methods(menu, "row")
        menu.tk_popup(x, y)

    def apply_view(view):
        """Restore columns, sort and filter from view, skipping what the table no longer has."""
        saved = [column for column in view.get("column_order", ()) if column in columns]
        order[:] = saved + [column for column in columns if column not in saved]
        visible.difference_update(view.get("hidden", ()))
        show_columns()
        set_sort({column: bool(ascending) for column, ascending in view.get("sort") or () if column in headings})
        query.set(view.get("filter") or "")

    entry.bind("<Return>", apply_filter)
    table.bind("<<TreeviewSelect>>", open_row)
    table.bind("<Button-1>", click_heading)
    bind_context_menu(table, open_menu, on_heading=open_header_menu)
    apply_view(view)
    render()
    fit_columns()
    if query.get().strip():
        apply_filter()  # On error the unfiltered rows stay, with the error in the status, and the filter is dropped.
    entry.focus_set()
    save_view()  # A view passed in replaces the saved one even if nothing changes.
    window.mainloop()


def main(argv=None) -> None:
    from .extra_methods import ARRAY_ACTIONS, EXTRA_VIEWERS

    parser = argparse.ArgumentParser(
        prog="lf-browse-table", description="Browse a pickled reference table read-only.")
    parser.add_argument("table", type=Path, help="reference table pickle")
    parser.add_argument("--max-rows", type=int, default=5000, help="rows to show at once (default: 5000)")
    args = parser.parse_args(argv)
    methods = DEFAULT_METHODS | EXTRA_VIEWERS | ARRAY_ACTIONS
    # Resolve modules used by saved store methods.
    for directory in (str(args.table.resolve().parent), str(Path.cwd())):
        if directory not in sys.path:
            sys.path.insert(0, directory)
    browse_table(args.table, max_rows=args.max_rows, methods=methods)
