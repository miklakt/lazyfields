"""Read-only Tk browser for reference tables; a selected row opens its group."""

import argparse
import inspect
import sys
from pathlib import Path
from reprlib import repr as short_repr

import pandas as pd

from .. import store  # noqa: F401  Registers the store accessor.
from .filesystem import panes
from .methods import ask_arguments, run_and_show


def _cell(column, value):
    if column == "lf_store_keys" and isinstance(value, (list, tuple)):
        return f"{len(value)} keys"
    return short_repr(value)[:80]


def _sorted(df, column, ascending):
    try:
        return df.sort_values(column, ascending=ascending, kind="stable")
    except TypeError:  # Mixed types in an object column.
        return df.sort_values(column, ascending=ascending, kind="stable", key=lambda s: s.astype(str))


def browse_table(reference_df: pd.DataFrame | str | Path, max_rows: int = 5000):
    """Show a table, or its pickle, with filtering and sorting; browse the selected row's group below.

    Right-clicking runs a registered method on the selected rows and shows its result.
    """
    import tkinter as tk
    from tkinter import messagebox, ttk

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

    columns = [column for column in reference_df.columns if column != "lf_path"]
    ids = [f"c{i}" for i in range(len(columns))]
    table = ttk.Treeview(top, columns=ids, show="tree headings")
    scroll = ttk.Scrollbar(top, orient="vertical", command=table.yview)
    xscroll = ttk.Scrollbar(top, orient="horizontal", command=table.xview)
    table.configure(yscrollcommand=scroll.set, xscrollcommand=xscroll.set)
    for heading in ("#0", *ids):
        table.column(heading, stretch=False)  # Stretching columns would never overflow, so nothing to scroll.
    xscroll.pack(side="bottom", fill="x")
    scroll.pack(side="right", fill="y")
    table.pack(fill="both", expand=True)

    bottom = ttk.Frame(split)
    filesystem = reference_df.store.filesystem
    set_root = panes(bottom, filesystem)
    split.add(top, weight=1)
    split.add(bottom, weight=2)

    view = {"df": reference_df, "sort": None}

    def render():
        df = view["df"]
        if view["sort"] is not None:
            df = _sorted(df, *view["sort"])
        view["shown"] = df.head(max_rows)
        table.delete(*table.get_children())
        for position, (_, row) in enumerate(view["shown"].iterrows()):
            table.insert("", "end", iid=str(position), text=row["lf_path"],
                         values=[_cell(column, row[column]) for column in columns])
        shown = f" (showing {max_rows})" if len(df) > max_rows else ""
        status.configure(text=f"{len(df)} / {len(reference_df)} rows{shown}")

    def apply_filter(event=None):
        expression = query.get().strip()
        try:
            view["df"] = reference_df.query(expression) if expression else reference_df
        except Exception as exc:
            status.configure(text=f"Error: {exc}")
            return
        render()

    def sort_by(column):
        ascending = view["sort"] != (column, True)
        view["sort"] = (column, ascending)
        for heading, name in (("#0", "lf_path"), *zip(ids, columns)):
            arrow = (" ▲" if ascending else " ▼") if name == column else ""
            table.heading(heading, text=name + arrow)
        render()

    def open_row(event=None):
        item = table.focus()
        if item:
            set_root("/" + view["shown"].iloc[int(item)]["lf_path"].lstrip("/"))

    menu = tk.Menu(table, tearoff=False)
    arguments = {}

    def run(name):
        positions = sorted(int(item) for item in table.selection())
        rows = view["shown"].iloc[positions[0] if len(positions) == 1 else positions]
        rows.__dict__["_filesystem"] = filesystem  # The store accessor's cache; spares rebuilding it.
        method = getattr(rows.store, name)
        parameters = list(inspect.signature(method).parameters.values())[1:]  # Without the row.
        target = rows["lf_path"] if len(positions) == 1 else f"{len(positions)} rows"

        def call(args, kwargs, text):
            window.configure(cursor="watch")
            window.update_idletasks()
            try:
                run_and_show(window, f"{target} · {name}({text})", lambda: method(*args, **kwargs))
            except Exception as exc:
                messagebox.showerror(name, f"{type(exc).__name__}: {exc}", parent=window)
            finally:
                window.configure(cursor="")

        if parameters:
            ask_arguments(window, name, parameters, arguments.setdefault(name, {}), call)
        else:
            call([], {}, "")

    def open_menu(event):
        item = table.identify_row(event.y)
        if not item:
            return
        if item not in table.selection():
            table.focus(item)
            table.selection_set(item)
        menu.delete(0, "end")
        names = reference_df.store.get_registered_methods()
        for name in names:
            menu.add_command(label=f"Run {name}", command=lambda name=name: run(name))
        if not names:
            menu.add_command(label="No registered methods", state="disabled")
        menu.tk_popup(event.x_root, event.y_root)

    for heading, name in (("#0", "lf_path"), *zip(ids, columns)):
        table.heading(heading, text=name, command=lambda name=name: sort_by(name))
    entry.bind("<Return>", apply_filter)
    table.bind("<<TreeviewSelect>>", open_row)
    table.bind("<Button-3>", open_menu)
    render()
    entry.focus_set()
    window.mainloop()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="lf-browse-table", description="Browse a pickled reference table read-only.")
    parser.add_argument("table", type=Path, help="reference table pickle")
    parser.add_argument("--max-rows", type=int, default=5000, help="rows to show at once (default: 5000)")
    args = parser.parse_args(argv)
    # Stored methods import modules by name, typically ones beside the table or in the working directory;
    # a console script would otherwise only search its own bin directory.
    for directory in (str(args.table.resolve().parent), str(Path.cwd())):
        if directory not in sys.path:
            sys.path.insert(0, directory)
    browse_table(args.table, max_rows=args.max_rows)
