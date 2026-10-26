"""Read-only Tk browser for reference tables; a selected row opens its group."""

import argparse
import inspect
import sys
from pathlib import Path

import pandas as pd

from .. import store  # noqa: F401  Registers the store accessor.
from .filesystem import bind_context_menu, one_line_repr, panes
from .methods import ask_arguments, run_and_show


def _cell(column, value):
    if column == "lf_mtime":
        return "" if pd.isna(value) else pd.Timestamp(value).isoformat(sep=" ", timespec="seconds")
    if column == "lf_store_keys" and isinstance(value, (list, tuple)):
        return f"{len(value)} keys"
    return one_line_repr(value, 80)


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
    headings = [("#0", "lf_path"), *zip(ids, columns)]
    table = ttk.Treeview(top, columns=ids, show="tree headings")
    scroll = ttk.Scrollbar(top, orient="vertical", command=table.yview)
    xscroll = ttk.Scrollbar(top, orient="horizontal", command=table.xview)
    table.configure(yscrollcommand=scroll.set, xscrollcommand=xscroll.set)
    for heading, name in headings:
        table.column(heading, stretch=False)  # Stretching columns would never overflow, so nothing to scroll.
        table.heading(heading, text=name, command=lambda name=name: sort_by(name))
    xscroll.pack(side="bottom", fill="x")
    scroll.pack(side="right", fill="y")
    table.pack(fill="both", expand=True)

    visible, order, names = set(ids), list(ids), dict(zip(ids, columns))

    def show_columns():
        table.configure(displaycolumns=[key for key in order if key in visible])

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
        choices = tk.Listbox(dialog, selectmode="multiple", exportselection=False, activestyle="dotbox")
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=choices.yview)
        choices.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        choices.pack(fill="both", expand=True)
        shown = []

        def refresh(all_columns=None):
            if all_columns is not None:
                visible.clear()
                visible.update(ids if all_columns else ())
            shown[:] = [key for key in order if search.get().casefold() in str(names[key]).casefold()]
            choices.delete(0, "end")
            for position, key in enumerate(shown):
                choices.insert("end", str(names[key]))
                if key in visible:
                    choices.selection_set(position)
            show_columns()

        def apply_selection(event):
            visible.difference_update(shown)
            visible.update(shown[position] for position in choices.curselection())
            show_columns()

        def move(step):
            """Move the focused column past its neighbour in the list, keeping its place in the order."""
            position = choices.index("active")
            if 0 <= position + step < len(shown):
                key = shown[position]
                order.remove(key)
                order.insert(order.index(shown[position + step]) + (step > 0), key)
                refresh()
                choices.activate(position + step)
                choices.see(position + step)
            return "break"  # Not also the plain arrow's focus move.

        ttk.Button(buttons, text="Show all", command=lambda: refresh(True)).pack(side="left")
        ttk.Button(buttons, text="Hide all", command=lambda: refresh(False)).pack(side="left")
        ttk.Button(buttons, text="Close", command=dialog.destroy).pack(side="right")
        choices.bind("<<ListboxSelect>>", apply_selection)
        choices.bind("<Alt-Up>", lambda event: move(-1))
        choices.bind("<Alt-Down>", lambda event: move(1))
        field.bind("<Down>", lambda event: choices.focus_set())
        search.trace_add("write", lambda *_: refresh())
        dialog.bind("<Escape>", lambda event: dialog.destroy())
        dialog.grab_set()
        refresh()
        field.focus_set()

    menubar = tk.Menu(window, tearoff=False)
    menubar.add_command(label="Columns", command=choose_columns)
    window.configure(menu=menubar)

    bottom = ttk.Frame(split)
    filesystem = reference_df.store.filesystem
    set_root = panes(bottom, filesystem, navigation_widgets=(table,))
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
        for heading, name in headings:
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

    def open_menu(widget, item, x, y):
        if item not in table.selection():
            table.focus(item)
            table.selection_set(item)
        menu.delete(0, "end")
        names = reference_df.store.get_registered_methods()
        for name in names:
            menu.add_command(label=f"Run {name}", command=lambda name=name: run(name))
        if not names:
            menu.add_command(label="No registered methods", state="disabled")
        menu.tk_popup(x, y)

    entry.bind("<Return>", apply_filter)
    table.bind("<<TreeviewSelect>>", open_row)
    bind_context_menu(table, open_menu)
    render()
    entry.focus_set()
    window.mainloop()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="lf-browse-table", description="Browse a pickled reference table read-only.")
    parser.add_argument("table", type=Path, help="reference table pickle")
    parser.add_argument("--max-rows", type=int, default=5000, help="rows to show at once (default: 5000)")
    args = parser.parse_args(argv)
    # Resolve modules used by saved store methods.
    for directory in (str(args.table.resolve().parent), str(Path.cwd())):
        if directory not in sys.path:
            sys.path.insert(0, directory)
    browse_table(args.table, max_rows=args.max_rows)
