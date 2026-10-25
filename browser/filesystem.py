"""Minimal read-only Tk filesystem browser with relocated mount markers."""

import argparse
from reprlib import repr as short_repr

from ..filesystem import FileSystem
from ..filesystem.specification import add_spec_arguments, filesystem_from_arguments
from .viewers import open_value, register_viewer, suitable_viewers


def one_line_repr(value, width=160):
    """A shortened one-line repr; floats, NumPy ones too, show up to 8 significant digits, such as 1.2345678e-12."""
    if getattr(value, "shape", None) == () and hasattr(value, "item"):  # A NumPy scalar or 0-D array.
        value = value.item()
    if isinstance(value, float):
        return f"{value:.8g}"
    return short_repr(value)[:width]


def type_name(value):
    """A dtype with any shape, such as float64 or float64 76×160; else the Python type, such as float.

    Values from JSON are plain Python values, so they show Python types; nested lists show their first
    elements' type and lengths like an array, such as list[float] 76×160.
    """
    dtype, shape = getattr(value, "dtype", None), getattr(value, "shape", None)
    if dtype is not None:
        return _array_name(dtype, shape)
    if value is None:
        return "None"
    if not isinstance(value, (list, tuple)):
        return type(value).__name__
    lengths, item = [], value
    while isinstance(item, (list, tuple)):
        lengths.append(len(item))
        if not item:
            break
        item = item[0]
    element = "" if isinstance(item, (list, tuple)) else f"[{type_name(item)}]"
    return f"{type(value).__name__}{element} {'×'.join(map(str, lengths))}"


def _array_name(dtype, shape):
    return f"{dtype} {'×'.join(map(str, shape))}" if shape else str(dtype)


def _cells(value):
    """Type and value columns of a row."""
    return type_name(value), one_line_repr(value)


_MARKERS = {"group": "⊞", "dataset": "▦", "scalar": "·", "relocated": "↪", "merged": "⋈"}


def browse_filesystem(filesystem):
    """Preview visible leaves; discover group children only when visited."""
    import tkinter as tk

    window = tk.Tk()
    window.title("lazyfields")
    panes(window, filesystem)("/")
    window.mainloop()


def panes(master, filesystem):
    """Pack tree and details panes into master; return a function that roots them at a group.

    Double-clicking a dataset in either pane opens its preferred viewer; right-clicking offers every
    suitable one and copying any node's path. Muted attributes follow its children or value; right-click
    copies their value.
    """
    import tkinter as tk
    from tkinter import ttk

    panes = ttk.Panedwindow(master, orient="horizontal")
    panes.pack(fill="both", expand=True)
    tree = ttk.Treeview(panes, show="tree")
    details = ttk.Treeview(panes, columns=("type", "value"), displaycolumns=("value", "type"))
    details.heading("#0", text="Node")
    details.heading("type", text="Type")
    details.heading("value", text="Value")
    details.column("type", width=110, stretch=False)
    details.tag_configure("attribute", foreground="gray")
    panes.add(tree, weight=1)
    panes.add(details, weight=2)
    ttk.Label(details, text="   ".join(f"{marker} {name}" for name, marker in _MARKERS.items())).place(
        relx=1, rely=1, anchor="se", x=-6, y=-3)
    kinds = {}
    children = {}
    previews = {}

    def insert(parent, path, name, kind=None, value=None):
        node = filesystem.node(path)
        if kind is None:
            kind = node.kind
        elif kind == "scalar" or kind == "dataset" and value is not None:
            previews[path] = _cells(value)
        kinds[path] = kind
        group = kind == "group"
        origin = node.origin
        marker = ("📄" if origin is not None and origin.infile_path is None else "📁" if node.is_directory
                  else _MARKERS[kind])
        prefix = "".join(f"{_MARKERS[name]} " for name, flag in (("relocated", node.is_relocated),
                                                                 ("merged", node.is_merged)) if flag)
        tree.insert(parent, "end", iid=path, text=f"{prefix}{marker} {name}")
        if group:
            tree.insert(path, "end")  # Children remain undiscovered until visited.
        else:
            children[path] = None
        return path

    def populate(item):
        if item not in children:
            tree.delete(*tree.get_children(item))
            children[item] = tuple(insert(item, f"{item.rstrip('/')}/{name}", name, kind, value)
                                   for name, kind, value in filesystem.items(item))
        return children[item]

    def columns(path):
        """Type and value; a dataset listed without its value gets them from its attrs and leading corner."""
        if kinds[path] == "group":
            return "", ""
        if path not in previews:
            try:
                attrs = filesystem.attrs(path)
                corner = tuple(slice(0, 8) for _ in attrs["shape"]) or None
                previews[path] = (_array_name(attrs["dtype"], attrs["shape"]),
                                  one_line_repr(filesystem.read(path, corner)))
            except Exception as exc:
                return "", f"Error: {exc}"
        return previews[path]

    def attributes(event=None):
        item = details.focus() if event else tree.focus()
        if item not in kinds:
            return
        details.delete(*(row for row in details.get_children() if "attribute" in details.item(row, "tags")))
        try:
            attrs = filesystem.attrs(item)
        except Exception as exc:
            attrs = {"Error": str(exc)}
        if attrs:
            details.insert("", "end", text="─" * 40, values=("─" * 20, "─" * 60), tags=("attribute", "separator"))
            for name, value in attrs.items():
                details.insert("", "end", text=name, values=_cells(value), tags=("attribute",))

    def show(event=None):
        item = tree.focus()
        if not item:
            return
        details.delete(*details.get_children())
        try:
            entries = populate(item)
            for child in (item,) if entries is None else entries:
                details.insert("", "end", iid=child, text=tree.item(child, "text"), values=columns(child))
            attributes()
        except Exception as exc:
            details.insert("", "end", iid="error", text="Error", values=("", str(exc)))

    def select(item):
        tree.selection_set(item)
        tree.focus(item)
        tree.see(item)

    def enter(event):
        item = event.widget.identify_row(event.y)
        if kinds.get(item) == "dataset":
            open_value(master, item, filesystem.read(item), attrs=filesystem.attrs(item, defaults=False))
        elif item and tree.exists(item):
            tree.item(tree.parent(item), open=True)
            select(item)

    def set_root(path):
        tree.delete(*tree.get_children())
        details.delete(*details.get_children())
        for cache in (kinds, children, previews):
            cache.clear()
        root = insert("", path, path.rstrip("/").rsplit("/", 1)[-1] or "/")
        tree.item(root, open=True)
        select(root)

    menu = tk.Menu(details, tearoff=False)

    def copy(path):
        details.clipboard_clear()
        details.clipboard_append(path)

    def open_menu(event):
        """Every suitable viewer for a dataset, and copying any node's path in the filesystem."""
        selected = details.identify_row(event.y)
        item = selected if selected in kinds else ""
        if not selected or "separator" in details.item(selected, "tags"):
            return
        details.selection_set(selected)
        menu.delete(0, "end")
        if kinds.get(item) == "dataset":
            for name, viewer in suitable_viewers("dataset", filesystem.attrs(item)):
                menu.add_command(label=f"Open with {name}", command=lambda viewer=viewer: open_value(
                    master, item, filesystem.read(item), viewer))
            menu.add_separator()
        menu.add_command(label=f"Copy path  {item}" if item else "Copy value",
                         command=lambda: copy(item or details.set(selected, "value")))
        menu.tk_popup(event.x_root, event.y_root)

    tree.bind("<<TreeviewOpen>>", show)
    tree.bind("<<TreeviewSelect>>", show)
    tree.bind("<Double-1>", enter)
    details.bind("<Double-1>", enter)
    details.bind("<<TreeviewSelect>>", attributes)
    details.bind("<Button-3>", open_menu)
    return set_root


def _show_tree(window, value):
    """Browse a mapping, or a list of mappings, like a filesystem group."""
    panes(window, FileSystem.from_value(value))("/")


register_viewer("Browser", lambda kind, attrs: kind == "group", _show_tree)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="lf-browse-fs", description="Browse a filesystem specification read-only.")
    add_spec_arguments(parser)
    browse_filesystem(filesystem_from_arguments(parser.parse_args(argv)))
