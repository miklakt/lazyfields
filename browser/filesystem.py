"""Minimal read-only Tk filesystem browser with relocated mount markers."""

import argparse
from datetime import datetime
from reprlib import repr as short_repr

from ..filesystem import FileSystem
from ..filesystem.specification import add_spec_arguments, filesystem_from_arguments
from ..readers import NodeInfo
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


def _file_value(path):
    """A file's size and modification time, such as 75.26 MB, 2026-10-26 11:52:13."""
    try:
        stat = path.stat()
    except OSError as exc:
        return f"Error: {exc}"
    size = stat.st_size
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            break
        size /= 1024
    return f"{size:.4g} {unit}, {datetime.fromtimestamp(stat.st_mtime).isoformat(sep=' ', timespec='seconds')}"


_MARKERS = {"group": "⊞", "dataset": "▦", "scalar": "·", "relocated": "↪", "merged": "⋈"}


def browse_filesystem(filesystem):
    """Preview visible leaves; discover group children only when visited."""
    import tkinter as tk

    window = tk.Tk()
    window.title("lazyfields")
    panes(window, filesystem)("/")
    window.mainloop()


def panes(master, filesystem, *, navigation_widgets=()):
    """Pack tree and details panes into master; return a function that roots them at a group.

    Double-clicking a dataset or scalar in either pane opens its preferred viewer, and a group goes into it;
    right-clicking any node offers every suitable one, its value read whole, and copying its path. The muted
    attributes of the node chosen in the tree follow its children or value; right-click copies their value.
    Backspace selects the parent; navigation_widgets share the shortcut.
    """
    import tkinter as tk
    from tkinter import ttk

    panes = ttk.Panedwindow(master, orient="horizontal")
    panes.pack(fill="both", expand=True)
    tree = ttk.Treeview(panes, show="tree")
    details = ttk.Treeview(panes, columns=("type", "value"), displaycolumns=("value", "type"))
    for column, text in (("#0", "Node"), ("type", "Type"), ("value", "Value")):
        details.heading(column, text=text)
    details.column("type", width=110, stretch=False)
    details.tag_configure("attribute", foreground="gray")
    panes.add(tree, weight=1)
    panes.add(details, weight=2)
    ttk.Label(details, text="   ".join(f"{marker} {name}" for name, marker in _MARKERS.items())).place(
        relx=1, rely=1, anchor="se", x=-6, y=-3)
    kinds, children, previews = {}, {}, {}

    def insert(parent, path, name, kind=None, value=None):
        node = filesystem.node(path)
        if kind is None:
            kind = node.kind
        elif kind == "scalar" or kind == "dataset" and value is not None:
            previews[path] = _cells(value)
        kinds[path] = kind
        origin, marker = node.origin, _MARKERS[kind]
        if origin is not None and origin.infile_path is None:
            file_format = filesystem.attrs(path, defaults=False).get("file_format", "file")
            marker, previews[path] = "📄", (file_format, _file_value(origin.file_path))
        elif node.is_directory:
            marker, previews[path] = "📁", ("directory", f"{len(node.listdir())} entries")
        prefix = "".join(f"{_MARKERS[name]} " for name, flag in (("relocated", node.is_relocated),
                                                                 ("merged", node.is_merged)) if flag)
        tree.insert(parent, "end", iid=path, text=f"{prefix}{marker} {name}")
        if kind == "group":
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
        preview = 8  # Entries read per axis: more than the value's shortened text shows.
        if kinds[path] == "group":
            return previews.get(path, ("", ""))
        if path not in previews:
            try:
                attrs = filesystem.attrs(path)
                corner = tuple(slice(0, preview) for _ in attrs["shape"])
                if len(corner) == 1:
                    corner = corner[0]
                previews[path] = (_array_name(attrs["dtype"], attrs["shape"]),
                                  one_line_repr(filesystem.read(path, corner or None)))
            except Exception as exc:
                return "", f"Error: {exc}"
        return previews[path]

    def show(event=None):
        item = tree.focus()
        if not item:
            return
        details.delete(*details.get_children())
        try:
            entries = populate(item)
            for child in (item,) if entries is None else entries:
                details.insert("", "end", iid=child, text=tree.item(child, "text"), values=columns(child))
            attrs = filesystem.attrs(item)
        except Exception as exc:
            details.insert("", "end", iid="error", text="Error", values=("", str(exc)))
            return
        if attrs:
            details.insert("", "end", text="─" * 40, values=("─" * 20, "─" * 60), tags=("attribute", "separator"))
            for name, value in attrs.items():
                details.insert("", "end", text=name, values=_cells(value), tags=("attribute",))

    def select(item):
        tree.selection_set(item)
        tree.focus(item)
        tree.see(item)
        show()

    def go_up(event=None):
        item = tree.focus()
        if item and item != "/":
            parent = item.rstrip("/").rsplit("/", 1)[0] or "/"
            if not tree.exists(parent):
                set_root(parent)
            else:
                tree.item(parent, open=True)
                select(parent)
        return "break"

    def open_node(path, view=None):
        open_value(master, path, filesystem.read(path), view, filesystem.attrs(path, defaults=False))

    def enter(event):
        item = event.widget.identify_row(event.y)
        if kinds.get(item) in ("dataset", "scalar"):
            open_node(item)
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
        """Every suitable viewer for a node, and copying its path in the filesystem."""
        selected = details.identify_row(event.y)
        item = selected if selected in kinds else ""
        if not selected or "separator" in details.item(selected, "tags"):
            return
        details.selection_set(selected)
        menu.delete(0, "end")
        if item:
            names = tuple(path.rsplit("/", 1)[-1] for path in populate(item) or ())  # Listed once, as the tree's.
            for name, viewer in suitable_viewers(NodeInfo(kinds[item], names, filesystem.attrs(item))):
                menu.add_command(label=f"Open with {name}", command=lambda viewer=viewer: open_node(item, viewer))
            menu.add_separator()
        menu.add_command(label=f"Copy path  {item}" if item else "Copy value",
                         command=lambda: copy(item or details.set(selected, "value")))
        menu.tk_popup(event.x_root, event.y_root)

    tree.bind("<<TreeviewOpen>>", show)
    tree.bind("<<TreeviewSelect>>", show)
    tree.bind("<Double-1>", enter)
    details.bind("<Double-1>", enter)
    details.bind("<Button-3>", open_menu)
    for widget in (tree, details, *navigation_widgets):
        widget.bind("<BackSpace>", go_up)
    return set_root


def _browse(value, attrs):
    """Browse a mapping, or a list of mappings, like a filesystem group with attrs at its root."""
    return lambda window: panes(window, FileSystem.from_value(value, attrs))("/")


register_viewer("Browser", lambda node: node.kind == "group", _browse)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="lf-browse-fs", description="Browse a filesystem specification read-only.")
    add_spec_arguments(parser)
    browse_filesystem(filesystem_from_arguments(parser.parse_args(argv)))
