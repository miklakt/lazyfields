"""Read-only Tk filesystem browser."""

import argparse
from dataclasses import replace
from datetime import datetime
from reprlib import repr as short_repr

from ..filesystem import FileSystem
from ..filesystem.specification import add_spec_arguments, filesystem_from_arguments
from .viewers import open_value, register_viewer, suitable_viewers


def one_line_repr(value, width=160):
    """A shortened one-line repr with up to eight significant digits for floats."""
    if getattr(value, "shape", None) == () and hasattr(value, "item"):  # A NumPy scalar or 0-D array.
        value = value.item()
    if isinstance(value, float):
        return f"{value:.8g}"
    return short_repr(value)[:width]


def type_name(value):
    """A type label with array or nested-list dimensions."""
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
    """Format a file's size and modification time."""
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


_KINDS = {"group": "⊞", "dataset": "▦", "scalar": "·"}
_FORMATS = {"directory": "📁"}  # Any other format is a file root.
_FLAGS = {"relocated": "↪", "merged": "⋈"}
_FACTS = ("format", "merged", "axis", "dtype", "shape", "type")  # Derived facts listed beside attributes.


def _collapsed(info):
    """Default collapsed when a group's type marks it reconstructed, not a real container; attrs overrides either way."""
    return info.attrs.get("collapsed", info.kind == "group" and info.type is not None)


def _label(info, name):
    """Tree text: flag markers, then a format or kind marker, then the name."""
    flags = "".join(f"{marker} " for flag, marker in _FLAGS.items() if getattr(info, flag))
    marker = _FORMATS.get(info.format, "📄") if info.format else _KINDS[info.kind]
    return f"{flags}{marker} {name}"


def bind_context_menu(widget, callback):
    """Open a row's context menu with right-click, Ctrl+Enter, Shift+F10, or Menu."""
    def mouse_menu(event):
        item = widget.identify_row(event.y)
        if item:
            widget.focus_set()
            widget.focus(item)
            callback(widget, item, event.x_root, event.y_root)
        return "break"

    def keyboard_menu(event):
        item = widget.focus()
        widget.see(item)
        if item and (bounds := widget.bbox(item)):
            x, y, width, height = bounds
            callback(widget, item, widget.winfo_rootx() + x + min(width, 20), widget.winfo_rooty() + y + height)
        return "break"

    widget.bind("<Button-3>", mouse_menu)
    for shortcut in ("<Control-Return>", "<Shift-F10>", "<Menu>"):
        widget.bind(shortcut, keyboard_menu)


def browse_filesystem(filesystem):
    """Preview visible leaves; discover group children only when visited."""
    import tkinter as tk

    window = tk.Tk()
    window.title("lazyfields")
    panes(window, filesystem)("/")
    window.mainloop()


def panes(master, filesystem, *, navigation_widgets=()):
    """Pack tree and details panes; return a function that selects their root.

    Double-click opens a node; right-click offers viewers and copying.
    Backspace selects the parent. navigation_widgets share that shortcut.
    Tab cycles through navigation_widgets, tree, and details.
    """
    import tkinter as tk
    from tkinter import ttk

    style = ttk.Style(master)
    for option, colour in (("background", "#c8c8c8"), ("foreground", "black")):  # Selection of an unfocused view.
        inherited = [entry for entry in style.map("Treeview", query_opt=option)
                     if entry[:2] != ("!disabled", "!selected")]
        style.map("Treeview", **{option: [("selected", "!focus", colour), *inherited]})

    split = ttk.Panedwindow(master, orient="horizontal")
    split.pack(fill="both", expand=True)
    tree = ttk.Treeview(split, show="tree")
    details = ttk.Treeview(split, columns=("type", "value"), displaycolumns=("value", "type"))
    for column, text in (("#0", "Node"), ("type", "Type"), ("value", "Value")):
        details.heading(column, text=text)
    details.column("type", width=110, stretch=False)
    details.tag_configure("attribute", foreground="gray")
    split.add(tree, weight=1)
    split.add(details, weight=2)
    ttk.Label(details, text="   ".join(f"{marker} {name}" for name, marker in {**_KINDS, **_FLAGS}.items())).place(
        relx=1, rely=1, anchor="se", x=-6, y=-3)
    infos, children, previews = {}, {}, {}

    def insert(parent, path, name, kind=None, value=None):
        info = infos[path] = filesystem.info(path, full=False)
        if kind == "scalar" or kind == "dataset" and value is not None:
            previews[path] = _cells(value)
        if info.format == "directory":
            previews[path] = ("directory", f"{len(filesystem.listdir(path))} entries")
        elif info.format and (origin := filesystem.node(path).origin) is not None:
            previews[path] = (info.format, _file_value(origin.file_path))
        tree.insert(parent, "end", iid=path, text=_label(info, name))
        # A tree's own root always expands, even as a record: it was opened to see inside it.
        if info.kind == "group" and (parent == "" or not _collapsed(info)):
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

    def described(path):
        """Info with a dataset's dtype and shape; a group's children are listed by the tree instead."""
        return filesystem.info(path, full=infos[path].kind != "group")

    def columns(path):
        """Type and value preview for a node."""
        preview = 8  # Entries read per axis: more than the value's shortened text shows.
        if infos[path].kind == "group":
            return previews.get(path, ("", ""))
        if path not in previews:
            try:
                info = described(path)
                corner = tuple(slice(0, preview) for _ in info.shape)
                if len(corner) == 1:
                    corner = corner[0]
                previews[path] = (_array_name(info.dtype, info.shape),
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
            info = described(item)
        except Exception as exc:
            details.insert("", "end", iid="error", text="Error", values=("", str(exc)))
            return
        facts = ((name, getattr(info, name)) for name in _FACTS if getattr(info, name) is not None)
        rows = [*facts, *info.attrs.items()]
        if rows:
            details.insert("", "end", text="─" * 40, values=("─" * 20, "─" * 60), tags=("attribute", "separator"))
            for name, value in rows:
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

    focus_order = (*navigation_widgets, tree, details)

    def cycle_focus(event):
        position = focus_order.index(event.widget)
        following = (focus_order[(position + offset) % len(focus_order)] for offset in range(1, len(focus_order) + 1))
        target = next((view for view in following if view.get_children()), event.widget)  # Empty views take no focus.
        target.focus_set()
        if not target.selection() and target.get_children():  # Else nothing shows where focus went.
            row = target.focus() or target.get_children()[0]
            target.focus(row)
            target.selection_set(row)
        return "break"

    def open_node(path, view=None):
        # Browsing stays lazy: a clone of its own source, not the value loaded.
        value = filesystem.clone(path) if view is _browse else filesystem.read(path)
        open_value(master, path, value, view, filesystem.attrs(path))

    def enter(event):
        item = event.widget.identify_row(event.y)
        if item in infos and infos[item].kind in ("dataset", "scalar"):
            open_node(item)
        elif item in infos and _collapsed(infos[item]):
            open_node(item, _browse)
        elif item and tree.exists(item):
            tree.item(tree.parent(item), open=True)
            select(item)

    def set_root(path):
        tree.delete(*tree.get_children())
        details.delete(*details.get_children())
        for cache in (infos, children, previews):
            cache.clear()
        root = insert("", path, path.rstrip("/").rsplit("/", 1)[-1] or "/")
        tree.item(root, open=True)
        select(root)

    menu = tk.Menu(details, tearoff=False)

    def copy(path):
        details.clipboard_clear()
        details.clipboard_append(path)

    def open_menu(widget, selected, x, y):
        """Offer viewers and copying for the selected row."""
        item = selected if selected in infos else ""
        if (widget is tree and not item) or "separator" in widget.item(selected, "tags"):
            return
        widget.selection_set(selected)
        menu.delete(0, "end")
        if item:
            names = tuple(path.rsplit("/", 1)[-1] for path in populate(item) or ())  # Listed once, as the tree's.
            for name, viewer in suitable_viewers(replace(described(item), children=names)):
                menu.add_command(label=f"Open with {name}", command=lambda viewer=viewer: open_node(item, viewer))
            menu.add_separator()
        menu.add_command(label=f"Copy path  {item}" if item else "Copy value",
                         command=lambda: copy(item or widget.set(selected, "value")))
        menu.tk_popup(x, y)

    tree.bind("<<TreeviewOpen>>", show)
    tree.bind("<<TreeviewSelect>>", show)
    tree.bind("<Double-1>", enter)
    details.bind("<Double-1>", enter)
    for widget in (tree, details):
        bind_context_menu(widget, open_menu)
    for widget in focus_order:
        widget.bind("<BackSpace>", go_up)
        widget.bind("<Tab>", cycle_focus)
    return set_root


def _browse(value, attrs):
    """Browse a lazy source as it stands, or an in-memory group with root attributes."""
    source = value if isinstance(value, FileSystem) else FileSystem.from_value(value, attrs=attrs)
    return lambda window: panes(window, source)("/")


register_viewer("Browser", lambda node: node.kind == "group", _browse)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="lf-browse-fs", description="Browse a filesystem specification read-only.")
    add_spec_arguments(parser)
    browse_filesystem(filesystem_from_arguments(parser.parse_args(argv)))
