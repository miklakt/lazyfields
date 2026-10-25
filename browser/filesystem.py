"""Minimal read-only Tk filesystem browser with relocated mount markers."""

import argparse
import json
from pathlib import Path

from ..filesystem import FileSystem
from .viewers import open_value, preview, suitable_viewers, type_name


def browse_filesystem(filesystem):
    """Preview visible leaves; discover group children only when visited."""
    import tkinter as tk

    window = tk.Tk()
    window.title("lazyfields")
    panes(window, filesystem)("/")
    window.mainloop()


def panes(master, filesystem):
    """Pack tree and details panes into master; return a function that roots them at a group.

    Double-clicking a dataset opens its preferred viewer; right-clicking offers every suitable one and
    copying any node's path. Muted attributes follow its children or value; right-click copies their value.
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
    ttk.Label(details, text="⊞ group   ▦ dataset   · scalar   ↪ relocated   ⋈ merged").place(
        relx=1, rely=1, anchor="se", x=-6, y=-3)
    kinds = {}
    children = {}
    previews = {}

    def insert(parent, path, name, kind=None, value=None):
        node = filesystem.node(path)
        if kind is None:
            kind = node.kind
        elif kind == "scalar" or kind == "dataset" and value is not None:
            previews[path] = type_name(value), preview(value)
        kinds[path] = kind
        group = kind == "group"
        origin = node.origin
        os_directory = group and origin is None and (filesystem._namespace_path / path.lstrip("/")).is_dir()
        marker = ("📄 " if origin is not None and origin.infile_path is None else "📁 " if os_directory
                  else {"group": "⊞ ", "dataset": "▦ ", "scalar": "· "}[kind])
        prefix = ("↪ " if node.is_relocated else "") + ("⋈ " if node.is_merged else "")
        tree.insert(parent, "end", iid=path, text=prefix + marker + name)
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
        if kinds[path] == "group":
            return "", ""
        if path not in previews:
            try:
                value = filesystem.read(path)
            except Exception as exc:
                return "", f"Error: {exc}"
            previews[path] = type_name(value), preview(value)
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
                details.insert("", "end", text=name, values=(type_name(value), preview(value)), tags=("attribute",))

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

    def enter(event):
        item = details.identify_row(event.y)
        if kinds.get(item) == "dataset":
            open_value(master, item, filesystem.read(item), attrs=filesystem.attrs(item, defaults=False))
        elif item and tree.exists(item):
            tree.item(tree.parent(item), open=True)
            tree.selection_set(item)
            tree.focus(item)
            tree.see(item)

    def set_root(path):
        tree.delete(*tree.get_children())
        details.delete(*details.get_children())
        for cache in (kinds, children, previews):
            cache.clear()
        root = insert("", path, path.rstrip("/").rsplit("/", 1)[-1] or "/")
        tree.item(root, open=True)
        tree.focus(root)
        tree.selection_set(root)

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
            attrs = filesystem.attrs(item)
            for name, viewer in suitable_viewers("dataset", attrs):
                menu.add_command(label=f"Open with {name}", command=lambda viewer=viewer: open_value(
                    master, item, filesystem.read(item), viewer, attrs))
            menu.add_separator()
        menu.add_command(label=f"Copy path  {item}" if item else "Copy value",
                         command=lambda: copy(item or details.set(selected, "value")))
        menu.tk_popup(event.x_root, event.y_root)

    tree.bind("<<TreeviewOpen>>", show)
    tree.bind("<<TreeviewSelect>>", show)
    details.bind("<Double-1>", enter)
    details.bind("<<TreeviewSelect>>", attributes)
    details.bind("<Button-3>", open_menu)
    return set_root


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="lf-browse-fs", description="Browse a filesystem specification read-only.")
    parser.add_argument("spec", help='specification JSON file, or inline JSON such as \'{"paths": ["*.h5"]}\'')
    parser.add_argument("--relative-to", type=Path,
                        help="base directory of specification paths "
                             "(default: the file's directory, or the working directory for inline JSON)")
    args = parser.parse_args(argv)

    if args.spec.lstrip().startswith("{"):
        spec, base = json.loads(args.spec), Path.cwd()
    else:
        spec, base = json.loads(Path(args.spec).read_text(encoding="utf-8")), Path(args.spec).resolve().parent
    browse_filesystem(FileSystem.from_spec(spec, relative_to=args.relative_to or base))
