"""Minimal read-only Tk explorer with relocated mount markers."""

from pprint import pformat
from reprlib import repr as short_repr

def explore(filesystem):
    """Preview visible leaves; discover group children only when visited."""
    import tkinter as tk

    window = tk.Tk()
    window.title("lazyfields")
    browser(window, filesystem)("/")
    window.mainloop()


def browser(master, filesystem):
    """Pack tree and details panes into master; return a function that roots them at a group."""
    import tkinter as tk
    from tkinter import ttk
    from tkinter.scrolledtext import ScrolledText

    panes = ttk.Panedwindow(master, orient="horizontal")
    panes.pack(fill="both", expand=True)
    tree = ttk.Treeview(panes, show="tree")
    details = ttk.Treeview(panes, columns=("value",))
    details.heading("#0", text="Node")
    details.heading("value", text="Value")
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
        elif kind != "group" and (kind == "scalar" or value is not None):
            previews[path] = short_repr(value)[:160]
        kinds[path] = kind
        group = kind == "group"
        origin = node.origin
        file_root = origin is not None and origin.infile_path is None
        os_directory = group and origin is None and (
            filesystem._namespace_path / path.lstrip("/")
        ).is_dir()
        if file_root:
            marker = "📄 "
        elif os_directory:
            marker = "📁 "
        else:
            marker = {"group": "⊞ ", "dataset": "▦ ", "scalar": "· "}[kind]
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
            children[item] = tuple(
                insert(item, f"{item.rstrip('/')}/{name}", name, kind, value)
                for name, kind, value in filesystem.items(item)
            )
        return children[item]

    def preview(path):
        if kinds[path] == "group":
            return ""
        if path not in previews:
            try:
                previews[path] = short_repr(filesystem.read(path))[:160]
            except Exception as exc:
                return f"Error: {exc}"
        return previews[path]

    def show_dataset(path, value):
        popup = tk.Toplevel(master)
        popup.title(path)
        text = ScrolledText(popup, width=100, height=30, wrap="word")
        text.pack(fill="both", expand=True)
        text.insert("1.0", pformat(value, width=100, sort_dicts=False))
        text.configure(state="disabled")

    def show(event=None):
        item = tree.focus()
        if not item:
            return
        details.delete(*details.get_children())
        try:
            entries = populate(item)
            if entries is None:
                details.insert("", "end", iid="value", text=tree.item(item, "text"),
                               values=(preview(item),))
            else:
                for child in entries:
                    details.insert("", "end", iid=child, text=tree.item(child, "text"),
                                   values=(preview(child),))
        except Exception as exc:
            details.insert("", "end", iid="error", text="Error", values=(str(exc),))

    def enter(event):
        item = details.identify_row(event.y)
        if item == "value":
            item = tree.focus()
        if kinds.get(item) == "dataset":
            show_dataset(item, filesystem.read(item))
        elif item and tree.exists(item):
            tree.item(tree.parent(item), open=True)
            tree.selection_set(item)
            tree.focus(item)
            tree.see(item)

    def set_root(path):
        tree.delete(*tree.get_children())
        details.delete(*details.get_children())
        kinds.clear()
        children.clear()
        previews.clear()
        root = insert("", path, path.rstrip("/").rsplit("/", 1)[-1] or "/")
        tree.item(root, open=True)
        tree.focus(root)
        tree.selection_set(root)

    tree.bind("<<TreeviewOpen>>", show)
    tree.bind("<<TreeviewSelect>>", show)
    details.bind("<Double-1>", enter)
    return set_root
