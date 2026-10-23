"""Minimal read-only Tk explorer with logical mount markers."""

from pprint import pformat
from reprlib import repr as short_repr

def explore(filesystem):
    """Preview visible leaves; discover group children only when visited."""
    import tkinter as tk
    from tkinter import ttk
    from tkinter.scrolledtext import ScrolledText

    window = tk.Tk()
    window.title("lazyfields")
    panes = ttk.Panedwindow(window, orient="horizontal")
    panes.pack(fill="both", expand=True)
    tree = ttk.Treeview(panes, show="tree")
    details = ttk.Treeview(panes, columns=("value",))
    details.heading("#0", text="Node")
    details.heading("value", text="Value")
    panes.add(tree, weight=1)
    panes.add(details, weight=2)
    kinds = {}
    children = {}
    previews = {}

    def insert(parent, path, name):
        node = filesystem.node(path)
        kind = kinds[path] = node.kind
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
        prefix = "↪ " if node.is_logical else ""
        tree.insert(parent, "end", iid=path, text=prefix + marker + name)
        if group:
            tree.insert(path, "end")  # Children remain undiscovered until visited.
        else:
            children[path] = None
        return path

    def populate(item):
        if item not in children:
            names = filesystem.listdir(item)
            tree.delete(*tree.get_children(item))
            children[item] = tuple(
                insert(item, f"{item.rstrip('/')}/{name}", name)
                for name in names
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
        popup = tk.Toplevel(window)
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

    root = insert("", "/", "/")
    tree.item(root, open=True)
    tree.focus(root)
    tree.bind("<<TreeviewOpen>>", show)
    tree.bind("<<TreeviewSelect>>", show)
    details.bind("<Double-1>", enter)
    tree.selection_set(root)
    window.mainloop()
