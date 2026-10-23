"""Minimal read-only Tk explorer for a filesystem with listdir() and read()."""

from reprlib import repr as short_repr

def explore(filesystem):
    """Block until closed; list only visited nodes and read only visited leaves."""
    import tkinter as tk
    from tkinter import ttk

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
    children = {}
    previews = {}

    def insert(parent, path, name):
        tree.insert(parent, "end", iid=path, text=name)
        tree.insert(path, "end")  # Expansion marker; no child inspection.
        return path

    def populate(item):
        if item not in children:
            try:
                names = filesystem.listdir(item)
            except NotADirectoryError:
                names = None
            tree.delete(*tree.get_children(item))
            children[item] = None if names is None else tuple(
                insert(item, f"{item.rstrip('/')}/{name}", name)
                for name in names
            )
        return children[item]

    def label(item):
        marker = ("▪ " if children[item] is None else "▸ ") if item in children else ""
        return marker + tree.item(item, "text")

    def show(event=None):
        item = tree.focus()
        if not item:
            return
        details.delete(*details.get_children())
        try:
            entries = populate(item)
            if entries is None:
                preview = short_repr(filesystem.read(item))
                previews[item] = preview[:160]
                details.insert("", "end", iid="value", text=label(item),
                               values=(preview,))
            else:
                for child in entries:
                    details.insert("", "end", iid=child, text=label(child),
                                   values=(previews.get(child, ""),))
        except Exception as exc:
            details.insert("", "end", iid="error", text="Error", values=(str(exc),))

    def enter(event):
        item = details.identify_row(event.y)
        if item and tree.exists(item):
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
