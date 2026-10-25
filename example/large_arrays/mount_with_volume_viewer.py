"""Explore the folder mount with a PyVista volume rendering viewer registered for 3D arrays."""

import numpy as np

from lazyfields.browser import register_viewer
from mount_by_folder import main


def is_volume(kind, attrs):
    return len(attrs["shape"]) == 3 and np.dtype(attrs["dtype"]).kind in "biuf"


def show_volume(window, value):
    """Render value in a PyVista window instead of the Tk one, which stays hidden to keep it interactive."""
    import pyvista as pv

    plotter = pv.Plotter(title=window.winfo_toplevel().title())
    plotter.add_volume(np.asarray(value), cmap="viridis")
    plotter.show_axes()
    plotter.show(interactive_update=True)
    window.winfo_toplevel().withdraw()

    def update():
        if not window.winfo_exists():
            return
        if plotter.iren.interactor.GetDone():  # The PyVista window was closed.
            window.winfo_toplevel().destroy()
            return
        plotter.update()
        window.after(30, update)

    window.bind("<Destroy>", lambda event: plotter.close())
    update()


register_viewer("Volume", is_volume, show_volume)

if __name__ == "__main__":
    main()
