import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Line:
    """Lightweight 3D line that can be updated with set_data_3d."""

    def __init__(self, ax, c="k", linewidth=1.5):
        self.ax = ax
        (self.artist,) = ax.plot([], [], [], color=c, linewidth=linewidth)

    def draw_from_to(self, x0, x1):
        self.artist.set_data_3d([x0[0], x1[0]], [x0[1], x1[1]], [x0[2], x1[2]])


class Arrow:
    """Arrow rendered as a line (much faster & easier to update than quiver)."""

    def __init__(self, ax, c="r", length=1.0, linewidth=2.0):
        self.ax = ax
        self.length = length
        (self.artist,) = ax.plot([], [], [], color=c, linewidth=linewidth)

    def draw_from_to(self, x, u):
        tip = x + u  # u already encodes direction & scale
        self.artist.set_data_3d([x[0], tip[0]], [x[1], tip[1]], [x[2], tip[2]])


class Dot3D:
    """A tiny filled disc in data units (replaces the DPI-scaled marker 'Sphere')."""

    def __init__(self, ax, r=0.01, c="k", resolution=40):
        self.ax = ax
        self.r = r
        self.color = c
        theta = np.linspace(0.0, 2 * np.pi, resolution, endpoint=True)
        # disc in the XY-plane, small radius r
        self.circle_local = np.vstack(
            [r * np.cos(theta), r * np.sin(theta), np.zeros_like(theta)]
        )
        self.coll = Poly3DCollection(
            [[(0.0, 0.0, 0.0)]], facecolor=c, edgecolor="none", alpha=1.0
        )
        # Safer sorting for visibility
        if hasattr(self.coll, "set_zsort"):
            self.coll.set_zsort("average")
        self.ax.add_collection3d(self.coll)

    def draw_at(self, x):
        x = np.asarray(x).reshape(
            3,
        )
        pts = self.circle_local + x[:, None]
        verts = [tuple(p) for p in pts.T]
        try:
            self.coll.set_verts([verts], closed=True)
        except TypeError:
            self.coll.set_verts([verts])


class Plate:
    def __init__(
        self,
        ax,
        r,
        c="k",
        x=np.array([0, 0, 0.0]),
        R=np.eye(3),
        resolution=60,
        alpha=1.0,
    ):
        self.ax = ax
        self.r = r
        self.color = c
        self.alpha = alpha

        theta = np.linspace(0.0, 2 * np.pi, resolution, endpoint=True)
        self.circle_local = np.vstack(
            [r * np.cos(theta), r * np.sin(theta), np.zeros_like(theta)]
        )

        self.collection = Poly3DCollection(
            [[(0.0, 0.0, 0.0)]],
            facecolor=self.color,
            edgecolor="none",  # was self.color
            linewidth=0.0,
            alpha=self.alpha,
        )
        if hasattr(self.collection, "set_zsort"):
            self.collection.set_zsort("average")

        ax.add_collection3d(self.collection)
        self.draw_at(x, R)

    def set_alpha(self, a: float):
        fc = self.collection.get_facecolor()
        if len(fc):
            r, g, b, _ = fc[0]
            self.collection.set_facecolor([(r, g, b, a)])

    def draw_at(self, x, R):
        pts = (R @ self.circle_local) + x[:, None]
        verts = [tuple(p) for p in pts.T]
        try:
            self.collection.set_verts([verts], closed=True)
        except TypeError:
            self.collection.set_verts([verts])
