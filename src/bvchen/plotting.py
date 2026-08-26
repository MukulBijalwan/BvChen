"""Plotting helpers for the bivariate Chen distribution.

Provides surface and contour plots of the absolutely continuous part of
the joint density (on the log scale, which is better behaved for
bathtub-shaped densities), as in Fig. 1 of the paper.
"""

from __future__ import annotations

import numpy as np

__all__ = ["surface_plot", "contour_plot"]


def _grid(xmax=3.0, n=120):
    x = np.linspace(1e-3, xmax, n)
    X, Y = np.meshgrid(x, x)
    return X, Y


def surface_plot(alpha1, alpha2, alpha3, beta, xmax=3.0, n=120,
                 log=True, ax=None):
    """3-D surface plot of the (log) joint density.

    Evaluates the absolutely continuous part of :math:`f` on a grid and
    draws a perspective surface, as in Fig. 1 of the paper.

    Returns
    -------
    matplotlib Axes3D
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d)
    import matplotlib.pyplot as plt

    from .distribution import dbvch

    X, Y = _grid(xmax, n)
    Z = np.asarray(dbvch(X, Y, alpha1, alpha2, alpha3, beta,
                         component="ac", log=log))
    # -inf on the diagonal of the log-density would break the mesh
    Z = np.nan_to_num(Z, neginf=-20.0)

    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none",
                           alpha=0.95, antialiased=True)
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    ax.set_zlabel(r"$\log f(z_1,z_2)$" if log else r"$f(z_1,z_2)$")
    ax.set_title(f"BvCh($\\alpha_1={alpha1:g}, \\alpha_2={alpha2:g}, "
                 f"\\alpha_3={alpha3:g}, \\beta={beta:g}$)")
    ax.figure.colorbar(surf, ax=ax, shrink=0.6)
    return ax


def contour_plot(alpha1, alpha2, alpha3, beta, xmax=3.0, n=120,
                 log=True, levels=12, ax=None):
    """Contour plot of the (log) joint density.

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt

    from .distribution import dbvch

    X, Y = _grid(xmax, n)
    Z = np.asarray(dbvch(X, Y, alpha1, alpha2, alpha3, beta,
                         component="ac", log=log))
    Z = np.nan_to_num(Z, neginf=-20.0)

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6))
    cf = ax.contourf(X, Y, Z, levels=levels, cmap="viridis")
    ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.5,
               alpha=0.5)
    ax.plot([0, xmax], [0, xmax], "r--", lw=1.2, label="diagonal (singular)")
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_title(f"BvCh contours ($\\beta={beta:g}$)")
    ax.legend(loc="upper left")
    ax.figure.colorbar(cf, ax=ax)
    ax.set_aspect("equal")
    return ax