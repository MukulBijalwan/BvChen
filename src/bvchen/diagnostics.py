"""Goodness-of-fit tools for the bivariate Chen law.

Implements the Kolmogorov-Smirnov testing strategy of Meintanis (2007),
applied to the two marginals and to :math:`\\min(Z_1, Z_2)`, together
with theoretical-vs-sample Q-Q diagnostics.

References
----------
Meintanis, S. G. (2007). Test of fit for Marshall-Olkin distributions
with applications. *J. Stat. Comput. Simul.* 77, 171--179.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .chen import pchen, qchen
from .distribution import pmarg1, pmarg2

__all__ = ["gof", "qqplot"]


def gof(z1, z2, alpha1, alpha2, alpha3, beta):
    """Kolmogorov-Smirnov goodness-of-fit tests for the BvCh model.

    Tests the null hypothesis that the data come from the
    BvCh(:math:`\\alpha_1, \\alpha_2, \\alpha_3, \\beta`) law by
    comparing :math:`Z_1`, :math:`Z_2` and :math:`\\min(Z_1, Z_2)`
    against their theoretical marginal CDFs.

    Returns
    -------
    dict
        ``ks_z1``, ``ks_z2``, ``ks_min``: the three ``KstestResult``
        objects.
    """
    z1 = np.asarray(z1, dtype=float).ravel()
    z2 = np.asarray(z2, dtype=float).ravel()
    if z1.size != z2.size:
        raise ValueError("z1 and z2 must have the same length")

    ks_z1 = stats.ks_1samp(z1, lambda q: pmarg1(q, alpha1, alpha2,
                                                alpha3, beta))
    ks_z2 = stats.ks_1samp(z2, lambda q: pmarg2(q, alpha1, alpha2,
                                                alpha3, beta))
    a_total = alpha1 + alpha2 + alpha3
    ks_min = stats.ks_1samp(np.minimum(z1, z2),
                            lambda q: pchen(q, a_total, beta))
    return {"ks_z1": ks_z1, "ks_z2": ks_z2, "ks_min": ks_min}


def qqplot(z1, z2, alpha1, alpha2, alpha3, beta, ax=None):
    """Theoretical Q-Q plot of marginals and minimum against sample data.

    Returns
    -------
    matplotlib Axes
        The axes with the three Q-Q panels (``z1``, ``z2``,
        ``min(z1, z2)``), each against its theoretical Chen quantiles.
    """
    import matplotlib.pyplot as plt

    z1 = np.sort(np.asarray(z1, dtype=float).ravel())
    z2 = np.sort(np.asarray(z2, dtype=float).ravel())
    zm = np.sort(np.minimum(z1, z2))
    n = z1.size
    if n < 2:
        raise ValueError("need at least two observations")
    pp = (np.arange(n) + 0.5) / n
    a_total = alpha1 + alpha2 + alpha3

    theory = {
        "z1": qchen(pp, alpha1 + alpha3, beta),
        "z2": qchen(pp, alpha2 + alpha3, beta),
        "min": qchen(pp, a_total, beta),
    }
    sample = {"z1": z1, "z2": z2, "min": zm}

    if ax is None:
        _, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax = np.atleast_1d(ax)
    all_vals = np.concatenate([theory["z1"], theory["z2"], theory["min"],
                               z1, z2, zm])
    hi = float(np.nanmax(all_vals)) * 1.05
    lims = (0.0, hi)
    a_total = alpha1 + alpha2 + alpha3
    panels = [("z1", alpha1 + alpha3, "Q-Q: Z1"),
              ("z2", alpha2 + alpha3, "Q-Q: Z2"),
              ("min", a_total, "Q-Q: min(Z1,Z2)")]
    for k, (name, a_par, title) in enumerate(panels):
        ax[k].plot(theory[name], sample[name], "o", ms=4,
                   mfc="steelblue", mec="steelblue")
        ax[k].plot([0, hi], [0, hi], "r-", lw=2)
        ax[k].set_title(title)
        ax[k].set_xlim(lims)
        ax[k].set_ylim(lims)
        ax[k].set_xlabel("theoretical quantile")
        ax[k].set_ylabel("sample quantile")
    ax[0].figure.tight_layout()
    return ax