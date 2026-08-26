"""bvchen: the bivariate Chen distribution.

A Python package implementing the bivariate Chen (BvCh) distribution of
Gupta, Pundir, Sharma and Mesfioui (2022), together with the univariate
Chen (2000) building blocks.

The distribution is built from three independent Chen components
:math:`U_i \\sim \\mathrm{Chen}(\\alpha_i, \\beta)` via
:math:`Z_1 = \\min(U_1, U_3)`, :math:`Z_2 = \\min(U_2, U_3)`.

Example
-------
>>> import bvchen
>>> dist = bvchen.BvChen(alpha1=0.8, alpha2=1.2, alpha3=0.5, beta=1.5)
>>> dist.pdf(0.5, 0.5)
>>> dist.rvs(10, random_state=42)
"""

from .chen import (chen_moment, dchen, fit_chen, hchen, pchen, qchen,
                   rchen, schen)
from .distribution import (BvChen, bvch_moment, dbvch, dcond1, dcond2,
                           dmarg1, dmarg2, hcond1, hcond2, kendall_tau,
                           pbvch, pmarg1, pmarg2, rbvch, sbvch, survcop)
from .estimation import BvChenFit, fit_bvchen, loglik_bvch, profile_beta
from .diagnostics import gof, qqplot

__version__ = "0.1.0"

__all__ = [
    # univariate Chen building blocks
    "dchen", "pchen", "schen", "qchen", "rchen", "hchen",
    "fit_chen", "chen_moment",
    # bivariate distribution
    "BvChen", "dbvch", "pbvch", "sbvch", "rbvch",
    "dmarg1", "dmarg2", "pmarg1", "pmarg2",
    "dcond1", "dcond2", "hcond1", "hcond2",
    "survcop", "kendall_tau", "bvch_moment",
    # estimation
    "fit_bvchen", "loglik_bvch", "profile_beta", "BvChenFit",
    # diagnostics
    "gof", "qqplot",
    "__version__",
]