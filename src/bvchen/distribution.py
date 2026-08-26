"""The bivariate Chen (BvCh) distribution.

Construction
------------
Let :math:`U_1, U_2, U_3` be independent Chen variables with common power
parameter :math:`\\beta` and scale parameters :math:`\\alpha_1, \\alpha_2,
\\alpha_3`, i.e. :math:`U_i \\sim \\mathrm{Chen}(\\alpha_i, \\beta)`.  Define

.. math::

    Z_1 = \\min(U_1, U_3), \\qquad Z_2 = \\min(U_2, U_3).

Then :math:`(Z_1, Z_2)` follows the *bivariate Chen* distribution
:math:`\\mathrm{BvCh}(\\alpha_1, \\alpha_2, \\alpha_3, \\beta)`.  It has

* absolutely continuous parts on :math:`\\{z_1 < z_2\\}` and
  :math:`\\{z_1 > z_2\\}` with total mass
  :math:`(\\alpha_1+\\alpha_2)/(\\alpha_1+\\alpha_2+\\alpha_3)`, and
* a singular part on the diagonal :math:`z_1 = z_2` with mass
  :math:`\\alpha_3/(\\alpha_1+\\alpha_2+\\alpha_3)`.

Joint survival (with :math:`u(z) = e^{z^\\beta}-1`):

.. math::

    S(z_1, z_2) = \\exp\\Big\\{
        -(\\alpha_1+\\alpha_3)\\,u(z_1)
        -(\\alpha_2+\\alpha_3)\\,u(z_2)
        + \\alpha_3\\,u\\big(\\min(z_1,z_2)\\big)
    \\Big\\}.

Marginals are again Chen: :math:`Z_1 \\sim \\mathrm{Chen}(\\alpha_1+\\alpha_3,
\\beta)` and :math:`Z_2 \\sim \\mathrm{Chen}(\\alpha_2+\\alpha_3, \\beta)`;
the minimum :math:`\\min(Z_1,Z_2) \\sim \\mathrm{Chen}(\\alpha_1+\\alpha_2+
\\alpha_3, \\beta)`.

References
----------
Gupta, P. K., Pundir, P. S., Sharma, V. K., Mesfioui, M. (2022). Bivariate
extension of bathtub-shaped distribution. *Life Cycle Reliability and
Safety Engineering* 11, 247--259.
"""

from __future__ import annotations

import numpy as np

from ._utils import log_singular, upr, validate_bvch
from .chen import dchen, pchen, qchen, schen

__all__ = [
    "BvChen",
    "dbvch",
    "sbvch",
    "pbvch",
    "rbvch",
    "dmarg1", "dmarg2", "pmarg1", "pmarg2",
    "dcond1", "dcond2", "hcond1", "hcond2",
    "survcop",
    "kendall_tau",
    "bvch_moment",
]


def _log_dens_ac_lo(x, y, alpha1, alpha2, alpha3, beta):
    """log f on {z1 < z2} = f_Ch(a1)(z1) * f_Ch(a2+a3)(z2)."""
    return (dchen(x, alpha1, beta, log=True)
            + dchen(y, alpha2 + alpha3, beta, log=True))


def _log_dens_ac_hi(x, y, alpha1, alpha2, alpha3, beta):
    """log f on {z1 > z2} = f_Ch(a1+a3)(z1) * f_Ch(a2)(z2)."""
    return (dchen(x, alpha1 + alpha3, beta, log=True)
            + dchen(y, alpha2, beta, log=True))


def dbvch(x, y, alpha1, alpha2, alpha3, beta, log=False,
          component="full"):
    """Joint density of the bivariate Chen distribution.

    Parameters
    ----------
    x, y : array_like
        Non-negative quantiles (recycled to a common length).
    alpha1, alpha2, alpha3, beta : float
        Parameters (all positive).
    log : bool, optional
        If True, the log-density is returned.
    component : {"full", "ac", "sing"}
        ``"full"`` evaluates the singular density on the diagonal and the
        absolutely-continuous density elsewhere; ``"ac"`` returns only the
        absolutely continuous part (zero on the diagonal); ``"sing"``
        returns only the singular diagonal density (zero off the diagonal).

    Returns
    -------
    ndarray
        Density values.
    """
    validate_bvch(alpha1, alpha2, alpha3, beta)
    if component not in ("full", "ac", "sing"):
        raise ValueError("component must be 'full', 'ac' or 'sing'")
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    scalar = xa.ndim == 0 and ya.ndim == 0
    n = max(xa.size, ya.size)
    x = np.broadcast_to(xa, (n,))
    y = np.broadcast_to(ya, (n,))

    val = np.zeros(n, dtype=float)
    diag = x == y
    lo = x < y
    hi = x > y

    if component != "sing":                       # absolutely continuous part
        with np.errstate(over="ignore", invalid="ignore"):
            val[lo] = np.exp(_log_dens_ac_lo(x[lo], y[lo], alpha1, alpha2,
                                             alpha3, beta))
            val[hi] = np.exp(_log_dens_ac_hi(x[hi], y[hi], alpha1, alpha2,
                                             alpha3, beta))
    if component != "ac":                         # singular diagonal part
        ld = log_singular(x[diag], alpha1, alpha2, alpha3, beta)
        with np.errstate(over="ignore", invalid="ignore"):
            val[diag] = np.exp(ld)

    if log:
        with np.errstate(divide="ignore"):
            val = np.log(val)
    if scalar:
        return float(val[0])
    return val
def sbvch(x, y, alpha1, alpha2, alpha3, beta):
    """Joint survival function :math:`P(Z_1 > x, Z_2 > y)`.

    .. math::
        S(x,y) = \\exp\\{-(\\alpha_1+\\alpha_3)u(x)
                 -(\\alpha_2+\\alpha_3)u(y)+\\alpha_3 u(\\min(x,y))\\}

    with :math:`u(z) = e^{z^\\beta}-1`.
    """
    validate_bvch(alpha1, alpha2, alpha3, beta)
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    scalar = xa.ndim == 0 and ya.ndim == 0
    n = max(xa.size, ya.size)
    x = np.broadcast_to(xa, (n,))
    y = np.broadcast_to(ya, (n,))

    out = np.zeros(n, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    xf = np.maximum(x[finite], 0.0)
    yf = np.maximum(y[finite], 0.0)
    m = np.minimum(xf, yf)
    with np.errstate(over="ignore", invalid="ignore"):
        exponent = (-(alpha1 + alpha3) * upr(xf, beta)
                    - (alpha2 + alpha3) * upr(yf, beta)
                    + alpha3 * upr(m, beta))
    out[finite] = np.exp(exponent)
    if scalar:
        return float(out[0])
    return out


def pbvch(x, y, alpha1, alpha2, alpha3, beta):
    """Joint CDF :math:`P(Z_1 \\le x, Z_2 \\le y)`.

    Computed through the inclusion-exclusion identity
    :math:`F = 1 - S_1 - S_2 + S`, which remains valid for laws with a
    singular component.
    """
    validate_bvch(alpha1, alpha2, alpha3, beta)
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    scalar = xa.ndim == 0 and ya.ndim == 0
    n = max(xa.size, ya.size)
    x = np.broadcast_to(xa, (n,))
    y = np.broadcast_to(ya, (n,))

    s1 = np.where(x == np.inf, 0.0, schen(x, alpha1 + alpha3, beta))
    s2 = np.where(y == np.inf, 0.0, schen(y, alpha2 + alpha3, beta))
    s = np.where((x == np.inf) | (y == np.inf), 0.0,
                 sbvch(x, y, alpha1, alpha2, alpha3, beta))
    out = np.clip(1.0 - s1 - s2 + s, 0.0, 1.0)
    if scalar:
        return float(out[0])
    return out


def rbvch(n, alpha1, alpha2, alpha3, beta, random_state=None):
    """Generate ``n`` draws from the bivariate Chen law.

    On the exponential scale :math:`U_i = e^{X_i^\\beta}-1 \\sim
    \\mathrm{Exp}(\\alpha_i)` the construction is a Marshall-Olkin
    competing-risks scheme:

    .. math::
        Z_1 = \\big(\\ln(1+\\min(E_0, E_1))\\big)^{1/\\beta}, \\qquad
        Z_2 = \\big(\\ln(1+\\min(E_0, E_2))\\big)^{1/\\beta}

    with independent :math:`E_i \\sim \\mathrm{Exp}(\\alpha_i)`.

    Returns
    -------
    ndarray of shape (n, 2)
        Columns ``z1`` and ``z2``.
    """
    validate_bvch(alpha1, alpha2, alpha3, beta)
    rng = np.random.default_rng(random_state)
    e0 = rng.exponential(scale=1.0 / alpha3, size=int(n))
    e1 = rng.exponential(scale=1.0 / alpha1, size=int(n))
    e2 = rng.exponential(scale=1.0 / alpha2, size=int(n))
    z1 = np.log1p(np.minimum(e0, e1)) ** (1.0 / beta)
    z2 = np.log1p(np.minimum(e0, e2)) ** (1.0 / beta)
    return np.column_stack([z1, z2])
def dmarg1(x, alpha1, alpha2, alpha3, beta, log=False):
    """Density of the first marginal :math:`Z_1 \\sim \\mathrm{Chen}(\\alpha_1+\\alpha_3, \\beta)`."""
    return dchen(x, alpha1 + alpha3, beta, log=log)


def dmarg2(y, alpha1, alpha2, alpha3, beta, log=False):
    """Density of the second marginal :math:`Z_2 \\sim \\mathrm{Chen}(\\alpha_2+\\alpha_3, \\beta)`."""
    return dchen(y, alpha2 + alpha3, beta, log=log)


def pmarg1(q, alpha1, alpha2, alpha3, beta):
    """CDF of the first marginal :math:`Z_1`."""
    return pchen(q, alpha1 + alpha3, beta)


def pmarg2(q, alpha1, alpha2, alpha3, beta):
    """CDF of the second marginal :math:`Z_2`."""
    return pchen(q, alpha2 + alpha3, beta)


class _CondResult(np.ndarray):
    """ndarray subclass carrying the diagonal atom of a conditional law."""

    atom: float = 0.0


def dcond1(z1, y, alpha1, alpha2, alpha3, beta):
    """Conditional density of :math:`Z_1` given :math:`Z_2 = y`.

    The conditional law has an atom at :math:`z_1 = y` (size returned in
    the ``atom`` attribute) and a piecewise continuous part:

    .. math::
        f_{Z_1|Z_2}(z_1|y) =
        \\begin{cases}
        f_{\\mathrm{Ch}(\\alpha_1)}(z_1), & z_1 < y,\\\\
        \\dfrac{f_{\\mathrm{Ch}(\\alpha_2)}(y)\\,f_{\\mathrm{Ch}(\\alpha_1+\\alpha_3)}(z_1)}
             {f_{\\mathrm{Ch}(\\alpha_2+\\alpha_3)}(y)}, & z_1 > y.
        \\end{cases}

    Returns
    -------
    ndarray
        Conditional density of the strictly off-diagonal part; the point
        mass :math:`P(Z_1 = y \\mid Z_2 = y)` is exposed as the
        ``atom`` attribute.
    """
    validate_bvch(alpha1, alpha2, alpha3, beta)
    z1a = np.atleast_1d(np.asarray(z1, dtype=float))
    fm2 = dmarg2(y, alpha1, alpha2, alpha3, beta)

    val = np.empty(z1a.shape, dtype=float)
    below = z1a < y
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        val[below] = dchen(z1a[below], alpha1, beta)
        above = ~below
        val[above] = (dchen(y, alpha2, beta)
                      * dchen(z1a[above], alpha1 + alpha3, beta) / fm2)

    # atom: g(y) / f_m2(y)
    atom = np.exp(log_singular(np.asarray(y, dtype=float),
                               alpha1, alpha2, alpha3, beta)) / fm2
    atom = float(np.nan_to_num(np.asarray(atom).ravel()[0],
                               nan=0.0, posinf=np.inf, neginf=0.0))
    out = np.asarray(val).view(_CondResult)
    out.atom = atom
    return out


def dcond2(z2, x, alpha1, alpha2, alpha3, beta):
    """Conditional density of :math:`Z_2` given :math:`Z_1 = x`.

    Symmetric to :func:`dcond1` with parameters 1 and 2 exchanged.
    """
    return dcond1(z2, x, alpha2, alpha1, alpha3, beta)


def _scond1(z, y, alpha1, alpha2, alpha3, beta):
    """Conditional survival :math:`P(Z_1 > z \\mid Z_2 = y)` (exact)."""
    fm2 = dmarg2(y, alpha1, alpha2, alpha3, beta)
    za = np.atleast_1d(np.asarray(z, dtype=float))
    out = np.empty(za.shape, dtype=float)
    below = za < y
    out[below] = 1.0 - pchen(za[below], alpha1, beta)
    notbelow = ~below
    out[notbelow] = ((pmarg2(y, alpha1, alpha2, alpha3, beta)
                      - pbvch(za[notbelow], y, alpha1, alpha2, alpha3,
                              beta)) / fm2)
    return out


def hcond1(z1, y, alpha1, alpha2, alpha3, beta):
    """Conditional hazard of the continuous part of :math:`Z_1 \\mid Z_2 = y`.

    Returns ``NaN`` exactly at ``z1 == y`` where only the point mass lives.
    """
    z1a = np.atleast_1d(np.asarray(z1, dtype=float))
    dens = np.asarray(dcond1(z1a, y, alpha1, alpha2, alpha3, beta))
    surv = _scond1(z1a, y, alpha1, alpha2, alpha3, beta)
    out = np.full(z1a.shape, np.nan)
    ok = (z1a != y) & (surv > 0)
    out[ok] = dens[ok] / surv[ok]
    return out


def hcond2(z2, x, alpha1, alpha2, alpha3, beta):
    """Conditional hazard of the continuous part of :math:`Z_2 \\mid Z_1 = x`."""
    return hcond1(z2, x, alpha2, alpha1, alpha3, beta)
def bvch_moment(r, s, alpha1, alpha2, alpha3, beta, tol=1e-8):
    r"""Raw joint moment :math:`E[Z_1^r Z_2^s]` of the BvCh law.

    Following the paper's Proposition 4, the moment splits into three
    integrals over :math:`\{z_1<z_2\}`, :math:`\{z_1>z_2\}` and the
    diagonal.  Each is computed on the exponential scale
    :math:`u = e^{z^\beta}-1` (where :math:`f_{\mathrm{Ch}(\alpha)}(z)\\,dz
    = \\alpha e^{-\\alpha u}du`), which is numerically very stable:

    .. math::
        E[Z_1^r Z_2^s] = I_1 + I_2 + I_3,

    .. math::
        I_1 &= \\alpha_1(\\alpha_2+\\alpha_3)\\int_0^\\infty
               (\\ln(1+u_2))^{s/\\beta} e^{-(\\alpha_2+\\alpha_3)u_2}
               \\Big[\\int_0^{u_2}(\\ln(1+u_1))^{r/\\beta}
               e^{-\\alpha_1 u_1}du_1\\Big]du_2 \\\\
        I_2 &= \\alpha_2(\\alpha_1+\\alpha_3)\\int_0^\\infty
               (\\ln(1+u_1))^{r/\\beta} e^{-(\\alpha_1+\\alpha_3)u_1}
               \\Big[\\int_0^{u_1}(\\ln(1+u_2))^{s/\\beta}
               e^{-\\alpha_2 u_2}du_2\\Big]du_1 \\\\
        I_3 &= \\alpha_3\\int_0^\\infty (\\ln(1+u))^{(r+s)/\\beta}
               e^{-(\\alpha_1+\\alpha_2+\\alpha_3)u}du.
    """
    from scipy import integrate

    validate_bvch(alpha1, alpha2, alpha3, beta)
    if r < 0 or s < 0:
        raise ValueError("r and s must be non-negative")
    total = alpha1 + alpha2 + alpha3

    def inner_power(u1, lam, power):
        return np.log1p(u1) ** power * np.exp(-lam * u1)

    # I1: region z1 < z2
    def inner1(u2):
        val, _ = integrate.quad(inner_power, 0.0, u2,
                                args=(alpha1, r / beta),
                                epsabs=tol, epsrel=tol)
        return val

    def i1_inner(u2):
        return (np.log1p(u2) ** (s / beta)
                * np.exp(-(alpha2 + alpha3) * u2) * inner1(u2))

    if r == 0 and s == 0:
        part_lt = alpha1 / total          # P(Z1 < Z2)
    else:
        val, _ = integrate.quad(i1_inner, 0.0, np.inf,
                                epsabs=tol, epsrel=tol)
        part_lt = alpha1 * (alpha2 + alpha3) * val

    # I2: region z1 > z2
    def inner2(u1):
        val, _ = integrate.quad(inner_power, 0.0, u1,
                                args=(alpha2, s / beta),
                                epsabs=tol, epsrel=tol)
        return val

    def i2_inner(u1):
        return (np.log1p(u1) ** (r / beta)
                * np.exp(-(alpha1 + alpha3) * u1) * inner2(u1))

    if r == 0 and s == 0:
        part_gt = alpha2 / total          # P(Z1 > Z2)
    else:
        val, _ = integrate.quad(i2_inner, 0.0, np.inf,
                                epsabs=tol, epsrel=tol)
        part_gt = alpha2 * (alpha1 + alpha3) * val

    # I3: diagonal singular part
    if r == 0 and s == 0:
        part_diag = alpha3 / total        # P(Z1 = Z2)
    else:
        def diag_integrand(u):
            return (np.log1p(u) ** ((r + s) / beta)
                    * np.exp(-total * u))

        val, _ = integrate.quad(diag_integrand, 0.0, np.inf,
                                epsabs=tol, epsrel=tol)
        part_diag = alpha3 * val

    return part_lt + part_gt + part_diag
def survcop(u, v, alpha1, alpha2, alpha3):
    """Survival copula :math:`\\hat C(u,v)` of the BvCh law.

    .. math::
        \\hat C(u, v) = \\min\\big\\{
            v\\,u^{\\alpha_1/(\\alpha_1+\\alpha_3)},\\;
            u\\,v^{\\alpha_2/(\\alpha_2+\\alpha_3)}
        \\big\\},

    which is a Marshall-Olkin survival copula.  (The published paper
    misprints the exponents; the mathematically correct form above
    matches the construction and the R reference implementation.)
    """
    for name, value in (("alpha1", alpha1), ("alpha2", alpha2),
                        ("alpha3", alpha3)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive")
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if np.any((u < 0) | (u > 1)) or np.any((v < 0) | (v > 1)):
        raise ValueError("u and v must lie in [0, 1]")
    return np.minimum(v * u ** (alpha1 / (alpha1 + alpha3)),
                      u * v ** (alpha2 / (alpha2 + alpha3)))


def kendall_tau(alpha1, alpha2, alpha3, beta, n=50000,
                random_state=None):
    """Monte-Carlo estimate of Kendall's tau.

    Uses :math:`\\tau = 4\\,E[F(Z_1, Z_2)] - 1` evaluated by simulation.
    Dependence increases with the relative size of :math:`\\alpha_3` and
    does not depend on :math:`\\beta`.
    """
    validate_bvch(alpha1, alpha2, alpha3, beta)
    sim = rbvch(n, alpha1, alpha2, alpha3, beta,
                random_state=random_state)
    w = pbvch(sim[:, 0], sim[:, 1], alpha1, alpha2, alpha3, beta)
    return float(4.0 * np.mean(w) - 1.0)
class BvChen:
    """Frozen bivariate Chen distribution with fixed parameters.

    Provides an object-oriented interface around the module functions.

    Parameters
    ----------
    alpha1, alpha2, alpha3, beta : float
        Parameters of the BvCh law (all positive).  ``alpha3`` is the
        common-shock parameter driving the singular (tie) component.
    """

    def __init__(self, alpha1, alpha2, alpha3, beta):
        validate_bvch(alpha1, alpha2, alpha3, beta)
        self.alpha1 = float(alpha1)
        self.alpha2 = float(alpha2)
        self.alpha3 = float(alpha3)
        self.beta = float(beta)

    # -- aliases for convenience --------------------------------------
    @property
    def alpha(self):
        """Alias of ``alpha3`` (the dependence/ties parameter)."""
        return self.alpha3

    @property
    def params(self):
        """Tuple ``(alpha1, alpha2, alpha3, beta)``."""
        return (self.alpha1, self.alpha2, self.alpha3, self.beta)

    def __repr__(self):
        return (f"BvChen(alpha1={self.alpha1:.6g}, "
                f"alpha2={self.alpha2:.6g}, alpha3={self.alpha3:.6g}, "
                f"beta={self.beta:.6g})")

    # -- distribution functions ---------------------------------------
    def pdf(self, x, y, log=False, component="full"):
        """Joint density; see :func:`dbvch`."""
        return dbvch(x, y, self.alpha1, self.alpha2, self.alpha3,
                     self.beta, log=log, component=component)

    def logpdf(self, x, y, component="full"):
        """Log joint density; see :func:`dbvch`."""
        return dbvch(x, y, self.alpha1, self.alpha2, self.alpha3,
                     self.beta, log=True, component=component)

    def sf(self, x, y):
        """Joint survival function; see :func:`sbvch`."""
        return sbvch(x, y, self.alpha1, self.alpha2, self.alpha3,
                     self.beta)

    def cdf(self, x, y):
        """Joint CDF; see :func:`pbvch`."""
        return pbvch(x, y, self.alpha1, self.alpha2, self.alpha3,
                     self.beta)

    def rvs(self, n=1, random_state=None):
        """Random generation; see :func:`rbvch`."""
        return rbvch(n, self.alpha1, self.alpha2, self.alpha3,
                     self.beta, random_state=random_state)
# -- marginals -----------------------------------------------------
    def marginal1_pdf(self, x, log=False):
        """Density of :math:`Z_1 \\sim \\mathrm{Chen}(\\alpha_1+\\alpha_3, \\beta)`."""
        return dmarg1(x, self.alpha1, self.alpha2, self.alpha3,
                      self.beta, log=log)

    def marginal2_pdf(self, x, log=False):
        """Density of :math:`Z_2 \\sim \\mathrm{Chen}(\\alpha_2+\\alpha_3, \\beta)`."""
        return dmarg2(x, self.alpha1, self.alpha2, self.alpha3,
                      self.beta, log=log)

    def marginal1_cdf(self, q):
        """CDF of :math:`Z_1`."""
        return pmarg1(q, self.alpha1, self.alpha2, self.alpha3,
                      self.beta)

    def marginal2_cdf(self, q):
        """CDF of :math:`Z_2`."""
        return pmarg2(q, self.alpha1, self.alpha2, self.alpha3,
                      self.beta)

    # -- conditionals ---------------------------------------------------
    def conditional1_pdf(self, z1, y):
        """Conditional density of :math:`Z_1 \\mid Z_2 = y` (with atom)."""
        return dcond1(z1, y, self.alpha1, self.alpha2, self.alpha3,
                      self.beta)

    def conditional2_pdf(self, z2, x):
        """Conditional density of :math:`Z_2 \\mid Z_1 = x` (with atom)."""
        return dcond2(z2, x, self.alpha1, self.alpha2, self.alpha3,
                      self.beta)

    def conditional1_hazard(self, z1, y):
        """Conditional hazard of the continuous part of :math:`Z_1 \\mid Z_2=y`."""
        return hcond1(z1, y, self.alpha1, self.alpha2, self.alpha3,
                      self.beta)

    def conditional2_hazard(self, z2, x):
        """Conditional hazard of the continuous part of :math:`Z_2 \\mid Z_1=x`."""
        return hcond2(z2, x, self.alpha1, self.alpha2, self.alpha3,
                      self.beta)
# -- dependence and moments -----------------------------------------
    def copula(self, u, v):
        """Survival copula; see :func:`survcop`."""
        return survcop(u, v, self.alpha1, self.alpha2, self.alpha3)

    def kendall_tau(self, n=50000, random_state=None):
        """Monte-Carlo Kendall's tau; see :func:`kendall_tau`."""
        return kendall_tau(self.alpha1, self.alpha2, self.alpha3,
                           self.beta, n=n, random_state=random_state)

    def moment(self, r, s):
        """Raw joint moment :math:`E[Z_1^r Z_2^s]`; see :func:`bvch_moment`."""
        return bvch_moment(r, s, self.alpha1, self.alpha2, self.alpha3,
                           self.beta)

    def mean(self):
        """Means :math:`(E[Z_1], E[Z_2])`."""
        from .chen import chen_moment
        e1 = chen_moment(1, self.alpha1 + self.alpha3, self.beta)
        e2 = chen_moment(1, self.alpha2 + self.alpha3, self.beta)
        return np.array([e1, e2])

    def var(self):
        r"""Variances :math:`(\mathrm{Var}(Z_1), \mathrm{Var}(Z_2))`."""
        from .chen import chen_moment
        e1, e2 = self.mean()
        v1 = chen_moment(2, self.alpha1 + self.alpha3, self.beta) - e1 ** 2
        v2 = chen_moment(2, self.alpha2 + self.alpha3, self.beta) - e2 ** 2
        return np.array([v1, v2])

    def cov(self):
        """2x2 variance-covariance matrix of :math:`(Z_1, Z_2)`."""
        from .chen import chen_moment
        ex = self.mean()
        m11 = chen_moment(2, self.alpha1 + self.alpha3, self.beta)
        m22 = chen_moment(2, self.alpha2 + self.alpha3, self.beta)
        cov12 = (bvch_moment(1, 1, self.alpha1, self.alpha2,
                             self.alpha3, self.beta) - ex[0] * ex[1])
        return np.array([[m11 - ex[0] ** 2, cov12],
                         [cov12, m22 - ex[1] ** 2]])

    def min_dist(self):
        r"""Distribution of :math:`\min(Z_1, Z_2)`.

        The minimum follows :math:`\mathrm{Chen}(\alpha_1+\alpha_2+
        \alpha_3,\ \beta)`.

        Returns
        -------
        dict
            Keys ``alpha``, ``beta`` and callables ``d``, ``p``, ``q``.
        """
        a = self.alpha1 + self.alpha2 + self.alpha3
        return {"alpha": a,
                "beta": self.beta,
                "d": lambda x: dchen(x, a, self.beta),
                "p": lambda q: pchen(q, a, self.beta),
                "q": lambda p: qchen(p, a, self.beta)}