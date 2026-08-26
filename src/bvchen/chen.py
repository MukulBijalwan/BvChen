"""Univariate Chen (2000) distribution — building blocks of the bivariate law.

The Chen distribution has CDF, survival, density and hazard functions

.. math::

    F(x)  &= 1 - \\exp\\{ \\alpha\\,(1 - e^{x^\\beta}) \\} \\\\
    S(x)  &= \\exp\\{ \\alpha\\,(1 - e^{x^\\beta}) \\} \\\\
    f(x)  &= \\alpha\\,\\beta\\, x^{\\beta-1}\\, e^{x^\\beta}\\,
              \\exp\\{ \\alpha\\,(1 - e^{x^\\beta}) \\} \\\\
    h(x)  &= \\alpha\\,\\beta\\, x^{\\beta-1}\\, e^{x^\\beta}

for :math:`x, \\alpha, \\beta > 0`.  Its hazard is bathtub-shaped for
:math:`\\beta < 1` and increasing for :math:`\\beta \\ge 1`.

References
----------
Chen, Z. (2000). A new two-parameter lifetime distribution with bathtub
shape or increasing failure rate function. *Statistics & Probability
Letters* 49(2), 155--161.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, optimize

__all__ = [
    "dchen", "pchen", "schen", "qchen", "rchen", "hchen", "fit_chen",
    "chen_moment",
]


def _log_density(x, alpha, beta):
    """Log-pdf computed in a numerically careful way."""
    x = np.asarray(x, dtype=float)
    logx = np.zeros_like(x)
    pos = x > 0
    logx[pos] = np.log(x[pos])
    # At x == 0 the term (beta-1)*log(x) has a well-defined limit along
    # with the density itself:
    #   beta > 1 : f(0) = 0 ;  beta == 1 : f(0) = alpha ;  beta < 1 : f -> inf
    pow_part = np.zeros_like(x)
    bm1 = beta - 1.0
    pow_part[pos] = bm1 * logx[pos]
    if bm1 > 0:
        pow_part[~pos] = -np.inf
    elif bm1 == 0:
        pow_part[~pos] = 0.0
    else:
        pow_part[~pos] = np.inf

    xb = x ** beta
    with np.errstate(over="ignore", invalid="ignore"):
        # alpha*(1 - e^{x^b}); overflow handled below
        exp_term = np.where(xb > 700.0, -np.inf, -alpha * np.expm1(xb))
        logf = (np.log(alpha) + np.log(beta) + xb
                + pow_part + exp_term)
    return logf


def dchen(x, alpha, beta, log=False):
    """Density of the Chen(``alpha``, ``beta``) distribution.

    Parameters
    ----------
    x : array_like
        Quantiles (non-negative).
    alpha, beta : float
        Scale (shape) and power parameters, both positive.
    log : bool, optional
        If True the log-density is returned.

    Returns
    -------
    ndarray
        Density values (or log-density if ``log=True``).
    """
    alpha = float(alpha)
    beta = float(beta)
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")
    logf = _log_density(x, alpha, beta)
    if log:
        return logf
    with np.errstate(over="ignore", invalid="ignore"):
        return np.exp(logf)


def pchen(q, alpha, beta):
    """CDF of the Chen(``alpha``, ``beta``) distribution."""
    alpha = float(alpha)
    beta = float(beta)
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")
    q = np.asarray(q, dtype=float)
    xb = q ** beta
    with np.errstate(over="ignore", invalid="ignore"):
        out = -np.expm1(-alpha * np.expm1(xb))
    # Clip tiny negatives from round-off.
    return np.clip(out, 0.0, 1.0)


def schen(q, alpha, beta):
    """Survival function of the Chen(``alpha``, ``beta``) distribution."""
    q = np.asarray(q, dtype=float)
    xb = q ** beta
    with np.errstate(over="ignore", invalid="ignore"):
        return np.exp(-alpha * np.expm1(xb))


def qchen(p, alpha, beta):
    """Quantile function of the Chen(``alpha``, ``beta``) distribution.

    The quantile is :math:`q(p) = [\\ln(1 - \\ln(1-p)/\\alpha)]^{1/\\beta}`.
    """
    alpha = float(alpha)
    beta = float(beta)
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")
    p = np.asarray(p, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p must lie in [0, 1]")
    with np.errstate(divide="ignore", invalid="ignore"):
        inner = 1.0 - np.log1p(-p) / alpha
    out = np.zeros_like(inner, dtype=float)
    ok = inner > 1.0
    out[ok] = np.log(inner[ok]) ** (1.0 / beta)
    return out


def rchen(n, alpha, beta, random_state=None):
    """Generate ``n`` variates from the Chen(``alpha``, ``beta``) law.

    Uses that :math:`U = e^{X^\\beta} - 1 \\sim \\mathrm{Exp}(\\alpha)`,
    so ``X = (log1p(U))**(1/beta)``.

    Parameters
    ----------
    n : int
        Number of variates.
    alpha, beta : float
        Parameters (positive).
    random_state : {int, np.random.Generator, None}, optional
        Seed or generator for reproducibility.
    """
    alpha = float(alpha)
    beta = float(beta)
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")
    rng = np.random.default_rng(random_state)
    u = rng.exponential(scale=1.0 / alpha, size=int(n))
    return np.log1p(u) ** (1.0 / beta)


def hchen(x, alpha, beta):
    """Hazard rate of the Chen(``alpha``, ``beta``) distribution."""
    alpha = float(alpha)
    beta = float(beta)
    x = np.asarray(x, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        out = alpha * beta * x ** (beta - 1) * np.exp(x ** beta)
    return out


def _ngll(par, x, x_pos):
    """Negative log-likelihood of the Chen law for a generic ``par``."""
    alpha, beta = par
    if alpha <= 0 or beta <= 0:
        return 1e300
    logf = _log_density(x, alpha, beta)
    bad = ~np.isfinite(logf)
    if np.any(bad & x_pos):
        return 1e300
    return -float(np.nansum(logf))


def fit_chen(x, start=None, tol=1e-8, maxit=100):
    """Maximum-likelihood fit of the two-parameter Chen distribution.

    Parameters
    ----------
    x : array_like
        Non-negative observations.
    start : tuple or dict, optional
        Starting values ``(alpha, beta)``; if omitted, method-of-moments
        style starts are constructed from the data.
    tol : float, optional
        Relative tolerance for the Nelder-Mead optimizer.
    maxit : int, optional
        Maximum iterations (scaled by the optimizer).

    Returns
    -------
    dict
        ``alpha``, ``beta``, ``loglik``, ``converged`` and ``se``
        (approximate standard errors from a numeric Hessian).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1-D array")
    if not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError("x must contain only finite non-negative values")
    if np.all(x == 0):
        raise ValueError("x contains only zeros; cannot fit")
    x_pos = x > 0

    if start is None:
        m1 = float(np.mean(x))
        m2 = float(np.mean(x ** 2))
        a0 = 1.0 / max(m1, 1e-3)
        b0 = max(0.5, np.log(2.0) / np.log(max(2.0 * m2 / max(m1 * m1, 1e-6),
                                               1.01)))
        start = (a0, b0)
    elif isinstance(start, dict):
        start = (start.get("alpha", 1.0), start.get("beta", 1.0))

    res = optimize.minimize(_ngll, np.asarray(start, dtype=float),
                            args=(x, x_pos), method="Nelder-Mead",
                            options={"xatol": tol, "fatol": tol,
                                     "maxiter": 20 * maxit})
    par = res.x

    # Approximate Hessian by finite differences for standard errors.
    h = 1e-5
    p = par
    f0 = _ngll(p, x, x_pos)
    H = np.full((2, 2), np.nan)
    H[0, 0] = (_ngll(p + np.array([h, 0.0]), x, x_pos) - 2 * f0
               + _ngll(p - np.array([h, 0.0]), x, x_pos)) / h ** 2
    H[1, 1] = (_ngll(p + np.array([0.0, h]), x, x_pos) - 2 * f0
               + _ngll(p - np.array([0.0, h]), x, x_pos)) / h ** 2
    H[0, 1] = H[1, 0] = (
        _ngll(p + np.array([h, h]), x, x_pos)
        - _ngll(p + np.array([h, -h]), x, x_pos)
        - _ngll(p + np.array([-h, h]), x, x_pos)
        + _ngll(p - np.array([h, h]), x, x_pos)) / (4 * h ** 2)

    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.array([np.nan, np.nan])

    return {"alpha": float(par[0]), "beta": float(par[1]),
            "loglik": -float(res.fun), "converged": bool(res.success),
            "se": se}


def chen_moment(k, alpha, beta, tol=1e-9):
    """Raw moment :math:`E[X^k]` of the Chen(``alpha``, ``beta``) law.

    Uses the exponential-scale representation :math:`u = e^{x^\\beta}-1`
    which turns the moment integral into the very stable one-dimensional
    integral

    .. math::
        E[X^k] = \\alpha \\int_0^\\infty \\big(\\ln(1+u)\\big)^{k/\\beta}
                 e^{-\\alpha u}\\, du.
    """
    alpha = float(alpha)
    beta = float(beta)
    k = float(k)
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")
    if k < 0:
        raise ValueError("k must be non-negative")

    def integrand(u):
        val = np.log1p(u) ** (k / beta) * np.exp(-alpha * u)
        return val

    if k == 0.0:
        return 1.0
    value, _ = integrate.quad(integrand, 0.0, np.inf,
                              limit=200, epsabs=tol, epsrel=tol)
    return alpha * value
