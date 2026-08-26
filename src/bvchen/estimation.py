"""Maximum-likelihood estimation of the bivariate Chen law via EM.

The observed-data log-likelihood has no closed-form maximiser, so the
paper (Section 8) proposes an expectation-conditional-maximisation (ECM)
scheme.  On the exponential scale :math:`t = e^{z^\\beta}-1` the latent
model is a Marshall-Olkin competing-risks scheme.  Each iteration:

* **E-step**: with the current :math:`\\alpha`'s, form the fractional
  pseudo-observations (probabilities :math:`\\varphi_1, \\varphi_2,
  \\upsilon_1, \\upsilon_2` described in the paper).
* **M-step**: re-estimate :math:`(\\alpha_1, \\alpha_2, \\alpha_3)` at
  the current :math:`\\beta`, then update :math:`\\beta` by
  one-dimensional numerical maximisation of the observed log-likelihood.

The implementation below (mirroring the validated R reference package)
maximises the observed log-likelihood directly over the three
:math:`\\alpha`'s at fixed :math:`\\beta` (Nelder-Mead on the log scale)
and then over :math:`\\beta` with the :math:`\\alpha`'s held fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize

from ._utils import log_singular, validate_bvch
from .chen import dchen, fit_chen

__all__ = ["loglik_bvch", "fit_bvchen", "profile_beta", "BvChenFit"]


def loglik_bvch(z1, z2, alpha1, alpha2, alpha3, beta):
    """Observed-data log-likelihood of the BvCh model.

    Points with :math:`z_1<z_2` contribute
    :math:`\\log f_{\\mathrm{Ch}(\\alpha_1)}(z_1)+\\log
    f_{\\mathrm{Ch}(\\alpha_2+\\alpha_3)}(z_2)` (and symmetrically for
    :math:`z_1>z_2`); ties contribute the log of the singular diagonal
    density.
    """
    validate_bvch(alpha1, alpha2, alpha3, beta)
    z1 = np.asarray(z1, dtype=float)
    z2 = np.asarray(z2, dtype=float)
    if z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have the same shape")
    if not np.all(np.isfinite(z1)) or not np.all(np.isfinite(z2)):
        raise ValueError("z1 and z2 must be finite")
    if np.any((z1 < 0) | (z2 < 0)):
        raise ValueError("z1 and z2 must be non-negative")

    lt = z1 < z2
    gt = z1 > z2
    eq = z1 == z2

    ll = np.zeros(z1.size, dtype=float)
    ll[lt] = (dchen(z1[lt], alpha1, beta, log=True)
              + dchen(z2[lt], alpha2 + alpha3, beta, log=True))
    ll[gt] = (dchen(z2[gt], alpha2, beta, log=True)
              + dchen(z1[gt], alpha1 + alpha3, beta, log=True))
    ll[eq] = log_singular(z1[eq], alpha1, alpha2, alpha3, beta)
    return float(np.sum(ll))


def _alphas_at_beta(z1, z2, beta, start):
    """Maximise the observed log-likelihood over the alphas at fixed beta."""
    start = np.asarray(start, dtype=float)

    def ngll(lp):
        a = np.exp(lp)
        try:
            ll = loglik_bvch(z1, z2, a[0], a[1], a[2], beta)
        except (ValueError, FloatingPointError):
            return 1e300
        if not np.isfinite(ll):
            return 1e300
        return -ll

    res = optimize.minimize(ngll, np.log(start), method="Nelder-Mead",
                            options={"xatol": 1e-7, "fatol": 1e-7,
                                     "maxiter": 2000, "maxfev": 4000})
    return np.exp(res.x)


def _default_start(z1, z2):
    """Starting values from univariate marginal fits (as in the paper)."""
    m1 = fit_chen(z1)
    m2 = fit_chen(z2)
    alpha1 = max(0.6 * m1["alpha"], 0.01)
    alpha2 = max(0.6 * m2["alpha"], 0.01)
    alpha3 = max(0.3 * min(m1["alpha"], m2["alpha"]), 0.01)
    beta = 0.5 * (m1["beta"] + m2["beta"])
    return np.array([alpha1, alpha2, alpha3, beta])
@dataclass
class BvChenFit:
    """Result of :func:`fit_bvchen`.

    Attributes
    ----------
    coefficients : np.ndarray
        ``(alpha1, alpha2, alpha3, beta)``.
    loglik : float
        Observed log-likelihood at the solution.
    iterations : int
        Number of EM iterations performed.
    converged : bool
        Whether the convergence tolerance was met.
    tau : float
        Monte-Carlo Kendall's tau implied by the fitted parameters.
    n : int
        Sample size.
    n_ties : int
        Number of observed ties :math:`z_{1i} = z_{2i}`.
    history : list
        Log-likelihood at each iteration.
    """

    coefficients: np.ndarray
    loglik: float = np.nan
    iterations: int = 0
    converged: bool = False
    tau: float = np.nan
    n: int = 0
    n_ties: int = 0
    history: list = field(default_factory=list)

    @property
    def alpha1(self):
        return float(self.coefficients[0])

    @property
    def alpha2(self):
        return float(self.coefficients[1])

    @property
    def alpha3(self):
        return float(self.coefficients[2])

    @property
    def beta(self):
        return float(self.coefficients[3])

    def __repr__(self):
        return (f"BvChenFit(alpha1={self.alpha1:.6g}, "
                f"alpha2={self.alpha2:.6g}, alpha3={self.alpha3:.6g}, "
                f"beta={self.beta:.6g}, loglik={self.loglik:.4f}, "
                f"converged={self.converged}, iterations={self.iterations})")
def fit_bvchen(z1, z2, start=None, tol=1e-8, maxit=200, verbose=False):
    """Fit the bivariate Chen distribution by the EM algorithm.

    Parameters
    ----------
    z1, z2 : array_like
        Paired non-negative observations.
    start : dict or sequence, optional
        Starting values ``(alpha1, alpha2, alpha3, beta)``; if omitted,
        values derived from univariate marginal fits are used.
    tol : float, optional
        Relative tolerance on the log-likelihood increase.
    maxit : int, optional
        Maximum number of EM iterations.
    verbose : bool, optional
        Print the iteration history.

    Returns
    -------
    BvChenFit
        Fitted coefficients and diagnostics.
    """
    z1 = np.asarray(z1, dtype=float).ravel()
    z2 = np.asarray(z2, dtype=float).ravel()
    if z1.size != z2.size:
        raise ValueError("z1 and z2 must have the same length")
    if z1.size == 0:
        raise ValueError("data must be non-empty")
    if not np.all(np.isfinite(z1)) or not np.all(np.isfinite(z2)):
        raise ValueError("data must be finite")
    if np.any((z1 < 0) | (z2 < 0)):
        raise ValueError("data must be non-negative")

    if start is None:
        alp0 = _default_start(z1, z2)
    elif isinstance(start, dict):
        alp0 = np.array([start.get("alpha1", 1.0),
                         start.get("alpha2", 1.0),
                         start.get("alpha3", 1.0),
                         start.get("beta", 1.0)])
    else:
        alp0 = np.asarray(start, dtype=float)
        if alp0.size != 4:
            raise ValueError("start must have 4 elements")
    if np.any(alp0 <= 0):
        raise ValueError("starting values must be positive")

    beta = float(alp0[3])
    alp = alp0[:3].copy()
    ll_old = -np.inf
    history = []
    converged = False

    for it in range(1, maxit + 1):
        # M-step (part 1): alphas at current beta
        alp = _alphas_at_beta(z1, z2, beta, alp)

        # M-step (part 2): beta by 1-D maximisation of the observed loglik
        def neg_ll_logbeta(logb, alp=alp):
            b = float(np.exp(logb))
            ll = loglik_bvch(z1, z2, alp[0], alp[1], alp[2], b)
            if not np.isfinite(ll):
                return 1e300
            return -ll

        res = optimize.minimize_scalar(
            neg_ll_logbeta, bounds=(np.log(1e-3), np.log(100.0)),
            method="bounded", options={"xatol": tol, "maxiter": 200})
        beta = float(np.exp(res.x))
        ll = loglik_bvch(z1, z2, alp[0], alp[1], alp[2], beta)
        history.append(ll)
        if verbose:
            print(f"iter {it:3d}  loglik = {ll:.6f}  beta = {beta:.4f}  "
                  f"alphas = {alp[0]:.4f}, {alp[1]:.4f}, {alp[2]:.4f}")
        if it > 1 and (ll - ll_old) < tol * (abs(ll_old) + tol):
            converged = True
            break
        ll_old = ll

    from .distribution import kendall_tau
    tau = kendall_tau(alp[0], alp[1], alp[2], beta, n=20000,
                      random_state=12345)
    return BvChenFit(coefficients=np.array([alp[0], alp[1], alp[2], beta]),
                     loglik=ll, iterations=len(history),
                     converged=converged, tau=tau,
                     n=z1.size, n_ties=int(np.sum(z1 == z2)),
                     history=history)


def profile_beta(z1, z2, betas=None, start=None):
    """Profile the observed log-likelihood over the power parameter beta.

    For each candidate :math:`\\beta` the :math:`\\alpha`'s are
    re-maximised (warm-started along the grid), so the returned curve is
    the profile log-likelihood.

    Returns
    -------
    dict
        ``grid`` : tuple ``(betas, prof_loglik)`` arrays;
        ``beta_hat`` : the maximising value.
    """
    z1 = np.asarray(z1, dtype=float).ravel()
    z2 = np.asarray(z2, dtype=float).ravel()
    if betas is None:
        betas = np.exp(np.linspace(np.log(0.05), np.log(20.0), 25))
    betas = np.asarray(betas, dtype=float)
    if start is None:
        start = np.array([1.0, 1.0, 1.0])

    a_cur = np.asarray(start, dtype=float)
    cold = np.array([1.0, 1.0, 1.0])
    vals = np.empty(betas.size)
    alphas = np.empty((betas.size, 3))
    for i, b in enumerate(betas):
        # warm-started optimisation, guarded by a cold restart: the
        # likelihood surface can be badly scaled for extreme beta, where
        # a long warm-start chain may wander off.
        a_warm = _alphas_at_beta(z1, z2, float(b), a_cur)
        a_cold = _alphas_at_beta(z1, z2, float(b), cold)
        ll_warm = loglik_bvch(z1, z2, a_warm[0], a_warm[1], a_warm[2],
                              float(b))
        ll_cold = loglik_bvch(z1, z2, a_cold[0], a_cold[1], a_cold[2],
                              float(b))
        if ll_warm >= ll_cold:
            a_cur, vals[i] = a_warm, ll_warm
        else:
            a_cur, vals[i] = a_cold, ll_cold
        alphas[i] = a_cur

    i = int(np.argmax(vals))
    lo = max(betas[max(i - 1, 0)], 1e-4)
    hi = betas[min(i + 1, betas.size - 1)]

    def neg_prof(b):
        # minimise the NEGATIVE profile log-likelihood (warm-started from
        # the alphas optimised at the grid maximiser)
        a = _alphas_at_beta(z1, z2, float(b), alphas[i])
        return -loglik_bvch(z1, z2, a[0], a[1], a[2], float(b))

    opt = optimize.minimize_scalar(neg_prof, bounds=(lo, hi),
                                   method="bounded",
                                   options={"xatol": 1e-5})
    return {"grid": (betas, vals), "beta_hat": float(opt.x)}
