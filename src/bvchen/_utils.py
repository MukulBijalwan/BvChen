"""Internal numeric helpers for the ``bvchen`` package.

This module is private API: nothing here is guaranteed to stay stable
between releases.
"""

from __future__ import annotations

import numpy as np


def validate_bvch(alpha1: float, alpha2: float, alpha3: float,
                  beta: float) -> None:
    """Validate the four strictly-positive parameters of the BvCh law.

    Raises
    ------
    ValueError
        If any parameter is not a positive finite real number.
    """
    for name, value in (("alpha1", alpha1), ("alpha2", alpha2),
                        ("alpha3", alpha3), ("beta", beta)):
        if not np.isscalar(value):
            raise ValueError(f"{name} must be a scalar, got {value!r}")
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number, "
                             f"got {value!r}")


def upr(z, beta):
    """Return ``u(z) = exp(z**beta) - 1``.

    The transform :math:`z \\mapsto e^{z^\\beta}-1` maps a Chen(alpha,
    beta) variate to an Exp(alpha) variate and is used everywhere in the
    bivariate construction to keep the numerics on a friendly scale.
    """
    z = np.asarray(z, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.expm1(np.power(z, beta))
    return out


def dupr(z, beta):
    """Return the derivative ``du/dz = beta * z**(beta-1) * exp(z**beta)``."""
    z = np.asarray(z, dtype=float)
    out = np.zeros_like(z)
    finite = np.isfinite(z)
    pos = finite & (z > 0)
    with np.errstate(over="ignore", invalid="ignore"):
        out[pos] = beta * z[pos] ** (beta - 1) * np.exp(z[pos] ** beta)
    # Boundary behaviour at z == 0 depends on beta.
    zero = finite & (z == 0)
    out[zero & (beta > 1)] = 0.0
    out[zero & (beta < 1)] = np.inf
    out[zero & (beta == 1)] = 1.0
    out[~finite] = np.inf
    return out


def log_singular(z, alpha1, alpha2, alpha3, beta):
    """Stable log of the diagonal singular density ``g(z)``.

    The singular part of the BvCh law lives on the diagonal
    :math:`z_1 = z_2 = z` and has density

    .. math::
        g(z) = \\alpha_3\\, u'(z)\\, \\exp\\{- (\\alpha_1+\\alpha_2+\\alpha_3)\\,u(z)\\}

    with :math:`u(z) = e^{z^\\beta}-1`.  The total mass on the diagonal is
    :math:`\\alpha_3/(\\alpha_1+\\alpha_2+\\alpha_3)`.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    total = alpha1 + alpha2 + alpha3
    t = np.power(z, beta)

    log_du = np.full_like(z, -np.inf)
    pos = np.isfinite(z) & (z > 0)
    with np.errstate(over="ignore", divide="ignore"):
        log_du[pos] = (np.log(beta) + (beta - 1) * np.log(z[pos]) + t[pos])

    with np.errstate(over="ignore", invalid="ignore"):
        out = np.log(alpha3) + log_du - total * np.expm1(t)

    zero = np.isfinite(z) & (z == 0)
    if np.any(zero):
        if beta > 1:
            out[zero] = -np.inf          # g(0) = 0
        elif beta < 1:
            out[zero] = np.inf           # g(0) = +infinity (integrable)
        else:
            out[zero] = np.log(alpha3)   # g(0) = alpha3
    out[~np.isfinite(z)] = np.nan
    return out