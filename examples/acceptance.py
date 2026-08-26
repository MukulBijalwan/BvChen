"""Final acceptance check of the installed bvchen package."""
import numpy as np
import bvchen
from bvchen import (BvChen, dbvch, pbvch, sbvch, rbvch, fit_bvchen,
                    gof, survcop, kendall_tau, bvch_moment, dchen, pchen,
                    qchen, rchen, schen)

print("package version:", bvchen.__version__)
dist = BvChen(0.8, 1.2, 0.5, 1.5)

# core identities
assert abs(sbvch(0, 0, 0.8, 1.2, 0.5, 1.5) - 1) < 1e-12
assert abs(pbvch(np.inf, np.inf, 0.8, 1.2, 0.5, 1.5) - 1) < 1e-12
lhs = pbvch(1.0, 2.0, 0.8, 1.2, 0.5, 1.5)
rhs = (1 - schen(1.0, 1.3, 1.5) - schen(2.0, 1.7, 1.5)
       + sbvch(1.0, 2.0, 0.8, 1.2, 0.5, 1.5))
assert abs(lhs - rhs) < 1e-10
assert abs(bvch_moment(0, 0, 0.8, 1.2, 0.5, 1.5) - 1) < 1e-12
assert abs(survcop(1, 1, 1, 1, 1) - 1) < 1e-15

# EM recovery
sim = rbvch(400, 0.8, 1.2, 0.5, 1.5, random_state=2024)
fit = fit_bvchen(sim[:, 0], sim[:, 1])
assert abs(fit.beta - 1.5) < 0.3 and fit.converged

# univariate round trip
p = np.array([0.1, 0.5, 0.9])
assert np.max(np.abs(pchen(qchen(p, 2.0, 1.5), 2.0, 1.5) - p)) < 1e-10

g = gof(sim[:, 0], sim[:, 1], *fit.coefficients)
print("EM fit        :", fit.coefficients.round(4))
print("KS p-values   :", [round(r.pvalue, 3) for r in g.values()])
print("tau           :", round(kendall_tau(0.8, 1.2, 0.5, 1.5,
                                         n=20000, random_state=1), 4))
print("\nALL FINAL CHECKS PASSED")