"""Quick-start example for the bvchen package.

Run:  python examples/quickstart.py
"""

import numpy as np

import bvchen
from bvchen import fit_bvchen, gof, qqplot, rbvch


def main() -> None:
    # ------------------------------------------------------------------
    # 1. A frozen bivariate Chen distribution
    # ------------------------------------------------------------------
    dist = bvchen.BvChen(alpha1=0.8, alpha2=1.2, alpha3=0.5, beta=1.5)
    print("distribution :", dist)

    print("\n-- distributional quantities -----------------------------")
    print("pdf(0.5, 0.7)          :", dist.pdf(0.5, 0.7))
    print("cdf(1.0, 1.5)          :", dist.cdf(1.0, 1.5))
    print("sf(1.0, 1.5)           :", dist.sf(1.0, 1.5))
    print("singular pdf at (1,1)  :", dist.pdf(1.0, 1.0, component="sing"))
    print("means                  :", np.round(dist.mean(), 4))
    print("covariance             :\n", np.round(dist.cov(), 4))
    print("Kendall tau            :",
          round(dist.kendall_tau(n=20000, random_state=0), 4))

    # ------------------------------------------------------------------
    # 2. Simulation
    # ------------------------------------------------------------------
    sim = rbvch(400, 0.8, 1.2, 0.5, 1.5, random_state=2024)
    tie_rate = float(np.mean(sim[:, 0] == sim[:, 1]))
    print("\n-- simulation --------------------------------------------")
    print("first rows:\n", np.round(sim[:5], 4))
    print(f"observed tie rate {tie_rate:.3f} "
          f"(theory alpha3/(sum alphas) = {0.5 / 2.5:.3f})")

    # ------------------------------------------------------------------
    # 3. Maximum-likelihood fitting via the EM algorithm
    # ------------------------------------------------------------------
    fit = fit_bvchen(sim[:, 0], sim[:, 1])
    print("\n-- EM fit -------------------------------------------------")
    print(fit)

    # ------------------------------------------------------------------
    # 4. Goodness of fit
    # ------------------------------------------------------------------
    print("\n-- goodness of fit ----------------------------------------")
    g = gof(sim[:, 0], sim[:, 1], *fit.coefficients)
    for name, res in g.items():
        print(f"{name}: KS={res.statistic:.4f}  p-value={res.pvalue:.4f}")

    # Q-Q panels (saved to file so the example runs headless)
    import matplotlib
    matplotlib.use("Agg")
    ax = qqplot(sim[:, 0], sim[:, 1], *fit.coefficients)
    ax[0].figure.savefig("qqplot.png", dpi=120)
    print("\nQ-Q panels written to qqplot.png")


if __name__ == "__main__":
    main()