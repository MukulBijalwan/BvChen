# bvchen — The Bivariate Chen Distribution

[![CI](https://github.com/MukulBijalwan/bvchen/actions/workflows/ci.yml/badge.svg)](https://github.com/MukulBijalwan/bvchen/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/bvchen.svg)](https://pypi.org/project/bvchen/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Authors:** Mukul Bijalwan ([mukulbijalwan555@gmail.com](mailto:mukulbijalwan555@gmail.com))
and Puneet Kumar Gupta ([puneetstat999@gmail.com](mailto:puneetstat999@gmail.com))

`bvchen` implements the **bivariate Chen (BvCh) distribution** introduced by
Gupta, Pundir, Sharma and Mesfioui (2022), together with the univariate
Chen (2000) building blocks. The distribution is constructed from three
independent Chen components `U_i ~ Chen(alpha_i, beta)` via

```
Z1 = min(U1, U3),   Z2 = min(U2, U3)
```

It has **bathtub-shaped marginals hazards** (for `beta < 1`), closed-form
joint survival, a singular component on the diagonal `z1 = z2` of mass
`alpha3 / (alpha1 + alpha2 + alpha3)`, and is the natural bivariate
extension of the bathtub-shaped Chen lifetime model.

## Features

- Joint density (`dbvch`), CDF (`pbvch`), survival function (`sbvch`) and
  random generation (`rbvch`)
- Marginal distributions: `Z1 ~ Chen(alpha1+alpha3, beta)`,
  `Z2 ~ Chen(alpha2+alpha3, beta)`
- Conditional densities with their diagonal atom and conditional hazards
- Raw joint moments `E[Z1^r Z2^s]` on a numerically stable exponential
  scale; means, variances and covariance matrix
- Survival copula (Marshall-Olkin form) and Kendall's tau
- Maximum-likelihood estimation via the **EM algorithm**
  (`fit_bvchen`) with profile likelihood for the power parameter
- Goodness-of-fit tools: Kolmogorov-Smirnov tests for both marginals and
  the minimum, Q-Q plots, density surface and contour plots

## Installation

```bash
pip install bvchen
```

For the plotting helpers install the extras:

```bash
pip install "bvchen[plot]"
```

## Quick start

```python
import numpy as np
import bvchen

# frozen distribution object
dist = bvchen.BvChen(alpha1=0.8, alpha2=1.2, alpha3=0.5, beta=1.5)

dist.pdf(0.5, 0.7)              # joint density at (z1, z2)
dist.cdf(1.0, 1.5)              # joint CDF
dist.sf(1.0, 1.5)               # joint survival
sim = dist.rvs(500, random_state=42)   # n x 2 matrix (columns z1, z2)

dist.mean()                     # [E(Z1), E(Z2)]
dist.cov()                      # variance-covariance matrix
dist.kendall_tau(random_state=0)
```

### Fitting to data with the EM algorithm

```python
from bvchen import rbvch, fit_bvchen, gof, qqplot

# simulate from a known law ...
data = rbvch(400, 0.8, 1.2, 0.5, 1.5, random_state=2024)

# ... and recover the parameters by maximum likelihood (EM)
fit = fit_bvchen(data[:, 0], data[:, 1])
print(fit.coefficients)     # close to (0.8, 1.2, 0.5, 1.5)
print(fit.loglik, fit.converged, fit.tau)

gof(data[:, 0], data[:, 1], *fit.coefficients)   # KS tests
qqplot(data[:, 0], data[:, 1], *fit.coefficients)  # Q-Q panels
```

### Functional interface

All routines are also available as plain functions mirroring the classic
R-style `d/p/q/r` naming:

```python
from bvchen import dbvch, pbvch, sbvch, rbvch
from bvchen import dcond1, hcond1, survcop, bvch_moment

dbvch(0.5, 0.5, 1, 1, 1, 1.5, component="sing")  # diagonal singular part
dcond1([0.4, 0.9, 1.2], y=1.0, alpha1=1, alpha2=1, alpha3=1, beta=1.5)
survcop(0.5, 0.6, 1, 1, 1)                        # survival copula
bvch_moment(r=1, s=1, alpha1=1, alpha2=1, alpha3=1, beta=1.5)
```

## Mathematical summary

With `u(z) = exp(z^beta) - 1`:

| Object | Formula |
| --- | --- |
| joint survival | `S(z1,z2) = exp{-(a1+a3)u(z1) - (a2+a3)u(z2) + a3 u(min(z1,z2))}` |
| density on `{z1<z2}` | `f_Ch(z1; a1) f_Ch(z2; a2+a3)` |
| density on `{z1>z2}` | `f_Ch(z1; a1+a3) f_Ch(z2; a2)` |
| singular density | `g(z) = a3 u'(z) exp{-A u(z)}`, mass `a3/A`, `A=a1+a2+a3` |
| marginals | `Z1~Ch(a1+a3,b)`, `Z2~Ch(a2+a3,b)`, `min(Z1,Z2)~Ch(A,b)` |
| survival copula | `C(u,v) = min(v u^(a1/(a1+a3)), u v^(a2/(a2+a3)))` |

## References

- Gupta, P. K., Pundir, P. S., Sharma, V. K., Mesfioui, M. (2022).
  *Bivariate extension of bathtub-shaped distribution.* Life Cycle
  Reliability and Safety Engineering 11, 247-259.
  <https://doi.org/10.1007/s41872-022-00193-4>
- Chen, Z. (2000). A new two-parameter lifetime distribution with bathtub
  shape or increasing failure rate function. *Statistics & Probability
  Letters* 49(2), 155-161.
- Meintanis, S. G. (2007). Test of fit for Marshall-Olkin distributions
  with applications. *J. Stat. Comput. Simul.* 77, 171-179.

> Note: equation (14) of the 2022 paper misprints the copula exponents;
> this package uses the mathematically correct Marshall-Olkin form above,
> consistent with the construction.

## Development

```bash
git clone https://github.com/MukulBijalwan/bvchen
cd bvchen
pip install -e ".[test]"
pytest
```

Build the distribution artifacts:

```bash
pip install build
python -m build
```

## Citation

If you use this package in your research, please cite the paper that
introduced the distribution:

```bibtex
@article{gupta2022bivariate,
  author  = {Gupta, Puneet Kumar and Pundir, Pramendra Singh and
             Sharma, Vikas Kumar and Mesfioui, M.},
  title   = {Bivariate extension of bathtub-shaped distribution},
  journal = {Life Cycle Reliability and Safety Engineering},
  year    = {2022},
  volume  = {11},
  pages   = {247--259},
  doi     = {10.1007/s41872-022-00193-4}
}
```

## License

MIT — see [LICENSE](LICENSE).