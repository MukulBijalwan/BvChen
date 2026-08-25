# BvChen

The **BvChen** package provides the density, distribution function,
survival function, random generation, marginal and conditional
distributions, dependence measures, moments, EM-based maximum
likelihood estimation and goodness-of-fit tools for the **bivariate
Chen distribution**, together with the univariate Chen (2000)
building blocks.

## Installation (from source)

```r
install.packages("path/to/BvChen", repos = NULL, type = "source")
```

## Quick start

```r
library(BvChen)

set.seed(2024)
sim <- rbvch(400, 0.8, 1.2, 0.5, 1.5)   # simulate

fit <- fitbvch(sim[, "z1"], sim[, "z2"]) # fit by EM
fit$coefficients

gofbvch(sim[, "z1"], sim[, "z2"], 0.8, 1.2, 0.5, 1.5)  # KS tests
qqbvch(sim[, "z1"], sim[, "z2"], 0.8, 1.2, 0.5, 1.5)   # Q-Q plots
```

See `vignette("bvchen")` for a full walk-through.
