test_that("univariate Chen: d/p/q/r consistency", {
  skip_on_cran()
  a <- 1.7; b <- 1.3
  x <- seq(0.01, 5, length.out = 200)
  # density integrates to 1
  int <- integrate(function(z) dchen(z, a, b), 0, Inf)$value
  expect_equal(int, 1, tolerance = 1e-6)
  # pchen vs schen
  expect_equal(schen(x, a, b), 1 - pchen(x, a, b))
  # quantiles invert the cdf
  p <- c(0.05, 0.3, 0.5, 0.9)
  expect_equal(pchen(qchen(p, a, b), a, b), p, tolerance = 1e-8)
})

test_that("fitchen recovers parameters", {
  skip_on_cran()
  set.seed(1)
  x <- rchen(1000, 2, 1.5)
  fit <- fitchen(x)
  expect_true(fit$converged)
  expect_lt(abs(fit$alpha - 2) / 2, 0.25)
  expect_lt(abs(fit$beta - 1.5) / 1.5, 0.2)
})

test_that("mixture weights of dbvch integrate correctly", {
  skip_on_cran()
  a1 <- 0.8; a2 <- 1.1; a3 <- 0.6; b <- 1.4
  tot <- a1 + a2 + a3
  ys <- seq(0.02, 6, length.out = 300)
  iv <- sapply(ys, function(y)
    stats::integrate(function(x) dbvch(x, y, a1, a2, a3, b), 0, y)$value)
  m_lt <- sum(diff(ys) * (head(iv, -1) + tail(iv, -1)) / 2)
  m_diag <- integrate(function(z)
    dbvch(z, z, a1, a2, a3, b, component = "sing"), 0, Inf)$value
  expect_equal(m_lt, a1 / tot, tolerance = 5e-3)
  # diagonal mass is a3 / total
  expect_equal(m_diag, a3 / tot, tolerance = 1e-4)
})

test_that("survival/cdf identities and boundaries", {
  skip_on_cran()
  a1 <- 1; a2 <- 1; a3 <- 1; b <- 1.2
  expect_equal(sbvch(0, 0, a1, a2, a3, b), 1)
  expect_equal(sbvch(Inf, Inf, a1, a2, a3, b), 0,
               tolerance = 1e-12)
  expect_equal(pbvch(Inf, Inf, a1, a2, a3, b), 1)
  z1 <- 1; z2 <- 2
  s1 <- schen(z1, a1 + a3, b); s2 <- schen(z2, a2 + a3, b)
  s <- sbvch(z1, z2, a1, a2, a3, b)
  expect_equal(pbvch(z1, z2, a1, a2, a3, b), 1 - s1 - s2 + s,
               tolerance = 1e-10)
})

test_that("rbvch marginals match Chen via KS", {
  skip_on_cran()
  set.seed(42)
  sim <- rbvch(3000, 1, 1.5, 0.7, 1.4)
  ks <- ks.test(sim[, "z1"], function(q) pmarg1(q, 1, 1.5, 0.7, 1.4))
  expect_gt(ks$p.value, 0.01)
  tie_rate <- mean(sim[, "z1"] == sim[, "z2"])
  expect_equal(tie_rate, 0.7 / (1 + 1.5 + 0.7), tolerance = 0.03)
})

test_that("copula properties and tau range", {
  skip_on_cran()
  expect_equal(survcopbvch(1, 1, 1, 1, 1), 1)
  expect_equal(survcopbvch(0, .5, 1, 1, 1), 0)
  set.seed(9)
  tau <- tau_bvch(1, 1, 0.01, 1.5, n = 5000, seed = 9)
  expect_gte(tau, -0.1); expect_lte(tau, 0.15)
  tau_big <- tau_bvch(1, 1, 10, 1.5, n = 5000, seed = 9)
  expect_gt(tau_big, tau)
})

test_that("moments are consistent with simulation and series", {
  skip_on_cran()
  set.seed(3)
  m1 <- mean(rchen(200000, 2, 1.5))
  expect_equal(mchen(1, 2, 1.5), m1, tolerance = 0.02)
  mn <- min_bvch(1, 1, 1, 1.5)
  expect_equal(mn$p(1), pchen(1, 3, 1.5))
})

test_that("EM fit recovers parameters and is start-invariant", {
  skip_on_cran()
  set.seed(2024)
  sim <- rbvch(600, 0.8, 1.2, 0.5, 1.5)
  f1 <- fitbvch(sim[, "z1"], sim[, "z2"])
  f2 <- fitbvch(sim[, "z1"], sim[, "z2"],
                start = list(alpha1 = 3, alpha2 = 0.3,
                             alpha3 = 2, beta = 1))
  expect_true(f1$converged)
  cf <- coef(f1)
  expect_lt(abs(cf["beta"] - 1.5), 0.25)
  expect_equal(coef(f1), coef(f2), tolerance = 1e-3)
})

test_that("gof does not reject well-specified data", {
  skip_on_cran()
  set.seed(5)
  sim <- rbvch(800, 1, 1, 1, 1.5)
  g <- gofbvch(sim[, "z1"], sim[, "z2"], 1, 1, 1, 1.5)
  expect_s3_class(g$ks_z1, "htest")
})
