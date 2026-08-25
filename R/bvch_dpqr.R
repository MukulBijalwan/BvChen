# ---------------------------------------------------------------
# Bivariate Chen distribution: CDF and random generation
# ---------------------------------------------------------------

#' @rdname bvch
#' @importFrom stats rexp
#' @export
pbvch <- function(x, y, alpha1, alpha2, alpha3, beta) {
  .validate_bvch(alpha1, alpha2, alpha3, beta)
  n <- max(length(x), length(y))
  x <- rep_len(as.numeric(x), n)
  y <- rep_len(as.numeric(y), n)
  # F = 1 - P(Z1>x) - P(Z2>y) + P(Z1>x, Z2>y)  (inclusion-exclusion;
  # valid also for distributions with a singular part)
  s1 <- ifelse(is.infinite(x), 0, schen(x, alpha1 + alpha3, beta))
  s2 <- ifelse(is.infinite(y), 0, schen(y, alpha2 + alpha3, beta))
  s  <- ifelse(is.infinite(x) | is.infinite(y), 0,
               sbvch(x, y, alpha1, alpha2, alpha3, beta))
  pmin(pmax(1 - s1 - s2 + s, 0), 1)
}

#' @rdname bvch
#' @importFrom stats rexp
#' @export
rbvch <- function(n, alpha1, alpha2, alpha3, beta) {
  stopifnot(length(n) == 1L, is.finite(n), n >= 0)
  .validate_bvch(alpha1, alpha2, alpha3, beta)
  # u(Xi) = e^{Xi^b} - 1 ~ Exp(ai)  =>  work on the Exp scale
  u0 <- stats::rexp(n, alpha3)
  u1 <- stats::rexp(n, alpha1)
  u2 <- stats::rexp(n, alpha2)
  z1 <- log1p(pmin(u0, u1))^(1 / beta)
  z2 <- log1p(pmin(u0, u2))^(1 / beta)
  out <- matrix(c(z1, z2), ncol = 2,
                dimnames = list(NULL, c("z1", "z2")))
  class(out) <- "bvch"
  out
}
