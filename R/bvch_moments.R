# ---------------------------------------------------------------
# Moments of the univariate and bivariate Chen distributions
#
# Univariate moment series (from expanding exp{-alpha e^{x^b}}):
#   E[Z^k] = alpha * e^alpha * Gamma(1 + k/beta) *
#            sum_{n>=1} (-1)^(n-1) alpha^n / n! * n^-(1 + k/beta)
# ---------------------------------------------------------------

#' @name Moments
#' @title Moments of the (Bivariate) Chen Distribution
#'
#' @param r,s non-negative orders of the moments \eqn{E[Z_1^r]},
#'   \eqn{E[Z_2^s]} and the cross moment \eqn{E[Z_1^r Z_2^s]}.
#' @param k moment order for the univariate integral.
#' @param alpha shape parameter (\eqn{\alpha > 0}) of the univariate
#'   Chen distribution.
#' @param tol stopping tolerance for the numerical integration.
#' @param alpha1,alpha2,alpha3 positive shape parameters of the three
#'   latent Chen variables.
#' @param beta positive common power parameter.
#'
#' @return
#'   \item{mchen}{numeric, \eqn{E[Z^k]} for \eqn{Z~Ch(alpha,beta)}.}
#'   \item{mbvch}{numeric, cross moment \eqn{E[Z_1^r Z_2^s]}.}
#'   \item{ebvch}{named vector of means \code{c(EZ1, EZ2)}.}
#'   \item{vbvch}{2x2 variance-covariance matrix.}
#'   \item{min_bvch}{list describing the minimum
#'     \eqn{M=\min(Z_1,Z_2)\sim Ch(\alpha_1+\alpha_2+\alpha_3,\beta)}
#'     with elements \code{alpha}, \code{beta}, and functions
#'     \code{d}, \code{p}, \code{q}.}
#'
#' @examples
#' mchen(1, 2, 1.5)                       # E[Z], Ch(2, 1.5)
#' mean(rchen(200000, 2, 1.5))            # agrees up to MC error
#'
#' mbvch(1, 1, 1, 1, 1, 1.5)
#' ebvch(1, 1, 1, 1.5)
#' vbvch(1, 1, 1, 1.5)
#'
#' mn <- min_bvch(1, 1, 1, 1.5)
#' mn$p(1)                                # P(min <= 1)
#' pchen(1, 3, 1.5)                       # identical
NULL

.mchen_integral <- function(k, alpha, beta, tol = 1e-10) {
  # E[Z^k] = int_0^inf z^k f(z) dz computed on the log scale
  stats::integrate(
    function(z) {
      lz <- suppressWarnings(dchen(z, alpha, beta, log = TRUE))
      ifelse(is.finite(lz), exp(k * log(z) + lz), 0)
    },
    lower = 0, upper = Inf, rel.tol = tol)$value
}

#' @rdname Moments
#' @export
mchen <- function(k, alpha, beta) {
  stopifnot(all(k >= 0), length(k) >= 1L)
  vapply(k, function(kk) .mchen_integral(kk, alpha, beta),
         numeric(1))
}

#' @rdname Moments
#' @importFrom stats integrate
#' @export
mbvch <- function(r, s, alpha1, alpha2, alpha3, beta,
                  tol = 1e-9) {
  .validate_bvch(alpha1, alpha2, alpha3, beta)
  stopifnot(r >= 0, s >= 0)

  .tri <- function(a_small, a_big, p, q) {
    # integral over {z_small < z_big} of z1^p z2^q
    #   * f_Ch(a_small)(z_small) * f_Ch(a_big)(z_big)
    inner <- function(x)
      stats::integrate(function(yy) yy^q *
                         dchen(yy, a_big, beta),
                       lower = x, upper = Inf)$value *
        dchen(x, a_small, beta) * x^p
    inner_v <- Vectorize(inner)
    stats::integrate(inner_v, lower = 0, upper = Inf,
                     rel.tol = tol)$value
  }
  part_lt <- .tri(alpha1, alpha2 + alpha3, r, s)
  part_gt <- .tri(alpha2, alpha1 + alpha3, s, r)

  diag_part <- stats::integrate(
    function(z) exp((r + s) * log(pmax(z, 1e-300)) +
                      .logdsing(z, alpha1, alpha2, alpha3, beta)),
    lower = 0, upper = Inf, rel.tol = tol)$value

  part_lt + part_gt + diag_part
}

#' @rdname Moments
#' @export
ebvch <- function(alpha1, alpha2, alpha3, beta) {
  c(EZ1 = mchen(1, alpha1 + alpha3, beta),
    EZ2 = mchen(1, alpha2 + alpha3, beta))
}

#' @rdname Moments
#' @export
vbvch <- function(alpha1, alpha2, alpha3, beta) {
  m11 <- mchen(2, alpha1 + alpha3, beta)
  m22 <- mchen(2, alpha2 + alpha3, beta)
  ex  <- ebvch(alpha1, alpha2, alpha3, beta)
  cov <- mbvch(1, 1, alpha1, alpha2, alpha3, beta) - ex["EZ1"] * ex["EZ2"]
  matrix(c(m11 - ex["EZ1"]^2, cov, cov, m22 - ex["EZ2"]^2), 2, 2,
         dimnames = list(c("Z1", "Z2"), c("Z1", "Z2")))
}

#' @rdname Moments
#' @export
min_bvch <- function(alpha1, alpha2, alpha3, beta) {
  .validate_bvch(alpha1, alpha2, alpha3, beta)
  a <- alpha1 + alpha2 + alpha3
  list(alpha = a, beta = beta,
       d = function(x) dchen(x, a, beta),
       p = function(q) pchen(q, a, beta),
       q = function(p) qchen(p, a, beta))
}
