# ---------------------------------------------------------------
# Bivariate Chen distribution: joint density and survival
#
# Construction:  Z1 = min(X0, X1), Z2 = min(X0, X2) with
#   Xi ~ Ch(ai, beta) independent.  Then
#   Z1 ~ Ch(a1 + a3, b),  Z2 ~ Ch(a2 + a3, b).
#
# Let u(z) = exp(z^beta) - 1 and u'(z) = beta z^(beta-1) exp(z^beta).
# Joint survival:
#   S(z1, z2) = exp{ -(a1+a3)u(z1) - (a2+a3)u(z2) + a3 u(m) },
#   m = min(z1, z2).
# Absolutely continuous part (z1 < z2):
#   f = f_Ch(a1,b)(z1) * f_Ch(a2+a3,b)(z2)   (symmetric in 1 <-> 2).
# Singular part on the diagonal z1 == z2 == z:
#   g(z) = a3 * u'(z) * exp{ -(a1+a2+a3) u(z) }
# with total mass a3 / (a1 + a2 + a3).
# ---------------------------------------------------------------

#' @title Internal helpers of the BvChen package
#'
#' @param z non-negative numeric vector.
#' @param beta power parameter (\eqn{\beta > 0}).
#'
#' @return Numeric vector; \code{.upr} is \eqn{e^{z^\beta}-1} and
#'   \code{.dupr} its derivative.
#' @keywords internal
.upr  <- function(z, beta) exp(z^beta) - 1

.dupr <- function(z, beta) {
  out <- rep(0, length(z))
  pos <- is.finite(z) & z > 0
  out[pos] <- beta * z[pos]^(beta - 1) * exp(z[pos]^beta)
  out[is.finite(z) & z == 0 & beta > 1] <- 0
  out[is.finite(z) & z == 0 & beta < 1] <- Inf
  out[!is.finite(z)] <- Inf
  out
}

# stable log of the diagonal singular density g(z)
.logdsing <- function(z, alpha1, alpha2, alpha3, beta) {
  A  <- alpha1 + alpha2 + alpha3
  t  <- z^beta
  # log u'(z) = log(beta) + (beta-1) log(z) + t ; guard z = 0
  ldupr <- rep(-Inf, length(z))
  pos <- is.finite(z) & z > 0
  ldupr[pos] <- log(beta) + (beta - 1) * log(z[pos]) + t[pos]
  out <- log(alpha3) + ldupr - A * (exp(t) - 1)
  out[z == 0] <- switch(sign(beta - 1) + 2,
                        -Inf, NA_real_, log(alpha3))
  out[!is.finite(z)] <- NaN
  out
}

.validate_bvch <- function(alpha1, alpha2, alpha3, beta) {
  stopifnot(
    length(beta) == 1L, is.finite(beta), beta > 0,
    all(c(alpha1, alpha2, alpha3) > 0),
    all(is.finite(c(alpha1, alpha2, alpha3)))
  )
  invisible(TRUE)
}

#' @rdname bvch
#' @title The Bivariate Chen Distribution
#'
#' @description Density (\code{dbvch}), joint CDF (\code{pbvch}),
#'   joint survival function (\code{sbvch}) and random generation
#'   (\code{rbvch}) for the bivariate Chen distribution built from
#'   three independent latent Chen components: \eqn{Z_1=\min(X_0,X_1)},
#'   \eqn{Z_2=\min(X_0,X_2)} with \eqn{X_i \sim Ch(\alpha_i,\beta)}.
#'
#' @details The joint law is a mixture of an absolutely continuous
#'   part living on \eqn{\{z_1<z_2\}} and \eqn{\{z_1>z_2\}}, carrying
#'   total mass \eqn{(\alpha_1+\alpha_2)/(\alpha_1+\alpha_2+\alpha_3)},
#'   and a singular part concentrated on the diagonal \eqn{z_1=z_2}
#'   with mass \eqn{\alpha_3/(\alpha_1+\alpha_2+\alpha_3)}.  On
#'   \eqn{z_1<z_2} the density factorises as
#'   \eqn{f_{Ch(\alpha_1,\beta)}(z_1)\,
#'        f_{Ch(\alpha_2+\alpha_3,\beta)}(z_2)}.
#'
#' @param x,y vectors of non-negative quantiles; recycled to common length.
#' @param n number of observations for \code{rbvch}.
#' @param alpha1,alpha2,alpha3 positive shape parameters of the three
#'   latent Chen variables.
#' @param beta positive common power parameter.
#' @param log logical; if TRUE the log-density of the continuous part
#'   is returned (\code{-Inf} on the diagonal).
#' @param component character selector for \code{dbvch}: \code{"full"}
#'   (default) evaluates the singular density on the diagonal and the
#'   continuous density elsewhere; \code{"ac"} evaluates only the
#'   absolutely continuous part (zero on the diagonal); \code{"sing"}
#'   evaluates only the diagonal singular density.
#'
#' @return
#'   \item{\code{dbvch}}{numeric vector of density values.}
#'   \item{\code{pbvch}}{numeric vector, \eqn{P(Z_1\le x, Z_2\le y)}.}
#'   \item{\code{sbvch}}{numeric vector, \eqn{P(Z_1>x, Z_2>y)}.}
#'   \item{\code{rbvch}}{an \eqn{n\times 2} numeric matrix with columns
#'     \code{z1}, \code{z2}, class \code{"bvch"}.}
#'
#' @references
#' Chen, Z. (2000). A new two-parameter lifetime distribution with
#' bathtub shape or increasing failure rate function.
#' \emph{Statistics & Probability Letters} 49, 155-161.
#' Meintanis, S. G. (2007). Test of fit for Marshall-Olkin
#' distributions with applications. \emph{J. Statist. Comput. Simul.}
#' 77, 171-179.
#'
#' @examples
#' # density pieces
#' dbvch(0.5, 1.5, 1, 1, 1, 1.5)      # region x < y
#' dbvch(1.5, 0.5, 1, 1, 1, 1.5)      # region x > y
#' dbvch(1.0, 1.0, 1, 1, 1, 1.5)      # singular diagonal density g(1)
#' dbvch(1.0, 1.0, 1, 1, 1, 1.5, component = "ac")   # 0 on diagonal
#'
#' # CDF from survival via inclusion-exclusion
#' s  <- sbvch(1, 2, 1, 1, 1, 1.5)
#' s1 <- schen(1, 2, 1.5); s2 <- schen(2, 2, 1.5)
#' all.equal(pbvch(1, 2, 1, 1, 1, 1.5), 1 - s1 - s2 + s)
#'
#' set.seed(1)
#' sim <- rbvch(500, 1, 1, 1, 1.5)
#' head(sim)
#' mean(sim[, "z1"] == sim[, "z2"])   # about a3/(a1+a2+a3) = 1/3
#'
#' @export
dbvch <- function(x, y, alpha1, alpha2, alpha3, beta,
                  log = FALSE, component = c("full", "ac", "sing")) {
  .validate_bvch(alpha1, alpha2, alpha3, beta)
  component <- match.arg(component)
  n <- max(length(x), length(y))
  x <- rep_len(as.numeric(x), n)
  y <- rep_len(as.numeric(y), n)

  val <- rep(0, n)
  diag <- (x == y)
  lo   <- (x < y)
  hi   <- (x > y)

  if (component != "sing") {           # absolutely continuous part
    val[lo] <- dchen(x[lo], alpha1, beta) *
               dchen(y[lo], alpha2 + alpha3, beta)
    val[hi] <- dchen(y[hi], alpha2, beta) *
               dchen(x[hi], alpha1 + alpha3, beta)
  }
  if (component != "ac") {             # singular diagonal part
    ld <- .logdsing(x[diag], alpha1, alpha2, alpha3, beta)
    val[diag] <- exp(ld)
  }
  if (log) log(val) else val
}

#' @rdname bvch
#' @export
sbvch <- function(x, y, alpha1, alpha2, alpha3, beta) {
  .validate_bvch(alpha1, alpha2, alpha3, beta)
  n <- max(length(x), length(y))
  x <- pmax(rep_len(as.numeric(x), n), 0)
  y <- pmax(rep_len(as.numeric(y), n), 0)
  m <- pmin(x, y)
  ex <- -(alpha1 + alpha3) * .upr(x, beta) -
        (alpha2 + alpha3) * .upr(y, beta) +
        alpha3 * .upr(m, beta)
  s <- exp(ex)
  s[is.nan(s)] <- 0          # Inf arguments: survival is exactly 0
  s
}
