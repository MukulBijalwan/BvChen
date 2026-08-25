# ---------------------------------------------------------------
# Bivariate Chen: marginal and conditional distributions
#   Z1 ~ Ch(a1 + a3, b),  Z2 ~ Ch(a2 + a3, b)
# ---------------------------------------------------------------

#' @name Marginals
#' @title Marginal Distributions of the Bivariate Chen Law
#'
#' @description Density and CDF of the marginals
#'   \eqn{Z_1 \sim Ch(\alpha_1+\alpha_3,\beta)} and
#'   \eqn{Z_2 \sim Ch(\alpha_2+\alpha_3,\beta)}.
#'
#' @param x,q non-negative quantiles.
#' @param y quantile at which the second marginal is evaluated.
#' @param alpha1,alpha2,alpha3 positive shape parameters of the three
#'   latent Chen variables.
#' @param beta positive common power parameter.
#'
#' @return Numeric vectors of density (\code{d*}) or probability
#'   (\code{p*}) values.
#'
#' @examples
#' dmarg1(1, 1, 1, 1, 1.5)
#' dchen(1, 2, 1.5)                   # identical by construction
#' pmarg2(c(0.5, 1, 2), 1, 1, 1, 1.5)
#' pchen(c(0.5, 1, 2), 2, 1.5)
#'
NULL

#' @rdname Marginals
#' @export
dmarg1 <- function(x, alpha1, alpha2, alpha3, beta)
  dchen(x, alpha1 + alpha3, beta)

#' @rdname Marginals
#' @export
dmarg2 <- function(y, alpha1, alpha2, alpha3, beta)
  dchen(y, alpha2 + alpha3, beta)

#' @rdname Marginals
#' @export
pmarg1 <- function(q, alpha1, alpha2, alpha3, beta)
  pchen(q, alpha1 + alpha3, beta)

#' @rdname Marginals
#' @export
pmarg2 <- function(q, alpha1, alpha2, alpha3, beta)
  pchen(q, alpha2 + alpha3, beta)

#' @name Conditional
#' @title Conditional Distributions of the Bivariate Chen Law
#'
#' @description Conditional density (\code{dcond}) and conditional
#'   hazard (\code{hcond}) of \eqn{Z_1} given \eqn{Z_2=y} (suffix 1)
#'   and of \eqn{Z_2} given \eqn{Z_1=x} (suffix 2).
#'
#' @details Given \eqn{Z_2=y}, the conditional law of \eqn{Z_1} has an
#'   atom at \eqn{z_1=y} of size \eqn{\pi(y)=g(y)/f_{m2}(y)}, where
#'   \eqn{g} is the diagonal singular density.  Its continuous part is
#'   piecewise:
#'   \itemize{
#'     \item \eqn{z_1<y}: \eqn{f_{Ch(\alpha_1,\beta)}(z_1)};
#'     \item \eqn{z_1>y}: \eqn{f_{Ch(\alpha_2,\beta)}(y)\,
#'           f_{Ch(\alpha_1+\alpha_3,\beta)}(z_1)/f_{m2}(y)},
#'   }
#'   with \eqn{f_{m2}} the marginal density of \eqn{Z_2}.
#'
#' @param x,y conditioning values (\code{x} conditions on
#'   \eqn{Z_1=x}, \code{y} on \eqn{Z_2=y}); must be positive.
#' @param z1,z2 values at which the conditional density/hazard of the
#'   response variable is evaluated (any non-negative value).
#' @param alpha1,alpha2,alpha3 positive shape parameters of the three
#'   latent Chen variables.
#' @param beta positive common power parameter.
#'
#' @return Numeric vector with attribute \code{"atom"} giving the
#'   conditional probability mass at \eqn{z_i = y}; the returned
#'   density itself refers to the strictly-off-diagonal part.
#'   \code{hcond*} return the conditional hazard rate of the
#'   absolutely continuous part (\code{NA} exactly at \eqn{z=y}).
#'
#' @examples
#' # conditional density of Z1 given Z2 = 1, below and above y = 1
#' round(dcond1(seq(0.4, 1.8, by = 0.2), 1, 1, 1, 1, 1.5), 4)
#' attr(dcond1(1, 1, 1, 1, 1, 1.5), "atom")   # point mass at z1 = y
#'
#' round(hcond1(seq(0.4, 1.8, by = 0.2), 1, 1, 1, 1, 1.5), 4)
NULL

# concrete implementations ------------------------------------------

#' @rdname Conditional
#' @export
dcond1 <- function(z1, y, alpha1, alpha2, alpha3, beta) {
  stopifnot(all(y > 0))
  fm2 <- dmarg2(y, alpha1, alpha2, alpha3, beta)
  below <- z1 < y
  val <- ifelse(below,
                dchen(z1, alpha1, beta),
                dchen(y, alpha2, beta) *
                  dchen(z1, alpha1 + alpha3, beta) / fm2)
  gd   <- exp(.logdsing(y, alpha1, alpha2, alpha3, beta))
  out <- as.numeric(val)
  attr(out, "atom") <- as.numeric(gd / fm2)
  out
}
#' @rdname Conditional
#' @export
dcond2 <- function(z2, x, alpha1, alpha2, alpha3, beta)
  dcond1(z2, x, alpha2, alpha1, alpha3, beta)

# conditional survival P(Z1 > z | Z2 = y), exact for the mixed law
.scond1 <- function(z, y, alpha1, alpha2, alpha3, beta) {
  fm2 <- dmarg2(y, alpha1, alpha2, alpha3, beta)
  ifelse(z < y,
         1 - pchen(z, alpha1, beta),
         (pmarg2(y, alpha1, alpha2, alpha3, beta) -
            pbvch(z, y, alpha1, alpha2, alpha3, beta)) / fm2)
}

#' @rdname Conditional
#' @details The conditional hazard of the absolutely continuous part is
#'   \eqn{h(z\mid y)=f_{Z_1|Z_2}(z\mid y)/P(Z_1>z\mid Z_2=y)}; it is
#'   \code{NA} exactly at \eqn{z=y}, where only the point mass lives.
#' @export
hcond1 <- function(z1, y, alpha1, alpha2, alpha3, beta) {
  stopifnot(all(y > 0))
  dens <- as.numeric(dcond1(z1, y, alpha1, alpha2, alpha3, beta))
  surv <- .scond1(z1, y, alpha1, alpha2, alpha3, beta)
  out <- rep(NA_real_, length(z1))
  ok <- z1 != y & surv > 0
  out[ok] <- dens[ok] / surv[ok]
  out
}

#' @rdname Conditional
#' @export
hcond2 <- function(z2, x, alpha1, alpha2, alpha3, beta)
  hcond1(z2, x, alpha2, alpha1, alpha3, beta)
