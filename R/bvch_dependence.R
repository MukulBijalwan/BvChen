# ---------------------------------------------------------------
# Bivariate Chen: survival copula, Kendall's tau, concordance
#
# In terms of the marginal survival levels u = S1(z1), v = S2(z2):
#   Chat(u, v) = min( u * v^(a3/(a2+a3)),  v * u^(a3/(a1+a3)) )
# ---------------------------------------------------------------

#' @title Survival Copula of the Bivariate Chen Law
#'
#' @param u,v vectors of marginal survival probabilities in \eqn{[0,1]};
#'   recycled to common length.
#' @param alpha1,alpha2,alpha3 positive shape parameters.
#'
#' @return The survival copula value \eqn{\hat C(u,v)}.
#'
#' @examples
#' # survival copula at a few points
#' survcopbvch(c(0.5, 0.8), c(0.5, 0.8), 1, 1, 1)
#' survcopbvch(0, 0.5, 1, 1, 1)            # boundary: 0
#' survcopbvch(1, 1, 1, 1, 1)              # comonotone corner: 1
#' @export
survcopbvch <- function(u, v, alpha1, alpha2, alpha3) {
  stopifnot(all(c(alpha1, alpha2, alpha3) > 0),
            all(u >= 0 & u <= 1), all(v >= 0 & v <= 1))
  # Marshall-Olkin form: min( u^{a1/(a1+a3)} v , u v^{a2/(a2+a3)} )
  pmin(v * u^(alpha1 / (alpha1 + alpha3)),
       u * v^(alpha2 / (alpha2 + alpha3)))
}

#' @rdname Dependence
#' @export
tau_bvch <- function(alpha1, alpha2, alpha3, beta,
                     n = 50000, seed = NULL) {
  .validate_bvch(alpha1, alpha2, alpha3, beta)
  if (!is.null(seed)) set.seed(seed)
  sim <- rbvch(n, alpha1, alpha2, alpha3, beta)
  w <- pbvch(sim[, "z1"], sim[, "z2"], alpha1, alpha2, alpha3, beta)
  as.numeric(4 * mean(w) - 1)
}

#' @rdname Dependence
#' @export
concordance_bvch <- function(alpha1a, alpha2a, alpha3a,
                             alpha1b, alpha2b, alpha3b, beta,
                             n = 50000, seed = NULL) {
  tau_a <- tau_bvch(alpha1a, alpha2a, alpha3a, beta, n, seed)
  tau_b <- tau_bvch(alpha1b, alpha2b, alpha3b, beta, n,
                    if (is.null(seed)) NULL else seed + 1L)
  tau_a <= tau_b
}

#' @name Dependence
#' @title Dependence Measures of the Bivariate Chen Law
#'
#' @description Kendall's tau (Monte-Carlo estimate via
#'   \eqn{\tau = 4\,E[F(Z_1,Z_2)] - 1}) and a concordance comparison of
#'   two parameter triples.  Dependence increases with the relative
#'   size of the common-shock parameter \eqn{\alpha_3} and does not
#'   depend on \eqn{\beta}.
#'
#' @param alpha1,alpha2,alpha3 positive shape parameters.
#' @param beta positive power parameter.
#' @param n number of Monte-Carlo draws for \code{tau_bvch}.
#' @param seed optional seed for reproducibility.
#' @param alpha1a,alpha2a,alpha3a first parameter triple of
#'   \code{concordance_bvch}.
#' @param alpha1b,alpha2b,alpha3b second parameter triple of
#'   \code{concordance_bvch}.
#'
#' @return \code{tau_bvch}: numeric estimate of Kendall's tau.
#'   \code{concordance_bvch}: TRUE if the first triple is
#'   concordance-smaller than the second.
NULL
