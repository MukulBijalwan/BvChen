# ---------------------------------------------------------------
# EM fitting of the bivariate Chen distribution
# ---------------------------------------------------------------

#' @rdname fitbvch
#' @title EM Estimation of the Bivariate Chen Distribution
#'
#' @description Fits the four-parameter bivariate Chen distribution
#'   \eqn{(\alpha_1,\alpha_2,\alpha_3,\beta)} to paired data with the
#'   EM algorithm.  On the exponential scale \eqn{t=e^{z^\beta}-1} the
#'   model is a Marshall-Olkin competing-risks scheme, so each
#'   iteration re-estimates the \eqn{\alpha}'s in closed form
#'   (expected failures over exposure) and updates \eqn{\beta} by
#'   one-dimensional numerical maximisation of the observed
#'   log-likelihood.  An observed tie \eqn{z_{1i}=z_{2i}} identifies
#'   the common-shock component exactly.
#'
#' @param z1,z2 paired non-negative observations.
#' @param start optional named list with starting values
#'   \code{alpha1}, \code{alpha2}, \code{alpha3}, \code{beta}; if
#'   omitted, values derived from marginal univariate fits are used.
#' @param tol relative tolerance on the log-likelihood increase.
#' @param maxit maximum number of EM iterations.
#' @param verbose logical; print the iteration history.
#'
#' @return An object of class \code{"bvchfit"}: list with elements
#'   \item{coefficients}{named vector \code{(alpha1, alpha2, alpha3,
#'     beta)}}
#'   \item{loglik}{observed log-likelihood at the solution}
#'   \item{iterations}{number of EM iterations performed}
#'   \item{converged}{logical convergence flag}
#'   \item{tau}{Kendall's tau implied by the fitted parameters}
#'   \item{n}{sample size}
#'   \item{n_ties}{number of observed ties \eqn{z_{1i}=z_{2i}}}
#'   \item{history}{iteration-wise log-likelihood values}
#'
#' @references Meintanis, S. G. (2007). Test of fit for
#'   Marshall-Olkin distributions with applications.  \emph{J. Statist.
#'   Comput. Simul.} 77, 171-179.
#'
#' @examples
#' set.seed(2024)
#' sim <- rbvch(400, 0.8, 1.2, 0.5, 1.5)
#' fit <- fitbvch(sim[, "z1"], sim[, "z2"])
#' round(fit$coefficients, 3)          # near (0.8, 1.2, 0.5, 1.5)
#' fit$loglik
#' fit$tau                             # dependence level
#'
#' # invariant to starting values
#' fit2 <- fitbvch(sim[, "z1"], sim[, "z2"],
#'                 start = list(alpha1 = 2, alpha2 = 2,
#'                              alpha3 = 2, beta = 1))
#' all.equal(coef(fit), coef(fit2), tolerance = 1e-4)
#'
#' @importFrom stats optimize
#' @export
fitbvch <- function(z1, z2, start = NULL,
                    tol = 1e-8, maxit = 200L, verbose = FALSE) {
  z1 <- as.numeric(z1); z2 <- as.numeric(z2)
  stopifnot(length(z1) == length(z2),
            all(is.finite(c(z1, z2))), all(c(z1, z2) >= 0))
  if (is.null(start)) {
    m1 <- fitchen(z1); m2 <- fitchen(z2)
    start <- list(alpha1 = max(m1$alpha * 0.6, 0.01),
                  alpha2 = max(m2$alpha * 0.6, 0.01),
                  alpha3 = max(pmin(m1$alpha, m2$alpha) * 0.3, 0.01),
                  beta  = mean(c(m1$beta, m2$beta)))
  }
  beta <- start$beta
  alp <- c(start$alpha1, start$alpha2, start$alpha3)
  ll_old <- -Inf
  hist <- numeric(0); converged <- FALSE
  for (it in seq_len(maxit)) {
    # --- update the alphas at the current beta (EM-type M-step) -----
    alp <- .alphas_at_beta(z1, z2, beta, alp)
    # --- update beta by maximising the observed log-likelihood ------
    obj <- function(lb) {
      b <- exp(lb)
      loglik_bvch(z1, z2, alp[1], alp[2], alp[3], b)
    }
    o <- stats::optimize(obj, log(c(1e-3, 100)), maximum = TRUE)
    beta <- exp(o$maximum)
    ll <- unname(loglik_bvch(z1, z2, alp[1], alp[2], alp[3], beta))
    hist <- c(hist, ll)
    if (verbose)
      cat(sprintf("iter %3d  loglik = %.6f  beta = %.4f\n",
                  it, ll, beta))
    if (ll - ll_old < tol * (abs(ll_old) + tol) && it > 1L) {
      converged <- TRUE
      break
    }
    ll_old <- ll
  }
  out <- list(
    coefficients = c(alpha1 = unname(alp[1]), alpha2 = unname(alp[2]),
                     alpha3 = unname(alp[3]), beta = unname(beta)),
    loglik = ll, iterations = length(hist), converged = converged,
    tau = tau_bvch(alp[1], alp[2], alp[3], beta, n = 20000),
    n = length(z1), n_ties = sum(z1 == z2), history = hist)
  class(out) <- "bvchfit"
  out
}

#' @rdname fitbvch
#' @param object an object of class \code{"bvchfit"}.
#' @param ... further arguments (unused).
#' @export
coef.bvchfit <- function(object, ...) object$coefficients

#' @rdname fitbvch
#' @param x an object of class \code{"bvchfit"}.
#' @param digits number of digits passed to \code{\link[base]{round}}.
#' @param ... further arguments (unused).
#' @export
print.bvchfit <- function(x, digits = max(3L, getOption("digits") - 3L),
                          ...) {
  cat("Bivariate Chen fit (EM algorithm)\n")
  cat(sprintf("  observations : %d (%d ties on the diagonal)\n",
              x$n, x$n_ties))
  cat("  coefficients :\n")
  print(round(x$coefficients, digits))
  cat(sprintf("  log-likelihood: %.4f   (iterations: %d%s)\n",
              x$loglik, x$iterations,
              ifelse(x$converged, ", converged", ", NOT converged")))
  cat(sprintf("  Kendall's tau : %.4f\n", x$tau))
  invisible(x)
}
