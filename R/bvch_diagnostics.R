# ---------------------------------------------------------------
# Goodness-of-fit and Q-Q diagnostics
# ---------------------------------------------------------------

#' @name gofbvch
#' @rdname gofbvch
#' @title Goodness-of-Fit Tests and Q-Q Plots
#'
#' @description Kolmogorov-Smirnov goodness-of-fit tests for the two
#'   marginals and for the minimum \eqn{\min(Z_1,Z_2)}, following the
#'   testing strategy of Meintanis (2007), together with Q-Q plots for
#'   \eqn{Z_1}, \eqn{Z_2} and the minimum.
#'
#' @param z1,z2 paired non-negative observations.
#' @param alpha1,alpha2,alpha3,beta fitted parameter values.
#' @param plot logical; if TRUE (default) the Q-Q plots are drawn.
#'
#' @return \code{gofbvch} returns a list with elements
#'   \code{ks_z1}, \code{ks_z2}, \code{ks_min}: each an object of
#'   class \code{"htest"} produced by \code{\link[stats]{ks.test}},
#'   comparing the data against the corresponding theoretical CDF.
#'
#'   \code{qqbvch} invisibly returns the sorted data and theoretical
#'   quantiles used in the plots, as a list with elements
#'   \code{z1}, \code{z2}, \code{min}.
#'
#' @references Meintanis, S. G. (2007). Test of fit for
#'   Marshall-Olkin distributions with applications.
#'   \emph{J. Statist. Comput. Simul.} 77, 171-179.
NULL

#' @rdname gofbvch
#' @importFrom graphics abline par plot
#' @importFrom stats ks.test qnorm
#' @export
gofbvch <- function(z1, z2, alpha1, alpha2, alpha3, beta) {
  z1 <- as.numeric(z1); z2 <- as.numeric(z2)
  stopifnot(length(z1) == length(z2),
            all(is.finite(c(z1, z2))), all(c(z1, z2) >= 0))
  ks_z1 <- stats::ks.test(z1, function(q)
    pmarg1(q, alpha1, alpha2, alpha3, beta))
  ks_z2 <- stats::ks.test(z2, function(q)
    pmarg2(q, alpha1, alpha2, alpha3, beta))
  mn <- min_bvch(alpha1, alpha2, alpha3, beta)
  ks_min <- stats::ks.test(pmin(z1, z2), mn$p)
  list(ks_z1 = ks_z1, ks_z2 = ks_z2, ks_min = ks_min)
}

#' @rdname gofbvch
#' @export
qqbvch <- function(z1, z2, alpha1, alpha2, alpha3, beta, plot = TRUE) {
  z1 <- sort(as.numeric(z1)); z2 <- sort(as.numeric(z2))
  zm <- sort(pmin(z1, z2)); n <- length(z1)
  stopifnot(n > 1L)
  pp <- (seq_len(n) - 0.5) / n
  mn <- min_bvch(alpha1, alpha2, alpha3, beta)
  th <- list(
    z1  = qchen(pp, alpha1 + alpha3, beta),
    z2  = qchen(pp, alpha2 + alpha3, beta),
    min = mn$q(pp))
  if (plot) {
    op <- par(mfrow = c(1, 3), pty = "s")
    on.exit(par(op), add = TRUE)
    lims <- range(c(th$z1, z1, th$z2, z2, th$min, zm), na.rm = TRUE)
    qq_one <- function(obs, theo, main) {
      plot(theo, obs, main = main,
           xlab = "theoretical quantile",
           ylab = "sample quantile", pch = 16, cex = 0.6,
           xlim = lims, ylim = lims)
      abline(0, 1, col = "red", lwd = 2)
    }
    qq_one(z1, th$z1, "Q-Q: Z1")
    qq_one(z2, th$z2, "Q-Q: Z2")
    qq_one(zm, th$min, "Q-Q: min(Z1,Z2)")
  }
  invisible(list(z1 = cbind(sample = z1, theory = th$z1),
                 z2 = cbind(sample = z2, theory = th$z2),
                 min = cbind(sample = zm, theory = th$min)))
}
