# ---------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------

#' @name plots
#' @rdname plots
#' @title Plots for the Bivariate Chen Distribution
#'
#' @description Contour (\code{plot}) and perspective (\code{persp})
#'   views of the absolutely continuous part of the joint density,
#'   plus a scatter-plot method for simulated data of class
#'   \code{"bvch"}.
#'
#' @param x for \code{plot.bvch}: an \code{n x 2} matrix of class
#'   \code{"bvch"} (as produced by \code{\link{rbvch}}) or an object
#'   of class \code{"bvchfit"}; for \code{persp.bvch} the same types.
#' @param alpha1,alpha2,alpha3,beta distribution parameters (required
#'   when \code{x} is a data matrix).
#' @param from,to range of the evaluation grid on both axes.
#' @param n.grid grid size per axis.
#' @param theta,phi viewing angles passed to \code{\link[graphics]{persp}}.
#' @param ... further graphical parameters.
#'
#' @return \code{plot.bvch} invisibly returns the evaluated density
#'   grid; \code{persp.bvch} invisibly returns the transformation
#'   matrix used.
#' @examples
#' set.seed(11)
#' sim <- rbvch(300, 1, 1, 1, 1.5)
#'
#' # contour of the AC density with the simulated scatter
#' plot(sim, 1, 1, 1, 1.5, to = 2.5)
#'
#' # 3-D perspective of the absolutely continuous part
#' persp(sim, 1, 1, 1, 1.5, to = 2)
NULL

.bvch_grid <- function(alpha1, alpha2, alpha3, beta,
                       from = 0.01, to = 3, n.grid = 60) {
  gx <- seq(from, to, length.out = n.grid)
  gz <- outer(gx, gx, function(a, b)
    dbvch(a, b, alpha1, alpha2, alpha3, beta, component = "ac"))
  list(x = gx, y = gx, z = gz)
}

#' Plot Method for Bivariate Chen Objects
#'
#' Draws a contour of the absolutely continuous joint density with the
#' simulated points (or the fitted parameters) superimposed.
#'
#' @param x object of class \code{"bvch"} or \code{"bvchfit"}.
#' @param alpha1,alpha2,alpha3,beta distribution parameters (required
#'   when \code{x} is a data matrix).
#' @param ... further arguments passed to methods.
#' @method plot bvch
#' @rawNamespace S3method(plot,bvch)
#' @export
plot.bvch <- function(x, alpha1 = NULL, alpha2 = NULL,
                      alpha3 = NULL, beta = NULL, ...) {
  fitted <- inherits(x, "bvchfit")
  if (fitted) {
    cf <- x$coefficients
    alpha1 <- cf[["alpha1"]]; alpha2 <- cf[["alpha2"]]
    alpha3 <- cf[["alpha3"]]; beta  <- cf[["beta"]]
    x <- NULL
  } else {
    if (is.null(alpha1) || is.null(alpha2) || is.null(alpha3) ||
        is.null(beta))
      stop("please provide alpha1, alpha2, alpha3 and beta")
  }
  .plot_bvch_data(x, alpha1, alpha2, alpha3, beta, ...)
}

.plot_bvch_data <- function(x, alpha1, alpha2, alpha3, beta,
                            from = 0.01, to = 3, n.grid = 80, ...) {
  g <- .bvch_grid(alpha1, alpha2, alpha3, beta, from, to, n.grid)
  graphics::contour(g$x, g$y, g$z,
                    xlab = expression(z[1]), ylab = expression(z[2]),
                    main = "Bivariate Chen: AC joint density", ...)
  if (!is.null(x)) {
    graphics::points(x[, "z1"], x[, "z2"], pch = 16, cex = 0.5,
                     col = grDevices::adjustcolor("steelblue", 0.4))
  }
  graphics::abline(0, 1, lty = 2, col = "red")
  invisible(g)
}

#' @rdname plots
persp.bvch <- function(x, alpha1, alpha2, alpha3, beta,
                       from = 0.05, to = 2.5, n.grid = 50,
                       theta = -30, phi = 25, ...) {
  if (inherits(x, "bvchfit")) {
    cf <- x$coefficients
    alpha1 <- cf["alpha1"]; alpha2 <- cf["alpha2"]
    alpha3 <- cf["alpha3"]; beta  <- cf["beta"]
  }
  g <- .bvch_grid(alpha1, alpha2, alpha3, beta, from, to, n.grid)
  tr <- graphics::persp(g$x, g$y, g$z, theta = theta, phi = phi,
                        xlab = "z1", ylab = "z2",
                        zlab = "f(z1,z2)", shade = 0.4,
                        col = "lightsteelblue", ...)
  invisible(tr)
}
