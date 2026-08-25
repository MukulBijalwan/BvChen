# ---------------------------------------------------------------
# Univariate Chen (2000) distribution: building blocks
#   F(x)   = 1 - exp{ alpha * (1 - exp(x^beta)) },   x >= 0
#   f(x)   = alpha * beta * x^(beta-1) * exp(x^beta)
#             * exp{ alpha * (1 - exp(x^beta)) }
#   h(x)   = alpha * beta * x^(beta-1) * exp(x^beta)
# Useful transform:  u(x) = exp(x^beta) - 1  ~  Exp(alpha)
# ---------------------------------------------------------------

#' @rdname Chen
#' @title The Univariate Chen Distribution
#'
#' @description Density, distribution function, quantile function,
#'   random generation, hazard function and maximum-likelihood fitting
#'   for the Chen (2000) distribution with shape parameter \code{alpha}
#'   and power parameter \code{beta}.
#'
#' @param x,q vector of quantiles (non-negative).
#' @param p vector of probabilities.
#' @param n number of observations.
#' @param alpha shape parameter (\eqn{\alpha > 0}).
#' @param beta power parameter (\eqn{\beta > 0}).
#' @param log logical; if TRUE, the log-density is returned.
#'
#' @return \code{dchen} gives the density, \code{pchen} the CDF,
#'   \code{schen} the survival function, \code{qchen} the quantile
#'   function, \code{rchen} generates random deviates, \code{hchen}
#'   gives the hazard rate and \code{fitchen} returns a fitted list
#'   with elements \code{alpha}, \code{beta} and \code{loglik}.
#'
#' @references Chen, Z. (2000). A new two-parameter lifetime distribution
#'   with bathtub shape or increasing failure rate function.
#'   \emph{Statistics & Probability Letters} 49, 155--161.
#'
#' @examples
#' dchen(1, alpha = 2, beta = 1.5)
#' pchen(c(0.5, 1, 2), 2, 1.5)
#' schen(1, 2, 1.5)                       # == 1 - pchen(1, 2, 1.5)
#' qchen(c(0.25, 0.5, 0.75), 2, 1.5)
#' hchen(seq(0.1, 2, length.out = 5), 2, 1.5)
#' set.seed(1); x <- rchen(500, 2, 1.5)
#' fit <- fitchen(x)                      # MLE close to (2, 1.5)
#' fit$alpha; fit$beta
#'
#' @export
dchen <- function(x, alpha, beta, log = FALSE) {
  stopifnot(length(alpha) == 1L, length(beta) == 1L, alpha > 0, beta > 0)
  xb <- x^beta
  val <- suppressWarnings(
    log(alpha) + log(beta) + (beta - 1) * log(x) + xb +
      alpha * (1 - exp(xb))
  )
  out <- exp(val)
  if (!log) out else val
}

#' @rdname Chen
#' @export
pchen <- function(q, alpha, beta) {
  stopifnot(alpha > 0, beta > 0)
  xb <- q^beta
  1 - exp(alpha * (1 - exp(xb)))
}

#' @rdname Chen
#' @export
schen <- function(q, alpha, beta) {
  1 - pchen(q, alpha, beta)
}

#' @rdname Chen
#' @export
qchen <- function(p, alpha, beta) {
  stopifnot(all(p >= 0 & p <= 1))
  # 1 - p = exp{a(1 - e^{x^b})}  =>  x = [ln(1 - ln(1-p)/a)]^{1/b}
  inner <- 1 - log(1 - p) / alpha       # >= 1
  ifelse(inner <= 1, 0, log(inner)^(1 / beta))
}

#' @rdname Chen
#' @export
rchen <- function(n, alpha, beta) {
  stopifnot(n >= 0, alpha > 0, beta > 0)
  # u(x) = e^{x^b} - 1 ~ Exp(alpha)
  u <- stats::rexp(n, rate = alpha)
  log1p(u)^(1 / beta)
}

#' @rdname Chen
#' @export
hchen <- function(x, alpha, beta) {
  stopifnot(alpha > 0, beta > 0)
  alpha * beta * x^(beta - 1) * exp(x^beta)
}

#' @name fitchen
#' @title Univariate Maximum-Likelihood Fit of the Chen Distribution
#'
#' @description Fits the two-parameter Chen distribution by numerical
#'   maximisation of the log-likelihood. Used internally to obtain
#'   starting values for \code{\link{fitbvch}}.
#'
#' @param x non-negative numeric vector of observations.
#' @param start optional named list with starting values \code{alpha},
#'   \code{beta}; sensible defaults are chosen from the data.
#' @param tol convergence tolerance used by \code{fitchen}.
#' @param maxit maximum number of Newton iterations in \code{fitchen}.
#' @return A list with components \code{alpha}, \code{beta},
#'   \code{loglik}, \code{converged} and \code{se} (approximate
#'   standard errors from the numerical Hessian).
#'
#' @examples
#' set.seed(42)
#' x <- rchen(300, alpha = 1.5, beta = 2)
#' fit <- fitchen(x)
#' fit$alpha; fit$beta; fit$loglik
#'
#' @export
fitchen <- function(x, start = NULL, tol = 1e-8, maxit = 100) {
  stopifnot(is.numeric(x), all(is.finite(x)), all(x >= 0))
  x <- as.numeric(x)
  n <- length(x)

  negll <- function(par) {
    a <- par[1]; b <- par[2]
    if (a <= 0 || b <= 0) return(.Machine$double.xmax)
    -sum(dchen(x, a, b, log = TRUE))
  }

  if (is.null(start)) {
    # method-of-moments style starting values
    m1 <- mean(x); m2 <- mean(x^2)
    start <- list(alpha = 1 / max(m1, 1e-3),
                  beta  = max(0.5, log(2) / log(max(2 * m2 / max(m1^2, 1e-6), 1.01))))
  }

  opt <- stats::optim(unlist(start), negll,
                      method = "Nelder-Mead",
                      control = list(reltol = tol, maxit = maxit * 20))
  h <- 1e-5
  H <- matrix(NA_real_, 2, 2)
  p <- opt$par
  f0 <- negll(p)
  H[1, 1] <- (negll(p + c(h, 0)) - 2 * f0 + negll(p - c(h, 0))) / h^2
  H[2, 2] <- (negll(p + c(0, h)) - 2 * f0 + negll(p - c(0, h))) / h^2
  H[1, 2] <- H[2, 1] <-
    (negll(p + c(h, h)) - negll(p + c(h, -h)) -
       negll(p + c(-h, h)) + negll(p - c(h, h))) / (4 * h^2)
  cov <- tryCatch(solve(H), error = function(e) matrix(NA_real_, 2, 2))

  list(alpha = p[1], beta = p[2], loglik = -opt$value,
       converged = opt$convergence == 0,
       se = sqrt(pmax(diag(cov), 0)))
}
