# ---------------------------------------------------------------
# Likelihood and EM estimation of the bivariate Chen distribution
#
# On the exponential scale ti = e^{zi^beta} - 1 the latent model is
# the classical Marshall-Olkin competing-risks model, so alpha hats
# have the closed form  alphahat_i = D_i / U_i  (failures/exposure),
# while beta is updated by maximising the observed log-likelihood.
# ---------------------------------------------------------------

#' @name Likelihood
#' @title Log-Likelihood and Profile Likelihood of the Bivariate Chen Model
#'
#' @description Observed-data log-likelihood and profile likelihood for
#'   the common power parameter \eqn{\beta}.
#'
#' @param z1,z2 numeric vectors of paired observations
#'   (\code{z1[i]}, \code{z2[i]}).
#' @param alpha1,alpha2,alpha3 positive shape parameters of the three
#'   latent Chen variables.
#' @param beta positive common power parameter.
#'
#' @return \code{loglik_bvch} returns the scalar log-likelihood;
#'   \code{profile_beta} returns a list with the evaluated grid
#'   (\code{beta}, \code{prof_loglik}) and the optimiser
#'   \code{beta_hat}.
#'
#' @examples
#' set.seed(7)
#' sim <- rbvch(300, 1, 1, 1, 1.5)
#' loglik_bvch(sim[, "z1"], sim[, "z2"], 1, 1, 1, 1.5)
#'
#' pb <- profile_beta(sim[, "z1"], sim[, "z2"])
#' pb$beta_hat                        # close to 1.5
#' head(pb$grid)
NULL

# one EM-type update of (alpha1, alpha2, alpha3) at fixed beta:
# numerical maximisation of the observed log-likelihood on the
# log-scale (smooth and well behaved in the alphas)
.alphas_at_beta <- function(z1, z2, beta, start) {
  o <- stats::optim(log(start), function(lp) {
    a <- exp(lp)
    ll <- suppressWarnings(
      tryCatch(loglik_bvch(z1, z2, a[1], a[2], a[3], beta),
               error = function(e) NA_real_))
    if (!is.finite(ll)) return(.Machine$double.xmax)
    -ll
  }, method = "Nelder-Mead",
     control = list(reltol = 1e-7, maxit = 600))
  exp(o$par)
}

#' @rdname Likelihood
#' @param betas candidate beta values for the profile grid; a sensible
#'   default grid over (0, 20] is used if omitted.
#' @param start named numeric vector with starting values for the
#'   alpha search inside \code{profile_beta}.

#' @importFrom stats optim
#' @export
profile_beta <- function(z1, z2, betas = NULL,
                         start = c(alpha1 = 1, alpha2 = 1, alpha3 = 1)) {
  z1 <- as.numeric(z1); z2 <- as.numeric(z2)
  stopifnot(length(z1) == length(z2),
            all(is.finite(c(z1, z2))), all(c(z1, z2) >= 0))
  prof <- function(b) {
    a <- .alphas_at_beta(z1, z2, b, start)
    loglik_bvch(z1, z2, a[1], a[2], a[3], b)
  }
  if (is.null(betas))
    betas <- exp(seq(log(0.05), log(20), length.out = 25))
  # warm-start the alpha search across the grid
  a_cur <- start
  vals <- numeric(length(betas))
  for (i in seq_along(betas)) {
    a_cur <- .alphas_at_beta(z1, z2, betas[i], a_cur)
    vals[i] <- loglik_bvch(z1, z2, a_cur[1], a_cur[2], a_cur[3], betas[i])
  }
  i <- which.max(vals)
  opt <- stats::optimize(prof,
                         lower = max(betas[max(i - 1, 1)], 1e-4),
                         upper = betas[min(i + 1, length(betas))],
                         maximum = TRUE)$maximum
  list(grid = data.frame(beta = betas, prof_loglik = vals),
       beta_hat = unname(opt))
}

#' @rdname Likelihood
#' @export
loglik_bvch <- function(z1, z2, alpha1, alpha2, alpha3, beta) {
  .validate_bvch(alpha1, alpha2, alpha3, beta)
  z1 <- as.numeric(z1); z2 <- as.numeric(z2)
  stopifnot(length(z1) == length(z2), all(is.finite(z1)),
            all(is.finite(z2)), all(z1 >= 0), all(z2 >= 0))
  lt <- z1 < z2; gt <- z1 > z2; eq <- z1 == z2
  ll <- numeric(length(z1))
  ll[lt] <- dchen(z1[lt], alpha1, beta, log = TRUE) +
            dchen(z2[lt], alpha2 + alpha3, beta, log = TRUE)
  ll[gt] <- dchen(z2[gt], alpha2, beta, log = TRUE) +
            dchen(z1[gt], alpha1 + alpha3, beta, log = TRUE)
  ll[eq] <- dbvch(z1[eq], z1[eq], alpha1, alpha2, alpha3, beta,
                  component = "sing", log = TRUE)
  sum(ll)
}


