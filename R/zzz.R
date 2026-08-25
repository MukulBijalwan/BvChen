# ---------------------------------------------------------------
# Delayed S3 registration: 'persp' lives in graphics, which is not
# loaded when the namespace is initialised with only base attached.
# ---------------------------------------------------------------

#' @importFrom utils packageVersion
.onLoad <- function(libname, pkgname) {
  if (requireNamespace("graphics", quietly = TRUE)) {
    registerS3method("persp", "bvch", persp.bvch,
                     envir = asNamespace("graphics"))
  }
}
