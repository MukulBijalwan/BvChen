if (requireNamespace("roxygen2", quietly = TRUE)) {
  roxygen2::roxygenise("C:/Users/Admin/.cline/data/workspaces/chat/BvChen")
  cat("roxygen done\n")
} else {
  cat("roxygen2 not installed\n")
}
