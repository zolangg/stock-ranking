# Ensure user library path exists & is used
userlib <- Sys.getenv("R_LIBS_USER")
if (!nzchar(userlib)) {
  userlib <- file.path(Sys.getenv("HOME"), "R", "library")
}
if (!dir.exists(userlib)) dir.create(userlib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(userlib, .libPaths()))

repos <- "https://cloud.r-project.org"
pkgs  <- c("jsonlite", "dbarts", "BART", "pROC", "caret")

need <- setdiff(pkgs, rownames(installed.packages()))
if (length(need)) install.packages(need, repos = repos)
