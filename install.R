# install.R  (put at the repo root)
pkgs <- c(
  "jsonlite",   # required by predict_bart_cli.R
  "dbarts",     # needed to predict with bart_model_A_*.rds
  "BART",       # needed to predict with bart_model_B_*.rds
  "pROC",       # only if you ever compute AUCs in R
  "caret"       # only if you refit caret models later
)
install.packages(pkgs, repos = "https://cloud.r-project.org")
