#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(jsonlite)
  library(dbarts)
  library(BART)
})

args <- commandArgs(trailingOnly = TRUE)
do_both <- "--both" %in% args || length(args) == 0
do_A    <- "--A" %in% args
do_B    <- "--B" %in% args
if (!do_A && !do_B) do_A <- do_B <- TRUE

# --- locate models under ./models (next to this script)
args <- commandArgs(trailingOnly = TRUE)
do_both <- "--both" %in% args || length(args) == 0
do_A    <- "--A" %in% args
do_B    <- "--B" %in% args
if (!do_A && !do_B) do_A <- do_B <- TRUE

# ----- resolve models directory -----
# 1) --models-dir=/path
models_arg <- grep("^--models-dir=", args, value = TRUE)
models_dir <- NA_character_
if (length(models_arg)) {
  models_dir <- sub("^--models-dir=", "", models_arg[1])
}

# 2) env MODELS_DIR
if (is.na(models_dir) || !nzchar(models_dir)) {
  env_md <- Sys.getenv("MODELS_DIR", unset = "")
  if (nzchar(env_md)) models_dir <- env_md
}

# 3) default: script_dir/models
if (is.na(models_dir) || !nzchar(models_dir)) {
  script_dir <- tryCatch(normalizePath(dirname(sys.frame(1)$ofile)),
                         error = function(e) getwd())
  models_dir <- file.path(script_dir, "models")
}

models_dir <- normalizePath(models_dir, mustWork = FALSE)

rds_A  <- file.path(models_dir, "bart_model_A_predDVol_ln.rds")
rds_Ap <- file.path(models_dir, "bart_model_A_predictors.rds")
rds_B  <- file.path(models_dir, "bart_model_B_FT.rds")
rds_Bp <- file.path(models_dir, "bart_model_B_predictors.rds")

if (!file.exists(rds_A) || !file.exists(rds_Ap) ||
    !file.exists(rds_B) || !file.exists(rds_Bp)) {
  stop(paste0(
    "Model files not found in: ", models_dir, "\nExpected:\n  ",
    basename(rds_A), "\n  ", basename(rds_Ap), "\n  ",
    basename(rds_B), "\n  ", basename(rds_Bp), "\n"
  ))
}

# ---- read JSON payload (array of rows)
txt <- readLines(file("stdin"))
payload <- fromJSON(paste(txt, collapse=""))
if (is.data.frame(payload)) df <- payload else df <- as.data.frame(payload)

eps <- 1e-6

# canonicalize inputs (expecting PMVolM, PMDolM, FloatM, GapPct, ATR, MCapM, Catalyst)
canon <- data.frame(
  PMVolM   = as.numeric(df$PMVolM  %||% NA),
  PMDolM   = as.numeric(df$PMDolM  %||% NA),
  FloatM   = as.numeric(df$FloatM  %||% NA),
  GapPct   = as.numeric(df$GapPct  %||% NA),
  ATR      = as.numeric(df$ATR     %||% NA),
  MCapM    = as.numeric(df$MCapM   %||% NA),
  Catalyst = as.integer(ifelse(is.na(df$Catalyst), 0, df$Catalyst))
)

# feature engineering (match training)
canon$FR        <- canon$PMVolM / pmax(canon$FloatM, eps)
canon$ln_pm     <- log(pmax(canon$PMVolM, eps))
canon$ln_pmdol  <- log(pmax(canon$PMDolM, eps))
canon$ln_fr     <- log(pmax(canon$FR, eps))
canon$ln_gapf   <- log(pmax(canon$GapPct, 0)/100 + eps)
canon$ln_atr    <- log(pmax(canon$ATR, eps))
canon$ln_mcap   <- log(pmax(canon$MCapM, eps))
canon$Catalyst  <- as.integer(canon$Catalyst != 0)
canon$ln_pmdol_per_mcap <- log(pmax(canon$PMDolM / pmax(canon$MCapM, eps), eps))

# helper to pull the exact columns in the saved order
pick <- function(df, cols) {
  cols <- intersect(cols, names(df))
  if (!length(cols)) stop("No matching predictors in payload for model.")
  df[, cols, drop = FALSE]
}

out <- list()

# ---- Model A: PredVol_M
if (do_A || do_both) {
  xA <- as.matrix(pick(canon, preds_A))
  predA_ln <- predict(bartA, newdata = xA)
  predA_ln <- if (is.matrix(predA_ln)) colMeans(predA_ln) else as.numeric(predA_ln)
  out$PredVol_M <- as.numeric(pmax(exp(predA_ln), eps))
}

# ---- Model B: FT probability (needs PredVol_M)
if (do_B || do_both) {
  if (is.null(out$PredVol_M)) {
    # compute with same A to avoid leakage
    xA <- as.matrix(pick(canon, preds_A))
    predA_ln <- predict(bartA, newdata = xA)
    predA_ln <- if (is.matrix(predA_ln)) colMeans(predA_ln) else as.numeric(predA_ln)
    out$PredVol_M <- as.numeric(pmax(exp(predA_ln), eps))
  }
  canon$PredVol_M <- out$PredVol_M
  xB <- as.matrix(pick(canon, preds_B))
  pr <- predict(bartB, newdata = xB)
  p  <- if (!is.null(pr$prob.test.mean)) as.numeric(pr$prob.test.mean)
        else if (!is.null(pr$ppost)) rowMeans(pr$ppost)
        else stop("Unexpected pbart predict output")
  out$FT_Prob <- as.numeric(pmin(pmax(p, 0), 1))
}

# emit NDJSON (or a single array)
cat(toJSON(as.data.frame(out), auto_unbox = FALSE))
