#!/usr/bin/env Rscript

# predict_bart_cli.R
# - Self-bootstraps packages into ./r-lib (writable)
# - Loads saved BART models
# - Reads JSON from stdin with numeric inputs
# - Outputs JSON with predictions

suppressWarnings(suppressMessages({
  # Ensure a writable user library
  libdir <- file.path(getwd(), "r-lib")
  if (!dir.exists(libdir)) dir.create(libdir, recursive = TRUE, showWarnings = FALSE)
  .libPaths(c(libdir, .libPaths()))
  
  # Minimal installer
  need <- c("jsonlite", "dbarts", "BART")
  for (p in need) {
    if (!requireNamespace(p, quietly = TRUE)) {
      install.packages(p, lib = libdir, repos = "https://cloud.r-project.org", quiet = TRUE)
    }
  }
  library(jsonlite)
  library(dbarts)  # bart()
  library(BART)    # pbart()
}))

# ---- Helpers ---------------------------------------------------------------
# Safe numeric
num <- function(x) {
  if (is.null(x)) return(NA_real_)
  if (is.numeric(x)) return(as.numeric(x))
  x <- gsub(",", "", as.character(x), fixed = TRUE)
  suppressWarnings(as.numeric(x))
}
nz <- function(x, eps = 1e-6) pmax(num(x), eps)

# Read predictors list (.rds may be a list or a character vector)
read_pred_list <- function(path) {
  obj <- readRDS(path)
  if (is.character(obj)) return(obj)
  if (is.list(obj) && !is.null(obj$predictors)) return(as.character(obj$predictors))
  stop(sprintf("Could not parse predictors in %s", path))
}

# ---- Load models -----------------------------------------------------------
# NOTE: expects files alongside this R script.
modelA_path  <- "bart_model_A_predDVol_ln.rds"
predsA_path  <- "bart_model_A_predictors.rds"
modelB_path  <- "bart_model_B_FT.rds"
predsB_path  <- "bart_model_B_predictors.rds"

if (!file.exists(modelA_path) || !file.exists(predsA_path) ||
    !file.exists(modelB_path) || !file.exists(predsB_path)) {
  stop("Model files not found in working directory.")
}

bartA  <- readRDS(modelA_path)
predsA <- read_pred_list(predsA_path)

bartB  <- readRDS(modelB_path)
predsB <- read_pred_list(predsB_path)

# ---- Read request (JSON on stdin) -----------------------------------------
# Expected input fields (numeric):
# PMVolM, PMDolM, FloatM, GapPct, ATR, MCapM, Catalyst
req_txt <- suppressWarnings(readLines(file("stdin")))
if (length(req_txt) == 0) stop("No JSON received on stdin.")
req <- jsonlite::fromJSON(paste(req_txt, collapse = "\n"))

PMVolM  <- nz(req$PMVolM)
PMDolM  <- nz(req$PMDolM)
FloatM  <- nz(req$FloatM)
GapPct  <- num(req$GapPct); GapFrac <- pmax(num(GapPct), 0) / 100
ATR     <- nz(req$ATR)
MCapM   <- nz(req$MCapM)
Catalyst <- as.integer(num(req$Catalyst) != 0)

# Feature engineering (match your R fitting code 1:1)
eps <- 1e-6
FR       <- PMVolM / pmax(FloatM, eps)
ln_pm    <- log(pmax(PMVolM, eps))
ln_pmdol <- log(pmax(PMDolM, eps))
ln_fr    <- log(pmax(FR, eps))
ln_gapf  <- log(GapFrac + eps)
ln_atr   <- log(pmax(ATR, eps))
ln_mcap  <- log(pmax(MCapM, eps))
ln_pmdol_per_mcap <- log(pmax(PMDolM / pmax(MCapM, eps), eps))

# Build a data.frame with all potential columns, then select by preds files
full_df <- data.frame(
  ln_pm = ln_pm,
  ln_pmdol = ln_pmdol,
  ln_fr = ln_fr,
  ln_gapf = ln_gapf,
  ln_atr = ln_atr,
  ln_mcap = ln_mcap,
  ln_pmdol_per_mcap = ln_pmdol_per_mcap,
  Catalyst = Catalyst,
  stringsAsFactors = FALSE
)

# ---- Predict Model A: Predicted Daily Volume (millions) --------------------
xA <- as.matrix(full_df[, predsA, drop = FALSE])
# dbarts::predict returns draws x n matrix, mean over draws:
predA_ln_draws <- predict(bartA, newdata = xA)
predA_ln_mean  <- if (is.matrix(predA_ln_draws)) colMeans(predA_ln_draws) else as.numeric(predA_ln_draws)
PredVol_M      <- pmax(exp(predA_ln_mean), eps)

# ---- Predict Model B: FT probability --------------------------------------
dfB <- cbind(full_df, PredVol_M = PredVol_M)
xB  <- as.matrix(dfB[, predsB, drop = FALSE])

# For pbart, predict() returns list with prob.test.mean
pb <- tryCatch(
  BART::predict(bartB, newdata = xB),
  error = function(e) NULL
)
if (!is.null(pb) && !is.null(pb$prob.test.mean)) {
  FT_Prob <- as.numeric(pb$prob.test.mean)
} else if (!is.null(bartB$prob.train.mean)) {
  # fallback (should be rare)
  FT_Prob <- rep(as.numeric(bartB$prob.train.mean)[1], length(PredVol_M))
} else {
  stop("Could not obtain FT probabilities from BART model.")
}

FT_Prob <- pmin(pmax(FT_Prob, 0), 1)
FT_Label <- ifelse(FT_Prob >= 0.5, "FT", "Fail")

# ---- Emit JSON -------------------------------------------------------------
out <- list(
  PredVol_M = PredVol_M,
  FT_Prob   = FT_Prob,
  FT_Label  = FT_Label
)
cat(jsonlite::toJSON(out, auto_unbox = TRUE))