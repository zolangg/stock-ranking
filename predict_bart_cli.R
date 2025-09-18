#!/usr/bin/env Rscript

suppressWarnings(suppressPackageStartupMessages({
  # only jsonlite is required; dbarts/BART are not needed for *prediction*
  # because we are only loading already-fitted models via readRDS + predict
  # (dbarts/BART provide predict methods when the object classes are resident,
  # but readRDS keeps the predict method attached to the object environment).
  # If jsonlite is missing, we bail out clearly.
  has_jsonlite <- requireNamespace("jsonlite", quietly = TRUE)
}))

die <- function(msg, code = 1L) {
  write(msg, file = stderr())
  quit("no", status = code)
}

if (!has_jsonlite) {
  die("jsonlite not installed. Install it (e.g., add 'jsonlite' to packages.txt) and redeploy.")
}

args <- commandArgs(trailingOnly = TRUE)
mode <- "--both"
models_dir <- "."

# parse args
for (a in args) {
  if (a %in% c("--both","--predictA","--predictB")) mode <- a
  if (startsWith(a, "--models_dir=")) {
    models_dir <- sub("^--models_dir=", "", a)
  }
}

# helpers
read_all <- function() {
  raw <- readLines(con = file("stdin"), warn = FALSE)
  txt <- paste(raw, collapse = "\n")
  if (nzchar(txt)) txt else "[]"
}

safe_num <- function(x) {
  if (is.null(x) || length(x) == 0) return(NA_real_)
  suppressWarnings(as.numeric(x))
}

# load models + predictor lists
load_model <- function(path) {
  if (!file.exists(path)) die(paste("Missing model file:", path))
  readRDS(path)
}

modelA_path <- file.path(models_dir, "bart_model_A_predDVol_ln.rds")
predsA_path <- file.path(models_dir, "bart_model_A_predictors.rds")
modelB_path <- file.path(models_dir, "bart_model_B_FT.rds")
predsB_path <- file.path(models_dir, "bart_model_B_predictors.rds")

bartA <- load_model(modelA_path)
preds_A <- load_model(predsA_path)
bartB <- load_model(modelB_path)
preds_B <- load_model(predsB_path)

# read payload
library(jsonlite)
payload_txt <- read_all()
inp <- tryCatch(jsonlite::fromJSON(payload_txt, simplifyDataFrame = TRUE), error = function(e) NULL)
if (is.null(inp)) die("Invalid JSON input on stdin.", 2L)

# coerce to data.frame
df <- as.data.frame(inp, stringsAsFactors = FALSE)

# expected raw columns (case from Streamlit):
# PMVolM (M shares), PMDolM ($M), FloatM (M), GapPct (%), ATR ($), MCapM ($M), Catalyst (0/1)
needed <- c("PMVolM","PMDolM","FloatM","GapPct","ATR","MCapM","Catalyst")
for (nm in needed) if (!nm %in% names(df)) df[[nm]] <- NA_real_

# numeric coercion
for (nm in needed) df[[nm]] <- safe_num(df[[nm]])

# feature engineering — MUST mirror training script
eps <- 1e-6
FR  <- df$PMVolM / pmax(df$FloatM, eps)
ln_pm   <- log(pmax(df$PMVolM, eps))
ln_pmdol<- log(pmax(df$PMDolM, eps))
ln_fr   <- log(pmax(FR, eps))
ln_gapf <- log(pmax(df$GapPct, 0)/100 + eps)
ln_atr  <- log(pmax(df$ATR, eps))
ln_mcap <- log(pmax(df$MCapM, eps))
ln_pmdol_per_mcap <- log(pmax(df$PMDolM / pmax(df$MCapM, eps), eps))

# Catalyst as integer 0/1
Catalyst <- ifelse(is.na(df$Catalyst), 0L, as.integer(df$Catalyst != 0))

# build feature frame with ALL engineered features
feat_all <- data.frame(
  ln_pm = ln_pm,
  ln_pmdol = ln_pmdol,
  ln_fr = ln_fr,
  ln_gapf = ln_gapf,
  ln_atr = ln_atr,
  ln_mcap = ln_mcap,
  ln_pmdol_per_mcap = ln_pmdol_per_mcap,
  Catalyst = Catalyst,
  PMVolM = df$PMVolM,
  PMDolM = df$PMDolM,
  FloatM = df$FloatM,
  GapPct = df$GapPct,
  ATR = df$ATR,
  MCapM = df$MCapM,
  stringsAsFactors = FALSE
)

# helper: select predictors robustly (in case lists are character)
sel_mat <- function(df, pred_names) {
  keep <- intersect(pred_names, colnames(df))
  if (!length(keep)) die(sprintf("None of the expected predictors found: %s", paste(pred_names, collapse=", ")))
  as.matrix(df[, keep, drop = FALSE])
}

# --- predict Model A (PredVol_M, in millions) ---
pred_vol_m <- rep(NA_real_, nrow(df))
if (mode %in% c("--both","--predictA","--predictB")) {
  XA <- sel_mat(feat_all, preds_A)
  # dbarts::bart predict returns draws x n OR vector
  drawsA <- tryCatch(predict(bartA, newdata = XA), error = function(e) NULL)
  if (is.null(drawsA)) die("Predict failed for Model A.", 3L)
  # posterior mean in ln-space -> exponentiate
  mu_ln <- if (is.matrix(drawsA)) colMeans(drawsA) else as.numeric(drawsA)
  pred_vol_m <- pmax(exp(mu_ln), eps)
}

# --- predict Model B (FT prob) ---
ft_prob <- rep(NA_real_, nrow(df))
if (mode %in% c("--both","--predictB")) {
  # ensure PredVol_M is available (from A) for the feature set B
  feat_all$PredVol_M <- pred_vol_m
  
  XB <- sel_mat(feat_all, preds_B)
  # BART::pbart predict returns list with prob.test.mean
  pb <- tryCatch({
    # if you saved bartB from BART::pbart, predict(bartB, newdata=...) should work:
    predict(bartB, newdata = XB)
  }, error = function(e) NULL)
  if (is.null(pb)) die("Predict failed for Model B.", 4L)
  
  if (!is.null(pb$prob.test.mean)) {
    ft_prob <- as.numeric(pb$prob.test.mean)
  } else if (!is.null(pb$ppost)) {
    ft_prob <- rowMeans(pb$ppost)
  } else {
    die("Unexpected pbart predict structure.", 5L)
  }
  # clamp
  ft_prob[!is.finite(ft_prob)] <- NA_real_
  ft_prob <- pmin(pmax(ft_prob, 0), 1)
}

# output JSON
out <- data.frame(
  PredVol_M = pred_vol_m,
  FT_Prob   = ft_prob
)
cat(jsonlite::toJSON(out, dataframe = "rows", auto_unbox = TRUE))