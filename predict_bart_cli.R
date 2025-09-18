#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(jsonlite)
  library(dbarts)   # bart()
  library(BART)     # pbart(), predict()
  library(readr)
})

args <- commandArgs(trailingOnly = TRUE)
# Modes: --both (default), --predvol, --ft
mode <- if (length(args)) args[1] else "--both"
if (!mode %in% c("--both","--predvol","--ft")) mode <- "--both"

# ---- Config / constants ----
eps <- 1e-6

# ---- Load models + predictor names (fail hard if missing) ----
A_model_path <- "bart_model_A_predDVol_ln.rds"
A_preds_path <- "bart_model_A_predictors.rds"
B_model_path <- "bart_model_B_FT.rds"
B_preds_path <- "bart_model_B_predictors.rds"

stopifnot(file.exists(A_model_path), file.exists(A_preds_path))
if (mode %in% c("--both","--ft")) {
  stopifnot(file.exists(B_model_path), file.exists(B_preds_path))
}

bartA   <- readRDS(A_model_path)
preds_A <- readRDS(A_preds_path)

if (mode %in% c("--both","--ft")) {
  bartB   <- readRDS(B_model_path)
  preds_B <- readRDS(B_preds_path)
}

# ---- Feature engineering (must mirror your R fitting code) ----
featurize <- function(df) {
  # expects raw columns:
  # PMVolM, PMDolM, FloatM, GapPct, ATR, MCapM, Catalyst (0/1)
  with(df, {
    FR      <- PMVolM / pmax(FloatM, eps)
    ln_pm   <- log(pmax(PMVolM, eps))
    ln_pmdol<- log(pmax(PMDolM, eps))
    ln_fr   <- log(pmax(FR, eps))
    ln_gapf <- log(pmax(GapPct, 0)/100 + eps)
    ln_atr  <- log(pmax(ATR, eps))
    ln_mcap <- log(pmax(MCapM, eps))
    ln_pmdol_per_mcap <- log(pmax(PMDolM / pmax(MCapM, eps), eps))
    Catalyst <- as.integer(ifelse(is.na(Catalyst), 0, Catalyst != 0))
    
    data.frame(
      PMVolM, PMDolM, FloatM, GapPct, ATR, MCapM, Catalyst,
      FR, ln_pm, ln_pmdol, ln_fr, ln_gapf, ln_atr, ln_mcap, ln_pmdol_per_mcap,
      check.names = FALSE
    )
  })
}

# ---- Read JSON from STDIN (array of rows) ----
incoming <- suppressWarnings(fromJSON(file("stdin"), simplifyDataFrame = TRUE))
if (!is.data.frame(incoming)) {
  stop("Input must be a JSON array of objects with numeric fields.")
}

# coerce all needed fields
need <- c("PMVolM","PMDolM","FloatM","GapPct","ATR","MCapM","Catalyst")
for (nm in need) if (!nm %in% names(incoming)) incoming[[nm]] <- NA_real_

incoming <- within(incoming, {
  PMVolM <- as.numeric(PMVolM); PMDolM <- as.numeric(PMDolM)
  FloatM <- as.numeric(FloatM); GapPct <- as.numeric(GapPct)
  ATR    <- as.numeric(ATR);    MCapM  <- as.numeric(MCapM)
  Catalyst <- as.numeric(Catalyst)
})

Xall <- featurize(incoming)

# ---- Model A: Predicted Daily Volume (millions) ----
pred_vol_m <- NULL
if (mode %in% c("--both","--predvol")) {
  Xa <- as.matrix(Xall[, preds_A, drop = FALSE])
  # dbarts::predict gives draws x n matrix; take column means
  pred_ln_draws <- predict(bartA, newdata = Xa)
  pred_ln_mean  <- if (is.matrix(pred_ln_draws)) colMeans(pred_ln_draws) else as.numeric(pred_ln_draws)
  pred_vol_m    <- pmax(exp(pred_ln_mean), eps)
}

# ---- Model B: FT probability ----
ft_prob <- NULL
if (mode %in% c("--both","--ft")) {
  # Must include PredVol_M as in training
  if (is.null(pred_vol_m)) {
    # if we’re in --ft mode without A: still compute A for consistency
    Xa <- as.matrix(Xall[, preds_A, drop = FALSE])
    pred_ln_draws <- predict(bartA, newdata = Xa)
    pred_ln_mean  <- if (is.matrix(pred_ln_draws)) colMeans(pred_ln_draws) else as.numeric(pred_ln_draws)
    pred_vol_m    <- pmax(exp(pred_ln_mean), eps)
  }
  Xall$PredVol_M <- pred_vol_m
  Xb <- as.matrix(Xall[, preds_B, drop = FALSE])
  
  pr <- predict(bartB, newdata = Xb)
  if (!is.null(pr$prob.test.mean)) {
    ft_prob <- as.numeric(pr$prob.test.mean)
  } else if (!is.null(pr$ppost)) {
    ft_prob <- rowMeans(pr$ppost)
  } else {
    stop("Unexpected structure returned by BART::predict on pbart model.")
  }
}

# ---- Emit JSON to STDOUT ----
out <- data.frame(
  PredVol_M = if (is.null(pred_vol_m)) NA_real_ else pred_vol_m,
  FT_Prob   = if (is.null(ft_prob))    NA_real_ else pmin(pmax(ft_prob, 0), 1),
  stringsAsFactors = FALSE
)
cat(toJSON(out, digits = 6))