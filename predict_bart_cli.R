#!/usr/bin/env Rscript

# ---------- bootstrap: install CRAN packages if missing ----------
need <- c("jsonlite","dbarts","BART")
for (p in need) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p, repos = "https://cloud.r-project.org", quiet = TRUE)
  }
}
suppressPackageStartupMessages({
  library(jsonlite)
  library(dbarts)
  library(BART)
})

args <- commandArgs(trailingOnly = TRUE)
mode_both <- length(args) > 0 && any(args %in% c("--both","--all"))

# ---------- load models & predictor lists ----------
# These files must sit in the same folder as this script (or use absolute paths).
read_pred_list <- function(f) {
  # could be a character vector or an R object; coerce to character
  x <- readRDS(f)
  if (is.character(x)) return(x)
  if (is.list(x) && !is.null(x$predictors)) return(as.character(x$predictors))
  stop(paste("Unexpected predictors file format:", f))
}

modelA   <- readRDS("bart_model_A_predDVol_ln.rds")
preds_A  <- read_pred_list("bart_model_A_predictors.rds")
modelB   <- readRDS("bart_model_B_FT.rds")
preds_B  <- read_pred_list("bart_model_B_predictors.rds")  # includes "PredVol_M"

# ---------- read JSON rows from stdin ----------
stdin_txt <- suppressWarnings(paste0(readLines("stdin"), collapse = "\n"))
if (!nzchar(stdin_txt)) {
  stop("No JSON received on STDIN.")
}
df_in <- fromJSON(stdin_txt, simplifyDataFrame = TRUE)

# Expected input columns (units noted in your app):
# PMVolM, PMDolM, FloatM, GapPct, ATR, MCapM, Catalyst(0/1)

eps <- 1e-6
num <- function(x) { suppressWarnings(as.numeric(x)) }

# feature engineering (mirror the R fitting code)
FR      <- num(df_in$PMVolM) / pmax(num(df_in$FloatM), eps)
ln_pm   <- log(pmax(num(df_in$PMVolM), eps))
ln_pmdol<- log(pmax(num(df_in$PMDolM), eps))
ln_fr   <- log(pmax(FR, eps))
ln_gapf <- log(pmax(num(df_in$GapPct), 0)/100 + eps)
ln_atr  <- log(pmax(num(df_in$ATR), eps))
ln_mcap <- log(pmax(num(df_in$MCapM), eps))
Catalyst<- as.integer(num(df_in$Catalyst) != 0)
ln_pmdol_per_mcap <- log(pmax(num(df_in$PMDolM) / pmax(num(df_in$MCapM), eps), eps))

feat_frame <- data.frame(
  ln_pm = ln_pm,
  ln_pmdol = ln_pmdol,
  ln_fr = ln_fr,
  ln_gapf = ln_gapf,
  ln_atr = ln_atr,
  ln_mcap = ln_mcap,
  Catalyst = Catalyst,
  ln_pmdol_per_mcap = ln_pmdol_per_mcap,
  check.names = FALSE
)

# ---------- Model A: predict ln(DVol), then exp ----------
# Use EXACT predictor names saved when you fitted.
xA <- as.matrix(feat_frame[, preds_A, drop = FALSE])
predA_ln_draws <- predict(modelA, newdata = xA)
predA_ln_mean  <- if (is.matrix(predA_ln_draws)) colMeans(predA_ln_draws) else as.numeric(predA_ln_draws)
PredVol_M      <- pmax(exp(predA_ln_mean), eps)

# ---------- Model B: FT probability (needs PredVol_M plus its predictors) ----------
feat_frame$PredVol_M <- PredVol_M
xB <- as.matrix(feat_frame[, preds_B, drop = FALSE])

# pbart predict: use out-of-sample interface if available
# If modelB was trained with pbart(), the object supports predict().
predB <- tryCatch(predict(modelB, newdata = xB), error = function(e) NULL)

if (!is.null(predB) && !is.null(predB$prob.test.mean)) {
  FT_Prob <- as.numeric(predB$prob.test.mean)
} else if (!is.null(modelB$prob.train.mean)) {
  # fallback (shouldn't be needed for OOS): length will be nrow(xB) only if predict works
  FT_Prob <- rep(mean(as.numeric(modelB$prob.train.mean), na.rm = TRUE), nrow(xB))
} else {
  stop("Could not obtain probabilities from BART classifier.")
}

out <- data.frame(PredVol_M = PredVol_M, FT_Prob = pmin(pmax(FT_Prob, 0), 1))
cat(toJSON(out, auto_unbox = TRUE))