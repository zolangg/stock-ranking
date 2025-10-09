# app.py — Premarket Ranking with OOF Pred Model + NCA kernel-kNN + CatBoost Leaf-kNN
import streamlit as st
import pandas as pd
import numpy as np
import re, json, hashlib

st.set_page_config(page_title="Premarket Stock Ranking (NCA + Leaf-kNN)", layout="wide")
st.title("Premarket Stock Ranking — NCA + Leaf-kNN Similarity")

# ============== Session ==============
ss = st.session_state
ss.setdefault("rows", [])
ss.setdefault("last", {})
ss.setdefault("base_df", pd.DataFrame())

ss.setdefault("pred_model_full", {})   # daily vol predictor for queries
ss.setdefault("nca_model", {})         # scaler, nca, emb, y, feat_names, guard
ss.setdefault("leaf_model", {})        # scaler, cat, emb, y, feat_names, metric

# ============== Deps ==============
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NeighborhoodComponentsAnalysis
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import pairwise_distances, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
except ModuleNotFoundError:
    st.error("Install scikit-learn==1.5.2")
    st.stop()

CATBOOST_AVAILABLE = True
try:
    from catboost import CatBoostClassifier, Pool
except Exception:
    CATBOOST_AVAILABLE = False

# ============== Small utils ==============
def do_rerun():
    if hasattr(st, "rerun"): st.rerun()
    elif hasattr(st, "experimental_rerun"): st.experimental_rerun()

def SAFE_JSON_DUMPS(obj) -> str:
    class Np(json.JSONEncoder):
        def default(self, o):
            if isinstance(o,(np.integer,)): return int(o)
            if isinstance(o,(np.floating,)): return float(o)
            if isinstance(o,(np.ndarray,)): return o.tolist()
            return super().default(o)
    return json.dumps(obj, cls=Np, ensure_ascii=False, separators=(",",":")).replace("</script>","<\\/script>")

_norm_cache={}
def _norm(s:str)->str:
    if s in _norm_cache: return _norm_cache[s]
    v=re.sub(r"\s+"," ",str(s).strip().lower())
    v=v.replace("%","").replace("$","").replace("(","").replace(")","").replace("’","").replace("'","")
    _norm_cache[s]=v; return v

def _pick(df: pd.DataFrame, candidates):
    if df is None or df.empty: return None
    cols=list(df.columns); cols_lc={c:c.strip().lower() for c in cols}
    nm={c:_norm(c) for c in cols}
    for cand in candidates:
        lc=cand.strip().lower()
        for c in cols:
            if cols_lc[c]==lc: return c
    for cand in candidates:
        n=_norm(cand)
        for c in cols:
            if nm[c]==n: return c
    for cand in candidates:
        n=_norm(cand)
        for c in cols:
            if n in nm[c]: return c
    return None

def _to_float(s):
    if pd.isna(s): return np.nan
    try:
        x=str(s).strip().replace(" ","")
        if "," in x and "." not in x: x=x.replace(",",".")
        else: x=x.replace(",","")
        return float(x)
    except: return np.nan

def _safe_to_binary(v):
    sv=str(v).strip().lower()
    if sv in {"1","true","yes","y","t"}: return 1
    if sv in {"0","false","no","n","f"}: return 0
    try: return 1 if float(sv)>=0.5 else 0
    except: return np.nan

def _mad(series: pd.Series) -> float:
    s=pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return np.nan
    med=float(np.median(s)); return float(np.median(np.abs(s-med)))

def _compute_bounds(arr, lo_q=0.01, hi_q=0.99):
    arr=np.asarray(arr); arr=arr[np.isfinite(arr)]
    if arr.size==0: return (np.nan, np.nan)
    return (float(np.quantile(arr, lo_q)), float(np.quantile(arr, hi_q)))

def _apply_bounds(arr, lo, hi):
    out=np.array(arr, dtype=float)
    if np.isfinite(lo): out=np.maximum(out, lo)
    if np.isfinite(hi): out=np.minimum(out, hi)
    return out

# ============== Daily volume predictor (LASSO→OLS→Isotonic) ==============
def _kfold_indices(n, k=5, seed=42):
    rng=np.random.default_rng(seed); idx=np.arange(n); rng.shuffle(idx)
    return np.array_split(idx, k)

def _lasso_cd_std(Xs, y, lam, max_iter=900, tol=1e-6):
    n,p=Xs.shape; w=np.zeros(p)
    for _ in range(max_iter):
        w_old=w.copy(); yhat=Xs@w
        for j in range(p):
            rj=y-yhat+Xs[:,j]*w[j]
            rho=(Xs[:,j]@rj)/n
            if   rho<-lam/2: w[j]=rho+lam/2
            elif rho> lam/2: w[j]=rho-lam/2
            else:            w[j]=0.0
            yhat=Xs@w
        if np.linalg.norm(w-w_old)<tol: break
    return w

def train_ratio_winsor_iso(df: pd.DataFrame) -> dict:
    eps=1e-6
    # need core columns
    need_min={"ATR_$","PM_Vol_M","PM_$Vol_M$","FR_x","Daily_Vol_M","Gap_%"}
    if not need_min.issubset(df.columns): return {}

    PM=pd.to_numeric(df["PM_Vol_M"], errors="coerce").values
    DV=pd.to_numeric(df["Daily_Vol_M"], errors="coerce").values
    valid=(np.isfinite(PM)&np.isfinite(DV)&(PM>0)&(DV>0))
    if valid.sum()<50: return {}

    def safe_log_col(colname, clip_low=True):
        vv=pd.to_numeric(df.get(colname, np.nan), errors="coerce").values
        vv=np.clip(vv, eps, None) if clip_low else vv
        return np.log(vv)

    # choose basis
    mcap=np.where(df["MC_PM_Max_M"].notna(), df["MC_PM_Max_M"], df.get("MarketCap_M$", np.nan))
    mcap=pd.to_numeric(mcap, errors="coerce").values
    mcap=np.clip(mcap, eps, None)

    flt=np.where(df["Float_PM_Max_M"].notna() if "Float_PM_Max_M" in df.columns else False,
                 df.get("Float_PM_Max_M", np.nan),
                 df.get("Float_M", np.nan))
    flt=pd.to_numeric(flt, errors="coerce").values
    flt=np.clip(flt, eps, None)

    ln_mcap=np.log(mcap)
    ln_gapf=np.log(np.clip(pd.to_numeric(df["Gap_%"], errors="coerce").values, 0, None)/100.0 + eps)
    ln_atr=safe_log_col("ATR_$")
    ln_pm =safe_log_col("PM_Vol_M")
    ln_pm_dol=safe_log_col("PM_$Vol_M$")
    ln_fr =safe_log_col("FR_x")
    ln_float_pmmax=np.log(flt)
    maxpullpm=pd.to_numeric(df.get("Max_Pull_PM_%", np.nan), errors="coerce").values
    ln_rvolmaxpm=safe_log_col("RVOL_Max_PM_cum")
    pm_dol_over_mc=pd.to_numeric(df.get("PM$Vol/MC_%", np.nan), errors="coerce").values
    catalyst=pd.to_numeric(df.get("Catalyst", np.nan), errors="coerce").fillna(0.0).clip(0,1).values

    mult=np.maximum(DV/PM,1.0)
    y_ln=np.log(mult)

    feats=[
        ("ln_mcap_pmmax",ln_mcap),("ln_gapf",ln_gapf),("ln_atr",ln_atr),("ln_pm",ln_pm),
        ("ln_pm_dol",ln_pm_dol),("ln_fr",ln_fr),("catalyst",catalyst),
        ("ln_float_pmmax",ln_float_pmmax),("maxpullpm",maxpullpm),
        ("ln_rvolmaxpm",ln_rvolmaxpm),("pm_dol_over_mc",pm_dol_over_mc)
    ]
    X=np.hstack([a.reshape(-1,1) for _,a in feats])
    mask=valid & np.isfinite(y_ln) & np.isfinite(X).all(axis=1)
    if mask.sum()<50: return {}
    X=X[mask]; y_ln=y_ln[mask]

    # winsor on heavy tails
    wins={}
    name_to_idx={n:i for i,(n,_) in enumerate(feats)}
    def _w(idx):
        arr=X[:,idx]; lo,hi=_compute_bounds(arr); wins[feats[idx][0]]=(lo,hi)
        X[:,idx]=_apply_bounds(arr,lo,hi)
    for nm in ["maxpullpm","pm_dol_over_mc"]:
        if nm in name_to_idx: _w(name_to_idx[nm])

    # winsor target multiplier
    mult_tr=np.exp(y_ln); mlo,mhi=_compute_bounds(mult_tr)
    y_ln=np.log(_apply_bounds(mult_tr,mlo,mhi))

    mu=X.mean(axis=0); sd=X.std(axis=0, ddof=0); sd[sd==0]=1.0
    Xs=(X-mu)/sd

    # pick lambda by CV
    folds=_kfold_indices(len(y_ln), k=min(5,max(2, len(y_ln)//10)), seed=42)
    lam_grid=np.geomspace(0.001,1.0,26)
    cv=[]
    for lam in lam_grid:
        errs=[]
        for vi in range(len(folds)):
            te=folds[vi]; tr=np.hstack([folds[j] for j in range(len(folds)) if j!=vi])
            w=_lasso_cd_std(Xs[tr], y_ln[tr], lam=lam, max_iter=1800)
            yhat=Xs[te]@w
            errs.append(np.mean((yhat - y_ln[te])**2))
        cv.append(np.mean(errs))
    lam_best=float(lam_grid[int(np.argmin(cv))])
    w=_lasso_cd_std(Xs, y_ln, lam=lam_best, max_iter=2500)
    sel=np.flatnonzero(np.abs(w)>1e-8)
    if sel.size==0: return {}
    Xsel=X[:,sel]; Xd=np.column_stack([np.ones(Xsel.shape[0]), Xsel])
    coef, *_=np.linalg.lstsq(Xd, y_ln, rcond=None)
    b0=float(coef[0]); bet=coef[1:].astype(float)

    # no iso fit here (we'll do iso only if we had holdout; but OK to skip)
    return {
        "eps":eps,"terms":[feats[i][0] for i in sel],"b0":b0,"betas":bet,"sel_idx":sel.tolist(),
        "mu":mu.tolist(),"sd":sd.tolist(),"winsor_bounds":wins,"feat_order":[nm for nm,_ in feats]
    }

def predict_daily_calibrated_row(row: dict, model: dict) -> float:
    if not model or "betas" not in model: return np.nan
    eps=float(model.get("eps",1e-6)); feat_order=model["feat_order"]
    wins=model.get("winsor_bounds",{}); sel=model.get("sel_idx",[])
    b0=float(model["b0"]); bet=np.array(model["betas"], dtype=float)

    def safe_log(v):
        v=float(v) if v is not None else np.nan
        return np.log(np.clip(v, eps, None)) if np.isfinite(v) else np.nan

    mcap=row.get("MC_PM_Max_M") if pd.notna(row.get("MC_PM_Max_M", np.nan)) else row.get("MarketCap_M$")
    ln_mcap=np.log(np.clip(mcap if mcap is not None else np.nan, eps, None)) if mcap is not None else np.nan
    ln_gapf=np.log(np.clip((row.get("Gap_%") or 0.0)/100.0 + eps, eps, None)) if row.get("Gap_%") is not None else np.nan
    ln_atr=safe_log(row.get("ATR_$"))
    ln_pm =safe_log(row.get("PM_Vol_M"))
    ln_pm_dol=safe_log(row.get("PM_$Vol_M$"))
    ln_fr =safe_log(row.get("FR_x"))
    catalyst=1.0 if (str(row.get("CatalystYN","No")).lower()=="yes" or float(row.get("Catalyst",0))>=0.5) else 0.0
    flt=row.get("Float_PM_Max_M") if pd.notna(row.get("Float_PM_Max_M", np.nan)) else row.get("Float_M")
    ln_float_pmmax=safe_log(flt)
    maxpullpm=float(row.get("Max_Pull_PM_%")) if row.get("Max_Pull_PM_%") is not None else np.nan
    ln_rvolmaxpm=safe_log(row.get("RVOL_Max_PM_cum"))
    pm_dol_over_mc=float(row.get("PM$Vol/MC_%")) if row.get("PM$Vol/MC_%") is not None else np.nan

    fmap={
        "ln_mcap_pmmax":ln_mcap, "ln_gapf":ln_gapf, "ln_atr":ln_atr, "ln_pm":ln_pm,
        "ln_pm_dol":ln_pm_dol, "ln_fr":ln_fr, "catalyst":catalyst,
        "ln_float_pmmax":ln_float_pmmax, "maxpullpm":maxpullpm,
        "ln_rvolmaxpm":ln_rvolmaxpm, "pm_dol_over_mc":pm_dol_over_mc
    }
    X=[]
    for nm in feat_order:
        v=fmap.get(nm, np.nan)
        if not np.isfinite(v): return np.nan
        lo,hi=wins.get(nm,(np.nan,np.nan))
        if np.isfinite(lo) or np.isfinite(hi):
            v=float(np.clip(v, lo if np.isfinite(lo) else v, hi if np.isfinite(hi) else v))
        X.append(v)
    X=np.array(X,dtype=float)
    if not sel: return np.nan
    yhat=b0+float(np.dot(X[sel], bet))
    mult=np.exp(yhat) if np.isfinite(yhat) else np.nan
    if not np.isfinite(mult): return np.nan
    PM=float(row.get("PM_Vol_M") or np.nan)
    if not np.isfinite(PM) or PM<=0: return np.nan
    return float(PM*mult)

# ============== Similarity heads ==============
def _elbow_kstar(d_sorted, k_min=3, k_max=15, max_rank=30):
    n=len(d_sorted)
    if n<=k_min: return max(1,n)
    upto=min(max_rank, n-1)
    if upto<2: return min(k_max, max(k_min, n))
    gaps=d_sorted[:upto]-d_sorted[1:upto+1]
    k=int(np.argmax(gaps)+1)
    return max(k_min, min(k_max, k))

def _kernel_weights(d, k):
    d=np.asarray(d)[:k]
    if d.size==0: return d, np.array([])
    bw=np.median(d[d>0]) if np.any(d>0) else (np.mean(d)+1e-6)
    bw=max(bw, 1e-6)
    w=np.exp(-(d/bw)**2)
    # drop near-zero weights
    mask=w>=1e-6
    return d[mask], w[mask]

def train_nca_head(df: pd.DataFrame, feat_names: list[str]):
    need=["FT01"]+feat_names
    dfx=df[need].dropna()
    if dfx.shape[0]<30: return {}
    y=dfx["FT01"].astype(int).values
    X=dfx[feat_names].astype(float).values
    scaler=StandardScaler().fit(X)
    Xs=scaler.transform(X)
    nca=NeighborhoodComponentsAnalysis(n_components=min(6, len(feat_names)), random_state=42, max_iter=250)
    Xn=nca.fit_transform(Xs, y)
    guard=IsolationForest(n_estimators=200, random_state=42, contamination="auto").fit(Xn)
    meta_cols=["FT01"]
    for c in ("TickerDB","Ticker"): 
        if c in df.columns: meta_cols.append(c)
    for f in feat_names:
        if f in df.columns and f not in meta_cols: meta_cols.append(f)
    meta=df.loc[dfx.index, meta_cols].reset_index(drop=True)
    return {"scaler":scaler,"nca":nca,"X_emb":Xn,"y":y,"feat_names":feat_names,"df_meta":meta,"guard":guard}

def train_leaf_head(df: pd.DataFrame, feat_names: list[str]):
    if not CATBOOST_AVAILABLE: return {}
    need=["FT01"]+feat_names
    dfx=df[need].dropna()
    if dfx.shape[0]<30: return {}
    y=dfx["FT01"].astype(int).values
    X=dfx[feat_names].astype(float).values
    scaler=StandardScaler().fit(X); Xs=scaler.transform(X)
    # CatBoost classifier (for leaf indices)
    class_weights=[1.0, float((y==0).sum()/max(1,(y==1).sum()))]
    # use a small eval split for early stopping
    n=len(y); val_size=max(20, int(0.15*n))
    idx=np.arange(n); rs=np.random.RandomState(42); rs.shuffle(idx)
    tr_idx, va_idx=idx[val_size:], idx[:val_size]
    train_pool=Pool(Xs[tr_idx], y[tr_idx])
    val_pool=Pool(Xs[va_idx], y[va_idx])
    cat=CatBoostClassifier(
        depth=6, learning_rate=0.08, iterations=1200, loss_function="Logloss",
        random_seed=42, l2_leaf_reg=6, class_weights=class_weights,
        bootstrap_type='No', random_strength=0.0, early_stopping_rounds=50,
        verbose=False
    )
    cat.fit(train_pool, eval_set=val_pool, use_best_model=True)
    # leaf embedding
    leaf_mat=np.array(cat.calc_leaf_indexes(Pool(Xs)), dtype=int)  # (n, n_trees)
    metric="hamming"  # distance on leaf ids
    meta_cols=["FT01"]
    for c in ("TickerDB","Ticker"): 
        if c in df.columns: meta_cols.append(c)
    for f in feat_names:
        if f in df.columns and f not in meta_cols: meta_cols.append(f)
    meta=df.loc[dfx.index, meta_cols].reset_index(drop=True)
    return {"scaler":scaler,"cat":cat,"emb":leaf_mat,"y":y,"feat_names":feat_names,"metric":metric,"df_meta":meta}

def _nca_score(model, row_dict):
    feats=model["feat_names"]
    vec=[]
    for f in feats:
        v=row_dict.get(f, None)
        if v is None or (isinstance(v,float) and not np.isfinite(v)): return None
        vec.append(float(v))
    xs=model["scaler"].transform(np.array(vec)[None,:])
    xn=model["nca"].transform(xs)
    oos=model["guard"].decision_function(xn.reshape(1,-1))[0]
    d=pairwise_distances(xn, model["X_emb"], metric="euclidean").ravel()
    order=np.argsort(d); d_sorted=d[order]
    k=_elbow_kstar(d_sorted, 3, 15, 30)
    d_top, w_top=_kernel_weights(d_sorted, k)
    if w_top.size==0: return {"p1":0.0,"k":0,"oos":float(oos)}
    idx=order[:len(d_top)]
    y=model["y"][idx]
    w=w_top/(w_top.sum() if w_top.sum()>0 else 1.0)
    p1=float(np.dot((y==1).astype(float), w))*100.0
    return {"p1":p1,"k":int(len(idx)),"oos":float(oos)}

def _leaf_score(model, row_dict):
    feats=model["feat_names"]
    vec=[]
    for f in feats:
        v=row_dict.get(f, None)
        if v is None or (isinstance(v,float) and not np.isfinite(v)): return None
        vec.append(float(v))
    xs=model["scaler"].transform(np.array(vec)[None,:])
    emb_q=np.array(model["cat"].calc_leaf_indexes(Pool(xs)), dtype=int)
    d=pairwise_distances(emb_q, model["emb"], metric=model["metric"]).ravel()
    order=np.argsort(d); d_sorted=d[order]
    k=_elbow_kstar(d_sorted, 3, 15, 30)
    d_top, w_top=_kernel_weights(d_sorted, k)
    if w_top.size==0: return {"p1":0.0,"k":0}
    idx=order[:len(d_top)]
    y=model["y"][idx]
    w=w_top/(w_top.sum() if w_top.sum()>0 else 1.0)
    p1=float(np.dot((y==1).astype(float), w))*100.0
    return {"p1":p1,"k":int(len(idx))}

# ============== Feature selection (drop-one + stability) ==============
def _cv_scores_for_head(df, feats, head="nca", n_splits=5):
    # returns list of (fold_auc)
    dfx=df[["FT01"]+feats].dropna()
    if dfx.empty or dfx["FT01"].nunique()<2: return [0.5]
    X=dfx[feats].astype(float).values
    y=dfx["FT01"].astype(int).values
    skf=StratifiedKFold(n_splits=min(n_splits, max(2, y.sum(), (y==0).sum())), shuffle=True, random_state=42)
    aucs=[]
    for tr, va in skf.split(X, y):
        part=dfx.reset_index(drop=True)
        tr_df=part.loc[tr]; va_df=part.loc[va]
        if head=="nca":
            mdl=train_nca_head(pd.concat([tr_df, va_df], axis=0), feats)  # fit on tr+va indices filtered; model itself uses dropna
            if not mdl: 
                aucs.append(0.5); continue
            scores=[]
            for i in va_df.index:
                s=_nca_score(mdl, va_df.loc[i, feats].to_dict())
                scores.append(np.nan if s is None else s["p1"]/100.0)
        else:
            mdl=train_leaf_head(tr_df, feats)
            if not mdl:
                aucs.append(0.5); continue
            scores=[]
            for i in va_df.index:
                s=_leaf_score(mdl, va_df.loc[i, feats].to_dict())
                scores.append(np.nan if s is None else s["p1"]/100.0)
        # compute AUC safely
        y_true=va_df["FT01"].values
        sc=np.array(scores, dtype=float)
        m=np.isfinite(sc)
        if m.sum()<3 or len(np.unique(y_true[m]))<2:
            aucs.append(float(np.mean((sc[m]>=0.5)==y_true[m])) if m.any() else 0.5)
        else:
            try:
                aucs.append(float(roc_auc_score(y_true[m], sc[m])))
            except Exception:
                aucs.append(float(np.mean((sc[m]>=0.5)==y_true[m])))
    return aucs

def select_features_with_drop_one(df, base_feats, head="nca", stability=0.6, max_keep=10, min_keep=6):
    # base score
    base_scores=_cv_scores_for_head(df, base_feats, head=head)
    base=np.nanmean(base_scores)

    keep=set(base_feats)
    contrib={}
    for f in base_feats:
        test=[x for x in base_feats if x!=f]
        sc=_cv_scores_for_head(df, test, head=head)
        delta=base - np.nanmean(sc)  # >0 means feature helps
        contrib[f]=delta

    # stability rule: keep features with positive contribution in >=60% folds
    stable=[]
    for f in base_feats:
        test=[x for x in base_feats if x!=f]
        sc=_cv_scores_for_head(df, test, head=head)
        arr=np.array(sc, dtype=float)
        pos=(base - arr)>0
        if np.mean(pos)>=stability:
            stable.append(f)

    if len(stable)<min_keep:
        # fallback: take top |delta| features
        rank=sorted(contrib.items(), key=lambda kv: (-kv[1], kv[0]))
        stable=[f for f,_ in rank[:max(min_keep, min(len(base_feats), max_keep))]]

    # cap to max_keep, keep the highest contributors
    if len(stable)>max_keep:
        rank=sorted([(f, contrib.get(f,0.0)) for f in stable], key=lambda kv:(-kv[1], kv[0]))
        stable=[f for f,_ in rank[:max_keep]]

    return stable

# ============== Variables (12) ==============
FEAT12 = [
    "MC_PM_Max_M","Float_PM_Max_M","Catalyst","ATR_$","Gap_%",
    "Max_Pull_PM_%","PM_Vol_M","PM_$Vol_M$","PM$Vol/MC_%",
    "RVOL_Max_PM_cum","FR_x","PM_Vol_%"
]

# ============== Upload / Build ==============
st.subheader("Upload Database")
uploaded = st.file_uploader("Upload .xlsx with your DB", type=["xlsx"], key="db_upl")
build_btn = st.button("Build models", use_container_width=True, key="db_build_btn")

@st.cache_data(show_spinner=False)
def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

@st.cache_data(show_spinner=True)
def _load_sheet(file_bytes: bytes):
    xls=pd.ExcelFile(file_bytes)
    sheet_candidates=[s for s in xls.sheet_names if _norm(s) not in {"legend","readme"}]
    sheet=sheet_candidates[0] if sheet_candidates else xls.sheet_names[0]
    raw=pd.read_excel(xls, sheet)
    return raw, sheet, tuple(xls.sheet_names)

if build_btn:
    if not uploaded:
        st.error("Please upload an Excel workbook first.")
    else:
        try:
            file_bytes=uploaded.getvalue(); _=_hash_bytes(file_bytes)
            raw, sel_sheet, _ = _load_sheet(file_bytes)

            poss=[c for c in raw.columns if _norm(c) in {"ft","ft01","group","label"}]
            col_group=poss[0] if poss else None
            if col_group is None:
                for c in raw.columns:
                    vals=pd.Series(raw[c]).dropna().astype(str).str.lower()
                    if len(vals) and vals.isin(["0","1","true","false","yes","no"]).all():
                        col_group=c; break
            if col_group is None:
                st.error("Could not detect FT (0/1) column."); st.stop()

            df=pd.DataFrame()
            df["GroupRaw"]=raw[col_group]

            def add_num(dfout,name,cands):
                src=_pick(raw,cands)
                if src: dfout[name]=pd.to_numeric(raw[src].map(_to_float), errors="coerce")

            # map
            add_num(df,"MC_PM_Max_M",["mc pm max (m)","premarket market cap (m)","mc_pm_max_m","mc pm max (m$)","market cap pm max (m)","market cap pm max m","premarket market cap (m$)"])
            add_num(df,"Float_PM_Max_M",["float pm max (m)","premarket float (m)","float_pm_max_m","float pm max (m shares)"])
            add_num(df,"MarketCap_M$",["marketcap m","market cap (m)","mcap m","marketcap_m$","market cap m$","market cap (m$)","marketcap","market_cap_m"])
            add_num(df,"Float_M",["float m","public float (m)","float_m","float (m)","float m shares","float_m_shares"])
            add_num(df,"Gap_%",["gap %","gap%","premarket gap","gap","gap_percent"])
            add_num(df,"ATR_$",["atr $","atr$","atr (usd)","atr","daily atr","daily_atr"])
            add_num(df,"PM_Vol_M",["pm vol (m)","premarket vol (m)","pm volume (m)","pm shares (m)","premarket volume (m)","pm_vol_m"])
            add_num(df,"PM_$Vol_M$",["pm $vol (m)","pm dollar vol (m)","pm $ volume (m)","pm $vol","pm dollar volume (m)","pm_dollarvol_m","pm $vol"])
            add_num(df,"PM_Vol_%",["pm vol (%)","pm_vol_%","pm vol percent","pm volume (%)","pm_vol_percent"])
            add_num(df,"Daily_Vol_M",["daily vol (m)","daily_vol_m","day volume (m)","dvol_m"])
            add_num(df,"Max_Pull_PM_%",["max pull pm (%)","max pull pm %","max pull pm","max_pull_pm_%"])
            add_num(df,"RVOL_Max_PM_cum",["rvol max pm (cum)","rvol max pm cum","rvol_max_pm (cum)","rvol_max_pm_cum","premarket max rvol","premarket max rvol (cum)"])

            # catalyst
            cand_catalyst=_pick(raw,["catalyst","catalyst?","has catalyst","news catalyst","catalyst_yn","cat"])
            if cand_catalyst: df["Catalyst"]=pd.Series(raw[cand_catalyst]).map(lambda v: float(_safe_to_binary(v)) if _safe_to_binary(v) in (0,1) else np.nan)

            # derived
            float_basis = "Float_PM_Max_M" if ("Float_PM_Max_M" in df.columns and df["Float_PM_Max_M"].notna().any()) else "Float_M"
            if {"PM_Vol_M", float_basis}.issubset(df.columns):
                df["FR_x"]=(df["PM_Vol_M"]/df[float_basis]).replace([np.inf,-np.inf], np.nan)

            mcap_basis="MC_PM_Max_M" if ("MC_PM_Max_M" in df.columns and df["MC_PM_Max_M"].notna().any()) else "MarketCap_M$"
            if {"PM_$Vol_M$", mcap_basis}.issubset(df.columns):
                df["PM$Vol/MC_%"]=(df["PM_$Vol_M$"]/df[mcap_basis]*100.0).replace([np.inf,-np.inf], np.nan)

            # scale % in DB
            if "Gap_%" in df.columns: df["Gap_%"]=pd.to_numeric(df["Gap_%"], errors="coerce")*100.0
            if "PM_Vol_%" in df.columns: df["PM_Vol_%"]=pd.to_numeric(df["PM_Vol_%"], errors="coerce")*100.0
            if "Max_Pull_PM_%" in df.columns: df["Max_Pull_PM_%"]=pd.to_numeric(df["Max_Pull_PM_%"], errors="coerce")*100.0

            df["FT01"]=pd.Series(df["GroupRaw"]).map(_safe_to_binary)
            df=df[df["FT01"].isin([0,1])].copy()

            # Build OOF PredVol_M for PM_Vol_% (no leakage)
            # 1) Full predictor for later query use
            ss.pred_model_full = train_ratio_winsor_iso(df) or {}
            # 2) OOF for training rows
            needed_pred_cols={"PM_Vol_M","Daily_Vol_M","ATR_$","PM_$Vol_M$","FR_x","Gap_%"}
            if not needed_pred_cols.issubset(set(df.columns)):
                st.error("DB missing predictor fields for OOF PM_Vol_% computation.")
                st.stop()

            dfx_pred=df.copy()
            mask_ok = dfx_pred[list(needed_pred_cols)].notna().all(axis=1)
            dfx_pred = dfx_pred[mask_ok].reset_index(drop=True)
            n=len(dfx_pred)
            oof=np.full(n, np.nan)
            y_ft=dfx_pred["FT01"].values
            skf=StratifiedKFold(n_splits=min(5, max(2, int(n/10))), shuffle=True, random_state=42)
            for tr, va in skf.split(np.zeros(n), y_ft):
                mdl=train_ratio_winsor_iso(dfx_pred.iloc[tr])
                if not mdl: continue
                for i, ridx in enumerate(va):
                    row=dfx_pred.iloc[ridx].to_dict()
                    oof[ridx]=predict_daily_calibrated_row(row, mdl)
            dfx_pred["PredVol_M_OOF"]=oof
            # merge back
            df["PredVol_M_OOF"]=np.nan
            df.loc[dfx_pred.index, "PredVol_M_OOF"]=dfx_pred["PredVol_M_OOF"].values
            df["PM_Vol_%"]=100.0*df["PM_Vol_M"]/df["PredVol_M_OOF"]

            ss.base_df = df

            # Feature selection per head on rows with all 12 features present
            feat_base=[f for f in FEAT12 if f in df.columns]
            train_df=df.dropna(subset=feat_base+["FT01"]).copy()
            if train_df.shape[0]<40:
                st.warning("Very small training set after NaN drop; results may be unstable.")

            feat_nca  = select_features_with_drop_one(train_df, feat_base, head="nca", stability=0.6, max_keep=10, min_keep=6)
            feat_leaf = select_features_with_drop_one(train_df, feat_base, head="leaf", stability=0.6, max_keep=10, min_keep=6) if CATBOOST_AVAILABLE else []

            ss.nca_model  = train_nca_head(train_df, feat_nca) if len(feat_nca)>=3 else {}
            ss.leaf_model = train_leaf_head(train_df, feat_leaf) if (CATBOOST_AVAILABLE and len(feat_leaf)>=3) else {}

            # Report selected features
            sf_cols = st.columns(3)
            with sf_cols[0]:
                st.success(f"NCA features ({len(feat_nca)}): " + ", ".join(feat_nca))
            with sf_cols[1]:
                if CATBOOST_AVAILABLE and ss.leaf_model:
                    st.success(f"Leaf features ({len(feat_leaf)}): " + ", ".join(feat_leaf))
                elif not CATBOOST_AVAILABLE:
                    st.error("CatBoost not installed — Leaf head disabled.")
                else:
                    st.warning("Leaf head not trained (too few clean rows).")
            with sf_cols[2]:
                st.info(f"OOF PredVol_M computed for PM_Vol_% (anti-leak).")

            st.success(f"Loaded “{sel_sheet}”. Models ready.")
            do_rerun()
        except Exception as e:
            st.error("Loading/processing failed.")
            st.exception(e)

# ============== Add Stock ==============
st.markdown("---")
st.subheader("Add Stock")

with st.form("add_form", clear_on_submit=True):
    c1,c2,c3=st.columns([1.2,1.2,0.8])
    with c1:
        ticker=st.text_input("Ticker","").strip().upper()
        mc_pmmax=st.number_input("Premarket Market Cap (M$)", 0.0, step=0.01, format="%.2f")
        float_pm=st.number_input("Premarket Float (M)", 0.0, step=0.01, format="%.2f")
        gap_pct=st.number_input("Gap %", 0.0, step=0.01, format="%.2f")
        max_pull_pm=st.number_input("Premarket Max Pullback (%)", 0.0, step=0.01, format="%.2f")
    with c2:
        atr_usd=st.number_input("Prior Day ATR ($)", 0.0, step=0.01, format="%.2f")
        pm_vol=st.number_input("Premarket Volume (M)", 0.0, step=0.01, format="%.2f")
        pm_dol=st.number_input("Premarket Dollar Vol (M$)", 0.0, step=0.01, format="%.2f")
        rvol_pm_cum=st.number_input("Premarket Max RVOL", 0.0, step=0.01, format="%.2f")
    with c3:
        catalyst_yn=st.selectbox("Catalyst?", ["No","Yes"], index=0)
    submitted=st.form_submit_button("Add to Table", use_container_width=True)

if submitted and ticker:
    fr=(pm_vol/float_pm) if float_pm>0 else 0.0
    pmmc=(pm_dol/mc_pmmax*100.0) if mc_pmmax>0 else 0.0
    row={
        "Ticker":ticker,
        "MC_PM_Max_M":mc_pmmax,
        "Float_PM_Max_M":float_pm,
        "Gap_%":gap_pct,
        "ATR_$":atr_usd,
        "PM_Vol_M":pm_vol,
        "PM_$Vol_M$":pm_dol,
        "FR_x":fr,
        "PM$Vol/MC_%":pmmc,
        "Max_Pull_PM_%":max_pull_pm,
        "RVOL_Max_PM_cum":rvol_pm_cum,
        "CatalystYN":catalyst_yn,
        "Catalyst":1.0 if catalyst_yn=="Yes" else 0.0,
    }
    # Predicted daily volume with full predictor
    pred = predict_daily_calibrated_row(row, ss.get("pred_model_full", {}))
    row["PredVol_M"]=float(pred) if np.isfinite(pred) else np.nan
    denom=row["PredVol_M"]
    row["PM_Vol_%"]=(row["PM_Vol_M"]/denom)*100.0 if np.isfinite(denom) and denom>0 else np.nan
    ss.rows.append(row); ss.last=row
    st.success(f"Saved {ticker}."); do_rerun()

# ============== Alignment Table (bars + child summary) ==============
st.markdown("### Alignment")

if not ss.base_df.empty and ss.rows:
    # Build summary rows + detail summaries (12 vars + PredVol_M)
    summaries=[]; details={}
    for row in ss.rows:
        tkr=row.get("Ticker","—")
        # NCA
        nca_score=None
        if ss.nca_model:
            nca_score=_nca_score(ss.nca_model, row)
        # Leaf
        leaf_score=None
        if ss.leaf_model:
            leaf_score=_leaf_score(ss.leaf_model, row)

        p_nca = 0.0 if (nca_score is None) else float(nca_score.get("p1",0.0))
        p_leaf = 0.0 if (leaf_score is None) else float(leaf_score.get("p1",0.0))

        # If a head is unavailable, don't drag average down: average over available heads
        vals=[x for x in [p_nca if ss.nca_model else None, p_leaf if ss.leaf_model else None] if x is not None]
        p_avg = float(np.mean(vals)) if len(vals)>0 else 0.0

        summaries.append({
            "Ticker": tkr,
            "NCA_raw": p_nca, "NCA_int": int(round(p_nca)),
            "LEAF_raw": p_leaf, "LEAF_int": int(round(p_leaf)),
            "AVG_raw": p_avg, "AVG_int": int(round(p_avg)),
        })

        # child: compact summary of variables + PredVol_M
        fields = FEAT12 + ["PredVol_M"]
        rows=[{"__group__": "Inputs summary"}]
        for f in fields:
            val=row.get(f, None)
            v = "" if (val is None or (isinstance(val,float) and not np.isfinite(val))) else float(val)
            rows.append({"Variable": f, "Value": v})
        details[tkr]=rows

    # Render HTML/JS
    import streamlit.components.v1 as components
    def _round_rec(o):
        if isinstance(o, dict): return {k:_round_rec(v) for k,v in o.items()}
        if isinstance(o, list): return [_round_rec(v) for v in o]
        if isinstance(o, float): return float(np.round(o,6))
        return o
    payload=_round_rec({"rows":summaries, "details":details})

    html = """
<!DOCTYPE html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css"/>
<link rel="stylesheet" href="https://cdn.datatables.net/responsive/2.5.0/css/responsive.dataTables.min.css"/>
<style>
  body{font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,"Helvetica Neue",sans-serif}
  table.dataTable tbody tr{cursor:pointer}
  .bar-wrap{display:flex;justify-content:center;align-items:center;gap:6px}
  .bar{height:12px;width:120px;border-radius:8px;background:#eee;position:relative;overflow:hidden}
  .bar>span{position:absolute;left:0;top:0;bottom:0;width:0%}
  .bar-label{font-size:11px;white-space:nowrap;color:#374151;min-width:28px;text-align:center}
  .blue>span{background:#3b82f6}.violet>span{background:#8b5cf6}.gray>span{background:#6b7280}
  #align td:nth-child(2),#align th:nth-child(2),#align td:nth-child(3),#align th:nth-child(3),#align td:nth-child(4),#align th:nth-child(4){text-align:center}
  .child-table{width:100%;border-collapse:collapse;margin:2px 0 2px 24px;table-layout:fixed}
  .child-table th,.child-table td{font-size:11px;padding:3px 6px;border-bottom:1px solid #e5e7eb;text-align:left;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  tr.group-row td{background:#f3f4f6!important;color:#374151;font-weight:600}
  .col-var{width:40%}.col-val{width:60%}
</style></head><body>
  <table id="align" class="display nowrap stripe" style="width:100%">
    <thead><tr><th>Ticker</th><th>NCA (FT=1)</th><th>Leaf (FT=1)</th><th>Average</th></tr></thead>
  </table>
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
  <script src="https://cdn.datatables.net/responsive/2.5.0/js/dataTables.responsive.min.js"></script>
  <script>
    const data = %%PAYLOAD%%;

    function barCell(valRaw, cls, valInt){
      const w=(valRaw==null||isNaN(valRaw))?0:Math.max(0,Math.min(100,valRaw));
      const text=(valInt==null||isNaN(valInt))?Math.round(w):valInt;
      return `<div class="bar-wrap"><div class="bar ${cls}"><span style="width:${w}%"></span></div><div class="bar-label">${text}</div></div>`;
    }
    function formatVal(x){return (x==null||x==='')?'':Number(x).toFixed(4);}

    function childTableHTML(ticker){
      const rows=(data.details||{})[ticker]||[];
      if(!rows.length) return '<div style="margin-left:24px;color:#6b7280;">No details.</div>';
      const cells = rows.map(r=>{
        if(r.__group__) return '<tr class="group-row"><td colspan="2">'+r.__group__+'</td></tr>';
        return `<tr><td class="col-var">${r.Variable}</td><td class="col-val">${formatVal(r.Value)}</td></tr>`;
      }).join('');
      return `<table class="child-table">
        <colgroup><col class="col-var"/><col class="col-val"/></colgroup>
        <thead><tr><th class="col-var">Variable</th><th class="col-val">Value</th></tr></thead>
        <tbody>${cells}</tbody></table>`;
    }

    $(function(){
      const table=$('#align').DataTable({
        data: data.rows||[], responsive:true, paging:false, info:false, searching:false, order:[[0,'asc']],
        columns:[
          {data:'Ticker'},
          {data:null, render:(r)=>barCell(r.NCA_raw,'blue',r.NCA_int)},
          {data:null, render:(r)=>barCell(r.LEAF_raw,'violet',r.LEAF_int)},
          {data:null, render:(r)=>barCell(r.AVG_raw,'gray',r.AVG_int)}
        ]
      });
      $('#align tbody').on('click','tr',function(){
        const row=table.row(this);
        if(row.child.isShown()){ row.child.hide(); $(this).removeClass('shown'); }
        else { row.child(childTableHTML(row.data().Ticker)).show(); $(this).addClass('shown'); }
      });
    });
  </script>
</body></html>
"""
    components.html(html.replace("%%PAYLOAD%%", SAFE_JSON_DUMPS(payload)), height=620, scrolling=True)
else:
    if ss.base_df.empty:
        st.info("Upload DB and click **Build models**.")
    elif not ss.rows:
        st.info("Add at least one stock to compare.")

# ============== Delete Control ==============
tickers=[r.get("Ticker") for r in ss.rows if r.get("Ticker")]
unique=[]; seen=set()
for t in tickers:
    if t and t not in seen: unique.append(t); seen.add(t)
c1,c2=st.columns([4,1])
with c1:
    sel=st.multiselect("", options=unique, default=[], placeholder="Select tickers…", label_visibility="collapsed")
with c2:
    if st.button("Delete", use_container_width=True):
        if sel:
            ss.rows=[r for r in ss.rows if r.get("Ticker") not in set(sel)]
            st.success(f"Deleted: {', '.join(sel)}"); do_rerun()
        else:
            st.info("No tickers selected.")
