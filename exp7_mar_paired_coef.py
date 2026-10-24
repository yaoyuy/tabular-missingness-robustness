"""Exp7 (2026-10-23, reviewer-driven supplement):
1) MAR test-time missingness: missingness probability for every non-anchor
   feature depends ONLY on one observed anchor feature, which is never masked.
   p = sigmoid(-1.0 + direction * 1.2 * z_anchor), z_anchor standardised with
   TRAIN-fold mean/std of the anchor, clipped to [-3, 3]. Direction +1 = MAR+,
   -1 = MAR-. Anchor: WDBC mean radius (col 0), Wine alcohol (col 0),
   Digits centre pixel (col 28), Ionosphere col 0, Heart age (col 0).
   Seeds: MAR+ 3000+fold, MAR- 4000+fold (MCAR uses 1000+fold, so masks differ).
   This is MAR by construction: the mask depends on an observed value, never
   on a value that is itself masked in that run. It is NOT MNAR, NOT correlated
   missingness, and NOT training-time missingness; those remain unstudied.
2) Paired tests on auxiliary benchmarks: per-fold clean and 30% MCAR scores
   for all 4 models on all 5 benchmarks, with paired t, Wilcoxon, mean diff,
   SD of diff, Cohen dz and 95% CI (t, df=4). n=5, so CIs are wide and
   two-sided Wilcoxon can never reach p<0.05 with n=5 (minimum 0.0625);
   report that, do not hide it.
3) LR coefficient interpretation: full-data LR (standardised, C=1.0,
   max_iter=1000) on WDBC, plus 5-fold train-fold coefficients (mean +- SD,
   sign consistency). Descriptive only; coefficients are conditional
   associations under L2 regularisation, NOT causal effects, and correlated
   features share the effect.
Models, seeds, folds, imputation identical to exp1/exp3/exp5/exp6.
"""
import json, urllib.request, io, numpy as np, pandas as pd
from scipy import stats
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def mcar_mask(X, rate, seed):
    rng = np.random.RandomState(seed); Xm = X.copy().astype(float)
    Xm[rng.rand(*Xm.shape) < rate] = np.nan; return Xm

def mar_mask(Xte, Xtr, anchor_idx, direction, seed):
    rng = np.random.RandomState(seed)
    Xm = Xte.copy().astype(float)
    mu = Xtr[:, anchor_idx].mean(); sd = Xtr[:, anchor_idx].std() or 1.0
    z = np.clip((Xte[:, anchor_idx] - mu) / sd, -3, 3)
    p_row = sigmoid(-1.0 + direction * 1.2 * z)  # one p per row, applied to non-anchor cols
    mask = np.zeros_like(Xm, dtype=bool)
    for j in range(Xm.shape[1]):
        if j == anchor_idx:
            continue
        mask[:, j] = rng.rand(Xm.shape[0]) < p_row
    Xm[mask] = np.nan
    return Xm, float(mask.sum() / max(1, mask.shape[0] * (mask.shape[1] - 1)))

def make_models(binary, wdbc=False):
    lr = make_pipeline(StandardScaler(), SimpleImputer(strategy="median"), LogisticRegression(max_iter=1000, random_state=42))
    rf = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
    hgb = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(random_state=42))
    if wdbc:
        svm = make_pipeline(StandardScaler(), SimpleImputer(strategy="median"), SVC(C=1.0, probability=True, random_state=42))
    else:
        svm = make_pipeline(StandardScaler(), SimpleImputer(strategy="median"), SVC(C=1.0, random_state=42))
    return {"LR": lr, "RF": rf, "HGB": hgb, "SVM": svm}

def score(model, X, y, binary, wdbc=False, mname=""):
    if binary:
        # Only SVM (without probability=True) needs decision_function;
        # LR/RF/HGB all have predict_proba. The original code used
        # decision_function for every non-WDBC model and crashed on RF.
        if mname == "SVM" and not wdbc:
            return roc_auc_score(y, model.decision_function(X))
        return roc_auc_score(y, model.predict_proba(X)[:, 1])
    return accuracy_score(y, model.predict(X))

def load_all():
    out = {}
    d = load_breast_cancer(); out["WDBC"] = (d.data, 1 - d.target, True, 0, "mean radius", d.feature_names)
    w = load_wine(); out["Wine"] = (w.data, w.target, False, 0, "alcohol", w.feature_names)
    dg = load_digits(); out["Digits"] = (dg.data, dg.target, False, 28, "centre pixel (row 3, col 4)", None)
    try:
        with urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data", timeout=30) as r:
            df = pd.read_csv(io.StringIO(r.read().decode("utf-8", errors="ignore")), header=None)
        out["Ionosphere"] = (df.iloc[:, :-1].values.astype(float), LabelEncoder().fit_transform(df.iloc[:, -1].values), True, 0, "column 0", None)
    except Exception as e:
        print("Ionosphere FAILED", e)
    try:
        with urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", timeout=30) as r:
            df = pd.read_csv(io.StringIO(r.read().decode("utf-8", errors="ignore")), header=None)
        df = df.replace("?", np.nan).dropna()
        out["Heart"] = (df.iloc[:, :-1].values.astype(float), (df.iloc[:, -1].values.astype(int) > 0).astype(int), True, 0, "age", None)
    except Exception as e:
        print("Heart FAILED", e)
    return out

def paired(a, b):
    """a, b: per-fold arrays. Returns diff = a - b stats."""
    a = np.asarray(a, float); b = np.asarray(b, float); diff = a - b
    md = float(diff.mean()); sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    t, p_t = stats.ttest_rel(a, b) if sd > 0 else (0.0, 1.0)
    try:
        _, p_w = stats.wilcoxon(a, b)
    except Exception:
        p_w = float("nan")
    ci_half = 2.776 * sd / np.sqrt(len(diff)) if len(diff) > 1 else 0.0
    return {"mean_diff": round(md, 4), "sd_diff": round(sd, 4),
            "ci95": [round(md - ci_half, 4), round(md + ci_half, 4)],
            "cohen_dz": round(md / sd, 3) if sd > 0 else None,
            "t_p": round(float(p_t), 4) if p_t == p_t else None,
            "wilcoxon_p": round(float(p_w), 4) if p_w == p_w else None,
            "per_fold_diff": [round(float(x), 4) for x in diff]}

results = {"MAR": {}, "paired_aux": {}, "LR_coefficients": {}}
datasets = load_all()
for name, (X, y, binary, anchor, anchor_name, feat_names) in datasets.items():
    print(f"=== {name} shape={X.shape} anchor={anchor_name} (col {anchor}) ===", flush=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    per = {m: {"clean": [], "mcar30": [], "mar_plus": [], "mar_minus": []} for m in ["LR", "RF", "HGB", "SVM"]}
    mar_rates = {"plus": [], "minus": []}
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        wdbc = (name == "WDBC")
        models = make_models(binary, wdbc=wdbc)
        Xm_mcar = mcar_mask(Xte, 0.30, 1000 + fold)
        Xm_plus, rate_plus = mar_mask(Xte, Xtr, anchor, +1, 3000 + fold)
        Xm_minus, rate_minus = mar_mask(Xte, Xtr, anchor, -1, 4000 + fold)
        mar_rates["plus"].append(rate_plus); mar_rates["minus"].append(rate_minus)
        for mname, model in models.items():
            model.fit(Xtr, ytr)
            per[mname]["clean"].append(score(model, Xte, yte, binary, wdbc, mname))
            per[mname]["mcar30"].append(score(model, Xm_mcar, yte, binary, wdbc, mname))
            per[mname]["mar_plus"].append(score(model, Xm_plus, yte, binary, wdbc, mname))
            per[mname]["mar_minus"].append(score(model, Xm_minus, yte, binary, wdbc, mname))
    results["MAR"][name] = {
        "anchor": anchor_name, "anchor_col": anchor,
        "achieved_masked_fraction_plus": round(float(np.mean(mar_rates["plus"])), 4),
        "achieved_masked_fraction_minus": round(float(np.mean(mar_rates["minus"])), 4),
        "per_fold": {m: {k: [round(float(x), 4) for x in v] for k, v in d.items()} for m, d in per.items()},
        "mean": {m: {k: f"{np.mean(v):.4f} +- {np.std(v):.4f}" for k, v in d.items()} for m, d in per.items()},
    }
    # paired tests for this dataset
    p = {}
    for mname in ["LR", "RF", "HGB", "SVM"]:
        p[f"{mname}_clean_minus_mcar30"] = paired(per[mname]["clean"], per[mname]["mcar30"])
        p[f"{mname}_mcar30_minus_mar_plus"] = paired(per[mname]["mcar30"], per[mname]["mar_plus"])
        p[f"{mname}_mcar30_minus_mar_minus"] = paired(per[mname]["mcar30"], per[mname]["mar_minus"])
    for other in ["RF", "HGB", "SVM"]:
        p[f"LR_minus_{other}_mcar30"] = paired(per["LR"]["mcar30"], per[other]["mcar30"])
        p[f"LR_minus_{other}_mar_plus"] = paired(per["LR"]["mar_plus"], per[other]["mar_plus"])
        p[f"LR_minus_{other}_clean"] = paired(per["LR"]["clean"], per[other]["clean"])
    results["paired_aux"][name] = p
    print(name, json.dumps(results["MAR"][name]["mean"], ensure_ascii=False), flush=True)

# LR coefficients on WDBC
Xw, yw, _, _, _, fnames = datasets["WDBC"]
scaler = StandardScaler().fit(Xw)
lr_full = LogisticRegression(max_iter=1000, random_state=42).fit(scaler.transform(Xw), yw)
fold_coefs = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for tr, te in skf.split(Xw, yw):
    sc = StandardScaler().fit(Xw[tr])
    m = LogisticRegression(max_iter=1000, random_state=42).fit(sc.transform(Xw[tr]), yw[tr])
    fold_coefs.append(m.coef_[0])
fold_coefs = np.array(fold_coefs)
coef_rows = []
for i, fname in enumerate(fnames):
    c = float(lr_full.coef_[0][i])
    coef_rows.append({"feature": str(fname), "coef_full": round(c, 4),
                      "odds_ratio_full": round(float(np.exp(c)), 4),
                      "coef_fold_mean": round(float(fold_coefs[:, i].mean()), 4),
                      "coef_fold_sd": round(float(fold_coefs[:, i].std()), 4),
                      "sign_consistent_folds": int(np.sum(np.sign(fold_coefs[:, i]) == np.sign(c)))})
coef_rows.sort(key=lambda r: -abs(r["coef_full"]))
results["LR_coefficients"]["WDBC_full"] = {"intercept_full": round(float(lr_full.intercept_[0]), 4), "rows": coef_rows}
# pruned LR on the 7 stable core features (indices by name)
core = ["mean radius", "mean perimeter", "mean concave points", "worst radius", "worst perimeter", "worst area", "worst concave points"]
idx = [list(fnames).index(c) for c in core]
sc = StandardScaler().fit(Xw[:, idx])
lr_core = LogisticRegression(max_iter=1000, random_state=42).fit(sc.transform(Xw[:, idx]), yw)
fold_core = []
for tr, te in skf.split(Xw, yw):
    s2 = StandardScaler().fit(Xw[tr][:, idx])
    mm = LogisticRegression(max_iter=1000, random_state=42).fit(s2.transform(Xw[tr][:, idx]), yw[tr])
    fold_core.append(mm.coef_[0])
fold_core = np.array(fold_core)
results["LR_coefficients"]["WDBC_pruned_core7"] = {
    "intercept_full": round(float(lr_core.intercept_[0]), 4),
    "rows": [{"feature": core[i], "coef_full": round(float(lr_core.coef_[0][i]), 4),
              "odds_ratio_full": round(float(np.exp(lr_core.coef_[0][i])), 4),
              "coef_fold_mean": round(float(fold_core[:, i].mean()), 4),
              "coef_fold_sd": round(float(fold_core[:, i].std()), 4),
              "sign_consistent_folds": int(np.sum(np.sign(fold_core[:, i]) == np.sign(lr_core.coef_[0][i])))} for i in range(7)]}

with open("results7_mar_paired_coef.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("WROTE results7_mar_paired_coef.json")
print(json.dumps(results["LR_coefficients"]["WDBC_pruned_core7"], indent=2, ensure_ascii=False))
