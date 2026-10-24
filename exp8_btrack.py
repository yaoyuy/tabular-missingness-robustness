"""Exp8 B-track (2026-10-24): calibration, SVM full curves, timing table,
Holm correction of Table 9, MAR sensitivity, MNAR, missing-indicator /
HGB-native controls, multi-seed checks. Protocol identical to exp1/exp7:
seed 42 stratified 5-fold, median imputation fitted on train fold only,
MCAR seeds 1000+fold, MAR+ 3000+fold, MAR- 4000+fold, grouped 2000+fold.
Calibration uses predict_proba (SVM only on WDBC, where probability=True);
auxiliary SVM is excluded from calibration because its protocol uses the
decision function. ECE uses 10 equal-width bins. Holm family = the 20
Table-9 comparisons (LR clean-30% and LR vs RF/HGB/SVM at 30% per dataset).
MNAR: per-feature own-value mask p = sigmoid(-1.0 + 1.2*z_own), z from the
training fold, seed 5000+fold; the mask depends on the masked value itself.
"""
import json, time, urllib.request, io
import numpy as np, pandas as pd
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
from sklearn.base import BaseEstimator, TransformerMixin

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def mcar_mask(X, rate, seed):
    rng = np.random.RandomState(seed)
    Xm = X.copy().astype(float)
    Xm[rng.rand(*Xm.shape) < rate] = np.nan
    return Xm

def mar_mask(Xte, Xtr, anchor_idx, direction, seed, slope=1.2, anchor2=None):
    rng = np.random.RandomState(seed)
    Xm = Xte.copy().astype(float)
    a = anchor2 if anchor2 is not None else anchor_idx
    mu = Xtr[:, a].mean(); sd = Xtr[:, a].std() or 1.0
    z = np.clip((Xte[:, a] - mu) / sd, -3, 3)
    p_row = sigmoid(-1.0 + direction * slope * z)
    mask = np.zeros_like(Xm, dtype=bool)
    for j in range(Xm.shape[1]):
        if j == a:
            continue
        mask[:, j] = rng.rand(Xm.shape[0]) < p_row
    Xm[mask] = np.nan
    return Xm, float(mask.sum() / max(1, mask.shape[0] * (mask.shape[1] - 1)))

def mnar_mask(Xte, Xtr, seed):
    rng = np.random.RandomState(seed)
    Xm = Xte.copy().astype(float)
    mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0); sd[sd == 0] = 1.0
    z = np.clip((Xte - mu) / sd, -3, 3)
    p = sigmoid(-1.0 + 1.2 * z)
    mask = rng.rand(*Xm.shape) < p
    Xm[mask] = np.nan
    return Xm, float(mask.mean())

class AddIndicators(BaseEstimator, TransformerMixin):
    """Append binary missing indicators to an already-imputed matrix.
    Fitted on train-fold missing pattern (all observed -> zeros); at test
    time the indicators are recomputed from the masked test matrix."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = np.asarray(X, float)
        return np.hstack([X, np.isnan(X).astype(float)])
    # NOTE: indicators must be computed BEFORE imputation; handled by a
    # custom pipeline below (see indicator_pipeline) instead of this class.

def make_models(binary, wdbc=False):
    lr = make_pipeline(StandardScaler(), SimpleImputer(strategy="median"), LogisticRegression(max_iter=1000, random_state=42))
    rf = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
    hgb = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(random_state=42))
    if wdbc:
        svm = make_pipeline(StandardScaler(), SimpleImputer(strategy="median"), SVC(C=1.0, probability=True, random_state=42))
    else:
        svm = make_pipeline(StandardScaler(), SimpleImputer(strategy="median"), SVC(C=1.0, random_state=42))
    return {"LR": lr, "RF": rf, "HGB": hgb, "SVM": svm}

def load_all():
    out = {}
    d = load_breast_cancer(); out["WDBC"] = (d.data, 1 - d.target, True, 0)
    w = load_wine(); out["Wine"] = (w.data, w.target, False, 0)
    dg = load_digits(); out["Digits"] = (dg.data, dg.target, False, 28)
    try:
        with urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data", timeout=30) as r:
            df = pd.read_csv(io.StringIO(r.read().decode("utf-8", errors="ignore")), header=None)
        out["Ionosphere"] = (df.iloc[:, :-1].values.astype(float), LabelEncoder().fit_transform(df.iloc[:, -1].values), True, 0)
    except Exception as e:
        print("Ionosphere download failed, local fallback:", e, flush=True)
        from pathlib import Path
        df = pd.read_csv(Path.home() / "workspace/paper-repo/data/ionosphere.data", header=None)
        out["Ionosphere"] = (df.iloc[:, :-1].values.astype(float), LabelEncoder().fit_transform(df.iloc[:, -1].values), True, 0)
    try:
        with urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", timeout=30) as r:
            df = pd.read_csv(io.StringIO(r.read().decode("utf-8", errors="ignore")), header=None)
        df = df.replace("?", np.nan).dropna()
        out["Heart"] = (df.iloc[:, :-1].values.astype(float), (df.iloc[:, -1].values.astype(int) > 0).astype(int), True, 0)
    except Exception as e:
        print("Heart download failed, local fallback:", e, flush=True)
        from pathlib import Path
        df = pd.read_csv(Path.home() / "workspace/paper-repo/data/processed.cleveland.data", header=None)
        df = df.replace("?", np.nan).dropna()
        out["Heart"] = (df.iloc[:, :-1].values.astype(float), (df.iloc[:, -1].values.astype(int) > 0).astype(int), True, 0)
    return out

def score_of(model, X, y, binary, wdbc, mname):
    if binary:
        if mname == "SVM" and not wdbc:
            return roc_auc_score(y, model.decision_function(X))
        return roc_auc_score(y, model.predict_proba(X)[:, 1])
    return accuracy_score(y, model.predict(X))

def calib_of(model, X, y, binary):
    """Returns (brier, ece) for a fitted model with predict_proba."""
    P = model.predict_proba(X)
    if binary:
        p = P[:, 1]
        brier = float(np.mean((p - y) ** 2))
        ece = 0.0
        for lo in np.linspace(0, 1, 11)[:-1]:
            m = (p >= lo) & (p < lo + 0.1) if lo < 0.9 else (p >= 0.9) & (p <= 1.0)
            if m.sum() > 0:
                ece += (m.mean()) * abs(p[m].mean() - y[m].mean())
        return brier, float(ece)
    n = len(y)
    onehot = np.zeros_like(P); onehot[np.arange(n), y] = 1
    brier = float(np.mean(np.sum((P - onehot) ** 2, axis=1)))
    conf = P.max(axis=1); pred = P.argmax(axis=1); correct = (pred == y).astype(float)
    ece = 0.0
    for lo in np.linspace(0, 1, 11)[:-1]:
        m = (conf >= lo) & (conf < lo + 0.1) if lo < 0.9 else (conf >= 0.9) & (conf <= 1.0)
        if m.sum() > 0:
            ece += (m.mean()) * abs(conf[m].mean() - correct[m].mean())
    return brier, float(ece)

def holm(pvals):
    """pvals: list of (label, p). Returns list of (label, p, holm_threshold, reject)."""
    m = len(pvals)
    order = sorted(pvals, key=lambda x: x[1])
    out, stop = [], False
    for i, (lab, p) in enumerate(order):
        thr = 0.05 / (m - i)
        rej = (p <= thr) and not stop
        if not rej:
            stop = True
        out.append({"label": lab, "p": p, "holm_threshold": round(thr, 5), "reject": bool(rej)})
    return out

datasets = load_all()
R = {"calibration": {}, "svm_curves": {}, "timing": {}, "table9_paired": {}, "holm": {},
     "mar_sensitivity": {}, "mnar": {}, "indicator_hgbnative": {}, "multiseed": {}}

for name, (X, y, binary, anchor) in datasets.items():
    wdbc = (name == "WDBC")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cal = {m: {"clean": [], "mcar30": [], "mar_plus": [], "mar_minus": []} for m in ["LR", "RF", "HGB"] + (["SVM"] if wdbc else [])}
    cal_ece = {m: {"clean": [], "mcar30": [], "mar_plus": [], "mar_minus": []} for m in cal}
    svm_curve = {r: [] for r in [0.0, 0.1, 0.3, 0.5]}
    per_fold = {m: {"clean": [], "mcar30": []} for m in ["LR", "RF", "HGB", "SVM"]}
    timing = {m: {"fit_s": [], "pred_s": []} for m in ["LR", "RF", "HGB", "SVM"]}
    print(f"=== {name} ===", flush=True)
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        conds = {"clean": Xte}
        conds["mcar30"] = mcar_mask(Xte, 0.3, 1000 + fold)
        conds["mar_plus"], _ = mar_mask(Xte, Xtr, anchor, +1, 3000 + fold)
        conds["mar_minus"], _ = mar_mask(Xte, Xtr, anchor, -1, 4000 + fold)
        models = make_models(binary, wdbc)
        for mname, model in models.items():
            # timing on clean, fit once per fold per model
            t0 = time.perf_counter(); model.fit(Xtr, ytr); t1 = time.perf_counter()
            _ = model.predict(Xte); t2 = time.perf_counter()
            timing[mname]["fit_s"].append(t1 - t0); timing[mname]["pred_s"].append(t2 - t1)
            per_fold[mname]["clean"].append(score_of(model, Xte, yte, binary, wdbc, mname))
            per_fold[mname]["mcar30"].append(score_of(model, conds["mcar30"], yte, binary, wdbc, mname))
            if mname in cal:
                for cname in ["clean", "mcar30", "mar_plus", "mar_minus"]:
                    b, e = calib_of(model, conds[cname], yte, binary)
                    cal[mname][cname].append(b); cal_ece[mname][cname].append(e)
        # SVM curves: refit SVM per rate (same fold, fresh model)
        for rate in [0.0, 0.1, 0.3, 0.5]:
            Xm = Xte if rate == 0.0 else mcar_mask(Xte, rate, 1000 + fold)
            svm = make_models(binary, wdbc)["SVM"]
            svm.fit(Xtr, ytr)
            svm_curve[rate].append(score_of(svm, Xm, yte, binary, wdbc, "SVM"))
    R["calibration"][name] = {m: {c: [round(float(np.mean(v)), 4), round(float(np.std(v, ddof=0)), 4)] for c, v in cal[m].items()} for m in cal}
    R["calibration"][name + "_ECE"] = {m: {c: [round(float(np.mean(v)), 4), round(float(np.std(v, ddof=0)), 4)] for c, v in cal_ece[m].items()} for m in cal_ece}
    R["svm_curves"][name] = {str(r): [round(float(np.mean(v)), 4), round(float(np.std(v, ddof=0)), 4)] for r, v in svm_curve.items()}
    R["timing"][name] = {m: {"fit_s_mean": round(float(np.mean(v["fit_s"])), 4), "pred_s_mean": round(float(np.mean(v["pred_s"])), 4)} for m, v in timing.items()}
    # Table 9 paired + Holm inputs
    pairs = {}
    for ds_label, a_key, b_key in [("LR_clean_minus_30", "clean", "mcar30")]:
        pass
    t9 = {}
    def paired(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float); diff = a - b
        md = float(diff.mean()); sd = float(diff.std(ddof=0))
        t, p = stats.ttest_rel(a, b)
        return {"mean_diff": round(md, 4), "t_p": round(float(p), 5) if float(p) >= 0.000005 else float(p)}
    t9["LR_clean_minus_30"] = paired(per_fold["LR"]["clean"], per_fold["LR"]["mcar30"])
    for other in ["RF", "HGB", "SVM"]:
        t9[f"LR_minus_{other}_30"] = paired(per_fold["LR"]["mcar30"], per_fold[other]["mcar30"])
    R["table9_paired"][name] = t9
    print(f"  {name} svm curves:", R["svm_curves"][name], flush=True)

all_p = [(f"{ds}:{k}", v["t_p"]) for ds, dd in R["table9_paired"].items() for k, v in dd.items()]
R["holm"] = holm(all_p)
print("HOLM:", json.dumps(R["holm"], indent=1), flush=True)

# MAR sensitivity: WDBC + Wine, LR + HGB, alternate anchor and slopes
ANCHOR2 = {"WDBC": 2, "Wine": 1}  # WDBC worst radius? col2 = mean perimeter; use col 5? keep documented: WDBC col 2, Wine col 1
for name in ["WDBC", "Wine"]:
    X, y, binary, anchor = datasets[name]
    wdbc = (name == "WDBC")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sens = {}
    for variant, kw in [("anchor2_plus", dict(anchor2=ANCHOR2[name], direction=+1)),
                        ("anchor2_minus", dict(anchor2=ANCHOR2[name], direction=-1)),
                        ("slope06_plus", dict(slope=0.6, direction=+1)),
                        ("slope06_minus", dict(slope=0.6, direction=-1)),
                        ("slope18_plus", dict(slope=1.8, direction=+1)),
                        ("slope18_minus", dict(slope=1.8, direction=-1))]:
        scores = {"LR": [], "HGB": []}
        rates = []
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
            Xm, rate = mar_mask(Xte, Xtr, anchor, kw["direction"], 3000 + fold if kw["direction"] > 0 else 4000 + fold,
                                slope=kw.get("slope", 1.2), anchor2=kw.get("anchor2"))
            rates.append(rate)
            models = make_models(binary, wdbc)
            for mname in ["LR", "HGB"]:
                models[mname].fit(Xtr, ytr)
                scores[mname].append(score_of(models[mname], Xm, yte, binary, wdbc, mname))
        sens[variant] = {"masked_rate": round(float(np.mean(rates)), 4),
                         "LR": [round(float(np.mean(scores["LR"])), 4), round(float(np.std(scores["LR"], ddof=0)), 4)],
                         "HGB": [round(float(np.mean(scores["HGB"])), 4), round(float(np.std(scores["HGB"], ddof=0)), 4)]}
    R["mar_sensitivity"][name] = sens
    print(f"MAR sens {name}:", sens, flush=True)

# MNAR: all datasets, all models, median imputation
for name, (X, y, binary, anchor) in datasets.items():
    wdbc = (name == "WDBC")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {m: [] for m in ["LR", "RF", "HGB", "SVM"]}; rates = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        Xm, rate = mnar_mask(Xte, Xtr, 5000 + fold); rates.append(rate)
        models = make_models(binary, wdbc)
        for mname, model in models.items():
            model.fit(Xtr, ytr)
            scores[mname].append(score_of(model, Xm, yte, binary, wdbc, mname))
    R["mnar"][name] = {"masked_rate": round(float(np.mean(rates)), 4),
                       **{m: [round(float(np.mean(v)), 4), round(float(np.std(v, ddof=0)), 4)] for m, v in scores.items()}}
    print(f"MNAR {name}:", R["mnar"][name], flush=True)

# Missing-indicator (LR only, indicators computed from masked test matrix
# before imputation, train-fold indicators are all zero) + HGB-native (NaN in)
for name in ["WDBC", "Ionosphere"]:
    X, y, binary, anchor = datasets[name]
    wdbc = (name == "WDBC")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    out = {"LR_indicator_mcar30": [], "LR_indicator_mar_plus": [], "HGB_native_mcar30": [], "HGB_native_mar_plus": []}
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        for cond, key_lr, key_hgb in [("mcar30", "LR_indicator_mcar30", "HGB_native_mcar30"),
                                      ("mar_plus", "LR_indicator_mar_plus", "HGB_native_mar_plus")]:
            Xm = mcar_mask(Xte, 0.3, 1000 + fold) if cond == "mcar30" else mar_mask(Xte, Xtr, anchor, +1, 3000 + fold)[0]
            ind = np.isnan(Xm).astype(float)
            imp = SimpleImputer(strategy="median").fit(Xtr)
            Xtr_i = np.hstack([imp.transform(Xtr), np.zeros((len(Xtr), Xtr.shape[1]))])
            Xte_i = np.hstack([imp.transform(Xm), ind])
            lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
            lr.fit(Xtr_i, ytr)
            out[key_lr].append(roc_auc_score(yte, lr.predict_proba(Xte_i)[:, 1]))
            hgb = HistGradientBoostingClassifier(random_state=42)
            hgb.fit(Xtr, ytr)  # train data have no NaN; test NaN handled natively
            out[key_hgb].append(roc_auc_score(yte, hgb.predict_proba(Xm)[:, 1]))
    R["indicator_hgbnative"][name] = {k: [round(float(np.mean(v)), 4), round(float(np.std(v, ddof=0)), 4)] for k, v in out.items()}
    print(f"IND/HGBN {name}:", R["indicator_hgbnative"][name], flush=True)

# Multi-seed: split seeds 1 and 7, mask seeds shifted by +10000
for seed in [1, 7]:
    for name, (X, y, binary, anchor) in datasets.items():
        if name not in ["WDBC", "Ionosphere"]:
            continue
        wdbc = (name == "WDBC")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        s = {"LR_mcar30": [], "HGB_mcar30": [], "LR_mcar50": [], "HGB_mcar50": [], "RF_mcar30": []}
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
            models = make_models(binary, wdbc)
            for mname, model in models.items():
                model.fit(Xtr, ytr)
            for rate, suffix in [(0.3, "mcar30"), (0.5, "mcar50")]:
                Xm = mcar_mask(Xte, rate, 10000 + seed * 1000 + fold)
                for mname in ["LR", "HGB"]:
                    s[f"{mname}_{suffix}"].append(score_of(models[mname], Xm, yte, binary, wdbc, mname))
            Xm30 = mcar_mask(Xte, 0.3, 10000 + seed * 1000 + fold)
            s["RF_mcar30"].append(score_of(models["RF"], Xm30, yte, binary, wdbc, "RF"))
        R["multiseed"][f"{name}_seed{seed}"] = {k: [round(float(np.mean(v)), 4), round(float(np.std(v, ddof=0)), 4)] for k, v in s.items()}
    print(f"multiseed {seed} done", flush=True)

with open("results8_btrack.json", "w") as f:
    json.dump(R, f, indent=1)
print("SAVED results8_btrack.json")
