"""Experiment v2: does v1 workload hold up?
Adds, all with fixed seed 42 and no test leakage:
 A. Paired differences across 5 folds (WDBC): LR-full vs LR-pruned at 30% missing; LR vs HGB at 50% missing. Wilcoxon + paired t (scipy), reported with n=5 caveat.
 B. Feature-count ablation for LR: k = 4, 8, 12, 20, 30 (top-k by train-fold RF importance), clean AUC and 30%-missing AUC.
 C. Non-random missing for LR: mask 50% of (i) worst-* features, (ii) mean-* features, (iii) top-8 features only, on test fold.
 D. Permutation importance (fold 0, LR and RF, clean test) - top 10.
 E. Second dataset check: Wine (178, 13 features, 3 classes), LR vs RF, clean accuracy and 30% MCAR accuracy, 5-fold stratified.
 F. Fit-time (seconds, fold 0, indicative only, VM CPU).
"""
import json, time, numpy as np
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.inspection import permutation_importance
from scipy import stats

def mask_mcar(X, rate, seed):
    if rate == 0: return X.copy().astype(float)
    rng = np.random.RandomState(seed); Xm = X.copy().astype(float)
    Xm[rng.rand(*Xm.shape) < rate] = np.nan; return Xm

def mask_cols(X, cols, rate, seed):
    Xm = X.copy().astype(float); rng = np.random.RandomState(seed)
    sub = rng.rand(X.shape[0], len(cols)) < rate
    Xm[np.ix_(np.arange(X.shape[0]), cols)] = np.where(sub, np.nan, Xm[np.ix_(np.arange(X.shape[0]), cols)])
    return Xm

data = load_breast_cancer(); X, y = data.data, 1 - data.target
names = list(data.feature_names)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# A, B, C, F
lr_full_auc30, lr_pruned_auc30, lr_auc50, hgb_auc50 = [], [], [], []
ablation = {k: {"clean": [], "miss30": []} for k in [4, 8, 12, 20, 30]}
nonrand = {"worst50": [], "mean50": [], "top8_50": [], "clean": []}
fit_times = {}
worst_cols = [i for i, n in enumerate(names) if n.startswith("worst")]
mean_cols = [i for i, n in enumerate(names) if n.startswith("mean")]

for fold, (tr, te) in enumerate(skf.split(X, y)):
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    rf_imp = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    order = np.argsort(rf_imp.feature_importances_)[::-1]
    top8 = order[:8]
    def lr(): return make_pipeline(StandardScaler(), SimpleImputer(strategy="median"), LogisticRegression(max_iter=1000, random_state=42))
    def hgb(): return make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(random_state=42))
    # A
    m_lr = lr().fit(Xtr, ytr); m_hgb = hgb().fit(Xtr, ytr)
    med = np.median(Xtr, axis=0)
    def auc_of(model, Xm):
        # pipeline already imputes with train median via SimpleImputer fitted on train, so just predict
        return roc_auc_score(yte, model.predict_proba(Xm)[:, 1])
    Xm30 = mask_mcar(Xte, 0.30, 1000+fold); Xm50 = mask_mcar(Xte, 0.50, 1000+fold)
    lr_full_auc30.append(auc_of(m_lr, Xm30)); lr_auc50.append(auc_of(m_lr, Xm50)); hgb_auc50.append(auc_of(m_hgb, Xm50))
    # pruned LR: train on top8 only, median from train top8
    med8 = np.median(Xtr[:, top8], axis=0)
    def fill8(A):
        A = A[:, top8].astype(float).copy(); idx = np.isnan(A)
        if idx.any(): A[idx] = np.take(med8, np.where(idx)[1])
        return A
    m_pr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)).fit(fill8(Xtr), ytr)
    lr_pruned_auc30.append(roc_auc_score(yte, m_pr.predict_proba(fill8(Xm30))[:, 1]))
    # B ablation (LR only)
    for k in [4, 8, 12, 20, 30]:
        cols = order[:k]; medk = np.median(Xtr[:, cols], axis=0)
        def fillk(A, cols=cols, medk=medk):
            A = A[:, cols].astype(float).copy(); idx = np.isnan(A)
            if idx.any(): A[idx] = np.take(medk, np.where(idx)[1])
            return A
        mk = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)).fit(fillk(Xtr), ytr)
        ablation[k]["clean"].append(roc_auc_score(yte, mk.predict_proba(fillk(Xte))[:, 1]))
        ablation[k]["miss30"].append(roc_auc_score(yte, mk.predict_proba(fillk(Xm30))[:, 1]))
    # C non-random (full LR, pipeline imputer)
    nonrand["clean"].append(auc_of(m_lr, Xte))
    nonrand["worst50"].append(auc_of(m_lr, mask_cols(Xte, worst_cols, 0.50, 2000+fold)))
    nonrand["mean50"].append(auc_of(m_lr, mask_cols(Xte, mean_cols, 0.50, 2000+fold)))
    nonrand["top8_50"].append(auc_of(m_lr, mask_cols(Xte, list(top8), 0.50, 2000+fold)))
    if fold == 0:
        for label, mk in [("LR", lr()), ("RF", make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))), ("HGB", hgb())]:
            t0 = time.time(); mk.fit(Xtr, ytr); fit_times[label] = round(time.time()-t0, 3)

def paired(a, b):
    a, b = np.array(a), np.array(b); diff = a - b
    try: w = stats.wilcoxon(a, b)
    except Exception as e: w = str(e)
    t = stats.ttest_rel(a, b)
    return {"mean_a": round(float(np.mean(a)), 4), "mean_b": round(float(np.mean(b)), 4), "mean_diff_a_minus_b": round(float(np.mean(diff)), 4), "wilcoxon": str(w), "paired_t": str(t)}

# D permutation, fold 0 only
tr, te = list(skf.split(X, y))[0]; Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
perm = {}
for label, mk in [("LR", make_pipeline(StandardScaler(), SimpleImputer(strategy="median"), LogisticRegression(max_iter=1000, random_state=42))), ("RF", make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)))]:
    mk.fit(Xtr, ytr)
    r = permutation_importance(mk, Xte, yte, n_repeats=10, random_state=42, scoring="roc_auc")
    idx = np.argsort(r.importances_mean)[::-1][:10]
    perm[label] = [(names[i], round(float(r.importances_mean[i]), 4)) for i in idx]

# E wine
wine = load_wine(); Xw, yw = wine.data, wine.target
skfw = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
wine_res = {"LR": {"clean": [], "miss30": []}, "RF": {"clean": [], "miss30": []}}
for fold, (tr, te) in enumerate(skfw.split(Xw, yw)):
    Xtr, Xte, ytr, yte = Xw[tr], Xw[te], yw[tr], yw[te]
    for label, mk in [("LR", make_pipeline(StandardScaler(), SimpleImputer(strategy="median"), LogisticRegression(max_iter=1000, random_state=42))), ("RF", make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)))]:
        mk.fit(Xtr, ytr)
        wine_res[label]["clean"].append(accuracy_score(yte, mk.predict(Xte)))
        wine_res[label]["miss30"].append(accuracy_score(yte, mk.predict(mask_mcar(Xte, 0.30, 1000+fold))))

out = {
    "paired_LRfull_minus_LRpruned_at30": paired(lr_full_auc30, lr_pruned_auc30),
    "paired_LR_minus_HGB_at50": paired(lr_auc50, hgb_auc50),
    "ablation_LR": {str(k): {kk: f"{np.mean(v):.4f} ± {np.std(v):.4f}" for kk, v in d.items()} for k, d in ablation.items()},
    "nonrandom_LR_auc": {k: f"{np.mean(v):.4f} ± {np.std(v):.4f}" for k, v in nonrand.items()},
    "permutation_fold0_top10": perm,
    "wine_accuracy": {k: {kk: f"{np.mean(v):.4f} ± {np.std(v):.4f}" for kk, v in d.items()} for k, d in wine_res.items()},
    "fit_time_fold0_sec": fit_times,
    "note": "n_folds=5, so paired tests are indicative only, not proof of superiority; report them as such in the paper.",
}
print(json.dumps(out, indent=2, ensure_ascii=False))
with open("results2.json", "w", encoding="utf-8") as f: json.dump(out, f, indent=2, ensure_ascii=False)
