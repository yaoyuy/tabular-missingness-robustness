"""Prototype experiment for paper topic selection.
Dataset: Wisconsin Diagnostic Breast Cancer (WDBC) via sklearn, 569 samples, 30 features, 212 malignant / 357 benign.
Protocol (fixed seed 42, no leakage):
 - Stratified 5-fold CV for clean performance.
 - Missing-robustness: on each fold, train on train fold, mask test features MCAR at 0/10/30/50%, impute with TRAIN median only.
 - Pruned model: top-8 features by RandomForest importance computed on TRAIN fold only, then retrain.
All numbers are real outputs of this script, not hand-written.
"""
import json, numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

rng_global = np.random.RandomState(42)
data = load_breast_cancer()
X, y = data.data, data.target  # sklearn: 0=malignant, 1=benign; we flip so 1=malignant for AUC clarity
y = 1 - y
print(f"n={X.shape[0]} p={X.shape[1]} malignant={int(y.sum())} benign={int((1-y).sum())}")

def make_models():
    return {
        "LogisticRegression": make_pipeline(StandardScaler(), SimpleImputer(strategy="median"), LogisticRegression(max_iter=1000, random_state=42)),
        "RandomForest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        "HistGradientBoosting": make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(random_state=42)),
        "SVM_RBF": make_pipeline(StandardScaler(), SimpleImputer(strategy="median"), SVC(C=1.0, probability=True, random_state=42)),
    }

def mask_test(X_test, rate, seed):
    if rate == 0:
        return X_test.copy()
    rng = np.random.RandomState(seed)
    Xm = X_test.copy().astype(float)
    mask = rng.rand(*Xm.shape) < rate
    Xm[mask] = np.nan
    return Xm

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clean = {k: {"acc": [], "f1": [], "auc": []} for k in make_models()}
robust = {k: {r: [] for r in [0, 0.10, 0.30, 0.50]} for k in make_models()}
pruned_clean = {"RF_top8": [], "LR_top8": []}
pruned_robust30 = {"RF_top8": [], "LR_top8": []}
top_features_per_fold = []

for fold, (tr, te) in enumerate(skf.split(X, y)):
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    models = make_models()
    # RF importance on train only, for pruned models
    rf_imp = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    top8 = np.argsort(rf_imp.feature_importances_)[::-1][:8]
    top_features_per_fold.append([data.feature_names[i] for i in top8])
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        proba = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(Xte)
        clean[name]["acc"].append(accuracy_score(yte, pred))
        clean[name]["f1"].append(f1_score(yte, pred))
        clean[name]["auc"].append(roc_auc_score(yte, proba))
        for rate in [0, 0.10, 0.30, 0.50]:
            Xm = mask_test(Xte, rate, seed=1000+fold)
            proba_m = model.predict_proba(Xm)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(Xm)
            robust[name][rate].append(roc_auc_score(yte, proba_m))
    # pruned
    for label, clf in [("RF_top8", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
                       ("LR_top8", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)))]:
        # imputation for pruned: median from train top8
        med = np.median(Xtr[:, top8], axis=0)
        def fill(A):
            A = A[:, top8].astype(float).copy()
            idx = np.isnan(A)
            if idx.any():
                A[idx] = np.take(med, np.where(idx)[1])
            return A
        clf.fit(fill(Xtr), ytr)
        pred = clf.predict(fill(Xte))
        proba = clf.predict_proba(fill(Xte))[:, 1]
        pruned_clean[label].append(roc_auc_score(yte, proba))
        Xm = mask_test(Xte, 0.30, seed=1000+fold)
        proba_m = clf.predict_proba(fill(Xm))[:, 1]
        pruned_robust30[label].append(roc_auc_score(yte, proba_m))

def summarise(d):
    return {k: f"{np.mean(v):.4f} ± {np.std(v):.4f}" for k, v in d.items()}

out = {
    "clean": {k: summarise(v) for k, v in clean.items()},
    "robust_auc": {k: {str(r): f"{np.mean(v):.4f} ± {np.std(v):.4f}" for r, v in rates.items()} for k, rates in robust.items()},
    "pruned_clean_auc": summarise(pruned_clean),
    "pruned_robust30_auc": summarise(pruned_robust30),
    "top8_per_fold": top_features_per_fold,
}
print(json.dumps(out, indent=2, ensure_ascii=False))
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
