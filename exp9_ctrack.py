"""Exp9 C-track (2026-10-24): hyperparameter sensitivity on WDBC.
LR C in {0.1, 1.0, 10.0} and RF n_estimators in {100, 300, 500}, all else
identical to exp1 (seed 42, 5-fold, median imputation train-fold only,
MCAR masks 1000+fold). Reports clean and 30% MCAR AUC, 5-fold mean+-SD.
Also writes the data used for the C-track figures (MAR curves from
results7, coefficient forest data from results7 LR coefficients).
"""
import json, numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

d = load_breast_cancer(); X, y = d.data, 1 - d.target
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
out = {"LR_C": {}, "RF_trees": {}}
for C in [0.1, 1.0, 10.0]:
    clean, m30 = [], []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        rng = np.random.RandomState(1000 + fold)
        Xm = X[te].copy().astype(float); Xm[rng.rand(*Xm.shape) < 0.3] = np.nan
        m = make_pipeline(StandardScaler(), SimpleImputer(strategy="median"),
                          LogisticRegression(C=C, max_iter=1000, random_state=42))
        m.fit(X[tr], y[tr])
        clean.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        m30.append(roc_auc_score(y[te], m.predict_proba(Xm)[:, 1]))
    out["LR_C"][str(C)] = {"clean": [round(float(np.mean(clean)), 4), round(float(np.std(clean, ddof=0)), 4)],
                           "mcar30": [round(float(np.mean(m30)), 4), round(float(np.std(m30, ddof=0)), 4)]}
    print("LR C", C, out["LR_C"][str(C)], flush=True)
for n in [100, 300, 500]:
    clean, m30 = [], []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        rng = np.random.RandomState(1000 + fold)
        Xm = X[te].copy().astype(float); Xm[rng.rand(*Xm.shape) < 0.3] = np.nan
        m = make_pipeline(SimpleImputer(strategy="median"),
                          RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1))
        m.fit(X[tr], y[tr])
        clean.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        m30.append(roc_auc_score(y[te], m.predict_proba(Xm)[:, 1]))
    out["RF_trees"][str(n)] = {"clean": [round(float(np.mean(clean)), 4), round(float(np.std(clean, ddof=0)), 4)],
                               "mcar30": [round(float(np.mean(m30)), 4), round(float(np.std(m30, ddof=0)), 4)]}
    print("RF trees", n, out["RF_trees"][str(n)], flush=True)
with open("results9_ctrack.json", "w") as f:
    json.dump(out, f, indent=1)
print("SAVED results9_ctrack.json")
