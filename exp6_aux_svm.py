"""Aux SVM check (2026-10-23, pre-draft gap close):
Outline v2 required aux SVM before manuscript, but top-3 supplement did not cover it.
Run SVM_RBF (no probability=True, use decision_function for binary AUC to avoid
internal CV cost) on Wine / Digits / Ionosphere / Heart-binary, same protocol:
seed 42, stratified 5-fold, missing only on TEST (MCAR 30%), median from TRAIN only.
This fills the SVM column for aux clean + 30% only; full 0/10/50% aux SVM curves
remain pending and must be stated as such in the draft.
"""
import json, urllib.request, io, numpy as np, pandas as pd
from sklearn.datasets import load_wine, load_digits
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score

def mask(X, rate, seed):
    rng = np.random.RandomState(seed); Xm = X.copy().astype(float)
    Xm[rng.rand(*Xm.shape) < rate] = np.nan; return Xm

def run(X, y, binary, name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clean, miss30 = [], []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        m = make_pipeline(StandardScaler(), SimpleImputer(strategy='median'), SVC(C=1.0, random_state=42))
        m.fit(X[tr], y[tr])
        if binary:
            clean.append(roc_auc_score(y[te], m.decision_function(X[te])))
            miss30.append(roc_auc_score(y[te], m.decision_function(mask(X[te], 0.30, 1000+fold))))
        else:
            clean.append(accuracy_score(y[te], m.predict(X[te])))
            miss30.append(accuracy_score(y[te], m.predict(mask(X[te], 0.30, 1000+fold))))
    out = {'metric': 'AUC' if binary else 'Accuracy',
           'clean': f"{np.mean(clean):.4f} +- {np.std(clean):.4f}",
           'miss30': f"{np.mean(miss30):.4f} +- {np.std(miss30):.4f}",
           'drop': f"{np.mean(clean)-np.mean(miss30):.4f}"}
    print(name, out, flush=True)
    return out

results = {}
w = load_wine(); results['Wine_SVM'] = run(w.data, w.target, False, 'Wine')
d = load_digits(); results['Digits_SVM'] = run(d.data, d.target, False, 'Digits')
try:
    with urllib.request.urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data', timeout=30) as r:
        df = pd.read_csv(io.StringIO(r.read().decode('utf-8', errors='ignore')), header=None)
    Xi = df.iloc[:, :-1].values.astype(float); yi = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    results['Ionosphere_SVM'] = run(Xi, yi, True, 'Ionosphere')
except Exception as e:
    results['Ionosphere_SVM'] = f"FAILED: {e}"
try:
    with urllib.request.urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data', timeout=30) as r:
        df = pd.read_csv(io.StringIO(r.read().decode('utf-8', errors='ignore')), header=None)
    df = df.replace('?', np.nan).dropna()
    Xh = df.iloc[:, :-1].values.astype(float); yh = (df.iloc[:, -1].values.astype(int) > 0).astype(int)
    results['Heart_binary_SVM'] = run(Xh, yh, True, 'Heart_binary')
except Exception as e:
    results['Heart_binary_SVM'] = f"FAILED: {e}"
with open('results6_aux_svm.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print('WROTE results6_aux_svm.json')
