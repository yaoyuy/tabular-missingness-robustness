"""Verification rerun (2026-10-23, pre-draft check):
Heart binary and Pima OpenML numbers were in outline/memory but had NO script
or results file on disk (exp3 only did Heart 5-class and Pima download failed).
Rerun both with the SAME protocol as exp1/exp3: seed 42, stratified 5-fold,
missing only on TEST (MCAR 30%), median imputation from TRAIN only.
If numbers differ from outline, outline must be corrected, not the rerun hidden.
"""
import json, urllib.request, io, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

def mask(X, rate, seed):
    rng = np.random.RandomState(seed); Xm = X.copy().astype(float)
    Xm[rng.rand(*Xm.shape) < rate] = np.nan; return Xm

def run(X, y, name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    out = {'shape': list(X.shape), 'class_counts': [int((y==c).sum()) for c in sorted(np.unique(y))]}
    for label, mk in [('LR', lambda: make_pipeline(StandardScaler(), SimpleImputer(strategy='median'), LogisticRegression(max_iter=1000, random_state=42))),
                      ('RF', lambda: make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))),
                      ('HGB', lambda: make_pipeline(SimpleImputer(strategy='median'), HistGradientBoostingClassifier(random_state=42)))]:
        clean, miss30 = [], []
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            m = mk(); m.fit(X[tr], y[tr])
            clean.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
            miss30.append(roc_auc_score(y[te], m.predict_proba(mask(X[te], 0.30, 1000+fold))[:, 1]))
        out[label] = {'clean': f"{np.mean(clean):.4f} +- {np.std(clean):.4f}",
                      'miss30': f"{np.mean(miss30):.4f} +- {np.std(miss30):.4f}",
                      'drop': f"{np.mean(clean)-np.mean(miss30):.4f}"}
        print(name, label, out[label], flush=True)
    return out

results = {}
# Heart Cleveland, binary
try:
    with urllib.request.urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data', timeout=30) as r:
        df = pd.read_csv(io.StringIO(r.read().decode('utf-8', errors='ignore')), header=None)
    df = df.replace('?', np.nan).dropna()
    Xh = df.iloc[:, :-1].values.astype(float)
    yh_raw = df.iloc[:, -1].values.astype(int)
    yh = (yh_raw > 0).astype(int)
    print('Heart raw 5-class counts:', {int(c): int((yh_raw==c).sum()) for c in sorted(np.unique(yh_raw))}, 'binary:', {int(c): int((yh==c).sum()) for c in [0,1]})
    results['Heart_binary_verified'] = run(Xh, yh, 'Heart_binary')
except Exception as e:
    results['Heart_binary_verified'] = f"FAILED: {e}"
    print('Heart FAILED', e)
# Pima via OpenML data_id 37
try:
    from sklearn.datasets import fetch_openml
    p = fetch_openml(data_id=37, as_frame=True, parser='auto')
    Xp = p.data.values.astype(float)
    yp = LabelEncoder().fit_transform(p.target.values)
    print('Pima shape', Xp.shape, 'counts', {int(c): int((yp==c).sum()) for c in sorted(np.unique(yp))})
    results['Pima_OpenML_verified'] = run(Xp, yp, 'Pima_OpenML')
except Exception as e:
    results['Pima_OpenML_verified'] = f"FAILED: {e}"
    print('Pima FAILED', e)

with open('results5_verify.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print('WROTE results5_verify.json')
