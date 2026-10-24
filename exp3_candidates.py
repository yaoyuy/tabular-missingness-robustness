"""Candidate extra datasets, same protocol as exp1/exp2 (seed 42, 5-fold stratified,
missing only on test, median imputation from train, MCAR 30%).
Built-in: Digits. Downloaded from UCI (free, public):
 Pima diabetes, Heart Cleveland, Ionosphere, Sonar, Banknote.
If a download fails, that candidate is reported as failed, not invented.
"""
import json, urllib.request, io, numpy as np, pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

def mask(X, rate, seed):
    rng = np.random.RandomState(seed); Xm = X.copy().astype(float)
    Xm[rng.rand(*Xm.shape) < rate] = np.nan; return Xm

def load_uci(url, header=None):
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = r.read().decode('utf-8', errors='ignore')
        return pd.read_csv(io.StringIO(data), header=header)
    except Exception as e:
        return f"DOWNLOAD_FAILED: {e}"

datasets = {}
d = load_digits(); datasets['Digits (sklearn, 1797x64, 10 classes)'] = (d.data, d.target)

cands = [
    ('Pima Diabetes (UCI, 768x8, binary)', 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'),
    ('Heart Cleveland (UCI, 303x13, 5 classes 0-4)', 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'),
    ('Ionosphere (UCI, 351x34, binary)', 'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'),
    ('Sonar (UCI, 208x60, binary)', 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data'),
    ('Banknote (UCI, 1372x4, binary)', 'https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt'),
]
for name, url in cands:
    df = load_uci(url)
    if isinstance(df, str):
        datasets[name] = df; continue
    df = df.replace('?', np.nan).dropna()
    X = df.iloc[:, :-1].values.astype(float)
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    datasets[name] = (X, y)

results = {}
for name, data in datasets.items():
    if isinstance(data, str):
        results[name] = data; continue
    X, y = data
    if len(np.unique(y)) < 2 or X.shape[0] < 100:
        results[name] = f"SKIPPED: shape={X.shape}, classes={len(np.unique(y))}"; continue
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    binary = len(np.unique(y)) == 2
    res = {'shape': list(X.shape), 'classes': int(len(np.unique(y))), 'class_counts': [int((y==c).sum()) for c in sorted(np.unique(y))]}
    for label, mk_fn in [('LR', lambda: make_pipeline(StandardScaler(), SimpleImputer(strategy='median'), LogisticRegression(max_iter=1000, random_state=42))),
                         ('RF', lambda: make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))),
                         ('HGB', lambda: make_pipeline(SimpleImputer(strategy='median'), HistGradientBoostingClassifier(random_state=42)))]:
        clean, miss30 = [], []
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            mk = mk_fn(); mk.fit(X[tr], y[tr])
            if binary:
                clean.append(roc_auc_score(y[te], mk.predict_proba(X[te])[:, 1]))
                miss30.append(roc_auc_score(y[te], mk.predict_proba(mask(X[te], 0.30, 1000+fold))[:, 1]))
            else:
                clean.append(accuracy_score(y[te], mk.predict(X[te])))
                miss30.append(accuracy_score(y[te], mk.predict(mask(X[te], 0.30, 1000+fold))))
        res[label] = {'metric': 'AUC' if binary else 'Accuracy', 'clean': f"{np.mean(clean):.4f} ± {np.std(clean):.4f}", 'miss30': f"{np.mean(miss30):.4f} ± {np.std(miss30):.4f}", 'drop': f"{np.mean(clean)-np.mean(miss30):.4f}"}
    results[name] = res
    print(name, json.dumps(res, ensure_ascii=False))

print(json.dumps(results, indent=2, ensure_ascii=False))
with open('results3.json', 'w', encoding='utf-8') as f: json.dump(results, f, indent=2, ensure_ascii=False)
