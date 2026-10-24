"""Supplement for VJCS draft, top-3 items (2026-10-23, user chose VJCS first).
Protocol matches exp1/exp3: seed 42, stratified 5-fold, missing only on TEST,
imputer fitted on TRAIN fold only (no test leakage), MCAR rates 0/10/30/50%.
Part A: imputation control on WDBC (median vs KNN vs Iterative) for LR and RF;
  plus KNN-vs-median control on Wine / Digits / Ionosphere (LR only, to bound runtime).
Part B: feature-stability quantification on WDBC:
  - RF-importance Top-8 / Top-12 selection frequency across 5 folds, pairwise Jaccard.
  - Permutation importance for LR on held-out test, per fold, mean +- std across folds.
No invented numbers: every value below is a real output of this script.
"""
import json, urllib.request, io, numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (required by sklearn before IterativeImputer import)
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.inspection import permutation_importance

def mask(X, rate, seed):
    if rate == 0:
        return X.copy().astype(float)
    rng = np.random.RandomState(seed)
    Xm = X.copy().astype(float)
    Xm[rng.rand(*Xm.shape) < rate] = np.nan
    return Xm

def load_ionosphere():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            txt = r.read().decode('utf-8', errors='ignore')
        df = pd.read_csv(io.StringIO(txt), header=None)
        X = df.iloc[:, :-1].values.astype(float)
        y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
        return X, y
    except Exception as e:
        return None, f"DOWNLOAD_FAILED: {e}"

def run_fill(X, y, binary, name, imputers, models):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    out = {}
    for imp_name, imp_fn in imputers.items():
        out[imp_name] = {}
        for m_name, mk_fn in models.items():
            per_rate = {r: [] for r in [0, 0.10, 0.30, 0.50]}
            for fold, (tr, te) in enumerate(skf.split(X, y)):
                clf = mk_fn(imp_fn())
                clf.fit(X[tr], y[tr])
                for rate in per_rate:
                    Xm = mask(X[te], rate, seed=1000+fold)
                    if binary:
                        per_rate[rate].append(roc_auc_score(y[te], clf.predict_proba(Xm)[:, 1]))
                    else:
                        per_rate[rate].append(accuracy_score(y[te], clf.predict(Xm)))
            out[imp_name][m_name] = {str(r): f"{np.mean(v):.4f} +- {np.std(v):.4f}"
                                     for r, v in per_rate.items()}
            print(name, imp_name, m_name, out[imp_name][m_name], flush=True)
    return out

results = {}
# --- WDBC main: median / KNN / Iterative, LR + RF ---
w = load_breast_cancer(); Xw, yw = w.data, 1 - w.target
imputers_w = {
    'median': lambda: SimpleImputer(strategy='median'),
    'KNN_k5': lambda: KNNImputer(n_neighbors=5),
    'Iterative': lambda: IterativeImputer(max_iter=10, random_state=42),
}
models_w = {
    'LR': lambda imp: make_pipeline(StandardScaler(), imp, LogisticRegression(max_iter=1000, random_state=42)),
    'RF': lambda imp: make_pipeline(imp, RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
}
results['WDBC_fill_control_AUC'] = run_fill(Xw, yw, True, 'WDBC', imputers_w, models_w)

# --- Aux: median vs KNN, LR only ---
imputers_aux = {
    'median': lambda: SimpleImputer(strategy='median'),
    'KNN_k5': lambda: KNNImputer(n_neighbors=5),
}
models_lr = {'LR': lambda imp: make_pipeline(StandardScaler(), imp, LogisticRegression(max_iter=1000, random_state=42))}
wine = load_wine()
results['Wine_fill_control_Accuracy_LR'] = run_fill(wine.data, wine.target, False, 'Wine', imputers_aux, models_lr)
dg = load_digits()
results['Digits_fill_control_Accuracy_LR'] = run_fill(dg.data, dg.target, False, 'Digits', imputers_aux, models_lr)
Xi, yi = load_ionosphere()
if Xi is None:
    results['Ionosphere_fill_control'] = yi
else:
    results['Ionosphere_fill_control_AUC_LR'] = run_fill(Xi, yi, True, 'Ionosphere', imputers_aux, models_lr)

# --- Part B: WDBC stability ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
feature_names = list(w.feature_names)
top8_sets, top12_sets = [], []
from collections import Counter
cnt8, cnt12 = Counter(), Counter()
perm_per_fold = []  # list of arrays (30,) per fold
for fold, (tr, te) in enumerate(skf.split(Xw, yw)):
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(Xw[tr], yw[tr])
    order = np.argsort(rf.feature_importances_)[::-1]
    t8, t12 = set(order[:8].tolist()), set(order[:12].tolist())
    top8_sets.append(t8); top12_sets.append(t12)
    for i in t8: cnt8[feature_names[i]] += 1
    for i in t12: cnt12[feature_names[i]] += 1
    # LR permutation importance on held-out test, clean data
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)).fit(Xw[tr], yw[tr])
    pi = permutation_importance(lr, Xw[te], yw[te], n_repeats=3, random_state=42, scoring='roc_auc')
    perm_per_fold.append(pi.importances_mean)
    print(f"fold {fold} top8={[feature_names[i] for i in sorted(t8, key=lambda j: -rf.feature_importances_[j])]}", flush=True)

def jaccard_mean(sets):
    vals = []
    for i in range(len(sets)):
        for j in range(i+1, len(sets)):
            vals.append(len(sets[i] & sets[j]) / len(sets[i] | sets[j]))
    return float(np.mean(vals)), float(np.std(vals))

perm = np.array(perm_per_fold)  # 5 x 30
perm_mean = perm.mean(axis=0); perm_std = perm.std(axis=0)
perm_order = np.argsort(perm_mean)[::-1]
results['WDBC_stability'] = {
    'top8_selection_count_out_of_5': dict(sorted(cnt8.items(), key=lambda kv: -kv[1])),
    'top12_selection_count_out_of_5': dict(sorted(cnt12.items(), key=lambda kv: -kv[1])),
    'top8_pairwise_jaccard_mean_std': [round(float(jaccard_mean(top8_sets)[0]), 4), round(float(jaccard_mean(top8_sets)[1]), 4)],
    'top12_pairwise_jaccard_mean_std': [round(float(jaccard_mean(top12_sets)[0]), 4), round(float(jaccard_mean(top12_sets)[1]), 4)],
    'top8_intersection_all_folds': sorted([feature_names[i] for i in set.intersection(*top8_sets)]),
    'top12_intersection_all_folds': sorted([feature_names[i] for i in set.intersection(*top12_sets)]),
    'LR_permutation_importance_AUC_drop_mean_std_top12': [
        {'feature': feature_names[i], 'mean': round(float(perm_mean[i]), 4), 'std': round(float(perm_std[i]), 4)}
        for i in perm_order[:12]],
    'LR_permutation_importance_max_mean': round(float(perm_mean.max()), 4),
}
print(json.dumps(results['WDBC_stability'], indent=2, ensure_ascii=False))
with open('results4.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("WROTE results4.json")
