"""Repeated CV to test whether v2 paired differences are real or just 5-fold noise.
10 repeats x 5 folds = 50 paired observations, seed varies per repeat.
Comparisons: LR-full vs LR-pruned at 30% missing; LR vs HGB at 50% missing.
"""
import json, numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy import stats

data = load_breast_cancer(); X, y = data.data, 1 - data.target
def mask(X, rate, seed):
    rng = np.random.RandomState(seed); Xm = X.copy().astype(float)
    Xm[rng.rand(*Xm.shape) < rate] = np.nan; return Xm

diff_pruned, diff_hgb = [], []
for rep in range(10):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100+rep)
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(Xtr, ytr)
        top8 = np.argsort(rf.feature_importances_)[::-1][:8]
        lr = make_pipeline(StandardScaler(), SimpleImputer(strategy="median"), LogisticRegression(max_iter=1000, random_state=42)).fit(Xtr, ytr)
        hgb = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(random_state=42)).fit(Xtr, ytr)
        Xm30 = mask(Xte, 0.30, 5000+rep*10+fold); Xm50 = mask(Xte, 0.50, 6000+rep*10+fold)
        auc_lr30 = roc_auc_score(yte, lr.predict_proba(Xm30)[:, 1])
        auc_hgb50 = roc_auc_score(yte, hgb.predict_proba(Xm50)[:, 1])
        auc_lr50 = roc_auc_score(yte, lr.predict_proba(Xm50)[:, 1])
        med8 = np.median(Xtr[:, top8], axis=0)
        def fill8(A):
            A = A[:, top8].astype(float).copy(); idx = np.isnan(A)
            if idx.any(): A[idx] = np.take(med8, np.where(idx)[1])
            return A
        pr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)).fit(fill8(Xtr), ytr)
        auc_pr30 = roc_auc_score(yte, pr.predict_proba(fill8(Xm30))[:, 1])
        diff_pruned.append(auc_lr30 - auc_pr30)
        diff_hgb.append(auc_lr50 - auc_hgb50)

def report(d, label):
    d = np.array(d)
    return {"label": label, "n": len(d), "mean_diff": round(float(np.mean(d)), 4), "std_diff": round(float(np.std(d)), 4), "paired_t_p": float(stats.ttest_1samp(d, 0).pvalue), "wilcoxon_p": float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else None}

out = {"LRfull_minus_LRpruned_at30_repeated": report(diff_pruned, "LRfull-LRpruned@30%"), "LR_minus_HGB_at50_repeated": report(diff_hgb, "LR-HGB@50%")}
print(json.dumps(out, indent=2))
with open("results2b.json", "w") as f: json.dump(out, f, indent=2)
