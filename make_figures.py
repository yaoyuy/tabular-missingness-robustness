"""Nature-figure-optimised publication figures, ONLY from result files.
Backend: Python (matplotlib), saved preference already 'python'.
Contracts (one claim each, quantitative grid):
- Fig1: WDBC AUC falls with test-time missingness; LR/SVM stay above RF/HGB at 50%.
- Fig2: WDBC LR AUC is flat across feature counts; k=8 costs ~0.01 AUC at 30%.
- Fig3: Median imputation is not best; KNN/Iterative beat it, most starkly on Digits.
Uncertainty everywhere: 5-fold mean +/- 1 SD (same definition in every panel).
Exports per figure: SVG (editable text, primary), PDF (editable text), PNG 300 dpi.
Fig3 (multi-panel) runs the render-time panel-alignment gate before export.
Portable repo copy: run with any Python that has matplotlib installed
(requirements.txt). Do NOT hand-type result numbers.
"""
import json, re, pathlib, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from audit_panel_alignment import require_matplotlib_panel_alignment  # vendored, unchanged, see README

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'font.size': 9,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'legend.frameon': False,
})

base = pathlib.Path(__file__).parent
# Committed results live in results/; fall back to the script dir for the
# original flat layout so both layouts work.
results_dir = base / 'results' if (base / 'results').is_dir() else base
outdir = base / 'figures'
outdir.mkdir(exist_ok=True)

def mean_std(s):
    m = re.match(r'([\d.]+)\s*(?:±|\+-)\s*([\d.]+)', s)
    if m is None:
        raise ValueError(f'cannot parse mean±std: {s!r}')
    return float(m.group(1)), float(m.group(2))

def panel_label(ax, label):
    offset = ScaledTranslation(-4 / 72, 3 / 72, ax.figure.dpi_scale_trans)
    ax.text(0, 1, label, transform=ax.transAxes + offset,
            fontsize=9, fontweight='bold', ha='left', va='bottom')

def save(fig, stem):
    fig.savefig(outdir / f'{stem}.svg', bbox_inches='tight')
    fig.savefig(outdir / f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(outdir / f'{stem}.png', dpi=300, bbox_inches='tight')

BLUE, TEAL, GREEN, RED = '#0F4D92', '#42949E', '#8BCF8B', '#B64342'

# Fig1
r1 = json.loads((results_dir / 'results.json').read_text())
rates = [0, 10, 30, 50]
keys = [('LogisticRegression', 'LR', BLUE, 'o'),
        ('SVM_RBF', 'SVM-RBF', TEAL, 's'),
        ('RandomForest', 'RF', GREEN, '^'),
        ('HistGradientBoosting', 'HGB', RED, 'D')]
fig, ax = plt.subplots(figsize=(7, 4.2))
for key, label, color, marker in keys:
    d = r1['robust_auc'][key]
    means, stds = [], []
    for rate, rk in zip(rates, ['0', '0.1', '0.3', '0.5']):
        m, s = mean_std(d[rk]); means.append(m); stds.append(s)
    ax.errorbar(rates, means, yerr=stds, marker=marker, capsize=3,
                color=color, label=label, linewidth=1.5)
ax.set_xlabel('Test-time MCAR missingness (%, median imputation)')
ax.set_ylabel('WDBC AUC (5-fold mean ± SD)')
ax.set_ylim(0.94, 1.0)
ax.set_xticks(rates)
ax.tick_params(pad=5)
ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.18), handletextpad=1.5)
fig.tight_layout(); save(fig, 'fig1_wdbc_missing_curves'); plt.close(fig)

# Fig2
r2 = json.loads((results_dir / 'results2.json').read_text())
ks = [4, 8, 12, 20, 30]
clean, miss, clean_e, miss_e = [], [], [], []
for k in ks:
    m, s = mean_std(r2['ablation_LR'][str(k)]['clean']); clean.append(m); clean_e.append(s)
    m, s = mean_std(r2['ablation_LR'][str(k)]['miss30']); miss.append(m); miss_e.append(s)
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.errorbar(ks, clean, yerr=clean_e, marker='o', capsize=3, color=BLUE, label='Clean', linewidth=1.5)
ax.errorbar(ks, miss, yerr=miss_e, marker='s', capsize=3, color=TEAL, label='30% missing', linewidth=1.5)
ax.set_xlabel('Number of features (top-k, training-fold RF importance)')
ax.set_ylabel('WDBC LR AUC (5-fold mean ± SD)')
# Bottom 0.97 (not 0.98): the k=4 30% error bar extends below 0.98 and
# previously collided with the '4' tick label. Axis is truncated, as labelled.
ax.set_ylim(0.97, 1.0)
ax.set_xticks(ks)
ax.tick_params(pad=5)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, handletextpad=1.5)
fig.tight_layout(); save(fig, 'fig2_ablation'); plt.close(fig)

# Fig3 (multi-panel, panel labels a/b, error bars = 1 SD)
r4 = json.loads((results_dir / 'results4.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
ax = axes[0]
labels = ['Median', 'KNN k=5', 'Iterative']
v30, e30, v50, e50 = [], [], [], []
for lab in labels:
    key = {'Median': 'median', 'KNN k=5': 'KNN_k5', 'Iterative': 'Iterative'}[lab]
    m, s = mean_std(r4['WDBC_fill_control_AUC'][key]['LR']['0.3']); v30.append(m); e30.append(s)
    m, s = mean_std(r4['WDBC_fill_control_AUC'][key]['LR']['0.5']); v50.append(m); e50.append(s)
x = list(range(len(labels)))
ax.bar([i - 0.18 for i in x], v30, width=0.36, color=BLUE, label='30%', yerr=e30, capsize=3)
ax.bar([i + 0.18 for i in x], v50, width=0.36, color=TEAL, label='50%', yerr=e50, capsize=3)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('AUC (axis truncated at 0.95)')
ax.set_ylim(0.95, 1.0)
ax.set_title('WDBC LR, by imputation')
ax.tick_params(pad=5)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, handletextpad=1.5)
panel_label(ax, 'a')
ax = axes[1]
labels2 = ['Median', 'KNN k=5']
w30, f30, w50, f50 = [], [], [], []
for k in ['median', 'KNN_k5']:
    m, s = mean_std(r4['Digits_fill_control_Accuracy_LR'][k]['LR']['0.3']); w30.append(m); f30.append(s)
    m, s = mean_std(r4['Digits_fill_control_Accuracy_LR'][k]['LR']['0.5']); w50.append(m); f50.append(s)
x2 = list(range(len(labels2)))
ax.bar([i - 0.18 for i in x2], w30, width=0.36, color=BLUE, label='30%', yerr=f30, capsize=3)
ax.bar([i + 0.18 for i in x2], w50, width=0.36, color=TEAL, label='50%', yerr=f50, capsize=3)
ax.set_xticks(x2); ax.set_xticklabels(labels2)
ax.set_ylabel('Accuracy (axis truncated at 0.60)')
ax.set_ylim(0.60, 1.0)
ax.set_title('Digits LR, by imputation')
ax.tick_params(pad=5)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, handletextpad=1.5)
panel_label(ax, 'b')
fig.tight_layout()
require_matplotlib_panel_alignment(
    fig, json_out=str(outdir / 'fig3_fill_control.alignment.json'),
    overlay_svg=str(outdir / 'fig3_fill_control.alignment.svg'),
    tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
save(fig, 'fig3_fill_control'); plt.close(fig)
print('wrote', sorted(p.name for p in outdir.glob('fig*.svg')) )
