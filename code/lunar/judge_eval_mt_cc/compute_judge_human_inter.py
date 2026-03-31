#!/usr/bin/env python3
"""
Compute inter-rater agreement (alpha, kappa) and f1 scores for llm judges vs human majority.
- judge scores: out/
- conversations: data/
- majority labels: labels/majority_labels_cc.csv
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# paths (windows-compatible using pathlib)
base_dir = Path(__file__).parent
out_dir = base_dir / "out"
data_dir = base_dir / "data"
labels_dir = base_dir / "labels"

# load human majority labels
majority_labels_path = labels_dir / "majority_labels_cc.csv"
print(f"Loading majority labels from: {majority_labels_path}")
majority_df = pd.read_csv(majority_labels_path)
print(f"Loaded {len(majority_df)} conversations with majority labels")
print(f"Columns: {majority_df.columns.tolist()}")
print(majority_df.head())

# extract human majority votes
conv_ids = majority_df['conversation_id'].values
human_clarity_majority = majority_df['majority_user_clarity'].values
human_request_majority = majority_df['majority_user_request_orientedness'].values

n_items = len(conv_ids)
print(f"\nHuman majority (Clarity): {human_clarity_majority}")
print(f"Human majority (Request): {human_request_majority}")


# --- compute cohen's kappa ---
def cohen_kappa(ratings1, ratings2):
    """
    compute cohen's kappa for two raters.
    """
    categories = sorted(set(ratings1) | set(ratings2))
    n = len(ratings1)
    
    # confusion matrix
    conf_matrix = np.zeros((len(categories), len(categories)))
    for r1, r2 in zip(ratings1, ratings2):
        i = categories.index(r1)
        j = categories.index(r2)
        conf_matrix[i, j] += 1
    
    # observed agreement
    p_o = np.trace(conf_matrix) / n
    
    # expected agreement
    row_sums = np.sum(conf_matrix, axis=1)
    col_sums = np.sum(conf_matrix, axis=0)
    p_e = np.sum(row_sums * col_sums) / (n ** 2)
    
    if p_e == 1:
        return 1.0
    
    kappa = (p_o - p_e) / (1 - p_e)
    return kappa


# --- compute f1 score ---
def compute_f1_multiclass(y_true, y_pred, categories=None):
    """
    compute macro-averaged f1 score for multiclass classification.
    """
    if categories is None:
        categories = sorted(set(y_true) | set(y_pred))
    
    f1_scores = []
    for cat in categories:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cat and yp == cat)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != cat and yp == cat)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cat and yp != cat)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    return np.mean(f1_scores)


# --- compute krippendorff's alpha ---
def krippendorff_alpha(data, level='ordinal'):
    """
    compute krippendorff's alpha for inter-rater reliability.
    data: numpy array of shape (n_items, n_raters)
    level: 'nominal', 'ordinal', 'interval', 'ratio'
    """
    n_items, n_raters = data.shape
    
    # get all unique values
    categories = sorted(set(data.flatten()))
    n_categories = len(categories)
    
    # create coincidence matrix
    coincidence = np.zeros((n_categories, n_categories))
    
    for item in range(n_items):
        ratings = data[item, :]
        n_r = len(ratings)  # number of raters for this item
        if n_r < 2:
            continue
        for i, vi in enumerate(ratings):
            for j, vj in enumerate(ratings):
                if i != j:
                    ci = categories.index(vi)
                    cj = categories.index(vj)
                    coincidence[ci, cj] += 1 / (n_r - 1)
    
    # compute observed disagreement
    n_total = np.sum(coincidence)
    if n_total == 0:
        return 1.0
    
    # marginals
    nc = np.sum(coincidence, axis=1)
    
    # expected disagreement based on metric type
    if level == 'nominal':
        # nominal: delta_ck = 1 if c != k, else 0
        delta = 1 - np.eye(n_categories)
    elif level == 'ordinal':
        # ordinal difference function
        delta = np.zeros((n_categories, n_categories))
        cumsum = np.cumsum(nc)
        for c in range(n_categories):
            for k in range(n_categories):
                if c == k:
                    delta[c, k] = 0
                else:
                    # ordinal metric: sum of all values between c and k
                    low, high = min(c, k), max(c, k)
                    g = cumsum[high] - cumsum[low] + nc[low] + nc[high]
                    delta[c, k] = (g / 2) ** 2
    elif level == 'interval':
        delta = np.zeros((n_categories, n_categories))
        for c in range(n_categories):
            for k in range(n_categories):
                delta[c, k] = (categories[c] - categories[k]) ** 2
    else:
        raise ValueError(f"Unknown level: {level}")
    
    # observed disagreement
    D_o = np.sum(coincidence * delta) / n_total
    
    # expected disagreement
    D_e = 0
    for c in range(n_categories):
        for k in range(n_categories):
            D_e += nc[c] * nc[k] * delta[c, k]
    D_e /= (n_total * (n_total - 1))
    
    if D_e == 0:
        return 1.0
    
    alpha = 1 - D_o / D_e
    return alpha


# load llm judge reports from out/
judge_files = list(out_dir.glob("evaluation_report_*.json"))
print(f"\nFound {len(judge_files)} judge report files in {out_dir}:")
for f in judge_files:
    print(f"  - {f.name}")

judge_ratings = {}
for filepath in judge_files:
    # extract judge name from filename (e.g., evaluation_report_GPT_4O.json -> GPT-4o)
    judge_name = filepath.stem.replace("evaluation_report_", "").replace("_", "-")
    
    with open(filepath, encoding='utf-8') as f:
        data = json.load(f)
    
    clarity = []
    request = []
    for entry in data['per_file']:
        clarity.append(entry['produced']['Clarity'])
        request.append(entry['produced']['Request-orientedness'])
    
    judge_ratings[judge_name] = {
        'clarity': np.array(clarity),
        'request': np.array(request)
    }
    print(f"Loaded {judge_name}: {len(clarity)} evaluations")

# build human ratings matrices for alpha computation
# extract individual rater columns from majority_df
clarity_cols = [c for c in majority_df.columns if c.startswith('user_clarity__')]
request_cols = [c for c in majority_df.columns if c.startswith('user_request_orientedness__')]

clarity_matrix = majority_df[clarity_cols].values  # shape: (n_items, n_raters)
request_matrix = majority_df[request_cols].values

print(f"\nHuman raters: {len(clarity_cols)}")
print(f"Clarity matrix shape: {clarity_matrix.shape}")
print(f"Request matrix shape: {request_matrix.shape}")

# compute metrics for each judge
print("\n" + "="*60)
print("LLM JUDGE vs HUMAN MAJORITY AGREEMENT")
print("="*60)

results = {}
for judge_name, ratings in judge_ratings.items():
    judge_clarity = ratings['clarity']
    judge_request = ratings['request']
    
    # cohen's kappa (judge vs human majority)
    kappa_clarity = cohen_kappa(human_clarity_majority.tolist(), judge_clarity.tolist())
    kappa_request = cohen_kappa(human_request_majority.tolist(), judge_request.tolist())
    
    # krippendorff's alpha (judge added as additional rater alongside all humans)
    extended_clarity = np.column_stack([clarity_matrix, judge_clarity])
    extended_request = np.column_stack([request_matrix, judge_request])
    alpha_clarity = krippendorff_alpha(extended_clarity, level='ordinal')
    alpha_request = krippendorff_alpha(extended_request, level='ordinal')
    
    # f1 scores (judge vs human majority)
    f1_clarity = compute_f1_multiclass(human_clarity_majority.tolist(), judge_clarity.tolist(), categories=[0, 1, 2])
    f1_request = compute_f1_multiclass(human_request_majority.tolist(), judge_request.tolist(), categories=[0, 1, 2])
    
    results[judge_name] = {
        'alpha_clarity': alpha_clarity,
        'alpha_request': alpha_request,
        'kappa_clarity': kappa_clarity,
        'kappa_request': kappa_request,
        'f1_clarity': f1_clarity,
        'f1_request': f1_request,
    }
    
    print(f"\n{judge_name}:")
    print(f"  Clarity:              α={alpha_clarity:.2f}, κ={kappa_clarity:.2f}, F1={f1_clarity:.2f}")
    print(f"  Request-orientedness: α={alpha_request:.2f}, κ={kappa_request:.2f}, F1={f1_request:.2f}")

# generate latex table for f1 scores
print("\n" + "="*60)
print("LATEX TABLE: F1 SCORES")
print("="*60)

latex_f1 = r"""\begin{table}[t]
\centering
\caption{$F_1$ score for each judge candidate vs.\ human majority, for clarity and request-orientedness.}
\label{tab:f1-scores}
\begin{tabular}{l cc}
\toprule
Model & Clarity & Request-orientedness \\
\midrule
"""

for judge_name in sorted(results.keys()):
    r = results[judge_name]
    latex_f1 += f"{judge_name} & {r['f1_clarity']:.2f} & {r['f1_request']:.2f} \\\\\n"

latex_f1 += r"""\bottomrule
\end{tabular}
\end{table}
"""

print(latex_f1)

# generate latex table for kappa
print("\n" + "="*60)
print("LATEX TABLE: COHEN'S KAPPA")
print("="*60)

latex_kappa = r"""\begin{table}[t]
\centering
\caption{Cohen's $\kappa$ for each judge vs.\ human majority.}
\label{tab:kappa-scores}
\begin{tabular}{l cc}
\toprule
Model & Clarity & Request-orientedness \\
\midrule
"""

for judge_name in sorted(results.keys()):
    r = results[judge_name]
    latex_kappa += f"{judge_name} & {r['kappa_clarity']:.2f} & {r['kappa_request']:.2f} \\\\\n"

latex_kappa += r"""\bottomrule
\end{tabular}
\end{table}
"""

print(latex_kappa)

# compute human baseline (fleiss' kappa for multi-rater)
def fleiss_kappa(data):
    """compute fleiss' kappa for multi-rater agreement."""
    n_items, n_raters = data.shape
    categories = sorted(set(data.flatten()))
    n_categories = len(categories)
    
    count_matrix = np.zeros((n_items, n_categories))
    for i in range(n_items):
        for j, cat in enumerate(categories):
            count_matrix[i, j] = np.sum(data[i, :] == cat)
    
    p_j = np.sum(count_matrix, axis=0) / (n_items * n_raters)
    P_i = np.zeros(n_items)
    for i in range(n_items):
        P_i[i] = (np.sum(count_matrix[i, :] ** 2) - n_raters) / (n_raters * (n_raters - 1))
    
    P_bar = np.mean(P_i)
    P_e = np.sum(p_j ** 2)
    
    if P_e == 1:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)

# human inter-rater agreement
alpha_clarity_human = krippendorff_alpha(clarity_matrix, level='ordinal')
alpha_request_human = krippendorff_alpha(request_matrix, level='ordinal')
kappa_clarity_human = fleiss_kappa(clarity_matrix)
kappa_request_human = fleiss_kappa(request_matrix)

print("\n" + "="*60)
print("HUMAN INTER-RATER AGREEMENT")
print("="*60)
print(f"Clarity:              α={alpha_clarity_human:.2f}, Fleiss' κ={kappa_clarity_human:.2f}")
print(f"Request-orientedness: α={alpha_request_human:.2f}, Fleiss' κ={kappa_request_human:.2f}")

# generate combined latex table for alpha and kappa
print("\n" + "="*60)
print("LATEX TABLE: ALPHA & KAPPA (COMBINED)")
print("="*60)

latex_alpha_kappa = r"""\begin{table}[t]
\centering
\caption{Inter-rater and judge-human agreement.}
\label{tab:agreement}
\begin{tabular}{l cc cc}
\toprule
& \multicolumn{2}{c}{Request-orientation} & \multicolumn{2}{c}{Clarity} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
Model & $\alpha$ & $\kappa$ & $\alpha$ & $\kappa$ \\
\midrule
"""

for judge_name in sorted(results.keys()):
    r = results[judge_name]
    latex_alpha_kappa += f"{judge_name} & {r['alpha_request']:.2f} & {r['kappa_request']:.2f} & {r['alpha_clarity']:.2f} & {r['kappa_clarity']:.2f} \\\\\n"

latex_alpha_kappa += r"""\midrule
\textbf{Human}$^\dagger$ & """ + f"{alpha_request_human:.2f} & {kappa_request_human:.2f} & {alpha_clarity_human:.2f} & {kappa_clarity_human:.2f}" + r""" \\
\bottomrule
\end{tabular}

\vspace{0.5em}
\footnotesize{$^\dagger$Fleiss' $\kappa$ (""" + str(len(clarity_cols)) + r""" raters); others: Cohen's $\kappa$.}
\end{table}
"""

print(latex_alpha_kappa)

# save latex tables to .tex files
latex_alpha_kappa_path = out_dir / "agreement_alpha_kappa.tex"
with open(latex_alpha_kappa_path, 'w', encoding='utf-8') as f:
    f.write(latex_alpha_kappa)
print(f"\nAlpha/Kappa table saved to: {latex_alpha_kappa_path}")

latex_f1_path = out_dir / "f1_scores.tex"
with open(latex_f1_path, 'w', encoding='utf-8') as f:
    f.write(latex_f1)
print(f"F1 table saved to: {latex_f1_path}")

# save results to json
output_path = out_dir / "judge_vs_human_metrics.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump({
        'results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
        'human_majority_clarity': human_clarity_majority.tolist(),
        'human_majority_request': human_request_majority.tolist(),
    }, f, indent=2)

print(f"\nResults saved to: {output_path}")

# generate plot
try:
    import matplotlib.pyplot as plt
    
    models = sorted(results.keys())
    f1_clarity = [results[m]['f1_clarity'] for m in models]
    f1_request = [results[m]['f1_request'] for m in models]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(models))
    width = 0.35
    
    colour_clarity = '#4ECDC4'  # teal
    colour_request = '#5B5EA6'  # purple
    
    bars1 = ax.bar(x - width/2, f1_clarity, width, label='Clarity', color=colour_clarity, edgecolor='white')
    bars2 = ax.bar(x + width/2, f1_request, width, label='Request-orientedness', color=colour_request, edgecolor='white')
    
    # add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('F1 Score — Judge vs. Human Majority', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(loc='upper right', frameon=True)
    ax.set_ylim(0, 1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plot_path = out_dir / "f1_score_plot.pdf"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / "f1_score_plot.png", dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
except ImportError:
    print("matplotlib not available, skipping plot generation")