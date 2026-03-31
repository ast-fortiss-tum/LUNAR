import argparse
import json
import os
from collections import Counter
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

# NEW: plotting
import matplotlib.pyplot as plt


def ensure_parent_dir(path: str) -> None:
    # matches your snippet
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def fleiss_kappa_from_counts(M: np.ndarray) -> float:
    if M.ndim != 2:
        raise ValueError("Counts matrix must be 2D.")
    N, _ = M.shape
    if N == 0:
        raise ValueError("No items to score.")

    n = M.sum(axis=1)
    if not np.all(n == n[0]):
        raise ValueError("Each item must have the same number of ratings (row sums must match).")
    n = int(n[0])
    if n < 2:
        raise ValueError("Need at least 2 raters for Fleiss' kappa.")

    p = M.sum(axis=0) / (N * n)
    P_i = np.sum(M * (M - 1), axis=1) / (n * (n - 1))
    P_bar = P_i.mean()
    P_e = np.sum(p ** 2)

    denom = 1 - P_e
    if np.isclose(denom, 0.0):
        return float("nan")
    return float((P_bar - P_e) / denom)


def build_counts_matrix(df: pd.DataFrame, rater_cols: List[str]) -> Tuple[np.ndarray, List[int]]:
    cats = sorted({int(v) for c in rater_cols for v in df[c].dropna().astype(int).tolist()})
    if not cats:
        raise ValueError(f"No category values found for columns: {rater_cols}")

    cat_to_idx = {c: i for i, c in enumerate(cats)}
    M = np.zeros((len(df), len(cats)), dtype=int)

    for i in range(len(df)):
        vals = df.loc[df.index[i], rater_cols].tolist()
        for v in vals:
            if pd.isna(v):
                raise ValueError(f"Missing rating at row={i} in one of columns {rater_cols}")
            M[i, cat_to_idx[int(v)]] += 1

    expected = len(rater_cols)
    if not np.all(M.sum(axis=1) == expected):
        raise ValueError("Not all rows have the expected number of ratings.")
    return M, cats


def majority_vote(values: List[int], tie_break: str = "min") -> int:
    c = Counter(values)
    best = c.most_common()
    best_count = best[0][1]
    tied = sorted([label for label, cnt in best if cnt == best_count])

    if len(tied) == 1:
        return tied[0]
    if tie_break == "min":
        return tied[0]
    if tie_break == "max":
        return tied[-1]
    raise ValueError(f"Tie in majority vote: {tied} for values={values}")


# NEW: deviation computation per item
def vote_stats(values: List[int], tie_break: str = "min") -> Dict[str, Any]:
    """
    Returns per-item vote summary.
    disagreement_rate = 1 - (max_votes / n_raters)
    """
    counts = Counter(int(v) for v in values)
    n = len(values)
    max_votes = max(counts.values())
    maj = majority_vote(values, tie_break=tie_break)
    return {
        "majority_label": int(maj),
        "majority_fraction": float(max_votes / n),
        "disagreement_rate": float(1.0 - (max_votes / n)),
        "vote_counts": dict(sorted(counts.items(), key=lambda kv: kv[0])),
    }


# NEW: plotting helper
def plot_disagreement(df_dev: pd.DataFrame, title: str, out_path: str) -> None:
    """
    Simple bar plot of disagreement_rate per conversation_id.
    If there are many items, this will be wide; it's still a direct "per-question" view.
    """
    ensure_parent_dir(out_path)

    x = np.arange(len(df_dev))
    y = df_dev["disagreement_rate"].to_numpy()

    plt.figure(figsize=(max(10, len(df_dev) * 0.35), 4.5))
    plt.bar(x, y)
    plt.ylim(0, 1)
    plt.ylabel("Disagreement rate (1 - majority_fraction)")
    plt.title(title)

    # Label a manageable number of x ticks; otherwise it becomes unreadable
    if len(df_dev) <= 40:
        plt.xticks(x, df_dev["conversation_id"], rotation=90, fontsize=8)
    else:
        plt.xticks([])  # too many labels, hide them
        plt.xlabel("conversation_id (labels hidden; see CSV)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder with rater CSV files")
    ap.add_argument("--output_csv", required=True, help="Output merged CSV with majority labels")
    ap.add_argument("--output_json", required=True, help="Output JSON with kappa results")
    # NEW outputs
    ap.add_argument("--deviation_csv", required=True, help="Output CSV with per-question deviation stats")
    ap.add_argument("--clarity_plot", required=True, help="Output path for clarity plot PNG")
    ap.add_argument("--request_plot", required=True, help="Output path for request-orientedness plot PNG")

    ap.add_argument("--suffix", default=".csv", help="Only include files with this suffix (default: .csv)")
    ap.add_argument("--tie_break", choices=["min", "max", "error"], default="min")
    ap.add_argument(
        "--items_mode",
        choices=["intersection", "union_drop_missing", "union_error_on_missing"],
        default="intersection",
    )
    args = ap.parse_args()

    input_dir = args.input_dir
    files = sorted(
        f for f in os.listdir(input_dir)
        if f.endswith(args.suffix) and os.path.isfile(os.path.join(input_dir, f))
    )
    if len(files) < 2:
        raise ValueError("Need at least 2 rater CSV files to compute Fleiss' kappa.")

    required = ["conversation_id", "user_clarity", "user_request_orientedness"]

    rater_dfs = []
    raters_meta = []

    for f in files:
        path = os.path.join(input_dir, f)
        df = pd.read_csv(path)

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{f} missing required columns: {missing}")

        rater_id = os.path.splitext(f)[0]
        raters_meta.append({"file": f, "rater_id": rater_id})

        df = df[required].copy()
        df = df.rename(
            columns={
                "user_clarity": f"user_clarity__{rater_id}",
                "user_request_orientedness": f"user_request_orientedness__{rater_id}",
            }
        )
        rater_dfs.append(df)

    how = "inner" if args.items_mode == "intersection" else "outer"
    merged = rater_dfs[0]
    for df in rater_dfs[1:]:
        merged = merged.merge(df, on="conversation_id", how=how)

    clarity_cols = [c for c in merged.columns if c.startswith("user_clarity__")]
    req_cols = [c for c in merged.columns if c.startswith("user_request_orientedness__")]
    rating_cols = clarity_cols + req_cols

    if args.items_mode == "union_drop_missing":
        merged = merged.dropna(subset=rating_cols).copy()
    elif args.items_mode == "union_error_on_missing":
        if merged[rating_cols].isna().any().any():
            bad = merged.loc[merged[rating_cols].isna().any(axis=1), ["conversation_id"] + rating_cols]
            raise ValueError(
                "Missing ratings exist under union_error_on_missing.\n"
                f"Example missing rows:\n{bad.head(10).to_string(index=False)}"
            )

    if len(merged) == 0:
        raise ValueError("No items left to score after merge/missing handling.")

    # Compute kappas
    M_clarity, cats_clarity = build_counts_matrix(merged, clarity_cols)
    M_req, cats_req = build_counts_matrix(merged, req_cols)

    kappa_clarity = fleiss_kappa_from_counts(M_clarity)
    kappa_request = fleiss_kappa_from_counts(M_req)

    # Majority labels
    merged["majority_user_clarity"] = [
        majority_vote([int(v) for v in row], tie_break=args.tie_break)
        for row in merged[clarity_cols].to_numpy()
    ]
    merged["majority_user_request_orientedness"] = [
        majority_vote([int(v) for v in row], tie_break=args.tie_break)
        for row in merged[req_cols].to_numpy()
    ]

    # NEW: per-question deviation file
    clarity_dev_rows = []
    req_dev_rows = []

    for idx, row in merged.iterrows():
        conv_id = row["conversation_id"]

        clarity_vals = [int(row[c]) for c in clarity_cols]
        req_vals = [int(row[c]) for c in req_cols]

        cstats = vote_stats(clarity_vals, tie_break=args.tie_break)
        rstats = vote_stats(req_vals, tie_break=args.tie_break)

        clarity_dev_rows.append(
            {
                "conversation_id": conv_id,
                "dimension": "clarity",
                **{k: v for k, v in cstats.items() if k != "vote_counts"},
                "vote_counts_json": json.dumps(cstats["vote_counts"], ensure_ascii=False),
            }
        )
        req_dev_rows.append(
            {
                "conversation_id": conv_id,
                "dimension": "request_orientedness",
                **{k: v for k, v in rstats.items() if k != "vote_counts"},
                "vote_counts_json": json.dumps(rstats["vote_counts"], ensure_ascii=False),
            }
        )

    dev_df = pd.DataFrame(clarity_dev_rows + req_dev_rows)

    # Plots: one per dimension (per question)
    clarity_dev_df = dev_df[dev_df["dimension"] == "clarity"].sort_values("conversation_id").reset_index(drop=True)
    req_dev_df = dev_df[dev_df["dimension"] == "request_orientedness"].sort_values("conversation_id").reset_index(drop=True)

    plot_disagreement(
        clarity_dev_df,
        title="Per-question disagreement — Clarity",
        out_path=args.clarity_plot,
    )
    plot_disagreement(
        req_dev_df,
        title="Per-question disagreement — Request orientedness",
        out_path=args.request_plot,
    )

    # JSON output
    results: Dict[str, Any] = {
        "input_dir": input_dir,
        "n_items_scored": int(len(merged)),
        "n_raters": int(len(files)),
        "raters": raters_meta,
        "items_mode": args.items_mode,
        "tie_break_rule_for_majority": args.tie_break,
        "dimensions": {
            "clarity": {
                "rater_columns": clarity_cols,
                "categories": cats_clarity,
                "fleiss_kappa": kappa_clarity,
            },
            "request_orientedness": {
                "rater_columns": req_cols,
                "categories": cats_req,
                "fleiss_kappa": kappa_request,
            },
        },
        "outputs": {
            "merged_csv": args.output_csv,
            "kappa_json": args.output_json,
            "deviation_csv": args.deviation_csv,
            "clarity_plot": args.clarity_plot,
            "request_plot": args.request_plot,
        },
    }

    # Ensure output folders exist (as requested)
    ensure_parent_dir(args.output_json)
    ensure_parent_dir(args.output_csv)
    ensure_parent_dir(args.deviation_csv)
    ensure_parent_dir(args.clarity_plot)
    ensure_parent_dir(args.request_plot)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    merged.to_csv(args.output_csv, index=False)
    dev_df.to_csv(args.deviation_csv, index=False)


if __name__ == "__main__":
    main()