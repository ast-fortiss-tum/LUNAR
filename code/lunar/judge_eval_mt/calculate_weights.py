import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def metric_value(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if metric == "precision":
        return float(precision_score(y_true, y_pred, zero_division=0))
    if metric == "recall":
        return float(recall_score(y_true, y_pred, zero_division=0))
    raise ValueError(f"Unknown metric: {metric}")


def infer_threshold_by_sweep(
    scores: np.ndarray,
    y_true: np.ndarray,
    metric: str = "f1",
) -> dict:
    """
    Infer a single threshold th on the linear score such that:
      predict_critical = (score > th)

    We test candidate thresholds at:
      - midpoints between sorted unique scores
      - plus just-below-min and just-above-max
    """
    uniq = np.unique(scores)
    if len(uniq) == 1:
        candidates = np.array([uniq[0]])
    else:
        mids = (uniq[:-1] + uniq[1:]) / 2.0
        candidates = np.concatenate(([uniq[0] - 1e-9], mids, [uniq[-1] + 1e-9]))

    best = None
    for th in candidates:
        y_pred = (scores > th).astype(int)
        val = metric_value(y_true, y_pred, metric)

        # tie-breakers: prefer higher recall, then higher precision, then lower threshold magnitude
        rec = metric_value(y_true, y_pred, "recall")
        prec = metric_value(y_true, y_pred, "precision")

        cand = {
            "threshold": float(th),
            "metric": metric,
            "metric_value": float(val),
            "accuracy": metric_value(y_true, y_pred, "accuracy"),
            "precision": float(prec),
            "recall": float(rec),
            "f1": metric_value(y_true, y_pred, "f1"),
        }

        if best is None:
            best = cand
            continue

        if cand["metric_value"] > best["metric_value"]:
            best = cand
        elif np.isclose(cand["metric_value"], best["metric_value"]):
            # tie-break: higher recall
            if cand["recall"] > best["recall"]:
                best = cand
            elif np.isclose(cand["recall"], best["recall"]):
                # then higher precision
                if cand["precision"] > best["precision"]:
                    best = cand

    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="merged_with_majority.csv")
    ap.add_argument("--output_json", required=True, help="Full results JSON")
    ap.add_argument("--output_rule_json", required=True, help="Rule JSON (single inferred threshold)")
    ap.add_argument("--output_csv", default=None, help="Optional per-item score CSV")
    ap.add_argument(
        "--threshold_metric",
        choices=["accuracy", "f1", "precision", "recall"],
        default="f1",
        help="Metric used to infer the threshold from data by sweeping score thresholds",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)

    needed = [
        "conversation_id",
        "majority_user_clarity",
        "majority_user_request_orientedness",
        "majority_is_critical_user",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input_csv: {missing}")

    X = df[["majority_user_clarity", "majority_user_request_orientedness"]].astype(float).to_numpy()
    y = df["majority_is_critical_user"].astype(int).to_numpy()

    # Fit logistic regression to get weights
    clf = LogisticRegression(solver="liblinear", C=1e6, fit_intercept=True)
    clf.fit(X, y)

    intercept = float(clf.intercept_[0])
    w_cl, w_ro = [float(v) for v in clf.coef_[0]]

    # Score in the exact form you requested (no intercept)
    scores = w_cl * X[:, 0] + w_ro * X[:, 1]

    # Infer a single threshold from the data
    inferred = infer_threshold_by_sweep(scores, y, metric=args.threshold_metric)
    th = float(inferred["threshold"])
    y_pred = (scores > th).astype(int)

    # Also compute ROC AUC from probabilities (optional)
    try:
        proba = clf.predict_proba(X)[:, 1]
        auc = float(roc_auc_score(y, proba))
    except Exception:
        auc = None

    full_results = {
        "input_csv": args.input_csv,
        "n_items": int(len(df)),
        "features": ["majority_user_clarity", "majority_user_request_orientedness"],
        "label": "majority_is_critical_user",
        "logistic_regression": {
            "intercept_b": intercept,
            "weight_clarity": w_cl,
            "weight_request_orientedness": w_ro,
            "score_definition": "score = weight_clarity*CL + weight_request_orientedness*RO",
            "roc_auc_prob_model": auc,
        },
        "inferred_threshold": inferred,
    }

    # Rule-only JSON (single threshold)
    rule_only = {
        "features": {
            "CL": "majority_user_clarity",
            "RO": "majority_user_request_orientedness",
        },
        "rule": {
            "score": "weight_clarity*CL + weight_request_orientedness*RO",
            "predict_is_critical_if": "score > threshold",
            "weight_clarity": w_cl,
            "weight_request_orientedness": w_ro,
            "threshold": th,
            "threshold_inferred_by": {
                "method": "sweep_linear_score_thresholds",
                "optimize_metric": args.threshold_metric,
            },
        },
        "training_metrics_at_inferred_threshold": {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
        },
    }

    ensure_parent_dir(args.output_json)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    ensure_parent_dir(args.output_rule_json)
    with open(args.output_rule_json, "w", encoding="utf-8") as f:
        json.dump(rule_only, f, indent=2)

    if args.output_csv:
        out = df[["conversation_id"]].copy()
        out["CL"] = X[:, 0]
        out["RO"] = X[:, 1]
        out["y_true_critical"] = y
        out["score"] = scores
        out["y_pred_inferred_threshold"] = y_pred
        ensure_parent_dir(args.output_csv)
        out.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()