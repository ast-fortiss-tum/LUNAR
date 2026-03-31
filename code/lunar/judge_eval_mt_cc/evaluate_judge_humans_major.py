"""
Patch-style complete script showing the key change you requested:

- Expected (ground-truth) labels are NO LONGER read from each JSON's metadata["judge_dimensions"].
- Instead, we load majority labels from a separate CSV file and use those as expected.

NEW:
- Always create a timestamped run folder under --output_folder for version tracking
- Always write metadata.json with argv + parsed args + environment info
- Restore ✅ PASS / ❌ FAIL status icons in the per-file console output

Assumptions for the majority labels file (CSV):
- Has a column 'conversation_id' that matches the JSON filename (e.g., 'conv_0001.json')
  OR matches the JSON basename (we use basename matching).
- Has columns:
    - majority_user_clarity
    - majority_user_request_orientedness
  Optionally:
    - majority_is_critical_user (not used for the 2-dim judge eval unless you extend it)
"""

import json
import os
import glob
from collections import Counter
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import platform
import sys
import subprocess

import pandas as pd

from judge_eval_mt_cc.prompts import JUDGE_PROMPT_CONSTRAINTS_CC
from llm.config import DEBUG
from llm.llms import LLMType, pass_llm
from json_repair import repair_json
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import tiktoken


# ── Run folder + metadata (NEW) ───────────────────────────────────────────────
def create_run_folder(base_output_folder: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(base_output_folder, ts)
    os.makedirs(run_folder, exist_ok=False)
    return run_folder


def _try_git_commit(cwd: str) -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def write_metadata_json(run_folder: str, args: argparse.Namespace) -> str:
    meta = {
        "run_folder": os.path.abspath(run_folder),
        "created_at_iso": datetime.now().isoformat(),
        "argv": sys.argv,
        "args": vars(args),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "system": {
            "platform": platform.platform(),
        },
        "git": {
            "commit": _try_git_commit(os.getcwd()),
        },
    }
    path = os.path.join(run_folder, "metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return path


# ── Cost & Token Tracking ─────────────────────────────────────────────────────
MODEL_PRICING = {
    LLMType.GPT_4O: {"input": 2.50, "output": 10.00},
    LLMType.GPT_4: {"input": 30.00, "output": 60.00},
    LLMType.GPT_5_MINI: {"input": 0.50, "output": 1.50},
    LLMType.GPT_4O_MINI: {"input": 0.50, "output": 1.50},
    LLMType.CLAUDE_35_SONNET: {"input": 3.00, "output": 15.00},
    LLMType.CLAUDE_3_HAIKU: {"input": 0.25, "output": 1.25},
}


def count_tokens(text: str, model_type: LLMType = LLMType.GPT_4O) -> int:
    try:
        encoding_map = {
            LLMType.GPT_4O: "o200k_base",
            LLMType.GPT_4: "cl100k_base",
        }
        encoding_name = encoding_map.get(model_type, "cl100k_base")
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4


def calculate_cost(input_tokens: int, output_tokens: int, model_type: LLMType) -> float:
    if model_type not in MODEL_PRICING:
        return 0.0
    pricing = MODEL_PRICING[model_type]
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


class CostTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_cost = 0.0
        self.total_time = 0.0
        self.calls = 0

    def add(self, input_tokens: int, output_tokens: int, cost: float, duration: float):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_cost += cost
        self.total_time += duration
        self.calls += 1

    def get_summary(self) -> dict:
        return {
            "total_calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "total_time_seconds": round(self.total_time, 2),
            "avg_input_tokens_per_call": round(self.input_tokens / self.calls, 2) if self.calls > 0 else 0,
            "avg_output_tokens_per_call": round(self.output_tokens / self.calls, 2) if self.calls > 0 else 0,
            "avg_cost_per_call_usd": round(self.total_cost / self.calls, 4) if self.calls > 0 else 0,
            "avg_time_per_call_seconds": round(self.total_time / self.calls, 2) if self.calls > 0 else 0,
        }


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_scores(expected: list, produced: list) -> dict:
    accuracy = accuracy_score(expected, produced)
    precision = precision_score(expected, produced, average="weighted", zero_division=0)
    recall = recall_score(expected, produced, average="weighted", zero_division=0)
    f1 = f1_score(expected, produced, average="weighted", zero_division=0)
    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
    }


def compute_scores_per_dimension(expected: list, produced: list, dim_index: int) -> dict:
    exp_dim = [e[dim_index] for e in expected]
    prod_dim = [p[dim_index] for p in produced]
    return compute_scores(exp_dim, prod_dim)


# ── Majority label loader ─────────────────────────────────────────────────────
def load_majority_labels_csv(path: str) -> dict:
    """
    Returns dict: {conversation_id (basename): {"Clarity": int, "Request-orientedness": int}}
    """
    df = pd.read_csv(path)

    required_cols = [
        "conversation_id",
        "majority_user_clarity",
        "majority_user_request_orientedness",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"majority_csv missing required columns: {missing}")

    label_map = {}
    for _, row in df.iterrows():
        conv = os.path.basename(str(row["conversation_id"]))
        label_map[conv] = {
            "Clarity": int(row["majority_user_clarity"]),
            "Request-orientedness": int(row["majority_user_request_orientedness"]),
        }
    return label_map


# ── LLM Judge ─────────────────────────────────────────────────────────────────
def llm_validator_from_turns(
    turns: list[dict],
    n=1,
    llm_type=None,
    aggregator="mean",
    prompt_template=JUDGE_PROMPT_CONSTRAINTS_CC,
    cost_tracker: CostTracker = None,
):
    if llm_type is None:
        from llm.config import LLM_VALIDATOR
        llm_type = LLMType(LLM_VALIDATOR)

    assert n >= 1
    answers = []
    justifications = []

    history_str = "\n".join(f"User: {t['user']}\nSystem: {t['system']}" for t in turns)
    prompt_eval = prompt_template.format(history=history_str)

    input_tokens = count_tokens(prompt_eval, llm_type)

    global_attempts = 0
    while global_attempts < 3:
        try:
            for _ in range(n):
                if not DEBUG and (llm_type != LLMType.MOCK):
                    attempts = 0
                    success = False
                    while attempts < 5 and not success:
                        try:
                            start_time = time.time()
                            raw_answer = pass_llm(
                                prompt_eval,
                                temperature=0.5 if n > 1 else 0,
                                max_tokens=4096,
                                llm_type=llm_type,
                            )
                            duration = time.time() - start_time

                            output_tokens = count_tokens(raw_answer, llm_type)
                            cost = calculate_cost(input_tokens, output_tokens, llm_type)

                            if cost_tracker:
                                cost_tracker.add(input_tokens, output_tokens, cost, duration)

                            response_json = json.loads(repair_json(raw_answer))

                            scores_dict = response_json.get("scores", {})
                            scores = [
                                float(scores_dict.get("Clarity", 0)),
                                float(scores_dict.get("Request-orientedness", 0)),
                            ]

                            answers.append(scores)
                            justifications.append(
                                [
                                    response_json.get("justification_clarity", ""),
                                    response_json.get("justification_request_orientedness", ""),
                                ]
                            )
                            success = True
                        except (json.JSONDecodeError, KeyError, ValueError) as e:
                            attempts += 1
                            print(f"Parsing failed (attempt {attempts}/5): {e}")
                            time.sleep(0.5)
                    if not success:
                        print("Failed to parse scores after 5 attempts. Using default value.")
                        answers.append([0.0, 0.0])
                        if cost_tracker:
                            cost_tracker.add(input_tokens, 100, calculate_cost(input_tokens, 100, llm_type), 1.0)
                else:
                    answers.append([float(random.randint(0, 2)), float(random.randint(0, 2))])

            answers_array = np.array(answers)  # shape: (n, 2)

            if aggregator == "mean":
                mean_per_category = np.mean(answers_array, axis=0)
                final_scores = [round(v) for v in mean_per_category]
            elif aggregator == "majority":
                final_scores = []
                for dim_scores in answers_array.T:
                    rounded_scores = [round(s) for s in dim_scores]
                    counter = Counter(rounded_scores)
                    most_common = counter.most_common()
                    if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
                        final_scores.append(most_common[0][0])
                    else:
                        final_scores.append(round(np.mean(rounded_scores)))
            else:
                raise ValueError(f"Unknown aggregator: {aggregator}")

            return final_scores, answers, justifications
        except Exception as e:
            global_attempts += 1
            print(f"Validator exception: {e}")

    print("Validator fallback for turns")
    return np.array([0, 0]), [], []


# ── Single file evaluator ─────────────────────────────────────────────────────
def evaluate_file(
    file_path: str,
    majority_labels: dict,
    n: int = 1,
    aggregator: str = "mean",
    llm_type: LLMType = LLMType.GPT_4O,
    cost_tracker: CostTracker = None,
) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conv_id = os.path.basename(file_path)
    if conv_id not in majority_labels:
        raise KeyError(
            f"conversation_id '{conv_id}' not found in majority labels CSV. "
            f"Expected the CSV conversation_id to match JSON filename."
        )

    expected_clarity = majority_labels[conv_id]["Clarity"]
    expected_request = majority_labels[conv_id]["Request-orientedness"]

    metadata = data.get("metadata", {})
    turns = data.get("turns", [])

    final_scores, raw_answers, justifications = llm_validator_from_turns(
        turns=turns,
        n=n,
        aggregator=aggregator,
        llm_type=llm_type,
        cost_tracker=cost_tracker,
    )

    produced_clarity = int(final_scores[0])
    produced_request = int(final_scores[1])

    clarity_match = produced_clarity == expected_clarity
    request_match = produced_request == expected_request

    return {
        "file": os.path.basename(file_path),
        "individual_id": metadata.get("individual_id"),
        "expected": {"Clarity": expected_clarity, "Request-orientedness": expected_request},
        "produced": {"Clarity": produced_clarity, "Request-orientedness": produced_request},
        "match": {
            "Clarity": clarity_match,
            "Request-orientedness": request_match,
            "all_correct": clarity_match and request_match,
        },
        "justifications": {
            "Clarity": justifications[0][0] if justifications else "",
            "Request-orientedness": justifications[0][1] if justifications else "",
        },
        "raw_answers": raw_answers,
        "scores": compute_scores(
            expected=[expected_clarity, expected_request],
            produced=[produced_clarity, produced_request],
        ),
    }


# ── Output writers ────────────────────────────────────────────────────────────
def write_report(
    results: list[dict],
    agg_metrics: dict,
    agg_per_dim: dict,
    out_folder: str,
    n_files: int,
    model_name: str,
    cost_summary: dict,
):
    os.makedirs(out_folder, exist_ok=True)
    report = {
        "generated_at": datetime.now().isoformat(),
        "model": model_name,
        "total_files": n_files,
        "passed": sum(1 for r in results if r["match"]["all_correct"]),
        "failed": sum(1 for r in results if not r["match"]["all_correct"]),
        "cost_summary": cost_summary,
        "aggregate_metrics": {
            "overall": agg_metrics,
            "per_dimension": agg_per_dim,
        },
        "per_file": [
            {
                "file": r["file"],
                "individual_id": r["individual_id"],
                "expected": r["expected"],
                "produced": r["produced"],
                "match": r["match"],
                "scores": r["scores"],
                "justifications": r["justifications"],
            }
            for r in results
        ],
    }
    report_path = os.path.join(out_folder, f"evaluation_report_{model_name}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Report written to: {report_path}")
    return report_path


def write_bar_plot(agg_metrics: dict, agg_per_dim: dict, out_folder: str, model_name: str):
    os.makedirs(out_folder, exist_ok=True)

    metric_names = ["accuracy", "precision", "recall", "f1"]
    labels = ["Overall", "Clarity", "Request-orientedness"]
    values = [
        [agg_metrics[m] for m in metric_names],
        [agg_per_dim["Clarity"][m] for m in metric_names],
        [agg_per_dim["Request-orientedness"][m] for m in metric_names],
    ]

    x = np.arange(len(metric_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#4C72B0", "#55A868", "#C44E52"]
    for i, (label, vals, color) in enumerate(zip(labels, values, colors)):
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title(f"LLM Judge Evaluation — {model_name}")
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.capitalize() for m in metric_names])
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(out_folder, f"evaluation_metrics_{model_name}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"📊 Bar plot written to: {plot_path}")


# ── Run over folder ───────────────────────────────────────────────────────────
def run_evaluation_single_model(
    folder_path: str,
    out_folder: str,
    n: int,
    aggregator: str,
    llm_type: LLMType,
    majority_labels: dict,
    num_files: int = None,
):
    json_files = sorted(glob.glob(os.path.join(folder_path, "*.json")))
    if not json_files:
        print(f"No JSON files found in: {folder_path}")
        return None

    if num_files:
        json_files = json_files[:num_files]

    results = []
    all_expected = []
    all_produced = []
    cost_tracker = CostTracker()

    model_name = llm_type.name
    model_start_time = time.time()

    print(f"\n{'='*90}")
    print(f"Evaluating with model: {model_name}")
    print(f"{'='*90}")
    print(f"Files to evaluate: {len(json_files)} | n={n} | aggregator={aggregator}\n")
    print(f"{'File':<35} {'ID':<5} {'Exp C':>6} {'Prod C':>7} {'Exp R':>6} {'Prod R':>7} {'F1':>6}  {'Status'}")
    print("-" * 90)

    for file_path in json_files:
        try:
            result = evaluate_file(
                file_path=file_path,
                majority_labels=majority_labels,
                n=n,
                aggregator=aggregator,
                llm_type=llm_type,
                cost_tracker=cost_tracker,
            )
        except Exception as e:
            print(f"ERROR processing {os.path.basename(file_path)}: {e}")
            continue

        results.append(result)
        all_expected.append([result["expected"]["Clarity"], result["expected"]["Request-orientedness"]])
        all_produced.append([result["produced"]["Clarity"], result["produced"]["Request-orientedness"]])

        # ✅ restore pass/fail icons (what you asked for)
        status = "✅ PASS" if result["match"]["all_correct"] else "❌ FAIL"

        print(
            f"{result['file']:<35} "
            f"{str(result['individual_id']):<5} "
            f"{str(result['expected']['Clarity']):>6} "
            f"{str(result['produced']['Clarity']):>7} "
            f"{str(result['expected']['Request-orientedness']):>6} "
            f"{str(result['produced']['Request-orientedness']):>7} "
            f"{str(result['scores']['f1']):>6}  {status}"
        )

    model_total_time = time.time() - model_start_time

    if not results:
        print("No results to aggregate.")
        return None

    flat_expected = [v for pair in all_expected for v in pair]
    flat_produced = [v for pair in all_produced for v in pair]

    agg_metrics = compute_scores(flat_expected, flat_produced)
    agg_per_dim = {
        "Clarity": compute_scores_per_dimension(all_expected, all_produced, 0),
        "Request-orientedness": compute_scores_per_dimension(all_expected, all_produced, 1),
    }

    passed = sum(1 for r in results if r["match"]["all_correct"])
    failed = len(results) - passed
    cost_summary = cost_tracker.get_summary()

    print("-" * 90)
    print(f"\nAggregate Metrics (overall):")
    print(f"  Accuracy : {agg_metrics['accuracy']}")
    print(f"  Precision: {agg_metrics['precision']}")
    print(f"  Recall   : {agg_metrics['recall']}")
    print(f"  F1       : {agg_metrics['f1']}")

    print(f"\nPer-Dimension Metrics:")
    for dim, scores in agg_per_dim.items():
        print(f"  [{dim}]  Accuracy={scores['accuracy']}  Precision={scores['precision']}  Recall={scores['recall']}  F1={scores['f1']}")

    print(f"\nCost & Time Summary:")
    print(f"  Total API Calls      : {cost_summary['total_calls']}")
    print(f"  Total Cost           : ${cost_summary['total_cost_usd']:.4f}")
    print(f"  Total Time (API)     : {cost_summary['total_time_seconds']:.2f}s")
    print(f"  Total Time (Overall) : {model_total_time:.2f}s")

    print(f"\nResults: {passed} passed, {failed} failed out of {len(results)} files.")

    write_report(results, agg_metrics, agg_per_dim, out_folder, len(results), model_name, cost_summary)
    write_bar_plot(agg_metrics, agg_per_dim, out_folder, model_name)

    return {
        "results": results,
        "agg_metrics": agg_metrics,
        "agg_per_dim": agg_per_dim,
        "passed": passed,
        "failed": failed,
        "total": len(results),
        "cost_summary": cost_summary,
    }


def run_evaluation(
    folder_path: str,
    out_folder: str,
    n: int,
    aggregator: str,
    llm_types: list[LLMType],
    majority_labels: dict,
    num_files: int = None,
):
    all_model_results = {}
    for llm_type in llm_types:
        result = run_evaluation_single_model(
            folder_path=folder_path,
            out_folder=out_folder,
            n=n,
            aggregator=aggregator,
            llm_type=llm_type,
            majority_labels=majority_labels,
            num_files=num_files,
        )
        if result:
            all_model_results[llm_type.name] = result
    return all_model_results


# ── Argument Parser ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM judge performance on human survey majority labels (CSV)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--data_folder", "-d", type=str, default="./judge_eval_mt_cc/data")
    parser.add_argument("--majority_csv", type=str, required=True, help="CSV with majority labels per conversation_id")
    parser.add_argument("--output_folder", "-o", type=str, default="./judge_eval_mt_cc/out")

    parser.add_argument("--num_samples", "-n", type=int, default=1)
    parser.add_argument("--aggregator", "-a", type=str, choices=["mean", "majority"], default="mean")
    parser.add_argument("--llm_types", "-l", type=str, nargs="+", default=["GPT_4O"])
    parser.add_argument("--max_files", "-m", type=int, default=None)

    return parser.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    llm_type_enums = []
    for llm_type_str in args.llm_types:
        try:
            llm_type_enums.append(LLMType[llm_type_str])
        except KeyError:
            print(f"Error: Invalid LLM type '{llm_type_str}'")
            print(f"Available types: {', '.join([t.name for t in LLMType])}")
            raise SystemExit(1)

    # NEW: timestamped output folder + metadata.json
    base_out = args.output_folder
    run_out = create_run_folder(base_out)
    meta_path = write_metadata_json(run_out, args)

    majority_labels = load_majority_labels_csv(args.majority_csv)

    print("=" * 90)
    print("LLM Judge Evaluation - Using Majority Labels CSV")
    print("=" * 90)
    print(f"Data folder:    {args.data_folder}")
    print(f"Majority CSV:   {args.majority_csv}")
    print(f"Output folder:  {run_out}")
    print(f"Metadata:       {meta_path}")
    print(f"Num samples:    {args.num_samples}")
    print(f"Aggregator:     {args.aggregator}")
    print(f"LLM types:      {', '.join([t.name for t in llm_type_enums])}")
    print(f"Max files:      {args.max_files if args.max_files else 'all'}")
    print("=" * 90)
    print()

    run_evaluation(
        folder_path=args.data_folder,
        out_folder=run_out,
        n=args.num_samples,
        aggregator=args.aggregator,
        llm_types=llm_type_enums,
        majority_labels=majority_labels,
        num_files=args.max_files,
    )