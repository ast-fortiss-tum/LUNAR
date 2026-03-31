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
from judge_eval_mt_cc.prompts import JUDGE_PROMPT_CONSTRAINTS_CC
from llm.config import DEBUG
from llm.llms import LLMType, pass_llm
from json_repair import repair_json
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import tiktoken

# ── Cost & Token Tracking ─────────────────────────────────────────────────────
# Pricing per 1M tokens (as of 2024)
MODEL_PRICING = {
    LLMType.GPT_4O: {"input": 2.50, "output": 10.00},
    LLMType.GPT_4: {"input": 30.00, "output": 60.00},
    LLMType.GPT_5_MINI: {"input": 0.50, "output": 1.50},
    LLMType.GPT_4O_MINI: {"input": 0.50, "output": 1.50},
    LLMType.CLAUDE_35_SONNET: {"input": 3.00, "output": 15.00},
    LLMType.CLAUDE_3_HAIKU: {"input": 0.25, "output": 1.25}
}


def count_tokens(text: str, model_type: LLMType = LLMType.GPT_4O) -> int:
    """
    Count tokens in text using tiktoken.
    Falls back to approximation for non-OpenAI models.
    """
    try:
        # Map model types to tiktoken encoding names
        encoding_map = {
            LLMType.GPT_4O: "o200k_base",
            LLMType.GPT_4: "cl100k_base",
        }
        
        encoding_name = encoding_map.get(model_type, "cl100k_base")
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception:
        # Fallback: rough approximation (1 token ≈ 4 characters)
        return len(text) // 4


def calculate_cost(input_tokens: int, output_tokens: int, model_type: LLMType) -> float:
    """Calculate cost in USD for given token counts."""
    if model_type not in MODEL_PRICING:
        return 0.0
    
    pricing = MODEL_PRICING[model_type]
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    
    return input_cost + output_cost


class CostTracker:
    """Track tokens, costs, and time across evaluations."""
    
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
    """
    Computes accuracy, precision, recall, and F1 between
    expected and produced dimension scores.
    Uses weighted averaging to handle multi-class scores gracefully.
    """
    accuracy  = accuracy_score(expected, produced)
    precision = precision_score(expected, produced, average="weighted", zero_division=0)
    recall    = recall_score(expected, produced, average="weighted", zero_division=0)
    f1        = f1_score(expected, produced, average="weighted", zero_division=0)

    return {
        "accuracy":  round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall":    round(float(recall), 4),
        "f1":        round(float(f1), 4),
    }


def compute_scores_per_dimension(expected: list, produced: list, dim_index: int) -> dict:
    """Compute metrics for a single dimension (Clarity=0, Request-orientedness=1)."""
    exp_dim  = [e[dim_index] for e in expected]
    prod_dim = [p[dim_index] for p in produced]
    return compute_scores(exp_dim, prod_dim)


# ── LLM Judge ─────────────────────────────────────────────────────────────────
def llm_validator_from_turns(
        turns: list[dict],
        n=1,
        llm_type=None,
        aggregator="mean",
        prompt_template=JUDGE_PROMPT_CONSTRAINTS_CC,
        cost_tracker: CostTracker = None
):
    """
    Variant of llm_validator_conversation that accepts raw turns directly,
    skipping the need to instantiate a Conversation object.

    Each turn is expected to be: {"user": "...", "system": "..."}
    """
    if llm_type is None:
        from llm.config import LLM_VALIDATOR
        llm_type = LLMType(LLM_VALIDATOR)

    assert n >= 1
    answers = []
    justifications = []

    # Format the dialogue history string directly from turns
    history_str = "\n".join(
        f"User: {t['user']}\nSystem: {t['system']}"
        for t in turns
    )
    prompt_eval = prompt_template.format(history=history_str)

    # Count input tokens
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
                            # Track time
                            start_time = time.time()
                            
                            raw_answer = pass_llm(
                                prompt_eval,
                                temperature=0.5 if n > 1 else 0,
                                max_tokens=4096,
                                llm_type=llm_type
                            )
                            
                            # Calculate duration
                            duration = time.time() - start_time
                            
                            # Count output tokens
                            output_tokens = count_tokens(raw_answer, llm_type)
                            
                            # Calculate cost
                            cost = calculate_cost(input_tokens, output_tokens, llm_type)
                            
                            # Track costs and time
                            if cost_tracker:
                                cost_tracker.add(input_tokens, output_tokens, cost, duration)
                            
                            response_json = json.loads(repair_json(raw_answer))

                            scores_dict = response_json.get("scores", {})
                            scores = [
                                float(scores_dict.get("Clarity", 0)),
                                float(scores_dict.get("Request-orientedness", 0))
                            ]

                            answers.append(scores)
                            justifications.append([
                                response_json.get("justification_clarity", ""),
                                response_json.get("justification_request_orientedness", "")
                            ])
                            success = True
                        except (json.JSONDecodeError, KeyError, ValueError) as e:
                            attempts += 1
                            print(f"Parsing failed (attempt {attempts}/5): {e}")
                            time.sleep(0.5)
                    if not success:
                        print("Failed to parse scores after 5 attempts. Using default value.")
                        answers.append([0.0, 0.0])
                        if cost_tracker:
                            # Still count the failed attempt
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
def evaluate_file(file_path: str, n: int = 1, aggregator: str = "mean",
                  llm_type: LLMType = LLMType.GPT_4O, 
                  cost_tracker: CostTracker = None) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    expected_dims = metadata.get("judge_dimensions", {})

    expected_clarity = expected_dims.get("Clarity")
    expected_request = expected_dims.get("Request-orientedness")

    turns = data.get("turns", [])

    # Direct call — no Conversation object needed
    final_scores, raw_answers, justifications = llm_validator_from_turns(
        turns=turns,
        n=n,
        aggregator=aggregator,
        llm_type=llm_type,
        cost_tracker=cost_tracker
    )

    produced_clarity = final_scores[0]
    produced_request = final_scores[1]

    clarity_match = produced_clarity == expected_clarity
    request_match = produced_request == expected_request

    return {
        "file": os.path.basename(file_path),
        "individual_id": metadata.get("individual_id"),
        "expected": {
            "Clarity": expected_clarity,
            "Request-orientedness": expected_request,
        },
        "produced": {
            "Clarity": produced_clarity,
            "Request-orientedness": produced_request,
        },
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
def write_report(results: list[dict], agg_metrics: dict, agg_per_dim: dict,
                 out_folder: str, n_files: int, model_name: str,
                 cost_summary: dict):
    """Write a JSON report summarising all results and aggregate metrics."""
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
    """Write a grouped bar chart of metrics overall and per dimension."""
    os.makedirs(out_folder, exist_ok=True)

    metric_names = ["accuracy", "precision", "recall", "f1"]
    labels       = ["Overall", "Clarity", "Request-orientedness"]
    values       = [
        [agg_metrics[m]                            for m in metric_names],
        [agg_per_dim["Clarity"][m]                 for m in metric_names],
        [agg_per_dim["Request-orientedness"][m]    for m in metric_names],
    ]

    x     = np.arange(len(metric_names))
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
                ha="center", va="bottom", fontsize=8
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


def write_comparison_plot(all_model_results: dict, out_folder: str):
    """Create a comparison plot across all models."""
    os.makedirs(out_folder, exist_ok=True)
    
    models = list(all_model_results.keys())
    metric_names = ["accuracy", "precision", "recall", "f1"]
    
    # Extract overall metrics for each model
    model_metrics = {
        model: [data["agg_metrics"][m] for m in metric_names]
        for model, data in all_model_results.items()
    }
    
    x = np.arange(len(metric_names))
    width = 0.8 / len(models)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    for i, (model, metrics) in enumerate(model_metrics.items()):
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, metrics, width, label=model, color=colors[i], alpha=0.85)
        
        for bar, val in zip(bars, metrics):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=7, rotation=0
            )
    
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Overall Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metric_names])
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plot_path = os.path.join(out_folder, "model_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Comparison plot written to: {plot_path}")


def write_cost_comparison_plot(all_model_results: dict, out_folder: str):
    """Create a cost comparison plot across all models."""
    os.makedirs(out_folder, exist_ok=True)
    
    models = []
    costs = []
    f1_scores = []
    
    for model, data in all_model_results.items():
        models.append(model)
        costs.append(data["cost_summary"]["total_cost_usd"])
        f1_scores.append(data["agg_metrics"]["f1"])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Cost comparison
    colors_cost = plt.cm.Reds(np.linspace(0.4, 0.8, len(models)))
    bars1 = ax1.bar(range(len(models)), costs, color=colors_cost, alpha=0.85)
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Total Cost (USD)")
    ax1.set_title("Cost Comparison")
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    
    for bar, cost in zip(bars1, costs):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(costs) * 0.01,
            f"${cost:.4f}",
            ha="center", va="bottom", fontsize=9
        )
    
    # Cost-effectiveness (F1 per dollar)
    cost_effectiveness = [f1 / cost if cost > 0 else 0 for f1, cost in zip(f1_scores, costs)]
    colors_eff = plt.cm.Greens(np.linspace(0.4, 0.8, len(models)))
    bars2 = ax2.bar(range(len(models)), cost_effectiveness, color=colors_eff, alpha=0.85)
    ax2.set_xlabel("Model")
    ax2.set_ylabel("F1 Score per Dollar")
    ax2.set_title("Cost-Effectiveness (Higher is Better)")
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    
    for bar, eff in zip(bars2, cost_effectiveness):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(cost_effectiveness) * 0.01,
            f"{eff:.1f}",
            ha="center", va="bottom", fontsize=9
        )
    
    plt.tight_layout()
    plot_path = os.path.join(out_folder, "cost_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Cost comparison plot written to: {plot_path}")


def write_time_comparison_plot(all_model_results: dict, out_folder: str):
    """Create a time comparison plot across all models."""
    os.makedirs(out_folder, exist_ok=True)
    
    models = []
    total_times = []
    avg_times = []
    
    for model, data in all_model_results.items():
        models.append(model)
        total_times.append(data["cost_summary"]["total_time_seconds"])
        avg_times.append(data["cost_summary"]["avg_time_per_call_seconds"])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Total time comparison
    colors_total = plt.cm.Blues(np.linspace(0.4, 0.8, len(models)))
    bars1 = ax1.bar(range(len(models)), total_times, color=colors_total, alpha=0.85)
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Total Time (seconds)")
    ax1.set_title("Total Processing Time")
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    
    for bar, t_time in zip(bars1, total_times):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(total_times) * 0.01,
            f"{t_time:.1f}s",
            ha="center", va="bottom", fontsize=9
        )
    
    # Average time per call
    colors_avg = plt.cm.Purples(np.linspace(0.4, 0.8, len(models)))
    bars2 = ax2.bar(range(len(models)), avg_times, color=colors_avg, alpha=0.85)
    ax2.set_xlabel("Model")
    ax2.set_ylabel("Average Time per Call (seconds)")
    ax2.set_title("Average Processing Time per Evaluation")
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    
    for bar, a_time in zip(bars2, avg_times):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(avg_times) * 0.01,
            f"{a_time:.2f}s",
            ha="center", va="bottom", fontsize=9
        )
    
    plt.tight_layout()
    plot_path = os.path.join(out_folder, "time_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Time comparison plot written to: {plot_path}")


def write_comparison_report(all_model_results: dict, out_folder: str):
    """Write a comprehensive comparison report for all models."""
    os.makedirs(out_folder, exist_ok=True)
    
    comparison = {
        "generated_at": datetime.now().isoformat(),
        "models_compared": list(all_model_results.keys()),
        "summary": {
            model: {
                "overall_metrics": data["agg_metrics"],
                "per_dimension": data["agg_per_dim"],
                "passed": data["passed"],
                "failed": data["failed"],
                "total": data["total"],
                "cost_summary": data["cost_summary"],
            }
            for model, data in all_model_results.items()
        },
        "ranking": {
            metric: sorted(
                [(model, data["agg_metrics"][metric]) for model, data in all_model_results.items()],
                key=lambda x: x[1],
                reverse=True
            )
            for metric in ["accuracy", "precision", "recall", "f1"]
        },
        "cost_ranking": sorted(
            [(model, data["cost_summary"]["total_cost_usd"]) for model, data in all_model_results.items()],
            key=lambda x: x[1]
        ),
        "time_ranking": sorted(
            [(model, data["cost_summary"]["avg_time_per_call_seconds"]) for model, data in all_model_results.items()],
            key=lambda x: x[1]
        ),
        "cost_effectiveness_ranking": sorted(
            [
                (
                    model, 
                    data["agg_metrics"]["f1"] / data["cost_summary"]["total_cost_usd"] 
                    if data["cost_summary"]["total_cost_usd"] > 0 else 0
                ) 
                for model, data in all_model_results.items()
            ],
            key=lambda x: x[1],
            reverse=True
        )
    }
    
    report_path = os.path.join(out_folder, "comparison_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n📄 Comparison report written to: {report_path}")
    
    # Print summary table
    print("\n" + "=" * 140)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 140)
    print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Pass Rate':>10} {'Cost (USD)':>12} {'Avg Time':>10} {'F1/$':>8}")
    print("-" * 140)
    
    for model, data in all_model_results.items():
        pass_rate = data["passed"] / data["total"] if data["total"] > 0 else 0
        cost = data["cost_summary"]["total_cost_usd"]
        avg_time = data["cost_summary"]["avg_time_per_call_seconds"]
        f1_per_dollar = data["agg_metrics"]["f1"] / cost if cost > 0 else 0
        
        print(
            f"{model:<20} "
            f"{data['agg_metrics']['accuracy']:>10.4f} "
            f"{data['agg_metrics']['precision']:>10.4f} "
            f"{data['agg_metrics']['recall']:>10.4f} "
            f"{data['agg_metrics']['f1']:>10.4f} "
            f"{pass_rate:>10.2%} "
            f"${cost:>11.4f} "
            f"{avg_time:>9.2f}s "
            f"{f1_per_dollar:>8.1f}"
        )
    print("=" * 140)
    
    # Print cost breakdown
    print("\n" + "=" * 140)
    print("COST & TIME BREAKDOWN")
    print("=" * 140)
    print(f"{'Model':<20} {'Calls':>8} {'Input Tokens':>15} {'Output Tokens':>15} {'Avg In/Call':>12} {'Avg Out/Call':>13} {'Total Time':>12}")
    print("-" * 140)
    
    for model, data in all_model_results.items():
        cs = data["cost_summary"]
        print(
            f"{model:<20} "
            f"{cs['total_calls']:>8} "
            f"{cs['input_tokens']:>15,} "
            f"{cs['output_tokens']:>15,} "
            f"{cs['avg_input_tokens_per_call']:>12.1f} "
            f"{cs['avg_output_tokens_per_call']:>13.1f} "
            f"{cs['total_time_seconds']:>11.1f}s"
        )
    print("=" * 140)


# ── Run over folder ───────────────────────────────────────────────────────────
def run_evaluation_single_model(folder_path: str, out_folder: str, n: int, 
                                aggregator: str, llm_type: LLMType, 
                                num_files: int = None):
    """Run evaluation for a single model."""
    json_files = sorted(glob.glob(os.path.join(folder_path, "*.json")))

    if not json_files:
        print(f"No JSON files found in: {folder_path}")
        return None

    if num_files:
        json_files = json_files[:num_files]

    results      = []
    all_expected = []
    all_produced = []
    cost_tracker = CostTracker()

    model_name = llm_type.name
    
    # Track overall time for the entire model evaluation
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
                file_path, 
                n=n, 
                aggregator=aggregator, 
                llm_type=llm_type,
                cost_tracker=cost_tracker
            )
        except Exception as e:
            print(f"ERROR processing {os.path.basename(file_path)}: {e}")
            continue

        results.append(result)
        all_expected.append([result["expected"]["Clarity"], result["expected"]["Request-orientedness"]])
        all_produced.append([result["produced"]["Clarity"], result["produced"]["Request-orientedness"]])

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

    # Calculate total model evaluation time
    model_total_time = time.time() - model_start_time

    if not results:
        print("No results to aggregate.")
        return None

    # Aggregate metrics
    flat_expected = [v for pair in all_expected for v in pair]
    flat_produced = [v for pair in all_produced for v in pair]

    agg_metrics  = compute_scores(flat_expected, flat_produced)
    agg_per_dim  = {
        "Clarity":              compute_scores_per_dimension(all_expected, all_produced, 0),
        "Request-orientedness": compute_scores_per_dimension(all_expected, all_produced, 1),
    }

    passed = sum(1 for r in results if r["match"]["all_correct"])
    failed = len(results) - passed

    cost_summary = cost_tracker.get_summary()

    print("-" * 90)
    print(f"\n{'Aggregate Metrics (overall)':}")
    print(f"  Accuracy : {agg_metrics['accuracy']}")
    print(f"  Precision: {agg_metrics['precision']}")
    print(f"  Recall   : {agg_metrics['recall']}")
    print(f"  F1       : {agg_metrics['f1']}")

    print(f"\n{'Per-Dimension Metrics':}")
    for dim, scores in agg_per_dim.items():
        print(f"  [{dim}]  Accuracy={scores['accuracy']}  Precision={scores['precision']}  Recall={scores['recall']}  F1={scores['f1']}")

    print(f"\n{'Cost & Time Summary':}")
    print(f"  Total API Calls      : {cost_summary['total_calls']}")
    print(f"  Total Input Tokens   : {cost_summary['input_tokens']:,}")
    print(f"  Total Output Tokens  : {cost_summary['output_tokens']:,}")
    print(f"  Total Tokens         : {cost_summary['total_tokens']:,}")
    print(f"  Avg Input/Call       : {cost_summary['avg_input_tokens_per_call']:.1f}")
    print(f"  Avg Output/Call      : {cost_summary['avg_output_tokens_per_call']:.1f}")
    print(f"  Total Cost           : ${cost_summary['total_cost_usd']:.4f}")
    print(f"  Avg Cost/Call        : ${cost_summary['avg_cost_per_call_usd']:.4f}")
    print(f"  Total Time (API)     : {cost_summary['total_time_seconds']:.2f}s")
    print(f"  Total Time (Overall) : {model_total_time:.2f}s")
    print(f"  Avg Time/Call        : {cost_summary['avg_time_per_call_seconds']:.2f}s")

    print(f"\nResults: {passed} passed, {failed} failed out of {len(results)} files.")

    # Write outputs
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


def run_evaluation(folder_path: str, out_folder: str = "./judge_eval_mt_cc/out",
                   n: int = 1, aggregator: str = "mean",
                   llm_types: list[LLMType] = None,
                   num_files: int = None):
    """Run evaluation across multiple models."""
    
    if llm_types is None or len(llm_types) == 0:
        llm_types = [LLMType.GPT_4O]
    
    all_model_results = {}
    
    for llm_type in llm_types:
        result = run_evaluation_single_model(
            folder_path=folder_path,
            out_folder=out_folder,
            n=n,
            aggregator=aggregator,
            llm_type=llm_type,
            num_files=num_files
        )
        
        if result:
            all_model_results[llm_type.name] = result
    
    # If multiple models, create comparison outputs
    if len(all_model_results) > 1:
        write_comparison_plot(all_model_results, out_folder)
        write_cost_comparison_plot(all_model_results, out_folder)
        write_time_comparison_plot(all_model_results, out_folder)
        write_comparison_report(all_model_results, out_folder)
    
    return all_model_results


# ── Argument Parser ───────────────────────────────────────────────────────────
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate LLM judge performance on human survey results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data_folder",
        "-d",
        type=str,
        default="./judge_eval_mt_cc/data",
        help="Path to folder containing JSON conversation files"
    )
    
    parser.add_argument(
        "--output_folder",
        "-o",
        type=str,
        default="./judge_eval_mt_cc/out",
        help="Path to output folder for report and plots"
    )
    
    parser.add_argument(
        "--num_samples",
        "-n",
        type=int,
        default=1,
        help="Number of LLM samples per conversation (for ensemble voting)"
    )
    
    parser.add_argument(
        "--aggregator",
        "-a",
        type=str,
        choices=["mean", "majority"],
        default="mean",
        help="Aggregation strategy for multiple samples: 'mean' or 'majority'"
    )
    
    parser.add_argument(
        "--llm_types",
        "-l",
        type=str,
        nargs='+',
        default=["DEEPSEEK_V3_0324"],
        help="LLM type(s) to use for evaluation (e.g., GPT_4O GPT_4_TURBO CLAUDE_SONNET_3_5). Can specify multiple models."
    )
    
    parser.add_argument(
        "--max_files",
        "-m",
        type=int,
        default=None,
        help="Maximum number of files to evaluate (None = all files)"
    )
    
    return parser.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    
    # Convert string LLM types to enums
    llm_type_enums = []
    for llm_type_str in args.llm_types:
        try:
            llm_type_enums.append(LLMType[llm_type_str])
        except KeyError:
            print(f"Error: Invalid LLM type '{llm_type_str}'")
            print(f"Available types: {', '.join([t.name for t in LLMType])}")
            exit(1)
    
    print("=" * 90)
    print("LLM Judge Evaluation - Multi-Model with Cost & Time Tracking")
    print("=" * 90)
    print(f"Data folder:    {args.data_folder}")
    print(f"Output folder:  {args.output_folder}")
    print(f"Num samples:    {args.num_samples}")
    print(f"Aggregator:     {args.aggregator}")
    print(f"LLM types:      {', '.join([t.name for t in llm_type_enums])}")
    print(f"Max files:      {args.max_files if args.max_files else 'all'}")
    print("=" * 90)
    print()
    
    run_evaluation(
        folder_path=args.data_folder,
        out_folder=args.output_folder,
        n=args.num_samples,
        aggregator=args.aggregator,
        llm_types=llm_type_enums,
        num_files=args.max_files
    )