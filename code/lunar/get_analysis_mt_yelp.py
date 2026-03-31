from __future__ import annotations
import os
os.environ["WANDB__ARTIFACT_MAX_WORKERS"] = "1"
os.environ["WANDB_CACHE_DIR"] = "C:/wandb_cache"

import re
import os
from typing import Callable, Iterable, List

from opensbt.visualization.llm_figures_navi import (
    boxplots,
    last_values_table,
    plot_boxplots_by_algorithm_raw,
    plot_metric_vs_time,
    statistics_table
)

# -----------------------------
# Config
# -----------------------------
PROJECT = "NaviYelp"
TEAM = "mt-test"
OUTPUT_DIR = f"./wandb_analysis/{PROJECT}"

ALGORITHMS = ["nsga2", "rs", "sensei"]

# Run-name / tag filtering rules
ALLOWED_MODELS = ("gpt-5-chat", "claude-4-sonnet", "gpt-4o")
REQUIRED_SUBSTRING = "ipa_yelp"
EXTRA_ALLOWED_SUBSTRING = "ipa_yelp_1"  # bypass rule (kept from your logic)
TIME_MARKERS = ("04-00-00", "04_00_00", "05-00-00", "05_00_00")
MIN_SEED_FOR_TIME_MARKERS = 1
# -----------------------------
# Helpers
# -----------------------------
def parse_tags(run) -> dict:
    """
    Convert run.tags like ["seed:12", "sut:xyz"] into {"seed": "12", "sut": "xyz"}.

    Safely ignores malformed tags.
    """
    tags = {}
    for t in getattr(run, "tags", []) or []:
        if ":" not in t:
            continue
        k, v = t.split(":", 1)
        tags[k] = v
    return tags


def is_finished(run) -> bool:
    return getattr(run, "state", None) == "finished"


def run_name(run) -> str:
    return (getattr(run, "name", "") or "").lower()


def filter_mt(runs: Iterable) -> List:
    selected = []

    for run in runs:
        if not is_finished(run):
            continue

        name = run_name(run)
        if REQUIRED_SUBSTRING not in name:
            continue

        # tags = parse_tags(run)
        # tags = parse_tags(run)
        # seed = int(tags.get("seed", 0))
        
        # Look for a number next to 'seed' in either order
        match = re.search(r'(?:seed(\d+)|(\d+)seed)', name)
        if match:
            seed_str = match.group(1) or match.group(2)
            seed = int(seed_str)
        else:
            seed = 0  # fallback if no seed found

        has_time_marker = any(marker in name for marker in TIME_MARKERS)
        has_allowed_model = any(model in name for model in ALLOWED_MODELS)

        passes_main_rule = has_time_marker and seed >= MIN_SEED_FOR_TIME_MARKERS and has_allowed_model
        passes_extra_rule = EXTRA_ALLOWED_SUBSTRING in name

        if passes_main_rule or passes_extra_rule:
            selected.append(run)

    return selected


RUN_FILTERS: dict[str, Callable] = {
    # BMW
    "TestNaviIndustry": filter_mt,
    "NaviIndustry": filter_mt,
    # YELP
    "TestNaviYelp": filter_mt,
    "NaviYelp": filter_mt,
    # MIXED
    "MultiTurnTest": filter_mt,  # fixed leading space from your original dict key
    "MultiTurnTestDiscrete": filter_mt,
    "MultiTurnDiscrete": filter_mt,
}


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    th_response = 0.65
    th_efficiency = 0.65
    th_content = 0.75
    # boxplots(algorithms=ALGORITHMS, project=PROJECT,  size=(18, 4), metric="failures", 
    #          file_name=os.path.join(OUTPUT_DIR, "failures_final"),run_filters=RUN_FILTERS)
    
    # --- Statistics tables ---
    statistics_table(
        algorithms=ALGORITHMS,
        project=PROJECT,
        metric="failures",
        path=os.path.join(OUTPUT_DIR, "failures_stats.csv"),
        run_filters=RUN_FILTERS,
        th_efficiency=th_efficiency,
        th_response=th_response,
        th_content=th_content
    )

    statistics_table(
        algorithms=ALGORITHMS,
        project=PROJECT,
        metric="critical_ratio",
        path=os.path.join(OUTPUT_DIR, "critical_ratio_stats.csv"),
        run_filters=RUN_FILTERS,
        th_efficiency=th_efficiency,
        th_response=th_response,
        th_content=th_content
    )

    # --- Time series plots (currently disabled) ---
    # plot_metric_vs_time(
    #     project=PROJECT,
    #     size=(18, 4),
    #     metric="failures",
    #     file_name=os.path.join(OUTPUT_DIR, "failures_over_time"),
    #     run_filters=RUN_FILTERS,
    #     th_efficiency=th_efficiency,
    #     th_response=th_response,
    #     th_content=th_content,
    # )
    # plot_metric_vs_time(
    #     project=PROJECT,
    #     size=(18, 4),
    #     metric="critical_ratio",
    #     file_name=os.path.join(OUTPUT_DIR, "critical_ratio_over_time"),
    #     run_filters=RUN_FILTERS,
    #     th_efficiency=th_efficiency,
    #     th_response=th_response,
    #     th_content=th_content,
    # )

    # --- Final value summary ---
    last_values_table(
        project=PROJECT,
        metrics=["failures", "critical_ratio"],
        path=os.path.join(OUTPUT_DIR, "metrics.csv"),
        run_filters=RUN_FILTERS,
        only_response=False,
        th_efficiency=th_efficiency,
        th_response=th_response,
        th_content=th_content
    )

    # --- Boxplots (raw) ---
    plot_boxplots_by_algorithm_raw(
        title="Navi-YELP",
        project=PROJECT,
        metric="failures",
        save_path=os.path.join(OUTPUT_DIR, "failures_final_raw"),
        run_filters=RUN_FILTERS,
        only_response=False,
        th_efficiency=th_efficiency,
        th_response=th_response,
        th_content=th_content
    )

    plot_boxplots_by_algorithm_raw(
        title="Navi-YELP",
        project=PROJECT,
        metric="critical_ratio",
        save_path=os.path.join(OUTPUT_DIR, "critical_ratio_final_raw"),
        run_filters=RUN_FILTERS,
        only_response=False,
        th_efficiency=th_efficiency,
        th_response=th_response,
        th_content=th_content
    )

    # --- Diversity analysis (TODO) ---
    # diversity_report(PROJECT, TEAM, input=True, output_path=OUTPUT_DIR)
    # diversity_report(PROJECT, TEAM, input=False, output_path=OUTPUT_DIR)


if __name__ == "__main__":
    main()