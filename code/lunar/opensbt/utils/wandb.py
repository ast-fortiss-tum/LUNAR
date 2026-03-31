import io
import time
import os
import json
from typing import Callable, Optional, List, Dict, Tuple
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
import tqdm
from pymoo.core.algorithm import Algorithm

import wandb
from wandb.sdk.artifacts.artifact import Artifact
from wandb.sdk.wandb_run import Run
from wandb.apis.public import Api
import re

def wandb_log_csv(filename):
    # log only if file is not empty
    if (
        Path(filename).exists()
        and Path(filename).is_file()
        and os.stat(filename).st_size > 0
    ):
        try:
            df = pd.read_csv(filename)
            metric_table = wandb.Table(dataframe=df)
            metric_table_artifact = wandb.Artifact("metric_history", type="dataset")
            metric_table_artifact.add(metric_table, "metric_table")
            metric_table_artifact.add_file(filename)
            wandb.log({"log": metric_table})
            wandb.log_artifact(metric_table_artifact, name=filename)
        except io.UnsupportedOperation:
            print(f"Cannot log {filename}. Check if it is in .csv format.")
    else:
        print(f"{filename} does not exist or it is empty.")



def logging_callback(algorithm: Algorithm):
    all_population = algorithm.pop
    critical_all, _ = all_population.divide_critical_non_critical()
    wandb.log(
        {
            "population_size": len(all_population),
            "failures": len(critical_all),
            "critical_ratio": len(critical_all) / len(all_population),
            "timestamp": time.time()
        }
    )

def logging_callback_archive(algorithm: Algorithm):
    if hasattr(algorithm, "archive") and algorithm.archive is not None:
        all_population = algorithm.archive
        critical_all, _ = all_population.divide_critical_non_critical()
        wandb.log(
            {
                "test_size": len(all_population),
                "failures": len(critical_all),
                "critical_ratio": len(critical_all) / len(all_population) if len(all_population) > 0 else 0.0,
                "timestamp": time.time()
            }
        )

class TableCallback:
    def __init__(self):
        self.table = wandb.Table(columns=["test_size", "failures", "critical_ratio", "timestamp"],
                                 log_mode="MUTABLE")

    def log(self, algorithm: Algorithm):
        all_population = algorithm.archive
        critical_all, _ = all_population.divide_critical_non_critical()
        self.table.add_data(
            len(all_population),
            len(critical_all),
            len(critical_all) / len(all_population) if len(all_population) > 0 else 0.0,
            time.time()
        )
        wandb.log({"Summary Table": self.table})

    def __getstate__(self):
        """Return state for pickling (skip unpicklable parts)."""
        state = self.__dict__.copy()
        state["table"] = None
        return state

    def __setstate__(self, state):
        """Recreate skipped attributes after unpickling."""
        self.__dict__.update(state)
        if self.table is None:
            self.table = wandb.Table(columns=["test_size", "failures", "critical_ratio", "timestamp"],
                                 log_mode="MUTABLE")
import re

DEFAULT_LLM_WHEN_MISSING = "gpt-5-chat"

def get_sut(project, run, tags):
    """
    ALWAYS returns: "<sut_type>_<llm>" (lowercased)

    If run name has no llm token before the sample-count token (<digits>n|n<digits>),
    llm defaults to DEFAULT_LLM_WHEN_MISSING.
    """
    if project in ("SafeLLM",):
        sut_type = (tags.get("sut_type") or tags.get("sut", "unknown")).strip().lower()
        llm = (tags.get("llm") or tags.get("model") or DEFAULT_LLM_WHEN_MISSING).strip().lower()
        return f"{sut_type}_{llm}"

    parts = run.name.split("_")

    # find first "<digits>n" or "n<digits>"
    n_idx = None
    for i, tok in enumerate(parts):
        if re.fullmatch(r"(?:\d+n|n\d+)", tok):
            n_idx = i
            break
    if n_idx is None or n_idx < 2:
        raise Exception(f"Could not parse run name: {run.name}")

    # head = [algo, sut_type... , (optional llm)]
    head = parts[:n_idx]
    after_algo = head[1:]  # drop algo

    if len(after_algo) == 1:
        # only sut_type present
        sut_type = after_algo[0].strip().lower()
        llm = DEFAULT_LLM_WHEN_MISSING
    else:
        # assume last token is llm
        sut_type = "_".join(after_algo[:-1]).strip().lower()
        llm = after_algo[-1].strip().lower()

        # OPTIONAL safety: if what we thought is llm doesn't look like a model id,
        # treat it as part of sut_type and default the llm.
        if not re.search(r"(gpt|claude|gemini|llama|mistral|deepseek|qwen)", llm, re.IGNORECASE):
            sut_type = "_".join(after_algo).strip().lower()
            llm = DEFAULT_LLM_WHEN_MISSING

    return f"{sut_type}_{llm}"

def download_run_artifacts(path: str, 
                           filter_runs: Optional[Callable[[List[Run]], List[Run]]] = None) -> Dict[str, Dict[str, List[str]]]:
    runs = Api().runs(
        path,
        per_page = 1000
    )
    project = path.split("/")[-1]
    if filter_runs is not None:
        runs = filter_runs(runs)

    all_paths = defaultdict(lambda: defaultdict(list))
    missing_runs = []
    for run in runs:
        tags = {t.split(":")[0]: t.split(":")[1] for t in run.tags}
        
        local_path = os.path.join(
            "results",
            "artifacts",
            path,
            tags.get("sut", "unknown sut"),
            tags.get("algorithm", "unknown algo"),
            tags.get("seed", "no seed"),
            # run.name,
            run.id,
        )
        os.makedirs(local_path, exist_ok=True)
        if not os.path.exists(os.path.join(local_path, "run_history.csv")):
            run.history().to_csv(os.path.join(local_path, "run_history.csv"), index=False)
        all_paths[get_sut(project, run, tags)][(tags.get("algorithm", "unknown algo"), tags.get("features", "default"))].append(local_path)
        if not os.path.exists(os.path.join(local_path, "config.txt")):
            missing_runs.append((run, local_path))
    for run, path in tqdm.tqdm(missing_runs):
        print("path:",  path)
        artifacts = run.logged_artifacts()
        # if len(artifacts) > 5:
        #     previous = artifacts[0]
        #     for a in artifacts:
        #         if a.type == "output":
        #             a.download(path)
        #             previous.download(path)
        #             break
        #         else:
        #             previous = a

        # download all artifacts to ensure summary_results.csv is present
        for a in artifacts:
            a.download(path)
            with open(os.path.join(path, "tags.json"), "w") as f:
                json.dump(tags, f)
            run.history().to_csv(os.path.join(local_path, "run_history.csv"), index=False)
    return all_paths

def download_run_artifacts_relative(
    wb_project_path: str,
    local_root: str = "./opentest",
    filter_runs: Optional[Callable[[List[Run]], List[Run]]] = None,
    one_per_name: bool = False
) -> Dict[str, str]:
    api = Api()
    runs = api.runs(wb_project_path, per_page=1000)
    project_name = wb_project_path.split("/")[-1]

    if filter_runs is not None:
        runs = filter_runs(runs)

    runs = list(runs)

    # Optionally group by run.name
    if one_per_name:
        grouped = defaultdict(list)
        for run in runs:
            grouped[run.name].append(run)
        selected_runs = [max(group, key=lambda r: r.created_at) for group in grouped.values()]
    else:
        selected_runs = runs

    all_paths = defaultdict(lambda: defaultdict(list))

    # Track run.name values we have already satisfied (either found locally or downloaded)
    picked_names = set()

    for run in tqdm.tqdm(selected_runs):
        tags = {t.split(":")[0]: t.split(":")[1] for t in run.tags}
        sut_name = get_sut(project_name, run, tags)

        algo = tags.get("algorithm", "unknown_algo")
        feats = tags.get("features", "default")
        seed = tags.get("seed", "no_seed")

        run_name_dir = os.path.join(
            local_root,
            project_name,
            "artifacts",
            sut_name,
            algo,
            seed,
            run.name,
        )

        # (1) If one_per_name: if we already picked something for this name, skip
        if one_per_name and run.name in picked_names:
            continue

        # (2) If there is already a stored run under the run.name folder:
        #     remember its path, add to all_paths, and mark the name as picked.
        if os.path.isdir(run_name_dir) and any(os.scandir(run_name_dir)):
            # choose the first non-empty run.id directory found
            chosen = None
            for entry in os.scandir(run_name_dir):
                if entry.is_dir() and any(os.scandir(entry.path)):
                    chosen = entry.path
                    break

            if chosen is not None:
                all_paths[sut_name][(algo, feats)].append(chosen)
                if one_per_name:
                    picked_names.add(run.name)
                continue

        # (3) Otherwise download this run
        local_path = os.path.join(run_name_dir, run.id)
        os.makedirs(local_path, exist_ok=True)

        # If already downloaded for this run.id, just record it
        if any(os.scandir(local_path)):
            all_paths[sut_name][(algo, feats)].append(local_path)
            if one_per_name:
                picked_names.add(run.name)
            continue

        for artifact in run.logged_artifacts():
            artifact.download(local_path)

        with open(os.path.join(local_path, "tags.json"), "w") as f:
            json.dump(tags, f)

        history_df = run.history(keys=None, pandas=True)
        history_df.to_csv(os.path.join(local_path, "run_history.csv"), index=False)

        all_paths[sut_name][(algo, feats)].append(local_path)
        if one_per_name:
            picked_names.add(run.name)

    return all_paths

def get_summary(artifact_directory_path: str):
    try:
        summary_table = pd.read_csv(os.path.join(artifact_directory_path, "summary_results.csv"))
        summary = {
            attr: value for attr, value in
            zip(summary_table.Attribute, summary_table.Value)
        }
    except Exception as e:
        print("Summary file not available.")
        return None
    return summary


def get_run_table(artifact_directory_path: str, freq: str = "1min", interpolate_duplicates: bool = True):
    if os.path.exists(os.path.join(artifact_directory_path, "Summary Table.table.json")):
        with open(os.path.join(artifact_directory_path, "Summary Table.table.json")) as f:
            table_dict = json.load(f)
            df = pd.DataFrame(data=table_dict["data"], columns=table_dict["columns"])
    else:
        df = pd.read_csv(os.path.join(artifact_directory_path, "run_history.csv"))
        df = df[["failures", "test_size", "critical_ratio", "timestamp"]].dropna()
    summary = get_summary(artifact_directory_path)
    if interpolate_duplicates:
        no_duplicates_ratio = float(summary["Number Critical Scenarios (duplicate free)"]) \
            / float(summary["Number Critical Scenarios"])
        df["failures"] = df.failures * no_duplicates_ratio
        df["critical_ratio"] = df["critical_ratio"] * no_duplicates_ratio
    df["time"] = df.timestamp.mul(1e9).apply(pd.Timestamp)
    df.time = df.time - df.time.iloc[0]
    df = df.set_index("time")
    df = pd.concat([df, df.resample(freq).asfreq()]).sort_index().interpolate("index").resample(freq).first()
    return df