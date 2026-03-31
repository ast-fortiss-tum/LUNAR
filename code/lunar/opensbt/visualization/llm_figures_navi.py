import json
from pathlib import Path
import pickle
import random
from typing import Dict, List, Literal, Optional, Tuple

import tqdm

from wandb import Run
from opensbt.utils.wandb import download_run_artifacts, download_run_artifacts_relative, get_summary
from opensbt.visualization.utils import AnalysisDiversity, AnalysisPlots, AnalysisException, AnalysisTables
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict, Counter
import pandas as pd
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import tqdm
from llm.model.models import Utterance
# from llm.utils.embeddings_local import get_embedding as get_embedding_local
from llm.utils.embeddings_openai import get_embedding as get_embedding_openai

import os
import json
from typing import Tuple, Optional

algo_names_map = {
    ("gs", "astral"): "ASTRAL",
    ("gs", "extended"): "T-wise",
    "gs": "T-wise",
    "random" : "LUNAR-RS",
    "rs": "LUNAR_S",
    "nsga2": "LUNAR",
    "NSGA2" : "LUNAR-GA",
    "nsga2d": "NSGAIID",
    "sensei": "SENSEI",
    "SENSEI": "SENSEI"
}

metric_names = {
    "failures": "Number of Failures",
    "critical_ratio": "Failure Rate",
}

def convert_name(run_name: str) -> str:
    """
    Convert a run name into a standardized SUT string by scanning each
    word (separated by '_') in order and including all matching identifiers.
    """
    sut_keywords = {
        # "ipa": "",
        # "yelp": "",
        "gpt-4o": "GPT-4o",
        "gpt-5-chat": "GPT-5-Chat",
        "deepseek-v3-0324": "DeepSeek-V3",
        "industry": "",
        "mistral": "Mistral-7B",
        "qwen3" : "Qwen3-8B",
        "deepseek-v2" : "DeepSeek-V2-16B",
        "mistral-instruct" : "Mistral-Instruct-7B",
        "claude-4-sonnet": "Claude-4-Sonnet",
        "gemini-3-flash-preview": "Gemini-3-Flash",
    }

    words = run_name.split("_")
    sut_parts = []
    for w in words:
        w_lower = w.lower()
        if w_lower in sut_keywords:
            print(w_lower)
            kwd = sut_keywords[w_lower]
            if kwd != "":
                sut_parts.append(kwd)
    
    return "_".join(sut_parts) if sut_parts else "unknown_sut"

def get_algo_name(algo, features):
    algo_name = algo_names_map.get((algo, features), None)
    if algo_name is None:
        algo_name = algo_names_map[algo]
    return algo_name

def capitalize(name: str) -> str:
    return "_".join(word.capitalize() for word in name.split("_"))

def get_embeddings(artifact_directory_path: str, critical_only: bool = True,
                   local: bool = True, input: bool = True) -> Tuple[List[Utterance], np.ndarray]:
    file = "embeddings_a.pkl" if input else "embeddings_a_output.pkl"
    pickle_path = os.path.join(artifact_directory_path, file)
    if os.path.exists(pickle_path) and False:
        with open(pickle_path, "rb") as f:
            return pickle.load(f)
        
    # get_embedding = get_embedding_local if local else get_embedding_openai
    get_embedding = get_embedding_openai

    print(f"Calculating embeddings for {artifact_directory_path}")
    json_path = "all_critical_utterances.json" if critical_only else "all_utterances.json"
    with open(os.path.join(artifact_directory_path, json_path), "r", encoding="utf8") as f:
        data = json.load(f)
    utterances = []
    embeddings = []
    for obj in data:
        utterance = obj.get("utterance", {})
        fitness = obj.get("fitness", {})
        
        question = utterance.get("question", "")
        answer = utterance.get("answer", "")
        if answer is None:
            answer = ""
        if question is None:
            question = ""
        answer = answer.strip()
        question = question.strip()
        is_critical = obj.get("is_critical", None)
        content_fitness = fitness.get("content_fitness", None)
        answer_fitness = fitness.get("answer_fitness", None)
        other = obj.get("other", None)
        if question != "":
            if not critical_only:
                utterances.append((utterance, fitness, other))
                embeddings.append(
                    get_embedding(question).reshape(1, -1) if input else get_embedding(answer).reshape(1, -1))
                continue
            if is_critical and (content_fitness is None or content_fitness < 1.0):
                utterances.append((utterance, fitness, other))
                embeddings.append(
                    get_embedding(question).reshape(1, -1) if input else get_embedding(answer).reshape(1, -1))
                               
    embeddings = np.concatenate(embeddings)
    with open(pickle_path, "wb") as f:
        pickle.dump((utterances, embeddings), f)
    return utterances, embeddings

def get_real_tests(
    path_name: str,
    th_content: Optional[float] = 0.75,
    th_efficiency: Optional[float] = 0.65,
    th_response: Optional[float] = 0.65,
    only_response: bool = False,
) -> Tuple[int, int]:

    print("using threshold:", th_content, th_efficiency, th_response)
    print("only_response:", only_response)

    num_real_tests = 0
    num_real_fail = 0

    if "SENSEI" in path_name.upper():
        print("Processing SENSEI run for real test calculation.")
        report_path = os.path.join(path_name, "report.json")

        with open(report_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # SENSEI format: root object with "all_evaluated_conversations"
        data = payload.get("all_evaluated_conversations", []) or []
        if isinstance(data, dict):
            data = [data]

        for obj in data:
            conversation = obj.get("conversation", {}) or {}
            turns_dict = conversation.get("turns", {}) or {}

            # turns is typically a dict like {"T1": {...}, "T2": {...}}
            if isinstance(turns_dict, dict):
                turn_iter = turns_dict.values()
            else:
                # defensive: if turns unexpectedly comes as a list
                turn_iter = turns_dict if isinstance(turns_dict, list) else []

            # real test = at least one non-empty question in any turn
            has_any_question = any(
                (t.get("question", "") or "").strip()
                for t in turn_iter
                if isinstance(t, dict)
            )
            if not has_any_question:
                continue

            num_real_tests += 1

            def check_fitness_consistency(data):
                weights = data["other"]["fitness_conversation_scores"]["weights"]
                scores_dict = data["other"]["fitness_conversation_scores"]["scores"]
                
                # fixed order to match weights
                keys = ["Clarity", "Request-Orientedness"]
                scores = [scores_dict[k] for k in keys]
                
                weighted_sum = sum(w * s for w, s in zip(weights, scores))
                reported = data["fitness_scores"]["dimensions_fitness"]
                
                return weighted_sum == reported
            # Only one fitness score exists for SENSEI
            # (you are using th_response as the threshold for it)
            dimensions_fitness = (obj.get("fitness_scores", {}) or {}).get("dimensions_fitness", None)
            
            consistent = check_fitness_consistency(obj)

            # if consistent:
            #     # not normalized
            #     dimensions_fitness = dimensions_fitness / 2  # normalization fix

            if "industry" not in path_name.lower() and dimensions_fitness is not None and \
                "control" not in path_name.lower():
                # print("dimension halfing applied for non-BMW run:", path_name)
                # print("###############################")
                dimensions_fitness = dimensions_fitness / 2  # normalization fix

            # Thresholding:
            # - if only_response=True => only check "response threshold" (th_response) against SENSEI score
            # - else keep current behavior (still uses th_response for SENSEI as you implemented)
            if th_response is None:
                if dimensions_fitness is None or dimensions_fitness < 1.0:
                    num_real_fail += 1
            else:
                if dimensions_fitness is not None and dimensions_fitness <= th_response:
                    num_real_fail += 1

        return num_real_tests, num_real_fail

    # NON-SENSEI
    all_tests = os.path.join(path_name, "all_utterances.json")

    with open(all_tests, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # summary exists only for non-SENSEI
    summary = get_summary(path_name)
    if summary is not None:
        num_crit_dupfree = int(summary["Number Critical Scenarios (duplicate free)"])
        num_crit = int(summary["Number Critical Scenarios"])
        num_all_dupfree = int(summary["Number All Scenarios (duplicate free)"])
        num_all = int(summary["Number All Scenarios"])
    else:
        num_all = len(all_tests)
        num_all_dupfree =  len(all_tests)
        num_crit = 0
        num_crit_dupfree = 0

    data = payload
    if isinstance(data, dict):
        data = [data]

    for obj in data:
        conversation = obj.get("conversation", {}) or {}
        turns = conversation.get("turns", []) or []

        # real test = at least one non-empty question across turns
        has_any_question = any(
            (t.get("question", "") or "").strip()
            for t in turns
            if isinstance(t, dict)
        )
        if not has_any_question:
            continue

        num_real_tests += 1

        is_critical = obj.get("is_critical", None)
        fitness = obj.get("fitness", {}) or {}

        dimensions_fitness = fitness.get("dimensions_fitness", None)
        efficiency_fitness = fitness.get("efficiency_fitness", None)
        effectiveness_fitness = fitness.get("effectiveness_fitness", None)

        # If only_response=True, we *only* check effectiveness_fitness vs th_response
        if is_critical:
            if only_response:
                # mimic old "no threshold" behavior if th_response is None
                if th_response is None:
                    if dimensions_fitness is None or dimensions_fitness < 1.0:
                        num_real_fail += 1
                else:
                    if dimensions_fitness is not None and dimensions_fitness <= th_response:
                        num_real_fail += 1
            else:
                no_thresholds = (th_content is None and th_efficiency is None and th_response is None)

                if no_thresholds:
                    # old behavior equivalent: if missing or < 1.0
                    if dimensions_fitness is None or dimensions_fitness < 1.0:
                        num_real_fail += 1
                else:
                    dim_fail = (
                        th_content is not None
                        and dimensions_fitness is not None
                        and dimensions_fitness <= th_response
                    )
                    effi_fail = (
                        th_efficiency is not None
                        and efficiency_fitness is not None
                        and efficiency_fitness <= th_efficiency
                    )
                    eff_fail = (
                        th_response is not None
                        and effectiveness_fitness is not None
                        and effectiveness_fitness <= th_content
                    )

                    # fail if ANY metric violates its threshold
                    if dim_fail or effi_fail or eff_fail:
                        num_real_fail += 1

    # duplicate-free adjustments (keep as before)
    num_real_fail = num_real_fail - (num_crit - num_crit_dupfree)
    num_real_tests = num_real_tests - (num_all - num_all_dupfree)
    # num_real_fail = max(num_real_fail, 0)
    # num_real_tests = max(num_real_tests, 0)

    return num_real_tests, num_real_fail

def plot_metric_vs_time(
    project="SafeLLM",
    size=(16, 6),
    metric="failures",
    time_in_minutes=180,
    file_name="plot",
    run_filters = None,
    one_per_name: bool = False,
    experiments_folder: str = rf"C:\Users\levia\Documents\testing\LLM\opensbt-llm\wandb_download" 
):  
    print("Project name:", project)
    if project not in run_filters:
        raise AnalysisException(
            "Plesase implement runs filter for yout project in opensbt.visualization.llm_figures"
        )
    artifact_paths = download_run_artifacts_relative(f"mt-test/{project}", 
                                                     local_root=experiments_folder, 
                                                     filter_runs=run_filters[project],
                                                     one_per_name=one_per_name)

    print("len(artifact_paths):", len(artifact_paths))
    fig, axes = plt.subplots(1, len(artifact_paths), sharey="row", sharex="all")

    # Flatten axes to a 1D list for consistent indexing
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    # Fixed color map for algorithms
    algo_colors = {
        "STELLAR": "tab:green",
        "NSGAII": "tab:green",
        "T-wise": "tab:orange",
        "Random": "tab:blue",
        "ASTRAL": "tab:red",
        "nsga2":  "tab:green",
        "rs": "tab:blue",
        "sensei": "tab:red"
    }
    
    # add as needed
    fig.set_size_inches(*size)
    fig.supxlabel("Time [min]")
    # fig.supylabel(metric_names[metric])
    tick_count = time_in_minutes // 30 + 1
    ticks_kwargs = {"xticks": np.linspace(0, time_in_minutes, tick_count)}
    if metric == "critical_ratio":
        ticks_kwargs["yticks"] = np.linspace(0.0, 1.0, 6)
        ticks_kwargs["ylim"] = (0.0, 1.0)
    plt.setp(axes, **ticks_kwargs)
    for i, (sut, algos) in enumerate(artifact_paths.items()):
        for (algo, features), paths in algos.items():
            algo_name = get_algo_name(algo, features)

            print("algo_name:", algo_name)
            color = algo_colors.get(algo_name, None)

            dfs = [get_run_history_table(path) for path in paths]
            AnalysisPlots.plot_with_std(axes[i], 
                                        dfs,
                                        label=algo_name,
                                        metric=metric,
                                        color = color,
                                        target_time=time_in_minutes)
        axes[i].set_title(convert_name(sut))
        axes[i].set_box_aspect(1)
        axes[i].tick_params(axis="x", rotation=45)  # rotate x-axis labels
    
    axes[0].set_ylabel(metric_names[metric], labelpad=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    plt.tight_layout()

    # Set figure background to light gray

    # Set axes background to gray as well
    for ax in axes:
        ax.set_facecolor("0.93")  # slightly darker than figure background
        ax.grid(
            visible=True,
            which="both",
            color="white",  # light grid lines
            linestyle="-",
            linewidth=0.8
        )
           # Remove all borders (spines)
        for spine in ax.spines.values():
            spine.set_visible(False)
    plt.savefig(file_name)


def plot_metric_vs_time_separate(
    project="SafeLLM",
    size=(12, 6),
    metric="failures",
    time_in_minutes=180,
    file_name="plot",
    run_filters = None,
    one_per_name: bool = False,
    experiments_folder: str = rf"C:\Users\levia\Documents\testing\LLM\opensbt-llm\wandb_download" 
):  
    print("Project name:", project)
    if project not in run_filters:
        raise AnalysisException(
            "Plesase implement runs filter for yout project in opensbt.visualization.llm_figures"
        )
    artifact_paths = download_run_artifacts_relative(f"mt-test/{project}", 
                                                     local_root=experiments_folder, 
                                                     filter_runs=run_filters[project],
                                                     one_per_name=one_per_name)

    print("len(artifact_paths):", len(artifact_paths))
    algo_colors = {
        "STELLAR": "tab:green",
        "NSGAII": "tab:green",
        "T-wise": "tab:orange",
        "Random": "tab:blue",
        "ASTRAL": "tab:red"
    }
    fig = plt.figure()
    fig.set_size_inches(size[0], size[1])
    for i, (sut, algos) in enumerate(artifact_paths.items()):
        for (algo, features), paths in algos.items():
            algo_name = get_algo_name(algo, features)
            dfs = dfs = [get_run_history_table(path) for path in paths]
            AnalysisPlots.plot_with_std(plt, dfs, label=algo_name, metric=metric, target_time=time_in_minutes, color=algo_colors[algo_name])
        plt.title(convert_name(sut.capitalize()))
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.85) 
        plt.tight_layout()
        tick_count = time_in_minutes // 30 + 1
        plt.xticks(np.linspace(0, time_in_minutes, tick_count))
        plt.xlim((0, time_in_minutes))
        plt.xlabel("Time [min]")
        if metric.lower() == "critical_ratio":
            plt.ylim(0, 1)
        plt.ylabel(metric_names[metric])
        plt.tight_layout()
        path = file_name + f"/{convert_name(sut.capitalize())}.pdf"
        os.makedirs(file_name, exist_ok=True)
        plt.savefig(path, format="pdf")
        plt.cla()


def get_color_by_algo(algo):
    print("algo", algo)
    algo = algo.lower()

    if algo == "rs":
        return "tab:blue"
    elif algo == "gs":
        return "tab:orange"
    elif algo == "nsga2":
        return "tab:green"
    else:
        return "black"
    
def plot_metric_vs_time_ivan(
    project="SafeLLM",
    size=(18, 6),
    metric="failures",
    time_in_minutes=120,
    file_name="plot",
    run_filters=None,
    experiments_folder = "wandb_experiments",
    one_per_name=False,
    plot_legend = False,
    tight=False,
    th_content: float = 0.75,
    th_efficiency: float = 0.65,
    th_response: float = 0.65
):
    
    if project not in run_filters:
        raise AnalysisException(
            "Plesase implement runs filter for yout project in opensbt.visualization.llm_figures"
        )
    artifact_paths = download_run_artifacts_relative(f"mt-test/{project}", 
                                                     local_root=experiments_folder,
                                                     filter_runs = run_filters[project],
                                                     one_per_name=one_per_name)
    fig, axes = plt.subplots(1, len(artifact_paths), sharey="row", sharex="all")
    fig.set_size_inches(*size)

    # Ensure axes is always a list
    if len(artifact_paths) == 1:
        axes = [axes]
    # fig.supylabel(metric_names[metric])
    tick_count = time_in_minutes // 30 + 1
    ticks_kwargs = {"xticks": np.linspace(0, time_in_minutes, tick_count)}
    if metric == "critical_ratio":
        ticks_kwargs["yticks"] = np.linspace(0.0, 1.0, 6)
        ticks_kwargs["ylim"] = (0.0, 1.0)
    plt.setp(axes, **ticks_kwargs)
    for i, (sut, algos) in enumerate(artifact_paths.items()):
        for (algo, features), paths in algos.items():
            algo_name = get_algo_name(algo, features)
            color = get_color_by_algo(algo)
            dfs = [get_run_history_table(path, th_response=th_response, th_content=th_content) for path in paths]
            AnalysisPlots.plot_with_std(axes[i], dfs, label=algo_name, metric=metric, color = color, target_time=time_in_minutes)
        axes[i].set_title(convert_name(sut.capitalize()))
        axes[i].set_box_aspect(1)
        axes[i].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.85) 
    
    # Put y-label only next to first subplot
    axes[0].set_ylabel(metric_names[metric], labelpad=10, fontsize=20)

    if len(axes) > 1:
        fig.supxlabel("Time [min]", fontsize = 20)
    else:
        axes[0].set_xlabel("Time [min]", fontsize = 20)

    if plot_legend:
        legend_handles = {}
        for label, col in AnalysisPlots.label_colors.items():
            legend_handles[label] = plt.Line2D([0], [0], color=col, lw=10)
        fig.legend(legend_handles.values(), legend_handles.keys(), title="Labels", loc="upper right")
    if tight:
        plt.tight_layout()

    folder = os.path.dirname(file_name)
    
    os.makedirs(folder, exist_ok=True)
    plt.savefig(file_name)

def get_run_history_table(run_path: str, freq: str = "1min", 
                              th_content: float = 0.75,
                            th_efficiency: float = 0.65,
                            th_response: float = 0.65,
                            only_response: bool= False):
    history_file = os.path.join(run_path, "run_history.csv")
    if not os.path.exists(history_file):
        raise FileNotFoundError(f"No run_history.csv found in {run_path}")

    df = pd.read_csv(history_file)

    # Handle timestamps
    if "timestamp" not in df.columns and "_timestamp" in df.columns:
        df["timestamp"] = df["_timestamp"]

    # Convert to datetime
    df["time"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.drop_duplicates(subset="time").sort_values("time")

    # --- Normalize so each run starts at t=0 ---
    df["time"] = df["time"] - df["time"].iloc[0]

    # Set normalized time as index
    df = df.set_index("time")

    # Resample and interpolate on a uniform grid
    if isinstance(df.index, pd.TimedeltaIndex):
        df = df.resample(freq).first()
        df = df.interpolate(method="time")

    # interpolate based on real failures

    num_tests_all, num_real_fail = get_real_tests(path_name = run_path, th_efficiency=th_efficiency, 
                                                    th_content = th_content, th_response = th_response,
                                                    only_response=only_response)
    
    all_fail = df["failures"].iloc[-1]

    # print(f"per cent real fail: ", num_real_fail/all_fail)
    # print(f"ratio_real:", num_real_fail/num_tests_all)

    df["failures"] = df["failures"] * num_real_fail/all_fail
    df["critical_ratio"] = df.critical_ratio * num_real_fail/all_fail

    return df

def diversity_report(
        algorithms,
        project="SafeLLM",
        output_path="diversity",
        input: bool = True,
        max_num_clusters = 150,
        silhouette_threshold = 20,
        visualize: bool = True,
        mode: Literal["separated", "merged"] = "separated",
        num_seeds: Optional[int] = None,
        local_root: str = None,
        run_filters = None,
        save_folder= None
    ):
    if mode == "merged" and num_seeds is None:
        raise AnalysisException("Provide the number of seed in merged mode")
    output_path = os.path.join(output_path, "input" if input else "output")


    if local_root != None:
        artifact_paths = download_run_artifacts_relative(f"mt-test/{project}", 
                                                         local_root=local_root,
                                                         filter_runs = run_filters[project])
    else:
        artifact_paths = download_run_artifacts(f"mt-test/{project}", run_filters[project])
    
    print(f"Applying diversity analysis for {len(artifact_paths)} runs.")
    cached_embeddings = dict()
    suts = []
    result = {}
    if os.path.exists(os.path.join(output_path, "report.json")):
        with open(os.path.join(output_path, "report.json"), "r") as f:
            result = json.load(f)
    else:
        for i, (sut, algos) in enumerate(artifact_paths.items()):
            suts.append(sut)
            print(sut)
            result[sut] = dict()
            algo_names = [get_algo_name(*key) for key in algos.keys()]

            for algo_name, paths in zip(algo_names, algos.values()):
                avg_max_distances = []
                avg_distances = []
                for path in paths:
                    _, embeddings = get_embeddings(path)
                    cached_embeddings[path] = embeddings
                    avg_max_distances.append(AnalysisDiversity.average_max_distance(embeddings))
                    avg_distances.append(AnalysisDiversity.average_distance(embeddings))
                result[sut][algo_name] = {
                    "avg_max_distance": avg_max_distances,
                    "avg_distance": avg_distances,
                }
            for nm in algo_names:
                result[sut][nm]["coverage"] = []
                result[sut][nm]["entropy"] = []

            if mode == "separated":
                original_seeds = min([len(paths) for paths in algos.values()])
                if num_seeds is None:
                    num_seeds = original_seeds
                for seed in range(num_seeds):
                    algo_counts = dict()
                    to_cluster = []
                    for algo_name, paths in zip(algo_names, algos.values()):
                        embeddings = cached_embeddings[paths[seed % original_seeds]]
                        algo_counts[algo_name] = embeddings.shape[0]
                        to_cluster.append(embeddings)
                    to_cluster = np.concatenate(to_cluster)
                    max_num_of_clusters = (len(to_cluster) if max_num_clusters is None else max_num_clusters)
                    clusterer, labels, centers, _ = AnalysisDiversity.cluster_data(
                        data=to_cluster,
                        n_clusters_interval=(2, min(to_cluster.shape[0], max_num_of_clusters)),
                        seed=seed,
                        silhouette_threshold=silhouette_threshold,
                    )
                    coverage, entropy, _ = AnalysisDiversity.compute_coverage_entropy(
                        labels, centers, algo_names, algo_counts,
                    )
                    for nm in coverage:
                        result[sut][nm]["coverage"].append(coverage[nm])
                        result[sut][nm]["entropy"].append(entropy[nm])
                    os.makedirs(os.path.join(output_path, sut), exist_ok=True)
                    cluster_data = (
                        to_cluster,
                        algo_names,
                        algo_counts,
                        labels,
                        centers,
                        seed,
                    )
                    with open(os.path.join(output_path, sut, f"pickled_cluster_data_seed{seed}.pkl"), "wb") as f:
                        pickle.dump(cluster_data, f)
                    if visualize:
                        AnalysisPlots.plot_clusters(
                            to_cluster,
                            centers,
                            os.path.join(output_path, sut, f"clusters_seed{seed}"),
                            algo_names,
                            algo_counts,
                            seed=seed,
                        )
            elif mode == "merged":
                algo_counts = defaultdict(int)
                to_cluster = []
                for algo_name, paths in tqdm.tqdm(zip(algo_names, algos.values())):
                    for path in paths:
                        embeddings = cached_embeddings[path]
                        algo_counts[algo_name] += embeddings.shape[0]
                        to_cluster.append(embeddings)
                to_cluster = np.concatenate(to_cluster)
                max_num_of_clusters = (len(to_cluster) if max_num_clusters is None else max_num_clusters)
                for seed in range(num_seeds):
                    clusterer, labels, centers, _ = AnalysisDiversity.cluster_data(
                        data=to_cluster,
                        n_clusters_interval=(2, min(to_cluster.shape[0], max_num_of_clusters)),
                        seed=seed,
                        silhouette_threshold=silhouette_threshold,
                    )
                    coverage, entropy, _ = AnalysisDiversity.compute_coverage_entropy(
                        labels, centers, algo_names, algo_counts,
                    )
                    for nm in coverage:
                        result[sut][nm]["coverage"].append(coverage[nm])
                        result[sut][nm]["entropy"].append(entropy[nm])
                    os.makedirs(os.path.join(output_path, sut), exist_ok=True)   
                    cluster_data = (
                        to_cluster,
                        algo_names,
                        algo_counts,
                        labels,
                        centers,
                        seed,
                    )
                    with open(os.path.join(output_path, sut, f"pickled_cluster_data_seed{seed}.pkl"), "wb") as f:
                        pickle.dump(cluster_data, f)
                    if visualize:
                        AnalysisPlots.plot_clusters(
                            to_cluster,
                            centers,
                            os.path.join(output_path, sut, f"clusters_seed{seed}"),
                            algo_names,
                            algo_counts,
                            seed=seed,
                        )
            else:
                raise AnalysisException("Unknown mode")
            print(result)

        with open(os.path.join(output_path, "report.json"), "w") as f:
            json.dump(result, f)

    data_dict = defaultdict(list)
    suts = list(artifact_paths.keys())
    for algorithm in algorithms:
        data_dict["Algorithm"].append(algorithm)
        for sut in suts:
            for metric in [
                "avg_max_distance",
                "avg_distance",
                "coverage",
                "entropy",
            ]:
                if algorithm not in result[sut]:
                    mean = None
                else:
                    values = result[sut][algorithm].get(metric, None)
                    mean = np.mean(values) if values is not None else None
                data_dict[f"{sut}.{metric}"].append(mean)
    pd.DataFrame(data_dict).to_csv(os.path.join(output_path, "scores.csv"), index=False)

    
    for metric in [
                "avg_max_distance",
                "avg_distance",
                "coverage",
                "entropy",
            ]:
        statistics = {}
        for sut in suts:
            algo_names = list(result[sut].keys())
            values = [result[sut][algo][metric] for algo in algo_names]
            statistics[sut] = AnalysisTables.statistics(values, algo_names)
        data_dict = defaultdict(list)
        for i in range(len(algorithms)):
            for j in range(i + 1, len(algorithms)):
                data_dict["Algorithm 1"].append(algorithms[i])
                data_dict["Algorithm 2"].append(algorithms[j])
                for sut in suts:
                    stats = statistics[sut][algorithms[i]][algorithms[j]]
                    if len(stats) > 0:
                        data_dict[f"{sut.capitalize()}.P-Value"].append(stats[0])
                        data_dict[f"{sut.capitalize()}.Effect Size"].append(stats[1])
                    else:
                        data_dict[f"{sut.capitalize()}.P-Value"].append(None)
                        data_dict[f"{sut.capitalize()}.Effect Size"].append(None)
        df = pd.DataFrame(data_dict)
        df = df.dropna()
        df.to_csv(os.path.join(output_path, save_folder, f"{metric}_stats.csv"), index=False)        
    return result

def statistics_table(
    algorithms,
    project="SafeLLM",
    metric="failures",
    path="table.csv",
    run_filters = None,
    one_per_name: bool = False,
    experiments_folder: str = "wandb_download",
    only_response: bool = False,
    th_content: float = 0.75,
    th_efficiency: float = 0.65,
    th_response: float = 0.65
):     
    print("using project:", project)
    save_path = path
    if run_filters is not None and project not in run_filters:
        raise AnalysisException(
            "Plesase implement runs filter for yout project in opensbt.visualization.llm_figures"
        )
    artifact_paths = download_run_artifacts_relative(local_root=experiments_folder, 
                                                     wb_project_path=f"mt-test/{project}",
                                                     filter_runs=run_filters[project],
                                                     one_per_name=one_per_name)
    print("artifact_paths:", artifact_paths)
    # statistics: number of paths per algorithm
    statistics = defaultdict(int)

    for scenario_dict in artifact_paths.values():
        for (algo, _variant), paths in scenario_dict.items():
            statistics[algo] += len(paths)

    statistics = dict(statistics)
    print("artefact paths statistics:", statistics)

    statistics = {}
    suts = []
    
    for i, (sut, algos) in enumerate(artifact_paths.items()):
        suts.append(sut)
        print("algos:", algos)
        algo_names = [get_algo_name(*key) for key in algos.keys()]
        print("algos:", algo_names)
        if metric == "failures":
            # Compute number of failures for each path
            values = [
                [get_real_tests(path, only_response=only_response,
                                                                    th_content=th_content,
                                                                    th_efficiency=th_efficiency,
                                                                    th_response=th_response)[1] for path in paths]  # index 1 = num_real_fail
                for paths in algos.values()
            ]
            print("failures:", values)
        elif metric == "critical_ratio":
            # Compute ratio = num_real_fail / num_real_tests for each path
            values = []
            for paths in algos.values():
                algo_values = []
                for path in paths:
                    num_real_tests, num_real_fail = get_real_tests(path, only_response=only_response)
                    ratio = (
                        num_real_fail / num_real_tests
                        if num_real_tests > 0
                        else 0.0
                    )
                    algo_values.append(ratio)
                values.append(algo_values)
            print("critical_ratio:", values)

        else:
            raise AnalysisException(f"Unknown metric: {metric}")
        statistics[sut] = AnalysisTables.statistics(values, algo_names)

    print("statistics:", statistics)

    data = defaultdict(list)
    print("algorithms: ", algorithms)
    for i in range(len(algorithms)):
        for j in range(i + 1, len(algorithms)):
            data["Algorithm 1"].append(algorithms[i])
            data["Algorithm 2"].append(algorithms[j])
            for sut in suts:
                stats = statistics[sut][algorithms[i]][algorithms[j]]
                if len(stats) > 0:
                    data[f"{sut.capitalize()}.P-Value"].append(stats[0])
                    data[f"{sut.capitalize()}.Effect Size"].append(stats[1])
                else:
                    data[f"{sut.capitalize()}.P-Value"].append(None)
                    data[f"{sut.capitalize()}.Effect Size"].append(None)
                print(stats)
    df = pd.DataFrame(data)


    print(f"Before dropna: {len(df)} rows")

    # Drop only fully empty rows
    df = df.dropna(how="all")
    print(f"After dropna: {len(df)} rows")


    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df.to_csv(save_path, index=False)

def boxplots(
    algorithms,
    project="SafeLLM",
    size=(18, 6),
    metric="failures",
    file_name="plot",
    experiments_folder = "wandb_download",
    run_filters = None,
    one_per_name = False,
    th_content=0.75,
    th_response=0.65,
    th_efficiency=0.65,
    only_response=False
):
    if project not in run_filters:
        raise AnalysisException(
            "Plesase implement runs filter for yout project in opensbt.visualization.llm_figures"
        )
    artifact_paths = download_run_artifacts_relative(f"mt-test/{project}", 
                                                     local_root=experiments_folder,
                                                     filter_runs = run_filters[project],
                                                     one_per_name=one_per_name)
    if metric == "critical_ratio":
        fig, axes = plt.subplots(1, len(artifact_paths), sharey="row")
    else:
        fig, axes = plt.subplots(1, len(artifact_paths))
    fig.set_size_inches(*size)
    # fig.supylabel(metric_names[metric])
    # Ensure axes is always a list
    if len(artifact_paths) == 1:
        axes = [axes]
    ticks_kwargs = {}
    if metric == "critical_ratio":
        ticks_kwargs["yticks"] = np.linspace(0.0, 1.0, 6)
        ticks_kwargs["ylim"] = (0.0, 1.0)
        plt.setp(axes, **ticks_kwargs)
    for i, (sut, algos) in enumerate(artifact_paths.items()):
        name_to_dfs = {}
        for (algo, features), paths in algos.items():
            algo_name = get_algo_name(algo, features)
            dfs = [get_run_history_table(path, th_content= th_content,
                    th_response=th_response, th_efficiency=th_efficiency, only_response=only_response) for path in paths]
            name_to_dfs[algo_name] = dfs
        algo_names = []
        dfs_list = []
        for algorithm in algorithms:
            if algorithm in name_to_dfs:
                algo_names.append(algorithm)
                dfs_list.append(name_to_dfs[algorithm])
        AnalysisPlots.boxplot(axes[i], dfs_list, algo_names, metric=metric)
        axes[i].set_title(convert_name(sut.capitalize()))
        axes[i].set_box_aspect(1)
        axes[i].set_xticklabels(algo_names, rotation='vertical')
    
    # Put y-label only next to first subplot
    axes[0].set_ylabel(metric_names[metric], labelpad=10, fontsize = 20)

    legend_handles = {}
    for label, col in AnalysisPlots.label_colors.items():
        legend_handles[label] = plt.Line2D([0], [0], color=col, lw=10)
    
    # fig.legend(legend_handles.values(), legend_handles.keys(), title="Labels", loc="upper right")
    fig.tight_layout()
    plt.savefig(file_name)


def boxplots_separate(
    algorithms,
    project="SafeLLM",
    size=(18, 6),
    metric="failures",
    file_name="plot",
    experiments_folder = "wandb_download",
    run_filters = None,
    one_per_name = False,
    th_content=None,
    th_response=None
):
    if project not in run_filters:
        raise AnalysisException(
            "Plesase implement runs filter for yout project in opensbt.visualization.llm_figures"
        )
    artifact_paths = download_run_artifacts_relative(f"mt-test/{project}", 
                                                     local_root=experiments_folder,
                                                     filter_runs = run_filters[project],
                                                     one_per_name=one_per_name)
    fig = plt.figure()
    fig.set_size_inches(*size)
    # fig.supylabel(metric_names[metric])
    # Ensure axes is always a list
    for i, (sut, algos) in enumerate(artifact_paths.items()):
        name_to_dfs = {}
        for (algo, features), paths in algos.items():
            algo_name = get_algo_name(algo, features)
            dfs = [get_run_history_table(path, th_content= th_content,
                    th_response=th_response) for path in paths]
            name_to_dfs[algo_name] = dfs
        algo_names = []
        dfs_list = []
        for algorithm in algorithms:
            if algorithm in name_to_dfs:
                algo_names.append(algorithm)
                dfs_list.append(name_to_dfs[algorithm])
        AnalysisPlots.boxplot(plt, dfs_list, algo_names, metric=metric)
        plt.title(convert_name(sut.capitalize()))
        plt.xticks(ticks=np.arange(len(algo_names)) + 1, labels=algo_names, rotation='vertical')
        if metric.lower() == "critical_ratio":
            plt.ylim(0, 1)
        plt.ylabel(metric_names[metric])
        plt.tight_layout()
        path = file_name + f"/{convert_name(sut.capitalize())}.pdf"
        os.makedirs(file_name, exist_ok=True)
        plt.savefig(path, format="pdf")
        plt.cla()


def plot_boxplots_by_algorithm_raw(
    title: str = None,
    project: str = "SafeLLM",
    metric: str = "failures",
    run_filters=None,
    one_per_name: bool = False,
    experiments_folder: str = "wandb_download",
    save_path: str = "plots/boxplots_raw.png",
    only_response: bool = False,
    th_content: float = 0.75,
    th_efficiency: float = 0.65,
    th_response: float = 0.65
):
    """
    Create boxplots of raw run values for each algorithm (one subplot per algorithm).
    Uses the same artifact data structure as in last_values_table().
    """

    artifact_paths = download_run_artifacts_relative(
        local_root=experiments_folder,
        wb_project_path=f"mt-test/{project}",
        filter_runs=run_filters[project] if run_filters else None,
        one_per_name=one_per_name,
    )
    print("artifact_paths:", artifact_paths)
    all_data = []

    for sut, algos in artifact_paths.items():
        for (algo, features), paths in algos.items():
            algo_name = get_algo_name(algo, features)

            # Collect per-run metric values
            values = []
            if metric == "failures":
                for path in paths:
                    values.append(get_real_tests(path, only_response=only_response,
                                                                    th_content=th_content,
                                                                    th_efficiency=th_efficiency,
                                                                    th_response=th_response)[1])  # number of failures
            elif metric == "critical_ratio":
                for path in paths:
                    num_real_tests, num_real_fail = get_real_tests(path, only_response=only_response,
                                                                    th_content=th_content,
                                                                    th_efficiency=th_efficiency,
                                                                    th_response=th_response)
                    ratio = num_real_fail / num_real_tests if num_real_tests > 0 else 0.0
                    values.append(ratio)
            else:
                raise AnalysisException(f"Unknown metric: {metric}")

            # Store all runs for that algorithm/SUT
            for v in values:
                all_data.append({
                    "SUT": sut,
                    "Algorithm": algo_name,
                    "Value": v
                })

    df = pd.DataFrame(all_data)
    if df.empty:
        raise RuntimeError("No data found to plot boxplots.")

    # Define consistent colors for algorithms
    algo_colors = {
        # "STELLAR": "#1f77b4",
        # "NSGAII": "#1f77b4",
        # "Random": "#ff7f0e",
        # "T-wise": "#2ca02c",
        "LUNAR": "#1f77b4",
        "LUNAR_S": "#2ca02c",
        "SENSEI": "#ff7f0e",
    }

    suts = list(df["SUT"].unique())
    n_suts = len(suts)

    # Define the desired full order
    full_order = ["LUNAR", "LUNAR_S", "SENSEI"]

    # # Keep only the algorithms present in the dataset
    # algo_order = [algo for algo in full_order if algo in df["Algorithm"].unique()]
    algo_order = full_order

    # Ensure the column is categorical with this order
    df["Algorithm"] = pd.Categorical(df["Algorithm"], categories=algo_order, ordered=True)

    # Create one subplot per SUT
    fig, axes = plt.subplots(1, n_suts, figsize=(5 * n_suts, 5), sharey=True)

    if n_suts == 1:
        axes = [axes]

    print("sut df columns:", df.columns)
    print("df:", df)

    for ax, sut in zip(axes, suts):
        sut_df = df[df["SUT"] == sut]

        sns.boxplot(
            data=sut_df,
            x="Algorithm",
            y="Value",
            palette=algo_colors,
            ax=ax,
            width=0.6,
            order=algo_order,
            hue="Algorithm",  # or another categorical variable
        )

        ax.set_title(convert_name(sut), fontsize=23, weight="bold")  # title = model name
        ax.set_ylabel(metric.replace("_", " ").capitalize(), fontsize = 20)
        ax.set_xlabel("")  # remove x-axis label
        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)
        # Style: gray background, white grid, no borders
        ax.set_facecolor("0.9")
        ax.grid(visible=True, which="both", color="white", linestyle="-", linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_visible(False)

    if title is not None:
        plt.suptitle(title, fontsize=24, weight="bold")
        plt.subplots_adjust(top=0.92)  # adjust to make room for suptitle

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    print(f"Boxplots saved to: {save_path}")
    plt.close()

def last_values_table(
    project: str = "SafeLLM",
    metrics: list | str = "failures",
    path: str = "table.csv",
    run_filters=None,
    one_per_name: bool = False,
    experiments_folder: str = "wandb_download",
    only_response: bool = False,
    th_content: Optional[float] = 0.75,
    th_efficiency: Optional[float] = 0.65,
    th_response: Optional[float] = 0.50,
):
    save_path = path

    if run_filters is not None and project not in run_filters:
        raise AnalysisException(
            "Please implement runs filter for your project in opensbt.visualization.llm_figures"
        )

    artifact_paths = download_run_artifacts_relative(
        local_root=experiments_folder,
        wb_project_path=f"mt-test/{project}",
        filter_runs=run_filters[project] if run_filters else None,
        one_per_name=one_per_name,
    )

    if isinstance(metrics, str):
        metrics = [metrics]

    summary_data = []

    for sut, algos in artifact_paths.items():
        algo_names = [get_algo_name(*key) for key in algos.keys()]
        print("algos:", algo_names)

        for algo_name, paths in zip(algo_names, algos.values()):
            print("-------------")
            
            row = {"Algorithm": algo_name,"SUT": sut}
            print("sut:", sut)
            print("algo: ", algo_name)

            for metric in metrics:
                if metric == "failures":
                    values = [get_real_tests(path, only_response=only_response,
                                                                    th_content=th_content,
                                                                    th_efficiency=th_efficiency,
                                                                    th_response=th_response)[1] for path in paths]
                elif metric == "critical_ratio":                              
                    values = []
                    for path in paths:
                        num_real_tests, num_real_fail = get_real_tests(path, only_response=only_response,
                                                                    th_content=th_content,
                                                                    th_efficiency=th_efficiency,
                                                                    th_response=th_response)
                        ratio = num_real_fail / num_real_tests if num_real_tests > 0 else 0.0
                        values.append(ratio)
                else:
                    raise AnalysisException(f"Unknown metric: {metric}")
                print(f"values for metric {metric}:", values)
                row[f"{metric}_mean"] = np.mean(values) if values else np.nan
                row[f"{metric}_std"] = np.std(values, ddof=1) if len(values) > 1 else 0.0
                
            summary_data.append(row)
            print("row:", row)
            print("-------------")
            
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    algorithm_order = ["nsga2", "rs", "sensei"]
    print(summary_data)
    summary_df = pd.DataFrame(summary_data)
    summary_df["Algorithm"] = pd.Categorical(summary_df["Algorithm"], categories=algorithm_order, ordered=True)
    summary_df = summary_df.sort_values(by=["SUT", "Algorithm"]).reset_index(drop=True)

    summary_df.to_csv(save_path, index=False)
    print(f"\nSummary table of mean/std saved to: {save_path}")

    # Determine LaTeX column alignment dynamically
    n_metrics = len(metrics)
    col_format = "ll" + "cc" * n_metrics  # 2 left + 2 per metric (mean/std)
    
    latex_path = os.path.splitext(save_path)[0] + ".tex"
    summary_df.to_latex(
        latex_path,
        index=False,
        float_format="%.3f",
        caption=f"Summary of metrics per SUT and algorithm",
        label="tab:metric_summary",
        column_format=col_format
    )
    print(f"Summary table exported to LaTeX: {latex_path}")

def count_file_paths(nested_dict):
    total = 0
    for value in nested_dict.values():
        if isinstance(value, dict):
            total += count_file_paths(value)  # recurse into nested dict
        elif isinstance(value, list):
            total += len(value)  # count the file paths in the list
    return total

def diversity_report(
        algorithms,
        project="SafeLLM",
        output_path="diversity",
        input: bool = True,
        max_num_clusters = 150,
        silhouette_threshold = 20,
        visualize: bool = True,
        mode: Literal["separated", "merged"] = "separated",
        num_seeds: Optional[int] = None,
        local_root: str = None,
        one_per_name: bool = True,
        run_filters: Dict = None
    ):
    if mode == "merged" and num_seeds is None:
        raise AnalysisException("Provide the number of seed in merged mode")
    output_path = os.path.join(output_path, "input" if input else "output")

    if local_root != None:
        artifact_paths = download_run_artifacts_relative(f"mt-test/{project}", 
                                                            local_root=local_root,
                                                            filter_runs = run_filters[project],
                                                            one_per_name=one_per_name)
    else:
        artifact_paths = download_run_artifacts(f"mt-test/{project}", run_filters[project])
    print(f"Applying diversity analysis for {count_file_paths(artifact_paths)} runs.")
    cached_embeddings = dict()
    suts = []
    result = {}
    if os.path.exists(os.path.join(output_path, "report.json")):
        with open(os.path.join(output_path, "report.json"), "r") as f:
            result = json.load(f)
    else:
        for i, (sut, algos) in enumerate(artifact_paths.items()):
            suts.append(sut)
            print(sut)
            result[sut] = dict()
            algo_names = [get_algo_name(*key) for key in algos.keys()]

            for algo_name, paths in zip(algo_names, algos.values()):
                avg_max_distances = []
                avg_distances = []
                for path in paths:
                    _, embeddings = get_embeddings(path)
                    cached_embeddings[path] = embeddings
                    avg_max_distances.append(AnalysisDiversity.average_max_distance(embeddings))
                    avg_distances.append(AnalysisDiversity.average_distance(embeddings))
                result[sut][algo_name] = {
                    "avg_max_distance": avg_max_distances,
                    "avg_distance": avg_distances,
                }
            for nm in algo_names:
                result[sut][nm]["coverage"] = []
                result[sut][nm]["entropy"] = []

            if mode == "separated":
                original_seeds = min([len(paths) for paths in algos.values()])
                if num_seeds is None:
                    num_seeds = original_seeds
                for seed in range(num_seeds):
                    algo_counts = dict()
                    to_cluster = []
                    for algo_name, paths in zip(algo_names, algos.values()):
                        embeddings = cached_embeddings[paths[seed % original_seeds]]
                        algo_counts[algo_name] = embeddings.shape[0]
                        to_cluster.append(embeddings)
                    to_cluster = np.concatenate(to_cluster)
                    max_num_of_clusters = (len(to_cluster) if max_num_clusters is None else max_num_clusters)
                    clusterer, labels, centers, _ = AnalysisDiversity.cluster_data(
                        data=to_cluster,
                        n_clusters_interval=(2, min(to_cluster.shape[0], max_num_of_clusters)),
                        seed=seed,
                        silhouette_threshold=silhouette_threshold,
                    )
                    coverage, entropy, _ = AnalysisDiversity.compute_coverage_entropy(
                        labels, centers, algo_names, algo_counts,
                    )
                    for nm in coverage:
                        result[sut][nm]["coverage"].append(coverage[nm])
                        result[sut][nm]["entropy"].append(entropy[nm])
                    os.makedirs(os.path.join(output_path, sut), exist_ok=True)
                    cluster_data = (
                        to_cluster,
                        algo_names,
                        algo_counts,
                        labels,
                        centers,
                        seed,
                    )
                    with open(os.path.join(output_path, sut, f"pickled_cluster_data_seed{seed}.pkl"), "wb") as f:
                        pickle.dump(cluster_data, f)
                    if visualize:
                        AnalysisPlots.plot_clusters(
                            to_cluster,
                            centers,
                            os.path.join(output_path, sut, f"clusters_seed{seed}"),
                            algo_names,
                            algo_counts,
                            seed=seed,
                        )
            elif mode == "merged":
                algo_counts = defaultdict(int)
                to_cluster = []
                for algo_name, paths in tqdm.tqdm(zip(algo_names, algos.values())):
                    for path in paths:
                        embeddings = cached_embeddings[path]
                        algo_counts[algo_name] += embeddings.shape[0]
                        to_cluster.append(embeddings)
                to_cluster = np.concatenate(to_cluster)
                max_num_of_clusters = (len(to_cluster) if max_num_clusters is None else max_num_clusters)
                for seed in range(num_seeds):
                    pickled_data_path = os.path.join(output_path, sut, f"pickled_cluster_data_seed{seed}.pkl")
                    
                    
                    if os.path.exists(pickled_data_path):
                        with open(pickled_data_path, "rb") as f:
                            content = pickle.load(f)
                            (
                                to_cluster,
                                algo_names,
                                algo_counts,
                                labels,
                                centers,
                                seed,
                            ) = content       
                    else:
                        clusterer, labels, centers, _ = AnalysisDiversity.cluster_data(
                            data=to_cluster,
                            n_clusters_interval=(2, min(to_cluster.shape[0], max_num_of_clusters)),
                            seed=seed,
                            silhouette_threshold=silhouette_threshold,
                        )
                        os.makedirs(os.path.join(output_path, sut), exist_ok=True)   
                        cluster_data = (
                            to_cluster,
                            algo_names,
                            algo_counts,
                            labels,
                            centers,
                            seed,
                        )
                        with open(pickled_data_path, "wb") as f:
                            pickle.dump(cluster_data, f)

                    coverage, entropy, _ = AnalysisDiversity.compute_coverage_entropy(
                        labels, centers, algo_names, algo_counts,
                    )
                    for nm in coverage:
                        result[sut][nm]["coverage"].append(coverage[nm])
                        result[sut][nm]["entropy"].append(entropy[nm])                   
                    if visualize:
                        AnalysisPlots.plot_clusters(
                            to_cluster,
                            centers,
                            os.path.join(output_path, sut, f"clusters_seed{seed}"),
                            algo_names,
                            algo_counts,
                            seed=seed,
                        )
            else:
                raise AnalysisException("Unknown mode")
            print(result)

        with open(os.path.join(output_path, "report.json"), "w") as f:
            json.dump(result, f)

    data_dict = defaultdict(list)
    suts = list(artifact_paths.keys())
    for algorithm in algorithms:
        data_dict["Algorithm"].append(algorithm)
        for sut in suts:
            for metric in [
                "avg_max_distance",
                "avg_distance",
                "coverage",
                "entropy",
            ]:
                if algorithm not in result[sut]:
                    mean = None
                else:
                    values = result[sut][algorithm].get(metric, None)
                    mean = np.mean(values) if values is not None else None
                data_dict[f"{sut}.{metric}"].append(mean)
    pd.DataFrame(data_dict).to_csv(os.path.join(output_path, "scores.csv"), index=False)

    
    for metric in [
                "avg_max_distance",
                "avg_distance",
                "coverage",
                "entropy",
            ]:
        statistics = {}
        for sut in suts:
            algo_names = list(result[sut].keys())
            values = [result[sut][algo][metric] for algo in algo_names]
            statistics[sut] = AnalysisTables.statistics(values, algo_names)
        data_dict = defaultdict(list)
        for i in range(len(algorithms)):
            for j in range(i + 1, len(algorithms)):
                data_dict["Algorithm 1"].append(algorithms[i])
                data_dict["Algorithm 2"].append(algorithms[j])
                for sut in suts:
                    stats = statistics[sut][algorithms[i]][algorithms[j]]
                    if len(stats) > 0:
                        data_dict[f"{sut.capitalize()}.P-Value"].append(stats[0])
                        data_dict[f"{sut.capitalize()}.Effect Size"].append(stats[1])
                    else:
                        data_dict[f"{sut.capitalize()}.P-Value"].append(None)
                        data_dict[f"{sut.capitalize()}.Effect Size"].append(None)
        df = pd.DataFrame(data_dict)
        df = df.dropna()
        df.to_csv(os.path.join(output_path, f"{metric}_stats.csv"), index=False)        
    return result

# def diversity_report(
#         algorithms,
#         project="SafeLLM",
#         output_path="diversity",
#         input: bool = True,
#         max_num_clusters = 150,
#         silhouette_threshold = 20,
#         visualize: bool = False,
#         local_root: str = None,
#         run_filters: dict = None,
#         mode: Literal["separated", "merged"] = "merged",
#         num_seeds: Optional[int] = 10,
#         one_per_name: bool = True,
#         save_folder: str = ""
#     ):
#     output_path = os.path.join(output_path, "input" if input else "output")

#     if local_root != None:
#         artifact_paths = download_run_artifacts_relative(f"mt-test/{project}", 
#                                                          local_root=local_root,
#                                                          filter_runs = run_filters[project],
#                                                          one_per_name=one_per_name)
#     else:
#         artifact_paths = download_run_artifacts(f"mt-test/{project}", run_filters[project])
    
#     print(f"Applying diversity analysis for {len(artifact_paths)} runs.")
#     cached_embeddings = dict()
#     suts = []
#     result = {}
#     if os.path.exists(os.path.join(output_path, "report.json")):
#         with open(os.path.join(output_path, "report.json"), "r") as f:
#             result = json.load(f)
#     else:
#         for i, (sut, algos) in enumerate(artifact_paths.items()):
#             suts.append(sut)
#             print(sut)
#             result[sut] = dict()
#             algo_names = [get_algo_name(*key) for key in algos.keys()]

#             for algo_name, paths in zip(algo_names, algos.values()):
#                 avg_max_distances = []
#                 avg_distances = []
#                 for path in paths:
#                     _, embeddings = get_embeddings(path)
#                     cached_embeddings[(path, input)] = embeddings
#                     avg_max_distances.append(AnalysisDiversity.average_max_distance(embeddings))
#                     avg_distances.append(AnalysisDiversity.average_distance(embeddings))
#                 result[sut][algo_name] = {
#                     "avg_max_distance": avg_max_distances,
#                     "avg_distance": avg_distances,
#                 }
#             for nm in algo_names:
#                 result[sut][nm]["coverage"] = []
#                 result[sut][nm]["entropy"] = []

#             if mode == "separated":
#                 original_seeds = min([len(paths) for paths in algos.values()])
#                 if num_seeds is None:
#                     num_seeds = original_seeds
#                 for seed in range(num_seeds):
#                     algo_counts = dict()
#                     to_cluster = []
#                     for algo_name, paths in zip(algo_names, algos.values()):
#                         embeddings = cached_embeddings[(paths[seed % original_seeds], input)]
#                         algo_counts[algo_name] = embeddings.shape[0]
#                         to_cluster.append(embeddings)
#                     to_cluster = np.concatenate(to_cluster)
#                     max_num_of_clusters = (len(to_cluster) if max_num_clusters is None else max_num_clusters)
#                     clusterer, labels, centers, _ = AnalysisDiversity.cluster_data(
#                         data=to_cluster,
#                         n_clusters_interval=(2, min(to_cluster.shape[0], max_num_of_clusters)),
#                         seed=seed,
#                         silhouette_threshold=silhouette_threshold,
#                     )
#                     coverage, entropy, _ = AnalysisDiversity.compute_coverage_entropy(
#                         labels, centers, algo_names, algo_counts,
#                     )
#                     for nm in coverage:
#                         result[sut][nm]["coverage"].append(coverage[nm])
#                         result[sut][nm]["entropy"].append(entropy[nm])
#                     os.makedirs(os.path.join(output_path, sut), exist_ok=True)
#                     cluster_data = (
#                         to_cluster,
#                         algo_names,
#                         algo_counts,
#                         labels,
#                         centers,
#                         seed,
#                     )
#                     with open(pickled_data_path, "wb") as f:
#                         pickle.dump(cluster_data, f)
#                     if visualize:
#                         AnalysisPlots.plot_clusters(
#                             to_cluster,
#                             centers,
#                             os.path.join(output_path, sut, f"clusters_seed{seed}"),
#                             algo_names,
#                             algo_counts,
#                             seed=seed,
#                         )
#             elif mode == "merged":
#                 algo_counts = defaultdict(int)
#                 to_cluster = []
#                 for algo_name, paths in tqdm.tqdm(zip(algo_names, algos.values())):
#                     for path in paths:
#                         embeddings = cached_embeddings[(path, input)]
#                         algo_counts[algo_name] += embeddings.shape[0]
#                         to_cluster.append(embeddings)
#                 to_cluster = np.concatenate(to_cluster)
#                 max_num_of_clusters = (len(to_cluster) if max_num_clusters is None else max_num_clusters)
#                 for seed in range(num_seeds):
#                     pickled_data_path = os.path.join(output_path, sut, f"pickled_cluster_data_seed{seed}.pkl")
                    
                    
#                     if os.path.exists(pickled_data_path):
#                         with open(pickled_data_path, "rb") as f:
#                             content = pickle.load(f)
#                             (
#                                 to_cluster,
#                                 algo_names,
#                                 algo_counts,
#                                 labels,
#                                 centers,
#                                 seed,
#                             ) = content       
#                     else:
#                         clusterer, labels, centers, _ = AnalysisDiversity.cluster_data(
#                             data=to_cluster,
#                             n_clusters_interval=(2, min(to_cluster.shape[0], max_num_of_clusters)),
#                             seed=seed,
#                             silhouette_threshold=silhouette_threshold,
#                         )
#                         os.makedirs(os.path.join(output_path, sut), exist_ok=True)   
#                         cluster_data = (
#                             to_cluster,
#                             algo_names,
#                             algo_counts,
#                             labels,
#                             centers,
#                             seed,
#                         )
#                         with open(os.path.join(output_path, sut, f"pickled_cluster_data_seed{seed}.pkl"), "wb") as f:
#                             pickle.dump(cluster_data, f)

#                     coverage, entropy, _ = AnalysisDiversity.compute_coverage_entropy(
#                         labels, centers, algo_names, algo_counts,
#                     )
#                     for nm in coverage:
#                         result[sut][nm]["coverage"].append(coverage[nm])
#                         result[sut][nm]["entropy"].append(entropy[nm])                   
#                     if visualize:
#                         AnalysisPlots.plot_clusters(
#                             to_cluster,
#                             centers,
#                             os.path.join(output_path, sut, f"clusters_seed{seed}"),
#                             algo_names,
#                             algo_counts,
#                             seed=seed,
#                         )
#             else:
#                 raise AnalysisException("Unknown mode")
#             print(result)

#         with open(os.path.join(output_path, "report.json"), "w") as f:
#             json.dump(result, f)

#     data_dict = defaultdict(list)
#     suts = list(artifact_paths.keys())
#     for algorithm in algorithms:
#         data_dict["Algorithm"].append(algorithm)
#         for sut in suts:
#             for metric in [
#                 "avg_max_distance",
#                 "avg_distance",
#                 "coverage",
#                 "entropy",
#             ]:
#                 if algorithm not in result[sut]:
#                     mean = None
#                 else:
#                     values = result[sut][algorithm].get(metric, None)
#                     mean = np.mean(values) if values is not None else None
#                 data_dict[f"{sut}.{metric}"].append(mean)
#     pd.DataFrame(data_dict).to_csv(os.path.join(output_path, "scores.csv"), index=False)

    
#     for metric in [
#                 "avg_max_distance",
#                 "avg_distance",
#                 "coverage",
#                 "entropy",
#             ]:
#         statistics = {}
#         for sut in suts:
#             algo_names = list(result[sut].keys())
#             values = [result[sut][algo][metric] for algo in algo_names]
#             statistics[sut] = AnalysisTables.statistics(values, algo_names)
#         data_dict = defaultdict(list)
#         for i in range(len(algorithms)):
#             for j in range(i + 1, len(algorithms)):
#                 data_dict["Algorithm 1"].append(algorithms[i])
#                 data_dict["Algorithm 2"].append(algorithms[j])
#                 for sut in suts:
#                     stats = statistics[sut][algorithms[i]][algorithms[j]]
#                     if len(stats) > 0:
#                         data_dict[f"{sut.capitalize()}.P-Value"].append(stats[0])
#                         data_dict[f"{sut.capitalize()}.Effect Size"].append(stats[1])
#                     else:
#                         data_dict[f"{sut.capitalize()}.P-Value"].append(None)
#                         data_dict[f"{sut.capitalize()}.Effect Size"].append(None)
#         df = pd.DataFrame(data_dict)
#         df = df.dropna()
#         df.to_csv(os.path.join(output_path, save_folder, f"{metric}_stats.csv"), index=False)        
#     return result


def sample_from_clusters(
        algorithms,
        project="SafeLLM",
        output_path="cluster_samples",
        suts=None,
        input: bool = True,
        min_num_clusters = 2,
        max_num_clusters = 150,
        silhouette_threshold = -1,
        num_samples: Optional[int] = None,
        local_root: str = None,
        one_per_name: bool = True,
        run_filters: Dict = None
    ):
    output_path = os.path.join(output_path, "input" if input else "output")
    if local_root != None:
        artifact_paths = download_run_artifacts_relative(f"mt-test/{project}", 
                                                            local_root=local_root,
                                                            filter_runs = run_filters[project],
                                                            one_per_name=one_per_name)
    else:
        artifact_paths = download_run_artifacts(f"mt-test/{project}", run_filters[project])
    if suts is None:
        suts = list(artifact_paths.keys())
    if not isinstance(suts, list):
        suts = [suts]

    algo_counts = defaultdict(int)
    to_cluster = []
    all_utterances = []
    algorithms_order = []
    suts_order = []
    for sut in suts:
        algos = artifact_paths[sut]
        algo_names = [get_algo_name(*key) for key in algos.keys()]
        for algo_name, paths in tqdm.tqdm(zip(algo_names, algos.values())):
            for path in paths:
                utterances, embeddings = get_embeddings(path, input=input)
                algo_counts[algo_name] += embeddings.shape[0]
                to_cluster.append(embeddings)
                all_utterances += [
                    {
                        "u": u[0],
                        "f": u[1],
                        "o": u[2],
                    } for u in utterances
                ]
                suts_order += [sut for u in utterances]
                algorithms_order += [algo_name for u in utterances]
    to_cluster = np.concatenate(to_cluster)
    all_utterances = np.array(all_utterances)
    max_num_of_clusters = (len(to_cluster) if max_num_clusters is None else max_num_clusters)
    clusterer, labels, centers, _ = AnalysisDiversity.cluster_data(
        data=to_cluster,
        n_clusters_interval=(min_num_clusters, min(to_cluster.shape[0], max_num_of_clusters)),
        silhouette_threshold=silhouette_threshold,
        seed=random.randint(0, 932932)
    )
    labels = np.array(labels)
    result = {}
    for label in range(len(centers)):
        label_mask = labels == label
        all_testcases = np.array(list(zip(all_utterances, algorithms_order, suts_order)))
        testcases = all_testcases[label_mask]
        if num_samples:
            sampled_raw = random.choices(testcases, k=num_samples)
        else:
            sampled_raw = testcases
        sampled = [
            {
                "utterance": s[0],
                "algorithm": s[1],
                "sut": s[2],
            } for s in sampled_raw
        ]
        result[label] = sampled
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "cluster_samples.json"), "w") as f:
        json.dump(result, f, indent=4)
    label_algo_counts = defaultdict(lambda: defaultdict(int))
    label_algo_shares = defaultdict(lambda: defaultdict(float))
    for label in range(len(centers)):
        df_data = defaultdict(list)
        label_mask = labels == label
        all_testcases = np.array(list(zip(all_utterances, algorithms_order, suts_order)))
        testcases = all_testcases[label_mask]
        for t in testcases:
            u = Utterance.model_validate(t[0]["u"])
            fitness = t[0]["f"]
            other = t[0]["o"]
            df_data["Algorithm"].append(t[1])
            df_data["Sut"].append(t[2])
            df_data["Question"].append(u.question)
            df_data["Answer"].append(u.answer)
            df_data["Fitness"].append(fitness)
            df_data["Other"].append(other)
            df_data["Utterance"].append(u)
            label_algo_counts[label][t[1]] += 1
        total = len(testcases)
        for algorithm, count in label_algo_counts[label].items():
            label_algo_shares[label][algorithm] = count / total
        
        df = pd.DataFrame(df_data)
        df.to_csv(os.path.join(output_path, f"cluster_{label}.csv"), index=False)

    with open(os.path.join(output_path, "algorithm_counts.json"), "w") as f:
        json.dump(label_algo_counts, f, indent=4)
    with open(os.path.join(output_path, "algorithm_shares.json"), "w") as f:
        json.dump(label_algo_shares, f, indent=4)          