from opensbt.utils.wandb import download_run_artifacts, download_run_artifacts_relative, get_run_table, get_summary
from opensbt.visualization.utils import AnalysisPlots, AnalysisException, AnalysisTables, AnalysisDiversity
import matplotlib.pyplot as plt
import os
import numpy as np
import json
from collections import defaultdict
import pandas as pd
from typing import List, Tuple, Literal, Optional
import pickle
import tqdm

from llm.utils.embeddings_local import get_embedding as get_embedding_local
from llm.utils.embeddings_openai import get_embedding as get_embedding_openai
from llm.model.models import Utterance

def filter_safety(runs):
    res = []
    take = False
    for run in runs:
        if run.id == "wqpwm2wj":
            take = True
        if take and run.state == "finished":
            res.append(run)
    return res


def filter_mt(runs):
    res = []
    take = False
    for run in runs:
        # if run.id == "x4gpto3o":
        #     take = True
        take = True
        if take and run.state == "finished":
            res.append(run)
    return res


run_filters = {"SafeLLM": filter_safety, 
               "MultiTurnTest": filter_mt,
                "TestNaviIndustry": filter_mt,
                "TestNaviYelp": filter_mt,
                "NaviIndustry": filter_mt,
                "NaviYelp": filter_mt,
                "MultiTurnTestDiscrete": filter_mt,
                "MultiTurnDiscrete": filter_mt

}


def convert_name(run_name: str) -> str:
    """
    Convert a run name into a standardized SUT string by scanning each
    word (separated by '_') in order and including all matching identifiers.
    """
    sut_keywords = {
        "ipa": "",
        "yelp": "",
        "gpt-4o": "GPT-4o",
        "gpt-5-chat": "GPT-5-Chat",
        "deepseek-v3-0324": "DeepSeek-V3",
        "industry": "",
        "mistral": "Mistral-7B",
        "qwen3" : "Qwen3-8B",
        "deepseek-v2" : "DeepSeek-V2-16B"
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


metric_names = {
    "failures": "Number of Failures",
    "critical_ratio": "Critical Ratio",
}

algo_names_map = {
    ("gs", "astral"): "ASTRAL",
    ("gs", "extended"): "T-wise",
    "gs": "T-wise",
    "rs": "Random",
    "nsga2": "MT-Test",  # used to be STELLAR
    "nsga2d": "NSGAIID",
    "sensei": "SENSEI"
}

algorithms = [
    "ASTRAL",
    "T-wise",
    "Random",
    "MT-Test",  # STELLAR
    "NSGAIID",
    "NSGAII"
]


def get_algo_name(algo, features):
    algo_name = algo_names_map.get((algo, features), None)
    if algo_name is None:
        algo_name = algo_names_map[algo]
    return algo_name


def plot_metric_vs_time(
    project="SafeLLM",
    team="opentest",
    size=(18, 6),
    metric="failures",
    time_in_minutes=120,
    file_name="plot",
):
    if project not in run_filters:
        raise AnalysisException(
            "Plesase implement runs filter for yout project in opensbt.visualization.llm_figures"
        )
    artifact_paths = download_run_artifacts(f"{team}/{project}", run_filters[project])
    fig, axes = plt.subplots(1, len(artifact_paths), sharey="row", sharex="all")
    if len(artifact_paths) == 0:
        print("No artifact paths found for the given project and team.")
        return 
    if len(artifact_paths) == 1:
        axes = [axes]
    fig.set_size_inches(*size)
    fig.supxlabel("Time, min")
    fig.supylabel(metric_names[metric])
    tick_count = time_in_minutes // 30 + 1
    ticks_kwargs = {"xticks": np.linspace(0, time_in_minutes, tick_count)}
    if metric == "critical_ratio":
        ticks_kwargs["yticks"] = np.linspace(0.0, 1.0, 6)
        ticks_kwargs["ylim"] = (0.0, 1.0)
    plt.setp(axes, **ticks_kwargs)
    for i, (sut, algos) in enumerate(artifact_paths.items()):
        for (algo, features), paths in algos.items():
            algo_name = get_algo_name(algo, features)
            dfs = [get_run_table(path) for path in paths]
            AnalysisPlots.plot_with_std(axes[i], dfs, label=algo_name, metric=metric, target_time=time_in_minutes)
        axes[i].set_title(convert_name(sut.capitalize()))
        axes[i].set_box_aspect(1)
        axes[i].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.85) 
    legend_handles = {}
    for label, col in AnalysisPlots.label_colors.items():
        legend_handles[label] = plt.Line2D([0], [0], color=col, lw=10)
    # fig.legend(legend_handles.values(), legend_handles.keys(), title="Labels", loc="upper right")
    plt.tight_layout()
    plt.savefig(file_name, format="pdf")


def plot_metric_vs_time_separate(
    project="SafeLLM",
    team="opentest",
    size=(12, 6),
    metric="failures",
    time_in_minutes=120,
    file_name="plot",
):
    if project not in run_filters:
        raise AnalysisException(
            "Plesase implement runs filter for yout project in opensbt.visualization.llm_figures"
        )
    artifact_paths = download_run_artifacts(f"{team}/{project}", run_filters[project])
    fig = plt.figure()
    fig.set_size_inches(size[0], size[1])
    for i, (sut, algos) in enumerate(artifact_paths.items()):
        for (algo, features), paths in algos.items():
            algo_name = get_algo_name(algo, features)
            dfs = [get_run_table(path) for path in paths]
            AnalysisPlots.plot_with_std(plt, dfs, label=algo_name, metric=metric, target_time=time_in_minutes)
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

def boxplots(
    project="SafeLLM",
    team="opentest",
    size=(18, 6),
    metric="failures",
    file_name="plot",
):
    if project not in run_filters:
        raise AnalysisException(
            "Plesase implement runs filter for yout project in opensbt.visualization.llm_figures"
        )
    artifact_paths = download_run_artifacts(f"{team}/{project}", run_filters[project])
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
            dfs = [get_run_table(path) for path in paths]
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
    plt.savefig(file_name, format="pdf")


def boxplots_separate(
    project="SafeLLM",
    team="opentest",
    size=(12, 6),
    metric="failures",
    file_name="plot",
):
    if project not in run_filters:
        raise AnalysisException(
            "Plesase implement runs filter for yout project in opensbt.visualization.llm_figures"
        )
    artifact_paths = download_run_artifacts(f"{team}/{project}", run_filters[project])
    fig = plt.figure()
    fig.set_size_inches(*size)
    # fig.supylabel(metric_names[metric])
    # Ensure axes is always a list
    for i, (sut, algos) in enumerate(artifact_paths.items()):
        name_to_dfs = {}
        for (algo, features), paths in algos.items():
            algo_name = get_algo_name(algo, features)
            dfs = [get_run_table(path) for path in paths]
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


def statistics_table(
    project="SafeLLM",
    team="opentest",
    metric="failures",
    path="table.csv"
):
    if project not in run_filters:
        raise AnalysisException(
            "Plesase implement runs filter for yout project in opensbt.visualization.llm_figures"
        )
    artifact_paths = download_run_artifacts(f"{team}/{project}", run_filters[project])
    print("artifact_paths:", artifact_paths)
    statistics = {}
    suts = []
    for i, (sut, algos) in enumerate(artifact_paths.items()):
        suts.append(sut)
        algo_names = [get_algo_name(*key) for key in algos.keys()]
        if metric == "failures":
            values = [
                [float(get_summary(path)["Number Critical Scenarios (duplicate free)"]) for path in paths] for paths in algos.values()
            ]
        elif metric == "critical_ratio":
            values = [
                [float(get_summary(path)["Number Critical Scenarios (duplicate free)"]) / float(get_summary(path)["Number All Scenarios"]) for path in paths] for paths in algos.values()
            ]
        else:
            raise AnalysisException("Unknown metric")
        statistics[sut] = AnalysisTables.statistics(values, algo_names)
    data = defaultdict(list)
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
    df = pd.DataFrame(data)
    # df = df.dropna()
    df.to_csv(path, index=False)


def get_embeddings(artifact_directory_path: str, critical_only: bool = True,
                   local: bool = True, input: bool = True) -> Tuple[List[Utterance], np.ndarray]:
    file = "embeddings.pkl" if input else "embeddings_output.pkl"
    pickle_path = os.path.join(artifact_directory_path, file)
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            return pickle.load(f)
    
    print(f"Calculating embeddings for {artifact_directory_path}")
    json_path = "all_critical_utterances.json" if critical_only else "all_utterances.json"
    with open(os.path.join(artifact_directory_path, json_path), "r", encoding="utf8") as f:
        utterances_raw = json.load(f)
    utterances = [Utterance.model_validate(u["utterance"]) for u in utterances_raw]

    get_embedding = get_embedding_local if local else get_embedding_openai
    if input:
        embeddings = [
            np.array(get_embedding(u.question)).reshape(1, -1) for u in utterances if u.question is not None
        ]
    else:
        embeddings = [
            np.array(get_embedding(u.answer)).reshape(1, -1) for u in utterances if u.answer is not None
        ]
    embeddings = np.concatenate(embeddings)
    with open(pickle_path, "wb") as f:
        pickle.dump((utterances, embeddings), f)
    return utterances, embeddings


def diversity_report(
        project="SafeLLM",
        team="opentest",
        output_path="diversity",
        input: bool = True,
        max_num_clusters = 150,
        silhouette_threshold = 20,
        visualize: bool = True,
        mode: Literal["separated", "merged"] = "separated",
        num_seeds: Optional[int] = None,
        save_folder: str = None
    ):
    if mode == "merged" and num_seeds is None:
        raise AnalysisException("Provide the number of seed in merged mode")
    output_path = os.path.join(output_path, "input" if input else "output")

    artifact_paths = download_run_artifacts(f"{team}/{project}", run_filters[project])
    
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


def last_values_table(
    project: str = "SafeLLM",
    team: str = "opentest",
    metrics: list | str = "failures",
    path: str = "table.csv",
):
    save_path = path

    if run_filters is not None and project not in run_filters:
        raise AnalysisException(
            "Please implement runs filter for your project in opensbt.visualization.llm_figures"
        )

    artifact_paths = download_run_artifacts(f"{team}/{project}", run_filters[project])

    if isinstance(metrics, str):
        metrics = [metrics]

    summary_data = []

    for sut, algos in artifact_paths.items():
        algo_names = [get_algo_name(*key) for key in algos.keys()]
        print("algos:", algo_names)

        for algo_name, paths in zip(algo_names, algos.values()):
            row = {"Algorithm": algo_name,"SUT": sut}

            for metric in metrics:
                if metric == "failures":
                    values = [int(get_summary(path)["Number Critical Scenarios (duplicate free)"]) for path in paths]
                elif metric == "critical_ratio":
                    values = []
                    for path in paths:
                        ratio = float(get_summary(path)["Number Critical Scenarios (duplicate free)"]) / float(get_summary(path)["Number All Scenarios"])
                        values.append(ratio)
                else:
                    raise AnalysisException(f"Unknown metric: {metric}")

                row[f"{metric}_mean"] = np.mean(values) if values else np.nan
                row[f"{metric}_std"] = np.std(values, ddof=1) if len(values) > 1 else 0.0

            summary_data.append(row)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    algorithm_order = ["MT-Test", "Random", "T-wise", "ASTRAL"]
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