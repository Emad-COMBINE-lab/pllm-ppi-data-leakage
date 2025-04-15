# Code for "A flaw in using pre-trained pLLMs in protein-protein interaction inference models"
#
# Copyright (C) 2025 Joseph Szymborski
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
from pathlib import Path
from typing import Optional, Union, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
)
from autofigures.parse import (
    parse_dscript,
    parse_rapppid,
    parse_pipr,
    parse_sprint,
    parse_intrepppid,
    parse_richoux,
)

colours = [
    "#D81B60",
    "#1E88E5",
    "#FFC107",
    "#48820C",
    "#690455",
    "#FF8A10",
    "#22AB8C",
    "#ba27bc",
    "#ba27bc",
]

fancy_model_names = {
    "esm": "ESM2-650M",
    "prose": "ProSE",
    "proteinbert": "ProteinBERT",
    "prottrans_bert": "ProtBERT",
    "prottrans_t5": "ProtT5-XL",
    "squeezeprot_sp_strict": "SqueezeProt-SP-C3",
    "squeezeprot_sp_nonstrict": "SqueezeProt-SP-C1",
    "squeezeprot_u50": "SqueezeProt-U50",
    "rapppid": "RAPPPID",
    "intrepppid": "INTREPPPID",
    "dscript": "D-SCRIPT",
    "intrepppid": "INTREPPPID",
    "pipr": "PIPR",
    "sprint": "SPRINT",
    "richoux": "Richoux",
}


def get_script_dir():
    return Path(os.path.dirname(os.path.realpath(__file__)))


def default_paths(
    output_folder: Optional[Union[Path, str]] = None,
    data_folder: Optional[Union[Path, str]] = None,
):
    if isinstance(output_folder, str):
        output_folder = Path(output_folder)

    if isinstance(data_folder, str):
        data_folder = Path(data_folder)

    script_directory = get_script_dir()

    if output_folder is None:
        output_folder = script_directory / "../../out"

    if data_folder is None:
        data_folder = script_directory / "../../data"

    return output_folder, data_folder


def plot_style():
    plt.rcParams["font.family"] = "Inter"
    # plt.rcParams['font.family'] = 'Inclusive Sans'
    plt.rcParams["font.size"] = 13


def merge_scores(output_folder: Path, seeds: Optional[List] = None):
    seeds = [1, 2, 3] if seeds is None else seeds

    seed_dfs = []
    for seed in seeds:
        if not os.path.isfile(output_folder / f"tables/scores_s{seed}.csv"):
            raise IOError("Please run `get_scores` first.")
        else:
            seed_df = pd.read_csv(output_folder / f"tables/scores_s{seed}.csv")
            seed_df["seed"] = seed
            seed_dfs.append(seed_df)

    return pd.concat(seed_dfs)


def traditional_scores(data_folder: Path):
    dfs = [
        parse_dscript(
            data_folder
            / "results/traditional/dscript_9606x9606.eval.test.predictions.tsv"
        ),
        parse_intrepppid(
            data_folder
            / "results/traditional/intrepppid_8675309_9606_distinct-stylishly.json"
        ),
        parse_pipr(data_folder / "results/traditional/pipr_8675309_9606x9606_C3.json"),
        parse_rapppid(data_folder / "results/traditional/rapppid_C3_S8.json"),
        parse_sprint(data_folder / "results/traditional/sprint_9606x9606.c3.out"),
        parse_richoux(
            data_folder / "results/traditional/richoux_predict_8675309_9606x9606.pkl",
            data_folder
            / "results/traditional/richoux_test_labels_8675309_9606x9606.pkl",
        ),
    ]

    return pd.concat(dfs)


def traditional_metrics(data_folder: Path, model_name):
    df = pd.read_csv(data_folder / "results/traditional/summary.csv")

    metrics = dict()

    model_df = df[df["method"] == model_name]

    metrics["mcc"] = model_df["mcc_50"].values
    metrics["auroc"] = model_df["auroc"].values
    metrics["ap"] = model_df["ap"].values

    for metric_name in ["mcc", "auroc", "ap"]:
        metrics[f"avg_{metric_name}"] = np.mean(metrics[metric_name])
        metrics[f"std_{metric_name}"] = np.std(metrics[metric_name])

    return metrics


def get_metrics(df: pd.DataFrame, model_name: str, seeds: Optional[List] = None):
    seeds = [1, 2, 3] if seeds is None else seeds

    metric_fn = {
        "mcc": matthews_corrcoef,
        "auroc": roc_auc_score,
        "ap": average_precision_score,
        "f1": f1_score,
        "acc": accuracy_score,
    }

    metrics = {metric_name: [] for metric_name in metric_fn}

    for seed in seeds:
        c_df = df[(df.seed == seed) & (df.model_name == model_name)]

        for metric_name in metric_fn:
            if metric_name in ["mcc", "f1", "acc"]:
                y_hat = (c_df.y_hat > 0.5).astype(int)
            else:
                y_hat = c_df.y_hat

            metric = metric_fn[metric_name](c_df.label, y_hat)
            metrics[metric_name].append(metric)

    for metric_name in metrics.copy():
        metrics[f"avg_{metric_name}"] = np.mean(metrics[metric_name])
        metrics[f"std_{metric_name}"] = np.std(metrics[metric_name])

    return metrics
