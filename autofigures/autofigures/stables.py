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
from pathlib import Path
from typing import Optional, Union
import csv
import json

import numpy as np

from autofigures.utils import fancy_model_names, default_paths, get_metrics, merge_scores, sigfig, traditional_metrics
import pandas as pd


def all_stables(output_folder: Optional[Union[Path, str]] = None, data_folder: Optional[Union[Path, str]] = None):
    print("Table 1")
    table1(output_folder, data_folder)
    print("\tDONE")

    print("STable 2")
    stable2(output_folder, data_folder)
    print("\tDONE")

    print("STable 4")
    stable4(output_folder, data_folder)
    print("\tDONE")

    print("STable 5")
    stable5(output_folder, data_folder)
    print("\tDONE")

def table1(output_folder: Optional[Union[Path, str]] = None, data_folder: Optional[Union[Path, str]] = None):
    output_folder, data_folder = default_paths(output_folder, data_folder)

    df1 = merge_scores(output_folder, seeds=[1, 2, 3])
    df2 = merge_scores(output_folder, seeds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    metric_names = ["auroc", "ap", "mcc"]
    model_names = ["prottrans_t5", "esm", "prottrans_bert", "squeezeprot_u50", "prose", "proteinbert", "squeezeprot_sp_nonstrict", "squeezeprot_sp_strict", 'rapppid', 'dscript', 'richoux', 'sprint', 'pipr']

    with open(output_folder / "tables/table1.csv", "w", newline="") as f:
        csv_writer = csv.writer(f)

        for model_name in model_names:

            if model_name in ['rapppid', 'dscript', 'richoux', 'sprint', 'pipr']:
                marker = '✘'
                metrics = traditional_metrics(
                    data_folder, model_name
                )
            else:
                marker = '✔'
                if model_name in ["squeezeprot_sp_nonstrict", "squeezeprot_sp_strict"]:
                    df = df2
                else:
                    df = df1

                metrics = get_metrics(
                    df, model_name, seeds=[1, 2, 3]
                )

            row = [fancy_model_names[model_name], marker]

            for metric_name in metric_names:
                row += [sigfig(np.mean(metrics[metric_name]))]
                row += [sigfig(np.std(metrics[metric_name]))]

            csv_writer.writerow(row)


def stable2(output_folder: Optional[Union[Path, str]] = None, data_folder: Optional[Union[Path, str]] = None):

    output_folder, data_folder = default_paths(output_folder, data_folder)

    df = merge_scores(output_folder, seeds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    metrics = {}
    metric_names = ["mcc", "auroc", "ap", "f1", "acc"]

    for model_type in ["nonstrict", "strict"]:
        model_name = (
            "squeezeprot_sp_nonstrict"
            if model_type == "nonstrict"
            else "squeezeprot_sp_strict"
        )
        metrics[model_type] = get_metrics(
            df, model_name, seeds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )

    with open(output_folder / "tables/stable2.csv", "w", newline="") as f:
        csv_writer = csv.writer(f)

        csv_writer.writerow(["","MCC@50%", "MCC@50%", "AUROC", "AUROC", "AP", "AP", "F-1 Score", "F-1 Score", "Accuracy", "Accuracy"])
        csv_writer.writerow(["Seed", "Non-strict", "Strict", "Non-strict", "Strict","Non-strict", "Strict","Non-strict", "Strict","Non-strict", "Strict"])

        for seed in range(10):
            row = [seed + 1]
            for metric_name in metric_names:
                for strictness in ["nonstrict", "strict"]:
                    fig = sigfig(metrics[strictness][metric_name][seed])
                    row += [fig]

            csv_writer.writerow(row)

        for agg in ["Mean", "StDev"]:
            if agg == "Mean":
                agg_fn = np.mean
            elif agg == "StDev":
                agg_fn = np.std

            row = [agg]
            for metric_name in metric_names:
                for strictness in ["nonstrict", "strict"]:
                    fig = agg_fn(metrics[strictness][metric_name])
                    fig = sigfig(fig)
                    row += [fig]

            csv_writer.writerow(row)

def stable4(output_folder: Optional[Union[Path, str]] = None, data_folder: Optional[Union[Path, str]] = None):

    output_folder, data_folder = default_paths(output_folder, data_folder)

    raw = {'snc': None, 'kappa': None}

    with open(output_folder / "tables/snc.json") as f:
        raw['snc'] = json.load(f)

    with open(output_folder / "tables/kappa.json") as f:
        raw['kappa'] = json.load(f)

    plm_models = ["esm", "prottrans_t5", "prottrans_bert", "prose", "proteinbert", "dscript"]
    nonplm_models = ["rapppid", "richoux", "sprint", "pipr"]

    with open(output_folder / "tables/stable4.csv", "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["","", "SqueezeProt-SP (Non-strict)", "SqueezeProt-SP (Strict)"])
        csv_writer.writerow(["Method", "pLM-based?", "Cohen's Kappa", "SNC", "Cohen's Kappa", "SNC"])

        for plm_model in plm_models:
            kappa_nonstrict = raw['kappa']['plm_nonstrict']['kappa'][plm_model]
            snc_nonstrict = raw['snc']['plm_nonstrict']['concordance'][plm_model]
            kappa_strict = raw['kappa']['plm_strict']['kappa'][plm_model]
            snc_strict = raw['snc']['plm_strict']['concordance'][plm_model]
            row = [fancy_model_names[plm_model], '✔', sigfig(kappa_nonstrict), sigfig(snc_nonstrict), sigfig(kappa_strict), sigfig(snc_strict)]
            csv_writer.writerow(row)

        kappa_nonstrict = list(raw['kappa']['plm_nonstrict']['kappa'].values())
        snc_nonstrict = list(raw['snc']['plm_nonstrict']['concordance'].values())
        kappa_strict = list(raw['kappa']['plm_strict']['kappa'].values())
        snc_strict = list(raw['snc']['plm_strict']['concordance'].values())

        for agg in ["Mean", "StDev"]:
            if agg == "Mean":
                agg_fn = np.mean
            elif agg == "StDev":
                agg_fn = np.std

            row = ["", agg, sigfig(agg_fn(kappa_nonstrict)), sigfig(agg_fn(snc_nonstrict)), sigfig(agg_fn(kappa_strict)), sigfig(agg_fn(snc_strict))]
            csv_writer.writerow(row)

        for nonplm_model in nonplm_models:
            kappa_nonstrict = raw['kappa']['nonplm_nonstrict']['kappa'][nonplm_model]
            snc_nonstrict = raw['snc']['nonplm_nonstrict']['concordance'][nonplm_model]
            kappa_strict = raw['kappa']['nonplm_strict']['kappa'][nonplm_model]
            snc_strict = raw['snc']['nonplm_strict']['concordance'][nonplm_model]
            row = [fancy_model_names[nonplm_model], '✘', sigfig(kappa_nonstrict), sigfig(snc_nonstrict), sigfig(kappa_strict), sigfig(snc_strict)]
            csv_writer.writerow(row)

        kappa_nonstrict = list(raw['kappa']['nonplm_nonstrict']['kappa'].values())
        snc_nonstrict = list(raw['snc']['nonplm_nonstrict']['concordance'].values())
        kappa_strict = list(raw['kappa']['nonplm_strict']['kappa'].values())
        snc_strict = list(raw['snc']['nonplm_strict']['concordance'].values())

        for agg in ["Mean", "StDev"]:
            if agg == "Mean":
                agg_fn = np.mean
            elif agg == "StDev":
                agg_fn = np.std

            row = ["", agg, sigfig(agg_fn(kappa_nonstrict)), sigfig(agg_fn(snc_nonstrict)),
                   sigfig(agg_fn(kappa_strict)), sigfig(agg_fn(snc_strict))]
            csv_writer.writerow(row)

def stable5(output_folder: Optional[Union[Path, str]] = None, data_folder: Optional[Union[Path, str]] = None):
    output_folder, data_folder = default_paths(output_folder, data_folder)

    df = pd.read_csv(output_folder / "tables/prefix_random.csv")

    def get_avg(window_type, model_name, metric):
        return df[(df['type'] == window_type) & (df['name'] == model_name) & (df['metric'] == metric)].iloc[0].avg

    def perc_diff(x1, x2):
        return (x2 - x1) / x1

    model_names = ["proteinbert", "prottrans_bert", "prottrans_t5", "squeezeprot_sp_nonstrict", "squeezeprot_sp_strict"]

    with open(output_folder / "tables/stable5.csv", "w") as f:
        csv_writer = csv.writer(f)

        csv_writer.writerow(['Encoder', "% Change MCC", "% Change AUROC", "% Change AP", "% Change F1", "% Change Accuracy"])

        for model_name in model_names:
            row = [fancy_model_names[model_name]]
            for metric_name in ['mcc', 'auroc', 'ap', 'f1', 'acc']:
                prefix = get_avg('prefix', model_name, metric_name)
                random = get_avg('random', model_name, metric_name)
                diff = sigfig(perc_diff(prefix, random))
                row += [diff]

            csv_writer.writerow(row)