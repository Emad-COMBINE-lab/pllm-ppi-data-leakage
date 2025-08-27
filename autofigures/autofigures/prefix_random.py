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

from autofigures.utils import plot_style, default_paths, colours, get_metrics, merge_scores
from typing import Union, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import metrics as m
from csv import DictWriter
#import matplotlib.pyplot as plt
#import matplotlib.patheffects as pe


def compute_table(
    output_folder: Union[Path, str],
    df: pd.DataFrame
):
    model_names = [
        "squeezeprot_sp_nonstrict",
        "squeezeprot_sp_strict",
        "proteinbert",
        "prottrans_bert",
        "prottrans_t5",
    ]

    metric_names = ["mcc", "auroc", "ap", "f1", "acc"]

    metrics = {
        "prefix": dict(),
        "random": dict()
    }

    print("Computing Metrics...")
    for model_type in ["prefix", "random"]:
        for model_name in model_names:

            if model_type == "random":
                model_name2 = model_name + "_random"
            else:
                model_name2 = model_name

            metrics[model_type][model_name] = get_metrics(
                df, model_name2
            )

    out_path = output_folder / "tables/prefix_random.csv"
    
    print("Writing Table...")
    with open(out_path, 'w') as f:
        csv_writer = DictWriter(f, ['type', 'name', 'metric', 'avg', 'std', 'seed1', 'seed2', 'seed3'])

        csv_writer.writeheader()

        for model_type in ["prefix", "random"]:
            for model_name in model_names:
                for metric_name in metric_names:

                    value = metrics[model_type][model_name][metric_name]

                    csv_writer.writerow({
                        "type": model_type,
                        "name": model_name,
                        "metric": metric_name,
                        "avg": np.mean(value),
                        "std": np.std(value),
                        "seed1": value[0],
                        "seed2": value[1],
                        "seed3": value[2]
                    })


def prefix_random(
    output_folder: Optional[Union[Path, str]] = None,
    data_folder: Optional[Union[Path, str]] = None,
):
    output_folder, data_folder = default_paths(output_folder, data_folder)

    df = merge_scores(output_folder)

    compute_table(output_folder, df)

    
