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


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path
from typing import Optional, Union
from autofigures.utils import (
    default_paths,
    merge_scores,
    get_metrics,
    colours,
    plot_style,
    fancy_model_names,
)


def speed(
    output_folder: Optional[Union[Path, str]] = None,
    data_folder: Optional[Union[Path, str]] = None,
    y_metric: str = "mcc",
):
    plot_style()

    model_names = [
        "prottrans_t5",
        "esm",
        "prottrans_bert",
        "squeezeprot_u50",
        "proteinbert",
        "prose",
    ]

    output_folder, data_folder = default_paths(output_folder, data_folder)

    speed_df = pd.read_csv(data_folder / "results/speed.csv").set_index("model_name")
    scores_df = merge_scores(output_folder)

    results = dict()

    for model_name in model_names:
        model_metrics = get_metrics(scores_df, model_name)

        results[model_name] = model_metrics
        results[model_name]["aa_per_sec"] = float(
            speed_df.loc[model_name]["aa_per_sec"]
        )

    f, ax = plt.subplots()

    for model_idx, model_name in enumerate(model_names):
        model_results = results[model_name]

        colour = colours[model_idx]
        marker = "o"
        s = 100

        for i in range(3):
            ax.scatter(
                model_results["aa_per_sec"],
                model_results[y_metric][i],
                edgecolor="k",
                facecolor=colour,
                s=s,
                linewidth=1,
                marker=marker,
                label=model_name,
                alpha=0.7,
            )

        x_margin = 1.06
        y_margin = 1

        fancy_name = fancy_model_names[model_name]

        if model_name == "prottrans_t5":
            x_margin = 1.025
            y_margin = 1.07
        elif model_name == "squeezeprot_u50":
            x_margin = 0.70
            y_margin = 1.08
        elif model_name == "proteinbert":
            y_margin = 0.9
            x_margin = 1.1
        elif model_name == "prose":
            y_margin = 0.82
            x_margin = 0.8
        elif model_name == "esm":
            x_margin = 1.13
        elif model_name == "prottrans_bert":
            x_margin = 1.1
            y_margin = 0.98

        plt.annotate(
            fancy_name,
            (
                model_results["aa_per_sec"] * x_margin,
                model_results[f"avg_{y_metric}"] * y_margin,
            ),
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            horizontalalignment="left",
        )

    # ax.set_xscale('log')
    ax.set_ylim(0, 1)
    ax.set_xlabel("Tokens per Second\n(×10³)")
    ax.set_ylabel("Predictive Performance\n(MCC)")

    ax.set_xticks([5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000])
    ax.set_xticklabels([5, 10, 15, 20, 25, 30, 35, 40])

    ax.grid(which="both")
    ax.set_axisbelow(True)
    plt.savefig(output_folder / "figures/speed.svg")
