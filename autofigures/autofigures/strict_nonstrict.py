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

import numpy as np
from typing import Union, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from autofigures.utils import (
    colours,
    default_paths,
    plot_style,
    get_metrics,
    merge_scores,
)


def strict_nonstrict(
    output_folder: Optional[Union[Path, str]] = None,
    data_folder: Optional[Union[Path, str]] = None,
    add_markers: bool = True,
    first_metric: str = "auroc",
    second_metric: str = "mcc",
):
    if isinstance(add_markers, str):
        add_markers = bool(add_markers)

    plot_style()

    output_folder, data_folder = default_paths(output_folder, data_folder)

    df = merge_scores(output_folder, seeds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    f, axs = plt.subplots(1, 2, figsize=(6, 4.5), dpi=150)

    metrics = {}

    for model_type in ["nonstrict", "strict"]:
        model_name = (
            "squeezeprot_sp_nonstrict"
            if model_type == "nonstrict"
            else "squeezeprot_sp_strict"
        )
        metrics[model_type] = get_metrics(
            df, model_name, seeds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )

    for m, m_label in [
        (first_metric, "first_metric"),
        (second_metric, "second_metric"),
    ]:
        if m not in ["auroc", "mcc", "ap", "f1"]:
            raise ValueError(
                f"{m_label} must be one of ['auroc', 'mcc', 'ap', 'f1'], got '{m}'"
            )

    metric_names = [first_metric, second_metric]
    titles = ["C)", "D)"]
    width = 0.6

    for j in range(2):
        strict_mean = np.mean(metrics["strict"][metric_names[j]])
        nonstrict_mean = np.mean(metrics["nonstrict"][metric_names[j]])

        strict_std = np.std(metrics["strict"][metric_names[j]])
        nonstrict_std = np.std(metrics["nonstrict"][metric_names[j]])

        axs[j].bar(0, strict_mean, color=colours[1], edgecolor="k", lw=1, width=width)
        axs[j].bar(
            1,
            nonstrict_mean,
            color=colours[0],
            edgecolor="k",
            lw=1,
            width=width,
            hatch="//",
        )

        axs[j].set_xticks([0, 1], ["Strict", "Non-Strict"])
        axs[j].set_ylabel(metric_names[j].upper())
        axs[j].set_title(titles[j], loc="left", fontweight="bold")

        axs[j].set_xlim(-0.5, 1.5)

        if add_markers:
            axs[j].scatter(
                [0] * len(metrics["strict"][metric_names[j]]),
                metrics["strict"][metric_names[j]],
                ec="k",
                fc="none",
                s=100,
            )
            axs[j].scatter(
                [1] * len(metrics["nonstrict"][metric_names[j]]),
                metrics["nonstrict"][metric_names[j]],
                ec="k",
                fc="none",
                s=100,
            )
        else:
            axs[j].errorbar(0, strict_mean, yerr=strict_std, color="black", capsize=7)
            axs[j].errorbar(
                1, nonstrict_mean, yerr=nonstrict_std, color="black", capsize=7
            )

        print("---")
        print(f"Strict Mean ({metric_names[j]}) {strict_mean:.3}±{strict_std:.3}")
        print(
            f"Non-strict Mean ({metric_names[j]}) {nonstrict_mean:.3}±{nonstrict_std:.3}"
        )

        axs[j].grid()
        axs[j].set_axisbelow(True)

    for i in [0, 1]:
        if metric_names[i] == "mcc":
            axs[i].set_ylim(0, 0.9)
        else:
            axs[i].set_ylim(0.5, 0.9)

    plt.tight_layout()
    plt.savefig(output_folder / "figures/strict_nonstrict.svg")
