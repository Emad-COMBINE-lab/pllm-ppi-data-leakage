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

import matplotlib.pyplot as plt
from itertools import product
from typing import Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np
import math
from tqdm import tqdm
from autofigures.utils import fancy_model_names, default_paths, plot_style


def mcc(tp, fp, tn, fn):
    numerator = tp * tn - fp * fn
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / denominator


def bottom_tri(df):
    rows = []

    for row_idx, row in df.iterrows():
        if row.p1_len_bin > row.p2_len_bin:
            p1_len_bin = row.p1_len_bin
            p2_len_bin = row.p2_len_bin
        else:
            p1_len_bin = row.p2_len_bin
            p2_len_bin = row.p1_len_bin

        rows.append(
            {
                "p1_len": row.p1_len,
                "p2_len": row.p2_len,
                "p1_len_bin": p1_len_bin,
                "p2_len_bin": p2_len_bin,
                "y_hat": row.y_hat,
                "label": row.label,
                "tp": row.tp,
                "fp": row.fp,
                "tn": row.tn,
                "fn": row.fn,
            }
        )

    return pd.DataFrame(rows)


def length_heatmap(
    output_folder: Optional[Union[Path, str]] = None,
    data_folder: Optional[Union[Path, str]] = None,
):
    plot_style()

    output_folder, data_folder = default_paths(output_folder, data_folder)

    df_raw = pd.read_csv(output_folder / "tables/lengths_by_acc.csv")

    for model_name in tqdm(fancy_model_names):
        print(model_name)
        df = df_raw[df_raw["model_name"] == model_name]

        df = bottom_tri(df)

        df_summary = (
            df[["p1_len_bin", "p2_len_bin", "tp", "fp", "tn", "fn"]]
            .groupby(["p1_len_bin", "p2_len_bin"])
            .sum()
        )

        df_summary["n_pos"] = df_summary.apply(lambda row: row.tp + row.fp, axis=1)
        df_summary["n_neg"] = df_summary.apply(lambda row: row.tn + row.fn, axis=1)
        df_summary["n"] = df_summary.apply(
            lambda row: row.tp + row.fp + row.tn + row.fn, axis=1
        )
        df_summary["mcc"] = df_summary.apply(
            lambda row: mcc(row.tp, row.fp, row.tn, row.fn), axis=1
        )

        bin_size = 50
        bins = np.array([(x + 1) * bin_size for x in range(3000 // bin_size)])

        x = [x for x, _ in product(bins, bins)]
        y = [y for _, y in product(bins, bins)]

        scaling_fn = lambda x: 3 * x + 20

        f, ax = plt.subplots(figsize=(10, 10))
        # max 250
        # min 10
        plt.scatter(x, y, s=25, facecolor="#ccc", marker="x")
        plt.scatter(
            df_summary.index.get_level_values(0),
            df_summary.index.get_level_values(1),
            s=scaling_fn(df_summary.n),
            c=df_summary.mcc,
            edgecolor="#ccc",
            cmap="RdYlBu_r",
            vmin=-0.7,
            vmax=0.7,
        )
        plt.ylim(55, 1215)
        plt.xlim(55, 1215)
        plt.grid(False)
        plt.gca().set_aspect("equal")
        plt.xlabel("Protein 1 Sequence Length")
        plt.ylabel("Protein 2 Sequence Length")
        plt.colorbar(label="MCC", fraction=0.02, drawedges=False, location="right")
        # plt.title(f"MCC of Testing Interactions as a\nFunction of Amino-Acid Sequence Length ({fancy_model_names[model_name]})",
        #          loc="left", fontsize=16, fontweight='bold')
        plt.savefig(output_folder / f"figures/length_heatmap_{model_name}.svg")

    f, ax = plt.subplots(figsize=(10, 10))
    x = np.linspace(0, 100, 5)
    ax.scatter([600] * 5, np.arange(5) * 200, s=scaling_fn(x), c="black")
    plt.ylim(55, 1215)
    plt.xlim(55, 1215)
    plt.savefig(output_folder / f"figures/length_heatmap_sizes.svg")
