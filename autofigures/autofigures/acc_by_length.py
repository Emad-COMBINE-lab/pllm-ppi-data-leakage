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
from os.path import isfile
from typing import Optional, Union
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from autofigures.dataset import RapppidDatasetSeq
from autofigures.utils import (
    default_paths,
    plot_style,
    merge_scores,
    traditional_scores,
    fancy_model_names,
    colours,
)


def get_length_df(data_folder: Path, bins):
    dataset_path = (
        data_folder
        / "ppi/rapppid_[common_string_9606.protein.links.detailed.v12.0_upkb.csv]_Mz70T9t-4Y-i6jWD9sEtcjOr0X8=.h5"
    )

    print("loading dataset...")
    dataset = RapppidDatasetSeq(dataset_path, 3, "test")
    print("done.")

    lengths_rows = []

    for i in tqdm(range(len(dataset))):
        seq1, seq2, label = dataset[i]

        lengths_rows.append({"p1_len": len(seq1), "p2_len": len(seq2), "label": label})

    lengths_df = pd.DataFrame(lengths_rows)
    lengths_df["p1_len_bin"] = bins[np.digitize(lengths_df.p1_len, bins, right=True)]
    lengths_df["p2_len_bin"] = bins[np.digitize(lengths_df.p2_len, bins, right=True)]

    return lengths_df


def get_length_by_acc_df(
    lengths_df: pd.DataFrame, output_folder: Path, data_folder: Path
):
    pllm_df = merge_scores(output_folder, seeds=[1, 2, 3])
    trad_df = traditional_scores(data_folder)

    rows = []

    for ppi_df in [pllm_df, trad_df]:
        for model_name in tqdm(ppi_df.model_name.unique()):
            model_df = ppi_df[
                (ppi_df["model_name"] == model_name) & (ppi_df["seed"] == 1)
            ]
            for (length_idx, length_row), (ppi_idx, ppi_row) in zip(
                lengths_df.iterrows(), model_df.iterrows()
            ):
                pair_len = (
                    length_row.p1_len
                    if length_row.p1_len > length_row.p2_len
                    else length_row.p2_len
                )
                pair_bin = (
                    length_row.p1_len_bin
                    if length_row.p1_len_bin > length_row.p2_len_bin
                    else length_row.p2_len_bin
                )

                row = {
                    "p1_len": length_row["p1_len"],
                    "p2_len": length_row["p2_len"],
                    "p1_len_bin": length_row["p1_len_bin"],
                    "p2_len_bin": length_row["p2_len_bin"],
                    "pair_len": pair_len,
                    "pair_bin": pair_bin,
                    "y_hat": ppi_row["y_hat"],
                    "label": ppi_row["label"],
                    "model_name": ppi_row["model_name"],
                    "tp": 1 if ppi_row.label == 1 and ppi_row.y_hat >= 0.5 else 0,
                    "fp": 1 if ppi_row.label == 0 and ppi_row.y_hat >= 0.5 else 0,
                    "tn": 1 if ppi_row.label == 0 and ppi_row.y_hat < 0.5 else 0,
                    "fn": 1 if ppi_row.label == 1 and ppi_row.y_hat < 0.5 else 0,
                }

                rows.append(row)

    df = pd.DataFrame(rows)

    df["t"] = df["tp"] | df["tn"]
    df["f"] = df["fp"] | df["fn"]

    return df


def acc_by_length(
    output_folder: Optional[Union[Path, str]] = None,
    data_folder: Optional[Union[Path, str]] = None,
):
    plot_style()
    plt.rcParams["font.size"] = 9

    output_folder, data_folder = default_paths(output_folder, data_folder)
    bin_size = 50
    bins = np.array([(x + 1) * bin_size for x in range(3000 // bin_size)])

    if isfile(output_folder / "tables/pair_lengths.csv"):
        lengths_df = pd.read_csv(output_folder / "tables/pair_lengths.csv")
    else:
        lengths_df = get_length_df(data_folder, bins)
        lengths_df.to_csv(output_folder / "tables/pair_lengths.csv")

    if isfile(output_folder / "tables/lengths_by_acc.csv"):
        length_by_acc_df = pd.read_csv(output_folder / "tables/lengths_by_acc.csv")
    else:
        length_by_acc_df = get_length_by_acc_df(lengths_df, output_folder, data_folder)
        length_by_acc_df.to_csv(output_folder / "tables/lengths_by_acc.csv")

    model_names = [
        "squeezeprot_sp_strict",
        "squeezeprot_sp_nonstrict",
        "prottrans_bert",
        "esm",
        "proteinbert",
        "prottrans_t5",
        "prose",
    ]

    context_lengths = {
        "esm": 1024,
        "proteinbert": 1024,
        "prottrans_bert": 2048,
        "prottrans_t5": 512,
        "squeezeprot_u50": 512,
        "squeezeprot_sp_strict": 512,
        "squeezeprot_sp_nonstrict": 512,
        "prose": None,
    }

    f, axs = plt.subplots(8, 1, figsize=(8, 7), sharex=True)

    for idx, model_name in enumerate(model_names):
        if model_name == "squeezeprot_sp_strict":
            tr_color = colours[0]
            fr_color = "#620033"
        else:
            tr_color = colours[1]  # '#1E88E5'
            fr_color = "#023059"  # colours[4] #'#B61651'  # '#D81B60'

        counts_t, hist_bins = np.histogram(
            length_by_acc_df[
                (length_by_acc_df.model_name == model_name)
                & (length_by_acc_df["t"] == 1)
            ].pair_bin,
            bins,
        )
        counts_f, hist_bins = np.histogram(
            length_by_acc_df[
                (length_by_acc_df.model_name == model_name)
                & (length_by_acc_df["f"] == 1)
            ].pair_bin,
            bins,
        )

        print(model_name, counts_t, counts_f)
        counts_tr = counts_t / (counts_t + counts_f)
        counts_fr = counts_f / (counts_t + counts_f)

        axs[idx].bar(np.arange(59), counts_tr, width=1.0, color=tr_color)
        axs[idx].bar(
            np.arange(59), counts_fr, bottom=counts_tr, width=1.0, color=fr_color
        )

        xtick = np.array([i * 10 for i in range(7)])
        axs[idx].set_xticks(xtick, xtick * 50)

        xtick = np.array([i for i in range(60)])
        axs[idx].set_xticks(xtick, minor=True)

        axs[idx].set_xlim(0, 60)

        if model_name == "squeezeprot_sp_strict":
            fancy_name = "SqueezeProt-SP\n(strict)"
        elif model_name == "squeezeprot_sp_nonstrict":
            fancy_name = "SqueezeProt-SP\n(non-strict)"
        else:
            fancy_name = fancy_model_names[model_name]

        axs[idx].annotate(fancy_name, (0.6, 0.1), c="white", weight="semibold")

        context_length = context_lengths[model_name]

        if context_length is not None:
            axs[idx].axvline(context_length / 50, color="w", ls=":")

        if idx == 2:
            axs[idx].annotate(
                "Context Length",
                ((context_length / 50) - 8.5, 0.2),
                color="w",
                weight="semibold",
                fontsize=9,
            )
        elif context_length is not None:
            print(context_length)

            axs[idx].annotate(
                "Context Length",
                ((context_length / 50) * 1.02, 0.2),
                color="w",
                weight="semibold",
                fontsize=9,
            )

        if idx == 3:
            axs[idx].set_ylabel("Proportion")

    axs[-1].bar(np.arange(59), counts_t + counts_f, width=1.0, color="k")
    axs[-1].set_ylabel("# Pairs")

    plt.tight_layout()
    plt.xlabel("Max. Sequence Length")
    plt.savefig(output_folder / "figures/acc_by_length.svg")
