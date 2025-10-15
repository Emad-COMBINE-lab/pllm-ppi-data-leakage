# Code for "A flaw in using pre-trained pLMs in protein-protein interaction inference models"
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

from autofigures.dataset import RapppidDataset
from typing import Union, Optional
from pathlib import Path
from autofigures.utils import default_paths, plot_style, fancy_model_names, colours
from collections import defaultdict, Counter
from scipy.stats import pearsonr, false_discovery_control
import numpy as np
import pandas as pd
import csv
import gzip
import pickle
import os.path
import matplotlib.pyplot as plt
from palettable.cubehelix import get_map
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter

MODELS = ["esm", "prose", "proteinbert", "prottrans_bert", "prottrans_t5", "squeezeprot_u50", "squeezeprot_sp_strict", "squeezeprot_sp_nonstrict", "dscript"]
STRICT_INDEX = MODELS.index("squeezeprot_sp_strict")
NONSTRICT_INDEX = MODELS.index("squeezeprot_sp_nonstrict")


def count_coincident_points(model_expected_counts, model_observed_counts):
    """
    Count the number of coincident points at each (x, y) position.

    Parameters:
    model_expected_counts: array-like, x coordinates
    model_observed_counts: array-like, y coordinates

    Returns:
    numpy array of same size with count of points at each (x, y) position
    """
    # Convert to numpy arrays if they aren't already
    x = np.array(model_expected_counts)
    y = np.array(model_observed_counts)

    # Create (x, y) coordinate pairs as tuples
    points = list(zip(x, y))

    # Count occurrences of each unique (x, y) pair
    point_counts = Counter(points)

    # Create result array with count for each original point
    density_counts = np.array([point_counts[point] for point in points])

    return density_counts

def count_dataset_proteins(
    ppi_path: Union[Path, str]
):
    c_type = 3
    split = 'test'

    dataset = RapppidDataset(ppi_path, c_type, split)

    positives_dict = defaultdict(lambda: 0)
    negatives_dict = defaultdict(lambda: 0)

    for i in range(len(dataset)):
        p1, p2, label = dataset[i]

        if label == 1:
            positives_dict[p1] += 1
            positives_dict[p2] += 1

        if label == 0:
            negatives_dict[p1] += 1
            negatives_dict[p2] += 1

    return positives_dict, negatives_dict

def count_result_proteins(
    ppi_path: Union[Path, str],
    model_name: str,
    output_folder: Union[Path, str],
    data_folder: Union[Path, str]
):
    c_type = 3
    split = 'test'

    dataset = RapppidDataset(ppi_path, c_type, split)

    positives_dict = defaultdict(lambda: 0)
    negatives_dict = defaultdict(lambda: 0)

    if model_name == "dscript":
        seed_path = data_folder / f"results/traditional/dscript_9606x9606.eval.test.predictions.tsv"
        seed_df = pd.read_csv(seed_path, delimiter='\t', names=['p1', 'p2', 'y_true', 'y_hat'])
    else:
        seed_path = output_folder / f"tables/scores_s1.csv"
        seed_df = pd.read_csv(seed_path)
        seed_df = seed_df[seed_df.model_name == model_name]


    y_hats = []

    for row_idx, row in seed_df.iterrows():
        y_hats.append(row.y_hat)

    for i, y_hat in zip(range(len(dataset)), y_hats):
        p1, p2, label = dataset[i]

        if y_hat > 0.5:
            positives_dict[p1] += 1
            positives_dict[p2] += 1
        else:
            negatives_dict[p1] += 1
            negatives_dict[p2] += 1

    return positives_dict, negatives_dict

def compute_count_matrix(
    output_folder: Union[Path, str],
    data_folder: Union[Path, str],
    ppi_path: Union[Path, str],
    expected_cache_path: Union[Path, str], 
    observed_cache_path: Union[Path, str]
):

    print("Counting PPI positives...")
    ppi_positives_dict, ppi_negatives_dict = count_dataset_proteins(ppi_path)
    print("Counting model positives...")

    model_positives_dicts = dict()
    model_negatives_dict = dict()

    proteins = list(ppi_positives_dict.keys())

    for model_name in MODELS:
        model_positives_dicts[model_name], model_negatives_dict[model_name] = count_result_proteins(
            ppi_path,
            model_name,
            output_folder,
            data_folder
        )

    expected_counts = np.empty((len(proteins), len(MODELS)))
    observed_counts = np.empty((len(proteins), len(MODELS)))

    for model_idx, model_name in enumerate(MODELS):
        for protein_idx in range(len(proteins)):
            protein_name = proteins[protein_idx]

            obs_count = model_positives_dicts[model_name][protein_name]
            exp_count = ppi_positives_dict[protein_name]

            expected_counts[protein_idx][model_idx] = exp_count
            observed_counts[protein_idx][model_idx] = obs_count

    with gzip.open(expected_cache_path, 'wb') as f:
        pickle.dump(expected_counts, f)

    with gzip.open(observed_cache_path, 'wb') as f:
        pickle.dump(observed_counts, f)

    return expected_counts, observed_counts

def degree_correlation(
    output_folder: Optional[Union[Path, str]] = None,
    data_folder: Optional[Union[Path, str]] = None,
    cmap: str = "perceptual_rainbow_16"
):

    if cmap == "perceptual_rainbow_16":
        cmap = get_map("perceptual_rainbow_16").mpl_colormap
        cmap_name = "perceptual_rainbow_16"
    else:
        cmap_name = cmap

    plot_style()
    plt.rcParams['font.size'] = 8

    output_folder, data_folder = default_paths(output_folder, data_folder)

    ppi_path = data_folder / "ppi/rapppid_[common_string_9606.protein.links.detailed.v12.0_upkb.csv]_Mz70T9t-4Y-i6jWD9sEtcjOr0X8=.h5"

    expected_cache_path =  output_folder / "tables/expected_positive_protein_counts.pkl.gz"
    observed_cache_path = output_folder / "tables/observed_positive_protein_counts.pkl.gz"

    if not os.path.isfile(expected_cache_path) or not os.path.isfile(observed_cache_path):
        expected_counts, observed_counts = compute_count_matrix(output_folder, data_folder, ppi_path, expected_cache_path, observed_cache_path)
    else:
        print("Reading cache...")
        with gzip.open(expected_cache_path, 'rb') as f:
            expected_counts = pickle.load(f)

        with gzip.open(observed_cache_path, 'rb') as f:
            observed_counts = pickle.load(f)

    print("expected_counts", expected_counts.shape)
    print("observed_counts", observed_counts.shape)

    f, axs = plt.subplots(nrows=4, ncols=2, figsize=(8,16))

    for model_idx, ax in zip(range(len(MODELS)), axs.flat):

        ax.set_axisbelow(True)
        ax.grid()

        model_expected_counts = expected_counts[:, model_idx]
        model_observed_counts = observed_counts[:, model_idx]

        counts = count_coincident_points(model_expected_counts, model_observed_counts)

        idx = counts.argsort()
        x, y, counts = model_expected_counts[idx], model_observed_counts[idx], counts[idx]

        scatter = ax.scatter(
            x, 
            y,
            c=counts,
            cmap=cmap,
            marker='o', 
            #edgecolors='k',
            norm=LogNorm(vmin=1, vmax=250),
            s=10
        )

        ax.set_aspect('equal')

        ax.set_xticks(np.arange(0, 51, 10))
        ax.set_yticks(np.arange(0, 51, 10))

        cbar = plt.colorbar(scatter, ax=ax, shrink=1.0, aspect=15)

        cbar.set_ticks([1, 2, 5, 10, 25, 50, 100, 250])
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))

        if MODELS[model_idx] == "squeezeprot_sp_strict":
            model_name = "SqueezeProt-SP (Strict)"
        elif MODELS[model_idx] == "squeezeprot_sp_nonstrict":
            model_name = "SqueezeProt-SP (Non-strict)"
        else:
            model_name = fancy_model_names[MODELS[model_idx]]

        R = pearsonr(model_expected_counts, model_observed_counts)

        display = f"PCC = {R.statistic:.3}"

        ax.text(0.05, 0.95, display, fontsize=8, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left')

        ax.set_xlabel('True Degree')
        ax.set_ylabel(f"{model_name} Degree")
    
    plt.tight_layout()    
    plt.savefig(output_folder / f"figures/pos_counts.{cmap_name}.svg")

    for sp_idx in [STRICT_INDEX, NONSTRICT_INDEX]:

        f, axs = plt.subplots(nrows=3, ncols=2, figsize=(8,11))

        for model_idx, ax in zip(range(len(MODELS)), axs.flat):

            ax.set_axisbelow(True)
            ax.grid()

            sp_observed_counts = observed_counts[:, sp_idx]
            model_observed_counts = observed_counts[:, model_idx]

            counts = count_coincident_points(model_expected_counts, model_observed_counts)

            idx = counts.argsort()
            x, y, counts = sp_observed_counts[idx], model_observed_counts[idx], counts[idx]

            scatter = ax.scatter(
                x,
                y,
                c=counts,
                cmap=cmap,
                marker='o',
                # edgecolors='k',
                norm=LogNorm(vmin=1, vmax=250),
                s=10
            )

            ax.set_aspect('equal')

            ax.set_xticks(np.arange(0, 51, 10))
            ax.set_yticks(np.arange(0, 51, 10))

            cbar = plt.colorbar(scatter, ax=ax, shrink=1.0, aspect=15)

            cbar.set_ticks([1, 2, 5, 10, 25, 50, 100, 250])
            cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))
            
            R = pearsonr(sp_observed_counts, model_observed_counts)

            model_name = fancy_model_names[MODELS[model_idx]]
            display = f"PCC = {R.statistic:.3}"

            ax.text(0.05, 0.95, display, fontsize=8, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left')


            sp_label = "Strict" if sp_idx == STRICT_INDEX else "Non-strict"
            ax.set_ylabel(f"{model_name} Degree")
            ax.set_xlabel(f"Squeeze-SP ({sp_label}) Degree")
        
        plt.tight_layout()    
        plt.savefig(output_folder / f"figures/pos_counts.{sp_label}.{cmap_name}.svg")
        print(output_folder / f"figures/pos_counts.{sp_label}.{cmap_name}.svg")

    for sp_idx in [STRICT_INDEX, NONSTRICT_INDEX]:

        f, axs = plt.subplots(nrows=3, ncols=2, figsize=(8, 11))

        for model_idx, ax in zip(range(len(MODELS)), axs.flat):
            ax.set_axisbelow(True)
            ax.grid()

            sp_observed_counts = observed_counts[:, sp_idx]
            model_observed_counts = observed_counts[:, model_idx]

            counts = count_coincident_points(model_expected_counts, model_observed_counts)

            idx = counts.argsort()
            x, y, counts = sp_observed_counts[idx], model_observed_counts[idx], counts[idx]

            scatter = ax.scatter(
                y,
                x,
                c=counts,
                cmap=cmap,
                marker='o',
                # edgecolors='k',
                norm=LogNorm(vmin=1, vmax=250),
                s=10
            )

            ax.set_aspect('equal')

            ax.set_xticks(np.arange(0, 51, 10))
            ax.set_yticks(np.arange(0, 51, 10))

            cbar = plt.colorbar(scatter, ax=ax, shrink=1.0, aspect=15)

            cbar.set_ticks([1, 2, 5, 10, 25, 50, 100, 250])
            cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))

            R = pearsonr(sp_observed_counts, model_observed_counts)

            model_name = fancy_model_names[MODELS[model_idx]]
            display = f"PCC = {R.statistic:.3}"

            ax.text(0.05, 0.95, display, fontsize=8, transform=ax.transAxes, verticalalignment='top',
                    horizontalalignment='left')

            sp_label = "Strict" if sp_idx == STRICT_INDEX else "Non-strict"
            ax.set_xlabel(f"{model_name} Degree")
            ax.set_ylabel(f"Squeeze-SP ({sp_label}) Degree")

        plt.tight_layout()
        plt.savefig(output_folder / f"figures/pos_counts.{sp_label}.{cmap_name}_flip.svg")
        print(output_folder / f"figures/pos_counts.{sp_label}.{cmap_name}_flip.svg")

    R_matrix = np.empty((len(MODELS), len(MODELS)))
    R_p_matrix = np.empty((len(MODELS), len(MODELS)))

    for model1_idx in range(len(MODELS)):
        for model2_idx in range(len(MODELS)):

            model1_counts = observed_counts[:, model1_idx]
            model2_counts = observed_counts[:, model2_idx]

            hist1, bin_edges1 = np.histogram(model1_counts, bins='fd')
            hist2, _ = np.histogram(model2_counts, bins=bin_edges1)

            R = pearsonr(model1_counts, model2_counts)

            R_matrix[model1_idx][model2_idx] = R.statistic
            R_p_matrix[model1_idx][model2_idx] = false_discovery_control(R.pvalue, method="by")

            print(f"{MODELS[model1_idx]}", f"{MODELS[model2_idx]}", R.statistic, R.pvalue)

    with open(output_folder / f"tables/pos_counts.csv", "w") as f:

        fieldnames = ["model_1", "model_2", "R", "p_value"]
        csv_writer = csv.DictWriter(f, fieldnames)
        csv_writer.writeheader()

        for model1_idx in range(len(MODELS)):
            for model2_idx in range(len(MODELS)):
                csv_writer.writerow({
                    'model_1': MODELS[model1_idx], 
                    'model_2': MODELS[model2_idx],
                    'R': R_matrix[model1_idx][model2_idx],
                    "p_value": R_p_matrix[model1_idx][model2_idx]
                })

    f, ax = plt.subplots(nrows=1, ncols=1)

    upper_tri_idx = np.triu_indices(len(MODELS))
    R_matrix_z = R_matrix.copy()
    R_matrix_z[upper_tri_idx] = None

    for model1_idx in range(len(MODELS)):
        for model2_idx in range(len(MODELS)):
            text = ax.text(model1_idx, model2_idx, f"{R_matrix[model1_idx, model2_idx]:.2}", ha="center", va="center", color="w")
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    r_imshow = ax.imshow(R_matrix_z, cmap=cmap)
    
    model_names = [fancy_model_names[model_idx] for model_idx in MODELS]

    ax.set_xticks(np.arange(len(MODELS))[:-1], labels=model_names[:-1],
              rotation=45, ha="right", rotation_mode="anchor")

    ax.set_yticks(np.arange(len(MODELS))[1:], labels=model_names[1:])
    
    cbar = f.colorbar(r_imshow, ax=ax, shrink=0.9, label='Pearson Correlation')

    plt.savefig(output_folder / f"figures/pos_counts.heatmap.{cmap_name}.svg")

    # bar chart

    all_R = R_matrix_z.flatten()
    strict_R = R_matrix_z[STRICT_INDEX, :].flatten()
    nonstrict_R = R_matrix_z[NONSTRICT_INDEX, :].flatten()

    all_R_mean = np.nanmean(all_R)
    all_R_std = np.nanstd(all_R)
    strict_R_mean = np.nanmean(strict_R)
    strict_R_std = np.nanstd(strict_R)
    nonstrict_R_mean = np.nanmean(nonstrict_R)
    nonstrict_R_std = np.nanstd(nonstrict_R)

    print("R_means", f"{all_R_mean:.3}, {strict_R_mean:.3}, {nonstrict_R_mean:3}")
    print("R_std", f"{all_R_std:.3}, {strict_R_std:.3}, {nonstrict_R_std:3}")

    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))

    hatches = [
        None,
        '//',
        '.'
    ]

    colour_idx = [1,0,2]

    ls = ["Strict v. pLMs", "Non-strict v. pLMs", "pLMs v. pLMs"]

    for i, mean in enumerate([strict_R_mean, nonstrict_R_mean, all_R_mean]):
        """
        ax.bar(
            i,
            mean,
            width=0.5,
            color=colours[colour_idx[i]],
            edgecolor="k",
            lw=1,
            hatch=hatches[i]
        )
        """

        ax.barh(
            ls[i],
            mean,
            height=0.4,
            color=colours[colour_idx[i]],
            edgecolor="k",
            lw=1,
            hatch=hatches[i]
        )

    s = 200
    #ax.scatter(np.zeros(len(strict_R)), strict_R, ec="k", fc="none", s=s)
    #ax.scatter(np.ones(len(nonstrict_R)), nonstrict_R, ec="k", fc="none", s=s)
    #ax.scatter(np.ones(len(all_R)) + 1, all_R, ec="k", fc="none", s=s)

    ax.scatter(strict_R, np.zeros(len(strict_R)), ec="k", fc="none", s=s)
    ax.scatter(nonstrict_R, np.ones(len(nonstrict_R)), ec="k", fc="none", s=s)
    ax.scatter(all_R, np.ones(len(all_R)) + 1, ec="k", fc="none", s=s)

    #ax.set_xticks([0, 1, 2], ["Strict v. pLMs", "Non-strict v. pLMs", "pLMs v. pLMs"], rotation=45)
    ax.set_xlabel("Pearson correlation coefficient (PCC)")
    ax.set_axisbelow(True)
    ax.grid()

    plt.savefig(output_folder / f"figures/R_bars.svg")
