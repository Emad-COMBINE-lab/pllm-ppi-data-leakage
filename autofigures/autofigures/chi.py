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

from autofigures.dataset import RapppidDataset
from typing import Union, Optional
from pathlib import Path
from autofigures.utils import default_paths, plot_style, fancy_model_names
from collections import defaultdict
from scipy.stats import chisquare
import numpy as np
import pandas as pd
import json

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
    output_folder: Union[Path, str]
):
    c_type = 3
    split = 'test'

    dataset = RapppidDataset(ppi_path, c_type, split)

    positives_dict = defaultdict(lambda: 0)
    negatives_dict = defaultdict(lambda: 0)

    for seed in [1,2,3]:

        seed_path = output_folder / f"tables/scores_s{seed}.csv"
        seed_df = pd.read_csv(seed_path)

        y_hats = []

        for row_idx, row in seed_df[seed_df.model_name == model_name].iterrows():
            y_hats.append(row.y_hat)

        #assert len(y_hats) == len(dataset)

        for i, y_hat in zip(range(len(dataset)), y_hats):
            p1, p2, label = dataset[i]

            if y_hat > 0.5:
                positives_dict[p1] += 1
                positives_dict[p2] += 1
            else:
                negatives_dict[p1] += 1
                negatives_dict[p2] += 1

    for key in positives_dict.keys():
        positives_dict[key] = round(positives_dict[key] / 3)

    for key in negatives_dict.keys():
        negatives_dict[key] = round(negatives_dict[key] / 3)

    return positives_dict, negatives_dict

def chi(
    output_folder: Optional[Union[Path, str]] = None,
    data_folder: Optional[Union[Path, str]] = None,
):

    plot_style()

    output_folder, data_folder = default_paths(output_folder, data_folder)

    ppi_path = data_folder / "ppi/rapppid_[common_string_9606.protein.links.detailed.v12.0_upkb.csv]_Mz70T9t-4Y-i6jWD9sEtcjOr0X8=.h5"

    print("Counting PPI positives...")
    ppi_positives_dict, ppi_negatives_dict = count_dataset_proteins(ppi_path)
    print("Counting model positives...")

    model_positives_dicts = dict()
    model_negatives_dict = dict()

    proteins = list(ppi_positives_dict.keys())
    model_names = list(fancy_model_names.keys())

    for model_name in model_names:
        model_positives_dicts[model_name], model_negatives_dict[model_name] = count_result_proteins(
            ppi_path,
            model_name,
            output_folder
        )

    expected_counts = np.empty((len(proteins), len(model_names)))
    observed_counts = np.empty((len(proteins), len(model_names)))

    for protein_idx in range(len(proteins)):
        for model_idx in range(len(model_names)):
            
            model_name = model_names[model_idx]
            protein_name = proteins[protein_idx]

            obs_count = model_positives_dicts[model_name][protein_name]
            exp_count = ppi_positives_dict[protein_name]

            expected_counts[protein_idx][model_idx] = exp_count
            observed_counts[protein_idx][model_idx] = obs_count

    dof = (len(proteins)-1)*(len(model_names)-1)
    stat = chisquare(observed_counts, expected_counts, dof, sum_check=False)
    print("all by all")
    print(stat)

    for model_name in model_names:
        expected_counts = np.empty(len(proteins))
        observed_counts = np.empty(len(proteins))

        for protein_idx in range(len(proteins)):
            protein_name = proteins[protein_idx]

            obs_count = model_positives_dicts[model_name][protein_name]
            exp_count = ppi_positives_dict[protein_name]

            expected_counts[protein_idx] = exp_count
            observed_counts[protein_idx] = obs_count

        stat = chisquare(observed_counts, expected_counts, sum_check=False)
        chi_stat = stat.statistic
        pvalue = stat.pvalue
        print(f"{model_name=}")
        print(f"{chi_stat=}, {pvalue=}")

    

    