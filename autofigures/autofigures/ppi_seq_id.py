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

from autofigures.utils import colours, fancy_model_names, default_paths
import matplotlib.pyplot as plt
import numpy as np
import json
import gzip

model_names = [

    'squeezeprot_sp_nonstrict',
    'squeezeprot_sp_strict',
]

def ppi_seq_id(output_folder: Optional[Union[Path, str]] = None, data_folder: Optional[Union[Path, str]] = None,):

    output_folder, data_folder = default_paths(output_folder, data_folder)

    with open(data_folder / 'ppi_seq_id/metrics.json') as f:
        metrics = json.load(f)

    m_names = ['mcc', 'auroc', 'bacc', 'f1']

    fancy_metric_names = {
        "mcc": "Matthew's Correlation Coefficient (MCC)",
        "auroc": "Area Under Receiver-Operator Characteristic",
        "bacc": "Balanced Accuracy",
        "ap": "Average Precision",
        "f1": "F-1 Score"
    }

    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    plt.rcParams["font.family"] = "Inter"

    avgs = {m_name: {model_name: [] for model_name in model_names} for m_name in m_names}
    stds = {m_name: {model_name: [] for model_name in model_names} for m_name in m_names}

    num_proteins = []

    with gzip.open(data_folder/'ppi_seq_id/test_set.json.gz', 'rt') as f:
        test_set = json.load(f)
        for threshold in thresholds:
            num_proteins.append(len(test_set[str(threshold)]))

    for m_name in m_names:
        num_thresh = len(metrics['thresholds'])

        for model_name in model_names:
            for idx in range(num_thresh):
                ms = []

                for seed in seeds:
                    ms.append(metrics[model_name][str(seed)][m_name][idx])

                avg = np.mean(ms)
                avgs[m_name][model_name].append(avg)

                std = np.std(ms)
                stds[m_name][model_name].append(std)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for idx, model_name in enumerate(model_names):
            marker = 's' if model_name == "squeezeprot_sp_strict" else 'o'

            ax.plot(
                metrics['thresholds'],
                avgs[m_name][model_name],
                label=fancy_model_names[model_name],
                lw=2,
                ls=':' if model_name == "squeezeprot_sp_nonstrict" else '-',
                c=colours[1] if model_name == "squeezeprot_sp_strict" else colours[0],
                marker=marker
            )
            ax.fill_between(metrics['thresholds'],
                            np.array(avgs[m_name][model_name]) - np.array(stds[m_name][model_name]),
                            np.array(avgs[m_name][model_name]) + np.array(stds[m_name][model_name]),
                            color=colours[1] if model_name == "squeezeprot_sp_strict" else colours[0],
                            alpha=0.2)

        ax.set_ylabel(fancy_metric_names[m_name])

        min_ylim = 0 if m_name == 'mcc' else 0.5

        ax.set_ylim(min_ylim, 1)

        ax2 = ax.twinx()
        ax2.set_ylabel('Number of Proteins')
        # ax2.set_ylim(0,4751)
        # ax2.plot(metrics['thresholds'], metrics['esm'][str(1)]['n'], label='Number of Testing Edges', lw=2, ls='--', c='k')
        ax2.plot(metrics['thresholds'], num_proteins, label='Number of Testing Proteins', lw=2, ls='--', c='k', marker='^')

        ax.grid()
        ax.set_axisbelow(True)
        ax.legend(edgecolor="k", fancybox=False, loc='upper left')
        ax2.legend(edgecolor="k", fancybox=False, loc='lower right')
        ax.set_xlabel("Maximum Sequence Identity")
        plt.savefig(output_folder / f'figures/ppi_seq_id_{m_name}.svg')
