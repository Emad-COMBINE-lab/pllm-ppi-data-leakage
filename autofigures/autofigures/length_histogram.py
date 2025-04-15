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

import gzip
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, Union
from autofigures.utils import plot_style, default_paths

def length_histogram(output_folder: Optional[Union[Path, str]] = None, data_folder: Optional[Union[Path, str]] = None):
    plot_style()

    output_folder, data_folder = default_paths(output_folder, data_folder)

    path = data_folder / "lengths/length_histogram.csv.gz"

    histogram_str = None

    with gzip.open(path, 'rt') as f:
        for idx, line in enumerate(f):
            if idx == 1:
                histogram_str = line

    binvals = histogram_str.replace("{", "").replace("}", "").replace("\"", "").strip().split(" ")
    binvals = [x.replace(',', '').split('=') for x in binvals]

    values = np.array([int(value) for _, value in binvals])
    cumulative_values = []

    for idx, value in enumerate(values):
        if idx == 0:
            cumulative_values.append(value)
        else:
            cumulative_values.append(cumulative_values[idx - 1] + value)

    cumulative_values = np.array(cumulative_values)
    bins = [int(hbin) for hbin, _ in binvals]

    plt.plot(bins[::3], 100 * cumulative_values[::3] / cumulative_values[-1], c='k')
    plt.grid()
    plt.xscale('log')
    plt.xlabel("Maximum Sequence Length")
    plt.ylabel("Proportion of Proteins\n(%)")

    plt.axvline(283, color='#D81B60', label='Median')
    plt.axvline(512, color='#1E88E5', ls='--')
    plt.axvline(1024, color='#1E88E5', ls='--')
    plt.axvline(2048, color='#1E88E5', ls='--')

    x_margin = 1.8
    properties = {'boxstyle': 'larrow', 'facecolor': '#fcf7cc', 'alpha': 0.9}

    plt.text(512 * x_margin, 5, 'Context Length: 512\n18.1% of proteins truncated', bbox=properties)
    plt.text(1024 * x_margin, 25, 'Context Length: 1024\n3.36% of proteins truncated', bbox=properties)
    plt.text(2048 * x_margin, 45, 'Context Length: 2048\n0.453% of proteins truncated', bbox=properties)

    plt.savefig(output_folder / "figures/length_histogram.svg")