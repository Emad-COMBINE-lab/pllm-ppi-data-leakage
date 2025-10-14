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

from typing import Union, Optional
from pathlib import Path
from autofigures.get_scores import get_scores
from autofigures.strict_nonstrict import strict_nonstrict
from autofigures.speed import speed
from autofigures.concordance import concordance, concordance2
from autofigures.length_histogram import length_histogram
from autofigures.acc_by_length import acc_by_length
from autofigures.length_heatmap import length_heatmap
from autofigures.sars_cov2 import sars_cov2
from autofigures.mutation import mutation
from autofigures.kw import kw
from autofigures.degree_correlation import degree_correlation
from autofigures.prefix_random import prefix_random
from autofigures.ppi_seq_id import ppi_seq_id
from autofigures.stables import all_stables

def run_all(
    output_folder: Optional[Union[Path, str]] = None,
    data_folder: Optional[Union[Path, str]] = None,
):
    print("GET_SCORES")
    get_scores(output_folder, data_folder)
    print("\tDONE")

    print("SPEED")
    speed(output_folder, data_folder)
    print("\tDONE")

    print("STRICT_NONSTRICT")
    strict_nonstrict(output_folder, data_folder)
    print("\tDONE")

    print("PPI_SEQ_ID")
    ppi_seq_id(output_folder, data_folder)
    print("\tDONE")

    print("KW")
    kw(output_folder, data_folder)
    print("\tDONE")

    print("CONCORDANCE SNC")
    concordance(output_folder, data_folder)
    print("\tDONE")

    print("CONCORDANCE KAPPA")
    concordance(output_folder, data_folder, cohen_kappa=True)
    print("\tDONE")

    print("DEGREE_CORRELATION")
    degree_correlation(output_folder, data_folder)
    print("\tDONE")

    print("ACC_BY_LENGTH")
    acc_by_length(output_folder, data_folder)
    print("\tDONE")

    print("LENGTH_HISTOGRAM")
    length_histogram(output_folder, data_folder)
    print("\tDONE")

    print("LENGTH_HEATMAP")
    length_heatmap(output_folder, data_folder)
    print("\tDONE")

    print("SARS_COV2")
    sars_cov2(output_folder, data_folder)
    print("\tDONE")

    print("MUTATION")
    mutation(output_folder, data_folder)
    print("\tMUTATION")
