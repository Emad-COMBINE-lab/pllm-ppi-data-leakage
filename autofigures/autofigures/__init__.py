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
from autofigures.concordance import concordance
from autofigures.length_histogram import length_histogram
from autofigures.acc_by_length import acc_by_length
from autofigures.length_heatmap import length_heatmap
from autofigures.sars_cov2 import sars_cov2
from autofigures.mutation import mutation
from autofigures.kw import kw


def run_all(
    output_folder: Optional[Union[Path, str]] = None,
    data_folder: Optional[Union[Path, str]] = None,
):
    print("[START] GET_SCORES")
    get_scores(output_folder, data_folder)
    print("[DONE] GET_SCORES")

    print("[START] SPEED")
    speed(output_folder, data_folder)
    print("[DONE] SPEED")

    print("[START] STRICT_NONSTRICT")
    strict_nonstrict(output_folder, data_folder)
    print("[DONE] STRICT_NONSTRICT")

    print("[START] KW")
    kw(output_folder, data_folder)
    print("[DONE] KW")

    print("[START] CONCORDANCE")
    concordance(output_folder, data_folder)
    print("[DONE] CONCORDANCE")

    print("[START] ACC_BY_LENGTH")
    acc_by_length(output_folder, data_folder)
    print("[DONE] ACC_BY_LENGTH")

    print("[START] LENGTH_HISTOGRAM")
    length_histogram(output_folder, data_folder)
    print("[DONE] LENGTH_HISTOGRAM")

    print("[START] LENGTH_HEATMAP")
    length_heatmap(output_folder, data_folder)
    print("[DONE] LENGTH_HEATMAP")

    print("[START] SARS_COV2")
    sars_cov2(output_folder, data_folder)
    print("[DONE] SARS_COV2")

    print("[START] MUTATION")
    mutation(output_folder, data_folder)
    print("[DONE] MUTATION")
