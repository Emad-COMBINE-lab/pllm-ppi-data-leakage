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

import fire
from autofigures import (
    run_all,
    get_scores,
    strict_nonstrict,
    speed,
    concordance,
    length_histogram,
    acc_by_length,
    length_heatmap,
    sars_cov2,
    mutation,
    kw,
)

if __name__ == "__main__":
    fire.Fire(
        {
            "run_all": run_all,
            "get_scores": get_scores,
            "speed": speed,
            "strict_nonstrict": strict_nonstrict,
            "kw": kw,
            "concordance": concordance,
            "length_histogram": length_histogram,
            "length_heatmap": length_heatmap,
            "sars_cov2": sars_cov2,
            "mutation": mutation,
            "acc_by_length": acc_by_length,
        }
    )
