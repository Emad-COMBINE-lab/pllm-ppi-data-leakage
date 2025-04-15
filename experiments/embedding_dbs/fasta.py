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
from pathlib import Path
from typing import Tuple, Generator


def stream_fasta(fasta_path: Path) -> Generator[Tuple[str, str], None, None]:
    """
    Iterates over a FASTA file, returning a tuple of FASTA
    record name and sequence for each iteration.

    :param fasta_path: Path of the FASTA file to stream. If gzip'd, must have a .gz extension.
    """

    if str(fasta_path).endswith(".gz"):
        f = gzip.open(str(fasta_path), "rt")
    else:
        f = open(str(fasta_path), "rt")

    sequence = None

    for line in f:
        line = line.strip()
        if line.startswith(">"):
            if sequence is not None and sequence != "":
                yield name, sequence
            name = line[1:]
            sequence = ""
        else:
            sequence += line
