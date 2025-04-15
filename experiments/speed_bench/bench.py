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
import gzip
from tqdm import tqdm
from more_itertools import chunked
from time import time
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(1, "../encode_llm")


class Speed_bench(object):
    def _stream_fasta(self, fasta_path: Path):
        if str(fasta_path).endswith(".gz"):
            f = gzip.open(str(fasta_path), "rt")
        else:
            f = open(str(fasta_path), "rt")

        sequence = None

        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if sequence is not None:
                    yield name, sequence
                name = line[1:]
                sequence = ""
            else:
                sequence += line

    def run(self, model_name: str, fasta_path: Path, batch_size: Optional[int] = None):
        if model_name == "esm":
            print("Loading ESM...")
            import esmer

            encoder = esmer.encode_batch
        elif model_name == "prose":
            print("Loading ProSE...")
            import proser

            encoder = proser.encode_batch
        elif model_name == "proteinbert":
            print("Loading ProteinBERT...")
            import proteinberter

            encoder = proteinberter.encode_batch
        elif model_name == "prottrans_bert":
            print("Loading ProtBERT...")
            import prottrans_bert

            encoder = prottrans_bert.encode_batch
        elif model_name == "prottrans_t5":
            print("Loading ProtT5...")
            import prottrans_t5

            encoder = prottrans_t5.encode_batch
        elif model_name == "squeezebert":
            print("Loading SqueezeBERT...")
            import squeezebert

            encoder = squeezebert.encode_batch
        elif model_name == "rapppid":
            print("Loading RAPPPID...")
            import rapppid

            encoder = rapppid.encode_batch
        else:
            print("Unexpected model.")

        sequences = []

        total_length = 0
        total_seqs = 0

        print("Reading sequences...")
        for _, sequence in self._stream_fasta(fasta_path):
            sequence = sequence[:1500]
            total_length += len(sequence)
            total_seqs += 1
            sequences.append(sequence)

        if batch_size is None:
            print("Auto-batch sizing...")

            # find longest sequence
            seq_len_max = max([len(sequence) for sequence in sequences])

            dummy_seq = "M" * seq_len_max

            batch_size = 1

            while True:
                try:
                    print(f"Testing batch size of {batch_size}")

                    batch = [dummy_seq] * batch_size

                    encoder(batch, "cuda")

                    batch_size += 1
                except Exception:
                    print(f"Batch size found! ({batch_size})")
                    break

        print("Bench starting...")

        start = time()

        for sequence_batch in chunked(tqdm(sequences), batch_size):
            _ = encoder(sequence_batch, device="cuda")

        end = time() - start

        print("-----")
        print(f"TOTAL TIME:\t{end:.6} seconds")
        print(f"TOTAL AAS:\t{total_length}")
        print(f"TOTAL SEQS:\t{total_seqs}")
        print(f"TIME/AAS:\t{end / total_length:.6}")
        print(f"TIME/SEQS:\t{end / total_seqs:.6}")


if __name__ == "__main__":
    fire.Fire(Speed_bench(), name="Speed Bench")
