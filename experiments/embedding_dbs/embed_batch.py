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

import csv
import fire
import lmdb
import json
import tables
from tqdm import tqdm
from pathlib import Path
import sys

sys.path.insert(1, "../encode_llm")
from fasta import stream_fasta


def num_sequences_h5(input_path: Path):
    h5file = tables.open_file(input_path, mode="r")
    sequences_table = h5file.root.sequences
    i = 0
    for _ in sequences_table.iterrows():
        i += 1

    return i


def num_sequences_fasta(input_path: Path):
    num_seqs = 0

    for name, sequence in stream_fasta(input_path):
        num_seqs += 1

    return num_seqs


def num_sequences_csv(input_path: Path):
    num_seqs = 0

    with open(input_path) as f:
        reader = csv.DictReader(f)

        for row in reader:
            num_seqs += 1

    return num_seqs


def get_sequences_h5(input_path: Path, trunc: int = 1024):
    h5file = tables.open_file(input_path, mode="r")
    sequences_table = h5file.root.sequences

    for row_num, row in enumerate(sequences_table.iterrows()):
        if row_num % 100 == 0:
            print(f"{row_num} rows...")
        upkb_ac, seq = row[0], row[1][:trunc]
        yield upkb_ac.decode("utf8"), seq.decode("utf8")

    h5file.close()


def get_sequences_fasta(input_path: Path, trunc: int = 1024):
    for name, sequence in stream_fasta(input_path):
        yield name.split("|")[0], sequence


def get_sequences_csv(input_path: Path, trunc: int = 1024):
    with open(input_path) as f:
        reader = csv.DictReader(f)

        for row in reader:
            yield row["accession"], row["sequence"]


class Embed(object):
    def _embed(
        self,
        input_path: Path,
        output_path: Path,
        encoder,
        max_length: int,
        max_batch_size: int,
        device: str,
    ):
        if input_path.endswith(".fasta") or input_path.endswith(".fasta.gz"):
            num_sequences = num_sequences_fasta
            get_sequences = get_sequences_fasta
        elif input_path.endswith(".h5"):
            num_sequences = num_sequences_h5
            get_sequences = get_sequences_h5
        elif input_path.endswith(".csv"):
            num_sequences = num_sequences_csv
            get_sequences = get_sequences_csv

        print("Counting sequences...")
        total = num_sequences(input_path)

        db_env = lmdb.open(str(output_path))
        db_env.set_mapsize(1024 * 1024 * 1024 * 1024)  # 1 TB

        print("Embedding...")

        batch_seqs = []
        batch_acs = []

        for idx, (upkb_ac, seq) in tqdm(
            enumerate(get_sequences(input_path)), total=total
        ):
            seq = seq[:max_length]

            batch_seqs.append(seq)
            batch_acs.append(upkb_ac)

            if len(batch_seqs) > max_batch_size:
                vecs = encoder(batch_seqs, device=device)

                with db_env.begin(write=True) as txn:
                    for vec_ac, vec in zip(batch_acs, vecs):
                        txn.put(vec_ac.encode("utf8"), json.dumps(vec).encode("utf8"))

                batch_seqs = []
                batch_acs = []

        if len(batch_seqs) > 0:
            vecs = encoder(batch_seqs, device=device)

            with db_env.begin(write=True) as txn:
                for vec_ac, vec in zip(batch_acs, vecs):
                    txn.put(vec_ac.encode("utf8"), json.dumps(vec).encode("utf8"))

    def squeezeprot_sp_nonstrict(
        self,
        input_path: Path,
        max_batch_size: int,
        device="cpu",
        output_path: Path = Path("../../data/embeddings/squeezeprot-sp.nonstrict.lmdb"),
    ):
        print("loading SqueezeBERT-SP (Non-strict)")
        import squeezebert

        def encoder(seq, device):
            return squeezebert.encode_batch(
                seq,
                device=device,
                weights_path="../../data/chkpts/squeezeprot-sp.non-strict/checkpoint-1390896",
            )

        self._embed(input_path, output_path, encoder, 512, max_batch_size, device)

    def squeezeprot_sp_strict(
        self,
        input_path: Path,
        max_batch_size: int,
        device="cpu",
        output_path: Path = Path("../../data/embeddings/squeezeprot-sp.strict.lmdb"),
    ):
        print("loading SqueezeBERT-SP (Strict)")
        import squeezebert

        def encoder(seq, device):
            return squeezebert.encode_batch(
                seq,
                device=device,
                weights_path="../../data/chkpts/squeezeprot-sp.strict/checkpoint-1383824",
            )

        self._embed(input_path, output_path, encoder, 512, max_batch_size, device)

    def squeezeprot_u50(
        self,
        input_path: Path,
        max_batch_size: int,
        device="cpu",
        output_path: Path = Path("../../data/embeddings/squeezebert-u50.lmdb"),
    ):
        print("loading SqueezeBERT-U50")
        import squeezebert

        def encoder(seq, device):
            return squeezebert.encode(
                seq,
                device=device,
                weights_path="../../data/chkpts/squeezeprot-u50/checkpoint-2477920",
            )

        self._embed(input_path, output_path, encoder, 512, max_batch_size, device)

    def prottrans_bert(
        self,
        input_path: Path,
        max_batch_size: int,
        device="cpu",
        output_path: Path = Path("../../data/embeddings/prottrans_bert.lmdb"),
    ):
        print("loading ProtBERT")
        import prottrans_bert

        encoder = prottrans_bert.encode_batch

        self._embed(input_path, output_path, encoder, 512, max_batch_size, device)

    def prottrans_t5(
        self,
        input_path: Path,
        max_batch_size: int,
        device="cpu",
        output_path: Path = Path("../../data/embeddings/prottrans_t5.lmdb"),
    ):
        print("loading ProtT5")
        import prottrans_t5

        encoder = prottrans_t5.encode_batch

        self._embed(input_path, output_path, encoder, 512, max_batch_size, device)

    def proteinbert(
        self,
        input_path: Path,
        max_batch_size: int,
        device="cpu",
        output_path: Path = Path("../../data/embeddings/proteinbert.lmdb"),
    ):
        print("loading ProteinBERT")
        import proteinberter

        encoder = proteinberter.encode_batch

        self._embed(input_path, output_path, encoder, 1022, max_batch_size, device)

    def prose(
        self,
        input_path: Path,
        max_batch_size: int,
        device="cpu",
        output_path: Path = Path("../../data/embeddings/prose.lmdb"),
    ):
        print("loading ProSE")
        import proser

        encoder = proser.encode_batch

        self._embed(input_path, output_path, encoder, 9999999, max_batch_size, device)

    def esm(
        self,
        input_path: Path,
        max_batch_size: int,
        device="cpu",
        output_path: Path = Path("../../data/embeddings/esm.lmdb"),
    ):
        print("loading ESM")
        import esmer

        encoder = esmer.encode_batch

        self._embed(
            input_path, output_path, encoder, 1024, max_batch_size, device=device
        )

    def rapppid(
        self,
        input_path: Path,
        max_batch_size: int,
        device="cpu",
        output_path: Path = Path("../../data/embeddings/rapppid.lmdb"),
    ):
        print("loading RAPPPID")
        # if max_batch_size > 0:
        #    "RAPPPID batched encoding not implemented, running sequentially"
        #    max_batch_size = 0
        import rapppid

        encoder = rapppid.encode_batch

        self._embed(
            input_path, output_path, encoder, 3000, max_batch_size, device=device
        )


if __name__ == "__main__":
    fire.Fire(Embed(), name="Embed")
