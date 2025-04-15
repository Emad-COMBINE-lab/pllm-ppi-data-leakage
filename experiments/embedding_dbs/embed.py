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
import lmdb
import json
import tables
from tqdm import tqdm
from pathlib import Path
import sys

sys.path.insert(1, "/whale/projects/phd/llm/encode_llm")
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
        yield name, sequence


class Embed(object):
    def _embed(self, input_path: Path, output_path: Path, encoder, max_length: int):
        if input_path.endswith(".fasta") or input_path.endswith(".fasta.gz"):
            num_sequences = num_sequences_fasta
            get_sequences = get_sequences_fasta
        elif input_path.endswith(".h5"):
            num_sequences = num_sequences_h5
            get_sequences = get_sequences_h5

        print("Counting sequences...")
        total = num_sequences(input_path)

        db_env = lmdb.open(str(output_path))
        db_env.set_mapsize(1024 * 1024 * 1024 * 1024)  # 1 TB

        print("Embedding...")
        for idx, (upkb_ac, seq) in tqdm(
            enumerate(get_sequences(input_path)), total=total
        ):
            seq = seq[:max_length]
            vec = encoder(seq)

            with db_env.begin(write=True) as txn:
                txn.put(upkb_ac.encode("utf8"), json.dumps(vec).encode("utf8"))

    """
    def squeezebert_u90_c3(self, input_path: Path, output_path: Path = Path("squeezebert.u90.c3.lmdb")):

        print('loading SqueezeBERT')
        import squeezebert
        
        def encoder(seq):
            return squeezebert.encode(seq, "/whale/projects/phd/llm/squeezebert_models/squeezebert.uniref90.c3/checkpoint-2517000")

        self._embed(input_path, output_path, encoder, 512)

    def squeezebert_u90_c1(self, input_path: Path, output_path: Path = Path("squeezebert.u90.c1.lmdb")):

        print('loading SqueezeBERT')
        import squeezebert

        def encoder(seq):
            return squeezebert.encode(seq, "/whale/projects/phd/llm/squeezebert_models/squeezebert.uniref90.c1/checkpoint-2433000")

        self._embed(input_path, output_path, encoder, 512)

    def squeezebert_u50_c3(self, input_path: Path, output_path: Path = Path("squeezebert.u50.c3.lmdb")):

        print('loading SqueezeBERT')
        import squeezebert

        def encoder(seq):
            return squeezebert.encode(seq, "/whale/projects/phd/llm/squeezebert_models/squeezebert.uniref50.c3/checkpoint-2541300")

        self._embed(input_path, output_path, encoder, 512)
    """

    def squeezeprot_sp_nonstrict(
        self,
        input_path: Path,
        output_path: Path = Path("squeezeprot-sp.nonstrict.lmdb"),
    ):
        print("loading SqueezeBERT")
        import squeezebert

        def encoder(seq):
            return squeezebert.encode(
                seq,
                "/whale/projects/phd/llm/squeezebert_models/fixed/squeezeprot-sp.nonstrict/checkpoint-1390896",
            )

        self._embed(input_path, output_path, encoder, 512)

    def squeezeprot_sp_nonstrict_unlearn(
        self,
        input_path: Path,
        output_path: Path = Path("squeezeprot-sp.nonstrict.unlearn.lmdb"),
    ):
        print("loading SqueezeBERT")
        import squeezebert

        def encoder(seq):
            return squeezebert.encode(
                seq,
                "/whale/projects/phd/llm/squeezebert_models/fixed/squeezeprot-sp.nonstrict.unlearn/checkpoint-133000",
            )

        self._embed(input_path, output_path, encoder, 512)

    def squeezeprot_sp_strict(
        self, input_path: Path, output_path: Path = Path("squeezeprot-sp.strict.lmdb")
    ):
        print("loading SqueezeBERT")
        import squeezebert

        def encoder(seq):
            return squeezebert.encode(
                seq,
                "/whale/projects/phd/llm/squeezebert_models/fixed/squeezeprot-sp.strict/checkpoint-1383824",
            )

        self._embed(input_path, output_path, encoder, 512)

    def squeezebert_xl_u50_c3(
        self, input_path: Path, output_path: Path = Path("squeezebert.xl.u50.c3.lmdb")
    ):
        print("loading SqueezeBERT XL")
        import squeezebert

        def encoder(seq):
            return squeezebert.encode(
                seq,
                "/whale/projects/phd/llm/squeezebert_models/squeezebert.uniref50.c3.regression4/checkpoint-2477920",
            )

        self._embed(input_path, output_path, encoder, 512)

    def prottrans_bert(
        self, input_path: Path, output_path: Path = Path("prottrans_bert.lmdb")
    ):
        print("loading ProtTrans BERT")
        import prottrans_bert

        encoder = prottrans_bert.encode

        self._embed(input_path, output_path, encoder, 512)

    def prottrans_t5(
        self, input_path: Path, output_path: Path = Path("prottrans_t5.lmdb")
    ):
        print("loading ProtTrans T5")
        import prottrans_t5

        encoder = prottrans_t5.encode

        self._embed(input_path, output_path, encoder, 512)

    def proteinbert(
        self, input_path: Path, output_path: Path = Path("proteinbert.lmdb")
    ):
        print("loading ProteinBERT")
        import proteinberter

        encoder = proteinberter.encode

        self._embed(input_path, output_path, encoder, 1022)

    def prose(self, input_path: Path, output_path: Path = Path("prose.lmdb")):
        print("loading PROSE")
        import proser

        encoder = proser.encode

        self._embed(input_path, output_path, encoder, 9999999)

    def esm(self, input_path: Path, output_path: Path = Path("esm.lmdb")):
        print("loading ESM")
        import esmer

        encoder = esmer.encode

        self._embed(input_path, output_path, encoder, 1024)

    def rapppid(self, input_path: Path, output_path: Path = Path("rapppid.lmdb")):
        print("loading RAPPPID")
        import rapppid

        encoder = rapppid.encode

        self._embed(input_path, output_path, encoder, 64)


if __name__ == "__main__":
    fire.Fire(Embed(), name="Embed")
