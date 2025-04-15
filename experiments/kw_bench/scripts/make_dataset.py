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
import json
import csv
from pathlib import Path
from fasta import stream_fasta
import fire
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm
import lmdb
import sys

sys.path.insert(1, "../net")
from data import Tensorpack


def filter_list(els, flist):
    filtered_els = []

    for el in els:
        if el in flist:
            filtered_els.append(el)

    return filtered_els


def parse_str_list(str_list):
    return str_list[1:-1].split(", ")


def els2ids(els, flist):
    el_ids = []

    for el in els:
        el_id = list(flist).index(el)
        el_ids.append(el_id)

    return el_ids


def zopen(path, mode="r", **kwargs):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, **kwargs)
    else:
        if mode == "rt":
            mode = "r"
        elif mode == "wt":
            mode = "w"

        return open(path, mode, **kwargs)


def get_vec(name: str, txn):
    vec_serialized = txn.get(name.encode("utf8"))

    if vec_serialized is None:
        print(name, name.encode("utf8"))
    else:
        return Tensorpack.encode(torch.tensor(json.loads(vec_serialized)))


def main(
    kw_path: Path = Path("../../../data/kw/kw.json.gz"),
    kw_train_path: Path = Path("../../../data/kw/kw_train.csv.gz"),
    kw_test_path: Path = Path("../../../data/kw/kw_test.csv.gz"),
    kw_test_vecs_path: Path = Path("../../../data/kw/kw_test_vecs.csv.gz"),
    kw_train_vecs_path: Path = Path("../../../data/kw/kw_train_vecs.csv.gz"),
    kw_features_path: Path = Path("../../../data/kw/subloc_kw_feature.csv.gz"),
    eval_seqs_path: Path = Path(
        "../../../data/sequences/uniprot_sprot.strict.eval.fasta.gz"
    ),
    strict_database_path: Path = Path(
        "../../../data/embeddings/squeezeprot-sp.strict.lmdb"
    ),
    nonstrict_database_path: Path = Path(
        "../../../data/embeddings/squeezeprot-sp.nonstrict.lmdb"
    ),
):
    print("Reading keywords...")
    with zopen(kw_path, "rt") as f:
        kws = json.load(f)

    kws = kws["flatten(list(keyword_attr_id))"]

    print(f"\t A total of {len(set(kws))} keywords were found")

    kw_counter = Counter(kws)

    unique_kws = []
    counts_kws = []

    # we'll here remove super infrequent keywords
    for kw in set(kws):
        if kw_counter[kw] < 10:
            continue

        unique_kws.append(kw)
        counts_kws.append(kw_counter[kw])

    # sorting helps to keep keyword indices deterministic
    idx_sort = np.argsort(counts_kws)
    counts_kws = np.array(counts_kws)[idx_sort]
    unique_kws = np.array(unique_kws)[idx_sort]

    print(f"\t Narrowed down to {len(unique_kws)} keywords")

    print("Reading testing proteins...")
    testing_proteins = []

    for name, sequence in stream_fasta(eval_seqs_path):
        testing_proteins.append(name.split("|")[0])

    print(f"\t Loaded {len(testing_proteins)} testing proteins")

    print("Writing kw dataset...")
    with zopen(kw_train_path, "wt") as f_train:
        with zopen(kw_test_path, "wt") as f_test:
            fieldnames = ["accession", "keywords", "keyword_ids", "sequence"]

            writer_train = csv.DictWriter(f_train, fieldnames=fieldnames)
            writer_test = csv.DictWriter(f_test, fieldnames=fieldnames)

            writer_train.writeheader()
            writer_test.writeheader()

            with zopen(kw_features_path, newline="", mode="rt") as csvfile:
                reader = csv.DictReader(csvfile)

                for in_row in reader:
                    kws = filter_list(
                        parse_str_list(in_row["keyword_attr_id"]), unique_kws
                    )

                    if len(kws) == 0:
                        continue

                    out_row = {
                        "accession": in_row["entry_accession"],
                        "keywords": kws,
                        "keyword_ids": els2ids(kws, unique_kws),
                        "sequence": in_row["sequence"],
                    }

                    if in_row["entry_accession"] in testing_proteins:
                        writer_test.writerow(out_row)
                    else:
                        writer_train.writerow(out_row)

    print("Writing kw vecs...")
    num_rows = 0

    with zopen(kw_test_path, newline="", mode="rt") as f_in:
        reader = csv.DictReader(f_in)

        for _ in reader:
            num_rows += 1

    strict_db_env = lmdb.open(
        str(strict_database_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    strict_txn = strict_db_env.begin()

    nonstrict_db_env = lmdb.open(
        str(nonstrict_database_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    nonstrict_txn = nonstrict_db_env.begin()

    for kw_path, kw_vecs_path in [
        (kw_test_path, kw_test_vecs_path),
        (kw_train_path, kw_train_vecs_path),
    ]:
        with zopen(kw_path, newline="", mode="rt") as f_in:
            with zopen(kw_vecs_path, newline="", mode="wt") as f_out:
                fieldnames = [
                    "accession",
                    "keywords",
                    "keyword_ids",
                    "sequence",
                    "strict_vector",
                    "nonstrict_vector",
                ]
                reader = csv.DictReader(f_in)
                writer = csv.DictWriter(f_out, fieldnames=fieldnames)

                for row_num, row in enumerate(tqdm(reader)):
                    row["strict_vector"] = get_vec(row["accession"], strict_txn)
                    row["nonstrict_vector"] = get_vec(row["accession"], nonstrict_txn)

                    writer.writerow(row)


if __name__ == "__main__":
    fire.Fire(main)
