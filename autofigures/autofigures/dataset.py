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


import json
import lmdb
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import tables as tb
import torch
from pathlib import Path


class RapppidDatasetSeq(Dataset):
    def __init__(self, dataset_path, c_type, split):
        super().__init__()

        self.dataset_path = dataset_path
        self.c_type = c_type
        self.split = split

    def get_sequence(self, name: str):
        with tb.open_file(self.dataset_path) as dataset:
            seq = dataset.root.sequences.read_where(f'name=="{name}"')[0][1].decode(
                "utf8"
            )

        return seq

    def __getitem__(self, idx):
        with tb.open_file(self.dataset_path) as dataset:
            p1, p2, label = dataset.root["interactions"][f"c{self.c_type}"][
                f"c{self.c_type}_{self.split}"
            ][idx]

        p1 = p1.decode("utf8")
        p2 = p2.decode("utf8")

        p1_seq = self.get_sequence(p1)
        p2_seq = self.get_sequence(p2)

        return p1_seq, p2_seq, 1 if label else 0

    def __len__(self):
        with tb.open_file(self.dataset_path) as dataset:
            l = len(
                dataset.root["interactions"][f"c{self.c_type}"][
                    f"c{self.c_type}_{self.split}"
                ]
            )
        return l


class RapppidDataset2(Dataset):
    def __init__(self, dataset_path, database_path, c_type, split):
        super().__init__()

        self.dataset_path = dataset_path
        self.database_path = database_path
        self.c_type = c_type
        self.split = split

        self.db_env = lmdb.open(
            str(database_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.txn = self.db_env.begin()

    def get_vec(self, name: str):
        vec_serialized = self.txn.get(name.encode("utf8"))

        if vec_serialized is None:
            print(name, name.encode("utf8"))
        else:
            return json.loads(vec_serialized)

    def __getitem__(self, idx):
        with tb.open_file(self.dataset_path) as dataset:
            p1, p2, label = dataset.root["interactions"][f"c{self.c_type}"][
                f"c{self.c_type}_{self.split}"
            ][idx]

        p1 = p1.decode("utf8")
        p2 = p2.decode("utf8")

        p1_vec = self.get_vec(p1)
        p2_vec = self.get_vec(p2)

        return torch.tensor(p1_vec), torch.tensor(p2_vec), 1 if label else 0

    def __len__(self):
        with tb.open_file(self.dataset_path) as dataset:
            l = len(
                dataset.root["interactions"][f"c{self.c_type}"][
                    f"c{self.c_type}_{self.split}"
                ]
            )
        return l


class RapppidDataModule2(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        dataset_path: Path,
        database_path: Path,
        c_type: int,
        workers: int,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.database_path = database_path

        self.dataset_train = None
        self.dataset_test = None

        self.workers = workers

        self.c_type = c_type

        self.train = []
        self.test = []
        self.seqs = []

    def setup(self, stage=None):
        self.dataset_train = RapppidDataset2(
            self.dataset_path, self.database_path, self.c_type, "train"
        )
        self.dataset_val = RapppidDataset2(
            self.dataset_path,
            self.database_path,
            self.c_type,
            "val",
        )
        self.dataset_test = RapppidDataset2(
            self.dataset_path, self.database_path, self.c_type, "test"
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=False,
        )
