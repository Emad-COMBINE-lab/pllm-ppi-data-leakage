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
import json
import gzip
import msgpack
import lightning as L
from pathlib import Path
from base64 import b64encode, b64decode
from typing import Optional, Union
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class Tensorpack:
    def __init__(self):
        pass

    @staticmethod
    def encode(t: torch.Tensor, b64: bool = True) -> Union[str, bytes]:
        torch_dtypes = [
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        ]

        t_bytes = t.numpy().tobytes()
        t_dtype = torch_dtypes.index(t.dtype)
        t_shape = t.shape

        payload = msgpack.dumps([t_bytes, t_dtype, t_shape])

        if b64:
            payload = b64encode(payload).decode("utf-8")

        return payload

    @staticmethod
    def decode(payload: Union[str, bytes]) -> torch.Tensor:
        if isinstance(payload, str):
            payload = b64decode(payload.encode("utf-8"))

        numpy_dtypes = [
            bool,
            np.uint8,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.float16,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        ]

        t_bytes, dtype_idx, t_shape = msgpack.loads(payload)

        np_dtype = numpy_dtypes[dtype_idx]

        t_np = np.frombuffer(t_bytes, dtype=np_dtype).copy()
        t_flat = torch.from_numpy(t_np)
        t = t_flat.reshape(t_shape)

        return t


class KeywordDataset(Dataset):
    def __init__(
        self,
        row_file: Path,
        strict: bool,
        num_rows: Optional[int] = None,
        offset: Optional[int] = None,
        num_classes=1083,
    ):
        super().__init__()

        self.row_file = row_file
        self.strict = strict
        self.offset = offset
        self.num_classes = num_classes

        self.kw_ids, self.vecs = self.load_rows(row_file, num_rows)

    def load_rows(self, row_file: Path, num_rows: int):
        kw_ids = []
        vecs = []

        with gzip.open(row_file, mode="rt", newline="") as f:
            reader = csv.reader(f)

            for row_num, row in enumerate(reader):
                if self.offset is not None and row_num < self.offset:
                    continue

                kw_ids.append(row[2])

                if self.strict:
                    vec_column = 4
                else:
                    vec_column = 5

                vec = row[vec_column]

                vecs.append(vec)

                if num_rows is not None and row_num > num_rows:
                    break

        return kw_ids, vecs

    def __getitem__(self, idx):
        kw_ids = json.loads(self.kw_ids[idx])
        kw_ohs = torch.zeros(self.num_classes)
        kw_ohs[kw_ids] = 1

        return kw_ohs, Tensorpack.decode(self.vecs[idx])

    def __len__(self):
        return len(self.kw_ids)


class KeywordDataModule(L.LightningDataModule):
    def __init__(
        self, train_path: Path, test_path: Path, strict: bool, batch_size: int
    ):
        super().__init__()

        self.train_path = train_path
        self.test_path = test_path
        self.strict = strict
        self.batch_size = batch_size

    def setup(self, stage: str):
        num_test = 0

        with gzip.open(self.test_path, "rt") as f:
            reader = csv.reader(f)
            for line in reader:
                num_test += 1

        self.train_dataset = KeywordDataset(self.train_path, self.strict)
        self.val_dataset = KeywordDataset(
            self.test_path, self.strict, num_rows=num_test // 2
        )
        self.test_dataset = KeywordDataset(
            self.test_path, self.strict, offset=num_test // 2
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
