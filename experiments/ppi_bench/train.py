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

from logger import DictLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import RapppidDataModule2
from fc_net import PPINet
import torch
import json
import gzip
import fire
from tqdm import tqdm
import pandas as pd
from base64 import urlsafe_b64encode
from hashlib import sha1
from lightning.pytorch import seed_everything


def train(
    model_name,
    max_epochs,
    num_layers,
    pooling,
    dataset_file,
    database_path,
    batch_size,
    input_dim,
    c_level,
    workers,
    seed,
):
    seed_everything(seed)

    dict_logger = DictLogger(version=model_name)

    datamodule = RapppidDataModule2(
        batch_size, dataset_file, database_path, c_level, workers
    )
    datamodule.setup()

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"../../data/chkpts/ppi/{model_name}", save_top_k=1, monitor="val_loss"
    )
    trainer = pl.Trainer(
        precision=16,
        logger=dict_logger,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
    )  # , callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.00, patience=3, verbose=True,)])

    steps_per_epoch = len(datamodule.train_dataloader()) // batch_size
    net = PPINet(steps_per_epoch, max_epochs, input_dim, pooling, num_layers)

    num_params = sum(p.numel() for p in net.parameters())

    trainer.fit(model=net, datamodule=datamodule)

    # Test
    print("Testing...")
    net = net.to("cuda")
    dataset = datamodule.dataset_test
    rows = []

    for idx in tqdm(range(len(dataset)), total=len(dataset)):
        a_vec, b_vec, label = dataset[idx]
        a_vec, b_vec, label = (
            a_vec.to("cuda"),
            b_vec.to("cuda"),
            torch.tensor(label).to("cuda"),
        )

        y_hat = net(a_vec.unsqueeze(0), b_vec.unsqueeze(0))
        y_hat = torch.sigmoid(y_hat)[0].item()

        rows.append({"y_hat": y_hat, "label": label.item()})

    df = pd.DataFrame(rows)
    df.to_csv(f"../../data/chkpts/ppi/{model_name}/test.csv")

    return dict_logger, num_params


def make_hashable(o):
    # from https://stackoverflow.com/questions/5884066/hashing-a-dictionary
    if isinstance(o, (tuple, list)):
        return tuple((make_hashable(e) for e in o))

    if isinstance(o, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in o.items()))

    if isinstance(o, (set, frozenset)):
        return tuple(sorted(make_hashable(e) for e in o))

    return o


def hash_dict(d: dict) -> str:
    """
    Create a portable, deterministic hash of a dictionary d
    :param d: dictionary to hash
    :return: The hash of the dict
    """
    return urlsafe_b64encode(sha1(repr(make_hashable(d)).encode()).digest()).decode()


def main(
    max_epochs,
    num_layers,
    dataset_file,
    database_path,
    batch_size,
    input_dim,
    c_level,
    workers=16,
    pooling=None,
    seed=8675309,
):
    payload = {
        "max_epochs": max_epochs,
        "num_layers": num_layers,
        "pooling": pooling,
        "dataset_file": dataset_file,
        "database_path": database_path,
        "batch_size": batch_size,
        "input_dim": input_dim,
        "c_level": c_level,
        "workers": workers,
        "seed": seed,
    }

    model_name = hash_dict(payload)

    dict_logger, num_params = train(
        model_name,
        max_epochs,
        num_layers,
        pooling,
        dataset_file,
        database_path,
        batch_size,
        input_dim,
        c_level,
        workers,
        seed,
    )

    payload["logging"] = dict_logger.metrics
    payload["num_params"] = num_params

    with gzip.open(f"../../data/chkpts/ppi/{model_name}/hparams.json.gz", "wt") as f:
        json.dump(payload, f, indent=4)


if __name__ == "__main__":
    fire.Fire(main)
