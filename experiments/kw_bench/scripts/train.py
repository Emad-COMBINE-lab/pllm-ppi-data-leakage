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

import sys

sys.path.insert(1, "../net")
from data import KeywordDataModule
from net import Protein2KeywordNet
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import seed_everything
import fire
from pathlib import Path
import json


def train(
    nl_type: str = "mish",
    loss_type: str = "asl",
    label_weighting: str = "log",
    train_path: Path = Path("../../../data/kw/kw_train_vecs.csv.gz"),
    test_path: Path = Path("../../../data/kw/kw_test_vecs.csv.gz"),
    meta_path: Path = Path("../../../data/kw/kw_meta.json"),
    hparams_dir: Path = Path("../../../data/kw/hparams/"),
    chkpts_dir: Path = Path("../../../data/chkpts/kw/"),
    logs_dir: Path = Path("../../../data/chkpts/kw/logs/"),
):
    # DATA

    print("LOADING STRICT DATASET")
    strict_data_module = KeywordDataModule(train_path, test_path, True, 100)
    print("LOADING NON-STRICT DATASET")
    nonstrict_data_module = KeywordDataModule(train_path, test_path, False, 100)

    # TRAIN
    idx = 0

    for seed in range(10):
        for strict in [True, False]:
            idx += 1

            seed_everything(seed, workers=True)

            row = {
                "idx": idx,
                "seed": seed,
                "strict": strict,
                "label_weighting": f"{label_weighting}",
                "loss_type": loss_type,
                "nl_type": nl_type,
            }

            print(json.dumps(row, indent=4))

            with open(hparams_dir / f"model_{idx}.json", "w") as f:
                json.dump(row, f)

            checkpoint_callback = ModelCheckpoint(
                save_top_k=1, monitor="val_loss_epoch"
            )
            logger = CSVLogger(logs_dir, name="KeywordNet", version=f"{idx}")

            if strict:
                data_module = strict_data_module
            else:
                data_module = nonstrict_data_module

            net = Protein2KeywordNet(
                meta_path,
                label_weighting=label_weighting,
                loss_type=loss_type,
                nl_type=nl_type,
            )
            trainer = L.Trainer(
                deterministic=True,
                precision=16,
                limit_train_batches=100,
                max_epochs=20,
                logger=logger,
                callbacks=[checkpoint_callback],
            )
            trainer.fit(model=net, datamodule=data_module)


if __name__ == "__main__":
    fire.Fire(train)
