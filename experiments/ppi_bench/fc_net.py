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

import torch
from torch import nn
import pytorch_lightning as pl
from collections import OrderedDict
from weightdrop import WeightDrop
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryMatthewsCorrCoef,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryConfusionMatrix,
    BinaryF1Score,
)


class Encoder(nn.Module):
    def __init__(self, input_dim, pooling=None, num_layers=3):
        super().__init__()

        self.pooling = pooling
        self.embed_dim = 768
        self.input_dim = input_dim

        if pooling == "average":
            self.pooler = nn.AdaptiveAvgPool1d(self.embed_dim)
        elif pooling == "max":
            self.pooler = nn.AdaptiveMaxPool1d(self.embed_dim)
        elif pooling is None:
            self.embed_dim = self.input_dim
        else:
            raise ValueError(f'Invalid value for pooling: "{pooling}"')

        if num_layers == 1:
            self.net = nn.Sequential(
                WeightDrop(
                    nn.Linear(self.embed_dim, 64),
                    ["weight"],
                    dropout=0.25,
                    variational=False,
                )
            )
        elif num_layers == 2:
            self.net = nn.Sequential(
                WeightDrop(
                    nn.Linear(self.embed_dim, 32),
                    ["weight"],
                    dropout=0.25,
                    variational=False,
                ),
                nn.Mish(),
                WeightDrop(
                    nn.Linear(32, 64),
                    ["weight"],
                    dropout=0.25,
                    variational=False,
                ),
            )
        elif num_layers > 2:
            start_block = [
                nn.Dropout(p=0.2),
                WeightDrop(
                    nn.Linear(self.embed_dim, 32),
                    ["weight"],
                    dropout=0.25,
                    variational=False,
                ),
                nn.Mish(),
            ]
            mid_block = [
                nn.Dropout(p=0.2),
                WeightDrop(
                    nn.Linear(32, 32),
                    ["weight"],
                    dropout=0.25,
                    variational=False,
                ),
                nn.Mish(),
            ]
            end_block = [
                WeightDrop(
                    nn.Linear(32, 64),
                    ["weight"],
                    dropout=0.25,
                    variational=False,
                )
            ]

            net_list = start_block + mid_block * (num_layers - 2) + end_block
            self.net = nn.Sequential(*net_list)
        else:
            raise ValueError("num_layers must be 1 or greater")

    def forward(self, x):
        if self.pooling is not None and self.input_dim > self.embed_dim:
            x = self.pooler(x)

        return self.net(x)


class Classifier(nn.Module):
    def __init__(self, num_layers=3):
        super().__init__()

        self.num_layers = num_layers

        if self.num_layers == 1:
            self.net = nn.Sequential(
                OrderedDict(
                    [
                        ("do0", nn.Dropout(p=0.2)),
                        ("nl0", nn.Mish()),
                        ("fc0", nn.Linear(64, 1)),
                    ]
                )
            )
        elif self.num_layers == 2:
            self.net = nn.Sequential(
                OrderedDict(
                    [
                        ("nl0", nn.Mish()),
                        ("do0", nn.Dropout(p=0.2)),
                        ("fc0", nn.Linear(64, 32)),
                        ("nl1", nn.Mish()),
                        ("do1", nn.Dropout(p=0.2)),
                        ("fc1", nn.Linear(32, 1)),
                    ]
                )
            )
        elif self.num_layers > 2:
            start_block = [
                ("nl0", nn.Mish()),
                ("do0", nn.Dropout(p=0.2)),
                ("fc0", nn.Linear(64, 32)),
            ]
            end_block = [
                ("nl_out", nn.Mish()),
                ("do_out", nn.Dropout(p=0.2)),
                ("fc_out", nn.Linear(32, 1)),
            ]

            net_list = start_block

            for i in range(self.num_layers - 2):
                net_list += [
                    (f"nl_{i + 1}", nn.Mish()),
                    (f"do_{i + 1}", nn.Dropout(p=0.2)),
                    (f"fc_{i + 1}", nn.Linear(32, 32)),
                ]

            net_list += end_block
            net_dict = OrderedDict(net_list)

            self.net = nn.Sequential(net_dict)

    def forward(self, x):
        return self.net(x)


class PPINet(pl.LightningModule):
    def __init__(
        self, steps_per_epoch, num_epochs, input_dim, pooling=None, num_layers=3
    ):
        super().__init__()

        self.steps_per_epoch = steps_per_epoch
        self.num_epochs = num_epochs

        self.encoder = Encoder(input_dim, pooling, num_layers)
        self.classifier = Classifier(num_layers)
        self.loss = nn.BCEWithLogitsLoss()

        self.acc_fn = BinaryAccuracy()
        self.auroc_fn = BinaryAUROC()
        self.ap_fn = BinaryAveragePrecision()
        self.mcc_fn = BinaryMatthewsCorrCoef()
        self.conf_fn = BinaryConfusionMatrix()
        self.f1_fn = BinaryF1Score()

    def forward(self, a, b):
        a, b = torch.tensor(a).cuda(), torch.tensor(b).cuda()

        z_a = self.encoder(a)
        z_b = self.encoder(b)

        z = (z_a + z_b) / 2

        y_hat = self.classifier(z).squeeze(1)

        return y_hat

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        a, b, y = batch

        a, b, y = torch.tensor(a).cuda(), torch.tensor(b).cuda(), torch.tensor(y).cuda()

        z_a = self.encoder(a)
        z_b = self.encoder(b)

        z = (z_a + z_b) / 2

        y_hat = self.classifier(z).squeeze(1)

        loss = self.loss(y_hat, y.float())

        acc = self.acc_fn(y_hat, y.float())
        auroc = self.auroc_fn(y_hat, y.float())
        mcc = self.mcc_fn(y_hat, y.float())
        ap = self.ap_fn(y_hat, y.int())
        f1 = self.f1_fn(y_hat, y.float())

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_auroc", auroc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_mcc", mcc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_ap", ap, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_f1", f1, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        a, b, y = batch

        a, b, y = torch.tensor(a).cuda(), torch.tensor(b).cuda(), torch.tensor(y).cuda()

        z_a = self.encoder(a)
        z_b = self.encoder(b)

        z = (z_a + z_b) / 2

        y_hat = self.classifier(z).squeeze(1)

        loss = self.loss(y_hat, y.float())

        acc = self.acc_fn(y_hat, y.float())
        auroc = self.auroc_fn(y_hat, y.float())
        mcc = self.mcc_fn(y_hat, y.float())
        ap = self.ap_fn(y_hat, y.int())
        f1 = self.f1_fn(y_hat, y.float())

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_auroc", auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_mcc", mcc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_ap", ap, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_f1", f1, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def test_step(self, batch, batch_idx):
        # training_step defines the train loop.
        a, b, y = batch

        a, b, y = torch.tensor(a).cuda(), torch.tensor(b).cuda(), torch.tensor(y).cuda()

        z_a = self.encoder(a)
        z_b = self.encoder(b)

        z = (z_a + z_b) / 2

        y_hat = self.classifier(z).squeeze(1)

        loss = self.loss(y_hat, y.float())

        acc = self.acc_fn(y_hat, y.float())
        auroc = self.auroc_fn(y_hat, y.float())
        mcc = self.mcc_fn(y_hat, y.float())
        ap = self.ap_fn(y_hat, y.int())
        f1 = self.f1_fn(y_hat, y.float())

        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_auroc", auroc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=False)
        self.log("test_mcc", mcc, on_step=True, on_epoch=True, prog_bar=False)
        self.log("test_ap", ap, on_step=True, on_epoch=True, prog_bar=False)
        self.log("test_f1", mcc, on_step=True, on_epoch=True, prog_bar=False)

        return loss

    def configure_optimizers(self):
        """
        optimizer = Ranger21(
                self.parameters(),
                use_warmup=False,
                warmdown_active=False,
                lr=1e-2,
                weight_decay=2e-2,
                num_batches_per_epoch=self.steps_per_epoch,
                num_epochs=self.num_epochs,
                warmdown_start_pct=0.72,
            )
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.02)
        return optimizer
