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


import os
import torch
import json
from pathlib import Path
from torch import nn, optim
import lightning as L
from weightdrop import WeightDrop


class AsymmetricLoss(nn.Module):
    def __init__(self, weights=None, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg  # Focusing parameter for negative samples
        self.gamma_pos = gamma_pos  # Focusing parameter for positive samples
        self.clip = clip  # Probability margin for negative samples
        self.eps = eps  # For numerical stability
        self.register_buffer(
            "weights", weights if weights is not None else torch.ones(1)
        )

    def forward(self, x, y):
        # Input x is logits, y is targets
        # Apply sigmoid to get probabilities
        x_sigmoid = torch.sigmoid(x)

        # Separate positive and negative samples
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric clipping: clip the negative sample predictions
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic BCE loss computation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric focusing
        modulator_pos = torch.pow(1 - xs_pos, self.gamma_pos)
        modulator_neg = torch.pow(1 - xs_neg, self.gamma_neg)

        # Combine positive and negative losses with their respective modulators
        loss = modulator_pos * los_pos + modulator_neg * los_neg

        # Apply class weights
        if self.weights.shape != torch.Size([1]):
            weighted_loss = loss * self.weights.view(1, -1)  # Shape to [1, num_classes]
        else:
            weighted_loss = loss

        # Return mean loss across all elements
        return -weighted_loss.mean()


class Protein2KeywordNet(L.LightningModule):
    def __init__(
        self,
        meta_path: Path,
        embed_dim=768,
        out_dim=1083,
        num_workers=None,
        label_weighting=None,
        loss_type="asl",
        nl_type="mish",
    ):
        super().__init__()

        if nl_type == "mish":
            nl_fn = nn.Mish()
        elif nl_type == "relu6":
            nl_fn = nn.ReLU6()
        elif nl_type == "leaky_relu":
            nl_fn = nn.LeakyReLU()
        else:
            raise ValueError("Expected nl_type 'mish', 'relu6', 'leaky_relu'.")

        self.net = nn.Sequential(
            nl_fn,
            WeightDrop(
                nn.Linear(embed_dim, 256),
                ["weight"],
                dropout=0.25,
                variational=False,
            ),
            nl_fn,
            WeightDrop(
                nn.Linear(256, 512),
                ["weight"],
                dropout=0.25,
                variational=False,
            ),
            nl_fn,
            nn.Linear(512, out_dim),
        )

        self.out_dim = out_dim

        if num_workers is None:
            self.num_workers = max(1, int(len(os.sched_getaffinity(0)) * 0.9))
        else:
            self.num_workers = num_workers

        with open(meta_path) as f:
            meta = json.load(f)

        negative_examples = torch.tensor([540074 - kw["count"] for kw in meta["kws"]])
        positive_examples = torch.tensor([kw["count"] for kw in meta["kws"]])

        if label_weighting is None:
            kw_pos_weights = torch.ones(len(meta["kws"]))
        elif label_weighting == "linear":
            kw_pos_weights = negative_examples / positive_examples
        elif label_weighting == "log":
            kw_pos_weights = torch.log(1 + (negative_examples / positive_examples))
        elif label_weighting == "root":
            kw_pos_weights = torch.sqrt(negative_examples / positive_examples)

        if loss_type == "asl":
            self.loss_fn = AsymmetricLoss(kw_pos_weights)
        elif loss_type == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=kw_pos_weights)
        else:
            raise ValueError("loss_type must be one of asl or bce")

    def forward(self, vecs):
        return self.net(vecs)

    def step(self, batch):
        kw_ohs, vecs = batch

        y_hat = self.forward(vecs)

        loss = self.loss_fn(y_hat, kw_ohs.to(vecs.dtype))

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train_loss_epoch", loss, on_step=False, on_epoch=True)
        self.log("train_loss_step", loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("val_loss_epoch", loss, on_step=False, on_epoch=True)
        self.log("val_loss_step", loss, on_step=True, on_epoch=False)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("test_loss_epoch", loss, on_step=False, on_epoch=True)
        self.log("test_loss_step", loss, on_step=True, on_epoch=False)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-5)
        return optimizer
