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


from torch import nn
import pytorch_lightning as pl
from collections import OrderedDict
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryMatthewsCorrCoef,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryConfusionMatrix,
    BinaryF1Score,
)

from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers.logger import rank_zero_experiment
from collections import defaultdict
from pytorch_lightning.loggers import Logger
import torch
from torch.nn import Parameter


# https://github.com/mourga/variational-lstm/blob/master/weight_drop.py
class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=True):
        """
        Dropout class that is paired with a torch module to make sure that the SAME mask
        will be sampled and applied to ALL timesteps.
        :param module: nn. module (e.g. nn.Linear, nn.LSTM)
        :param weights: which weights to apply dropout (names of weights of module)
        :param dropout: dropout to be applied
        :param variational: if True applies Variational Dropout, if False applies DropConnect (different masks!!!)
        """
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        """
        Smerity code I don't understand.
        """
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):
        """
        This function renames each 'weight name' to 'weight name' + '_raw'
        (e.g. weight_hh_l0 -> weight_hh_l0_raw)
        :return:
        """
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + "_raw", Parameter(w.data))

    def _setweights(self):
        """
        This function samples & applies a dropout mask to the weights of the recurrent layers.
        Specifically, for an LSTM, each gate has
        - a W matrix ('weight_ih') that is multiplied with the input (x_t)
        - a U matrix ('weight_hh') that is multiplied with the previous hidden state (h_t-1)
        We sample a mask (either with Variational Dropout or with DropConnect) and apply it to
        the matrices U and/or W.
        The matrices to be dropped-out are in self.weights.
        A 'weight_hh' matrix is of shape (4*nhidden, nhidden)
        while a 'weight_ih' matrix is of shape (4*nhidden, ninput).
        **** Variational Dropout ****
        With this method, we sample a mask from the tensor (4*nhidden, 1) PER ROW
        and expand it to the full matrix.
        **** DropConnect ****
        With this method, we sample a mask from the tensor (4*nhidden, nhidden) directly
        which means that we apply dropout PER ELEMENT/NEURON.
        :return:
        """
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + "_raw")
            w = None

            if self.variational:
                #######################################################
                # Variational dropout (as proposed by Gal & Ghahramani)
                #######################################################
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                mask = mask.cuda()
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                #######################################################
                # DropConnect (as presented in the AWD paper)
                #######################################################
                w = torch.nn.functional.dropout(
                    raw_w, p=self.dropout, training=self.training
                )

            if not self.training:  # (*)
                w = w.data

            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


class DictLogger(Logger):
    def __init__(self, name=None, version=None):
        super().__init__()

        def def_value():
            return []

        # Defining the dict
        self.metrics = defaultdict(def_value)
        self._name = "DictLogger" if name is None else name
        self._version = "0.1" if version is None else version

    @property
    def name(self):
        return self._name

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        pass

    @property
    def version(self):
        # Return the experiment version, int or str.
        return self._version

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        for key in metrics.keys():
            self.metrics[key].append(metrics[key])

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        # If you implement this, remember to call `super().save()`
        # at the start of the method (important for aggregation of metrics)
        super().save()

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass


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
