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
from autofigures.fc_net import PPINet
from autofigures.dataset import RapppidDataset2
from autofigures.utils import default_paths
from lightning.pytorch import seed_everything
from sklearn import metrics as sk_metrics
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import Union, Optional
from pathlib import Path


def compute_model_scores(model_name: str, seed: int, data_folder: Path):
    if model_name == "esm":
        chkpt_paths = [
            data_folder
            / "chkpts/ppi/iRcKUvufAzeh9CuJAtFMgjRq8Yo=/epoch=66-step=20569.ckpt",
            data_folder
            / "chkpts/ppi/aLK3NQAPP9uHfEDhrBCaFXTjE9I=/epoch=92-step=28551.ckpt",
            data_folder
            / "chkpts/ppi/x5i3FQ9dA3iI6IoHUyCzgydVWgM=/epoch=88-step=27323.ckpt",
        ]
        db_path = data_folder / "embeddings/esm.lmdb"
        model_dim = 6165

    elif model_name == "prottrans_bert":
        chkpt_paths = [
            data_folder
            / "chkpts/ppi/DoetbgdKMxDuKdfRWOUHSGAXTf0=/epoch=80-step=24867-v1.ckpt",
            data_folder
            / "chkpts/ppi/jjvmU1efEjj1VWqJ8frthcAuiK4=/epoch=90-step=27937-v1.ckpt",
            data_folder
            / "chkpts/ppi/nHLZTP7wTAmtJSlpLtAYOH6gAj0=/epoch=80-step=24867-v1.ckpt",
        ]
        db_path = data_folder / "embeddings/prottrans_bert.lmdb"
        model_dim = 1024

    elif model_name == "prottrans_t5":
        chkpt_paths = [
            data_folder
            / "chkpts/ppi/Dvy3BYf00JQ18SHKOSdCu9zWojs=/epoch=97-step=30086.ckpt",
            data_folder
            / "chkpts/ppi/XAnrmSjztabYNXRBbdrU2uV8XRM=/epoch=86-step=26709.ckpt",
            data_folder
            / "chkpts/ppi/PUoe1MerYJArEl8h4d-P40V06zA=/epoch=89-step=27630.ckpt",
        ]
        db_path = data_folder / "embeddings/prottrans_t5.lmdb"
        model_dim = 1024

    elif model_name == "prose":
        chkpt_paths = [
            data_folder
            / "chkpts/ppi/xSIEaL28YQW14K0UhNoWmwX-_HU=/epoch=89-step=27630.ckpt",
            data_folder
            / "chkpts/ppi/7KwnY62-2a7UBaKz3jplmvRqSWk=/epoch=99-step=30700.ckpt",
            data_folder
            / "chkpts/ppi/3VHXtNHqEashvmxcWsXwCxgPqs0=/epoch=97-step=30086.ckpt",
        ]
        db_path = data_folder / "embeddings/prose.lmdb"
        model_dim = 6165

    elif model_name == "proteinbert":
        chkpt_paths = [
            data_folder
            / "chkpts/ppi/048O3lE7pCo4Y_qpQAZLfxYrz6Q=/epoch=46-step=14429.ckpt",
            data_folder
            / "chkpts/ppi/Ir_BXqrPDOutyG20qLfc2Sn4qoE=/epoch=91-step=28244.ckpt",
            data_folder
            / "chkpts/ppi/7x1W0IhBtFMoEdgUYTJCY6yuhoc=/epoch=79-step=24560.ckpt",
        ]
        db_path = data_folder / "embeddings/proteinbert.lmdb"
        model_dim = 1562

    elif model_name == "squeezeprot_sp_strict":
        chkpt_paths = [
            data_folder
            / "chkpts/ppi/c-HcTcy0NwO7JY8U3lu8Nva7d7o=/epoch=75-step=23332.ckpt",
            data_folder
            / "chkpts/ppi/hdhFlXCydBsLZrIicCM4LMrNWoE=/epoch=13-step=4298.ckpt",
            data_folder
            / "chkpts/ppi/r8TQa4FnoUdDvr2LPGZqZk4z6y4=/epoch=16-step=5219.ckpt",
            data_folder
            / "chkpts/ppi/vFjXUGbR0vMEu8Bu0j2C2445J2A=/epoch=60-step=18727.ckpt",
            data_folder
            / "chkpts/ppi/5yEzfPl2E2eT54OJXLF0K75z_3I=/epoch=14-step=4605.ckpt",
            data_folder
            / "chkpts/ppi/W9DvdTJ7bW1GaCV3TyJ7HZznwZw=/epoch=29-step=9210.ckpt",
            data_folder
            / "chkpts/ppi/ngRDPuiaLHeCXQJvigyMuE1tsmw=/epoch=42-step=13201.ckpt",
            data_folder
            / "chkpts/ppi/MFwIN5YJ_98AUYYGORMlCvgv8j4=/epoch=25-step=7982.ckpt",
            data_folder
            / "chkpts/ppi/yGeHPlv6AviEUwC4jssZy-FuTBM=/epoch=39-step=12280.ckpt",
            data_folder
            / "chkpts/ppi/s3SMyXYEAmNECJgqSfhKutisFLc=/epoch=85-step=26402.ckpt",
        ]
        db_path = data_folder / "embeddings/squeezeprot-sp.strict.lmdb"
        model_dim = 768

    elif model_name == "squeezeprot_sp_nonstrict":
        chkpt_paths = [
            data_folder
            / "chkpts/ppi/0KCyeB_K-ZCIlNwpl3rRopZyM0I=/epoch=49-step=15350.ckpt",
            data_folder
            / "chkpts/ppi/1c-2iUOfs3Ye_pV0WGtCApV1Rgg=/epoch=13-step=4298.ckpt",
            data_folder
            / "chkpts/ppi/8F6cxSJIb1syPfiZDK04DgCaP90=/epoch=54-step=16885.ckpt",
            data_folder
            / "chkpts/ppi/SsTLBBEPiGnpX91hunP9-E8w6jY=/epoch=64-step=19955.ckpt",
            data_folder
            / "chkpts/ppi/EdJtO7pKkYBjeeGajJ33-hySWfE=/epoch=35-step=11052.ckpt",
            data_folder
            / "chkpts/ppi/K51CpKE1he98de-DV7Pq67DE4ok=/epoch=91-step=28244.ckpt",
            data_folder
            / "chkpts/ppi/6U8njyd40iXoaBCx4WdL57FOX9o=/epoch=24-step=7675.ckpt",
            data_folder
            / "chkpts/ppi/7YxAquqSeLtB1I85ItoXQvtGsg8=/epoch=23-step=7368.ckpt",
            data_folder
            / "chkpts/ppi/qI4P2gmDPMvY6epwa6cIgG9PQlE=/epoch=12-step=3991.ckpt",
            data_folder
            / "chkpts/ppi/MmCcff1PHh0lOvl84LiytfrznMw=/epoch=23-step=7368.ckpt",
        ]
        db_path = data_folder / "embeddings/squeezeprot-sp.nonstrict.lmdb"
        model_dim = 768

    elif model_name == "squeezeprot_u50":
        chkpt_paths = [
            data_folder
            / "chkpts/ppi/uARWusUXomAV5qa0X9e2V4C5wwI=/epoch=83-step=25788.ckpt",
            data_folder
            / "chkpts/ppi/cpns6_wtKs93e6ekMXxkBilEGv4=/epoch=83-step=25788.ckpt",
            data_folder
            / "chkpts/ppi/OcZKrg-rYN5RsfC6TH9btqfdVV8=/epoch=80-step=24867.ckpt",
        ]
        db_path = data_folder / "embeddings/squeezebert.u50.lmdb"
        model_dim = 768
    else:
        raise ValueError("Unknown model name '{}'".format(model_name))

    if len(chkpt_paths) < seed:
        return []
    else:
        chkpt_path = chkpt_paths[seed - 1]

    dataset_path = (
        data_folder
        / "ppi/rapppid_[common_string_9606.protein.links.detailed.v12.0_upkb.csv]_Mz70T9t-4Y-i6jWD9sEtcjOr0X8=.h5"
    )
    dataset = RapppidDataset2(dataset_path, db_path, 3, "test")

    weights = torch.load(chkpt_path, weights_only=True)

    ppi_net = PPINet(1, 100, model_dim, pooling="average", num_layers=3).cuda()
    ppi_net.load_state_dict(weights["state_dict"])

    rows = []

    for idx in range(len(dataset)):
        a_vec, b_vec, label = dataset[idx]

        y_hat = ppi_net(a_vec.unsqueeze(0), b_vec.unsqueeze(0))
        y_hat = torch.sigmoid(y_hat)[0].item()

        rows.append(
            {
                "y_hat": y_hat,
                "label": label,
                "model_name": model_name,
            }
        )

    return rows


def compute_all_scores(seed: int, data_folder: Path):
    seed_everything(seed, workers=True)

    model_names = [
        "esm",
        "prottrans_bert",
        "prottrans_t5",
        "prose",
        "proteinbert",
        "squeezeprot_sp_strict",
        "squeezeprot_sp_nonstrict",
        "squeezeprot_u50",
    ]

    rows = []

    for model_name in tqdm(model_names):
        print(model_name)
        rows += compute_model_scores(model_name, seed, data_folder)

    return pd.DataFrame(rows)


def get_metrics(df):
    model_names = df.model_name.unique()

    metrics = {model_name: {} for model_name in model_names}

    for model_name in model_names:
        label = df[df["model_name"] == model_name].label
        yhat_bin = (np.array(df[df["model_name"] == model_name].y_hat) > 0.5).astype(
            int
        )
        yhat = df[df["model_name"] == model_name].y_hat

        mcc = sk_metrics.matthews_corrcoef(label, yhat_bin)
        auroc = sk_metrics.roc_auc_score(label, yhat)

        metrics[model_name]["mcc"] = mcc
        metrics[model_name]["auroc"] = auroc

    return metrics


def get_scores(
    output_folder: Optional[Union[Path, str]] = None,
    data_folder: Optional[Union[Path, str]] = None,
):
    output_folder, data_folder = default_paths(output_folder, data_folder)

    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        print(f"COMPUTING SEED {seed}")

        if os.path.isfile(output_folder / f"tables/scores_s{seed}.csv"):
            print(f"# SKIPPING SCORING FOR {seed}, FOUND FILE")
        else:
            df = compute_all_scores(seed, data_folder)
            df.to_csv(output_folder / f"tables/scores_s{seed}.csv")

        print("DONE.")
