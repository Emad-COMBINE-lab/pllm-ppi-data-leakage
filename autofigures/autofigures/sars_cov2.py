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

from pathlib import Path
from typing import Optional, Union
import torch
import lmdb
import json
import pandas as pd
from tqdm import tqdm
from os.path import isfile
import matplotlib.pyplot as plt
from sklearn import metrics as m
from scipy.stats import spearmanr
from autofigures.utils import default_paths, plot_style, fancy_model_names, colours
from autofigures.fc_net import PPINet
from autofigures.parse import parse_rapppid


def get_vec(env, protein_id: str):
    with env.begin() as txn:
        try:
            protein_vec = torch.tensor(
                json.loads(txn.get(protein_id.encode("utf8")).decode("utf8"))
            ).unsqueeze(0)
        except Exception as e:
            print(e)
            print(f"Error while getting {protein_id}")
            protein_vec = None

        return protein_vec


def get_vec_pairs(db_path: Path, ppi_path: Path):
    env = lmdb.open(str(db_path))

    ppi_df = pd.read_csv(ppi_path).dropna()

    for _, row in ppi_df.iterrows():
        bait_id, prey_id = row.Bait_RefSeq, row.Preys

        try:
            bait_vec = get_vec(env, bait_id)
        except KeyError:
            continue

        try:
            prey_vec = get_vec(env, prey_id)
        except KeyError:
            continue

        yield bait_vec, prey_vec, row


def get_scores(model_name: str, data_folder: Path, seed: int):
    if model_name == "esm":
        chkpt_paths = [
            data_folder
            / "chkpts/sars_cov2/iRcKUvufAzeh9CuJAtFMgjRq8Yo=/epoch=66-step=20569.ckpt",
            data_folder
            / "chkpts/sars_cov2/aLK3NQAPP9uHfEDhrBCaFXTjE9I=/epoch=92-step=28551.ckpt",
            data_folder
            / "chkpts/sars_cov2/x5i3FQ9dA3iI6IoHUyCzgydVWgM=/epoch=88-step=27323.ckpt",
        ]
        chkpt_path = chkpt_paths[seed - 1]

        db_path = data_folder / "embeddings/sars-cov-2/esm2_t33_650M_UR50D.lmdb"
        model_dim = 6165
        num_layers = 3
        pooling = "average"

    elif model_name == "prottrans_bert":
        chkpt_paths = [
            data_folder
            / "chkpts/sars_cov2/DoetbgdKMxDuKdfRWOUHSGAXTf0=/epoch=80-step=24867-v1.ckpt",
            data_folder
            / "chkpts/sars_cov2/jjvmU1efEjj1VWqJ8frthcAuiK4=/epoch=90-step=27937-v1.ckpt",
            data_folder
            / "chkpts/sars_cov2/nHLZTP7wTAmtJSlpLtAYOH6gAj0=/epoch=80-step=24867-v1.ckpt",
        ]
        chkpt_path = chkpt_paths[seed - 1]

        db_path = data_folder / "embeddings/sars-cov-2/prottrans_bert.lmdb"
        model_dim = 1024
        num_layers = 3
        pooling = "average"

    elif model_name == "prottrans_t5":
        chkpt_paths = [
            data_folder
            / "chkpts/sars_cov2/Dvy3BYf00JQ18SHKOSdCu9zWojs=/epoch=97-step=30086.ckpt",
            data_folder
            / "chkpts/sars_cov2/XAnrmSjztabYNXRBbdrU2uV8XRM=/epoch=86-step=26709.ckpt",
            data_folder
            / "chkpts/sars_cov2/PUoe1MerYJArEl8h4d-P40V06zA=/epoch=89-step=27630.ckpt",
        ]
        chkpt_path = chkpt_paths[seed - 1]

        db_path = data_folder / "embeddings/sars-cov-2/prottrans_t5.lmdb"
        model_dim = 1024
        num_layers = 3
        pooling = "average"

    elif model_name == "prose":
        chkpt_paths = [
            data_folder
            / "chkpts/sars_cov2/xSIEaL28YQW14K0UhNoWmwX-_HU=/epoch=89-step=27630.ckpt",
            data_folder
            / "chkpts/sars_cov2/7KwnY62-2a7UBaKz3jplmvRqSWk=/epoch=99-step=30700.ckpt",
            data_folder
            / "chkpts/sars_cov2/3VHXtNHqEashvmxcWsXwCxgPqs0=/epoch=97-step=30086.ckpt",
        ]
        chkpt_path = chkpt_paths[seed - 1]

        db_path = data_folder / "embeddings/sars-cov-2/prose.lmdb"
        model_dim = 6165
        num_layers = 3
        pooling = "average"

    elif model_name == "proteinbert":
        chkpt_paths = [
            data_folder
            / "chkpts/sars_cov2/048O3lE7pCo4Y_qpQAZLfxYrz6Q=/epoch=46-step=14429.ckpt",
            data_folder
            / "chkpts/sars_cov2/Ir_BXqrPDOutyG20qLfc2Sn4qoE=/epoch=91-step=28244.ckpt",
            data_folder
            / "chkpts/sars_cov2/7x1W0IhBtFMoEdgUYTJCY6yuhoc=/epoch=79-step=24560.ckpt",
        ]
        chkpt_path = chkpt_paths[seed - 1]

        db_path = data_folder / "embeddings/sars-cov-2/proteinbert.lmdb"
        model_dim = 1562
        num_layers = 3
        pooling = "average"

    elif model_name == "squeezeprot_sp_strict":  #'squeezebert_c3':
        chkpt_paths = [
            data_folder
            / "chkpts/sars_cov2/c-HcTcy0NwO7JY8U3lu8Nva7d7o=/epoch=75-step=23332.ckpt",
            data_folder
            / "chkpts/sars_cov2/hdhFlXCydBsLZrIicCM4LMrNWoE=/epoch=13-step=4298.ckpt",
            data_folder
            / "chkpts/sars_cov2/r8TQa4FnoUdDvr2LPGZqZk4z6y4=/epoch=16-step=5219.ckpt",
        ]
        chkpt_path = chkpt_paths[seed - 1]

        db_path = data_folder / "embeddings/sars-cov-2/squeezebert_strict.lmdb"
        model_dim = 768
        num_layers = 3
        pooling = None

    elif model_name == "squeezeprot_sp_nonstrict":  #'squeezebert_c1':
        chkpt_paths = [
            data_folder
            / "chkpts/sars_cov2/0KCyeB_K-ZCIlNwpl3rRopZyM0I=/epoch=49-step=15350.ckpt",
            data_folder
            / "chkpts/sars_cov2/1c-2iUOfs3Ye_pV0WGtCApV1Rgg=/epoch=13-step=4298.ckpt",
            data_folder
            / "chkpts/sars_cov2/8F6cxSJIb1syPfiZDK04DgCaP90=/epoch=54-step=16885.ckpt",
        ]
        chkpt_path = chkpt_paths[seed - 1]

        db_path = data_folder / "embeddings/sars-cov-2/squeezebert_nonstrict.lmdb"
        model_dim = 768
        num_layers = 3
        pooling = None

    elif model_name == "squeezeprot_u50":
        chkpt_paths = [
            data_folder
            / "chkpts/sars_cov2/uARWusUXomAV5qa0X9e2V4C5wwI=/epoch=83-step=25788.ckpt",
            data_folder
            / "chkpts/sars_cov2/cpns6_wtKs93e6ekMXxkBilEGv4=/epoch=83-step=25788.ckpt",
            data_folder
            / "chkpts/sars_cov2/OcZKrg-rYN5RsfC6TH9btqfdVV8=/epoch=80-step=24867.ckpt",
        ]
        chkpt_path = chkpt_paths[seed - 1]

        db_path = data_folder / "embeddings/sars-cov-2/squeezeprot_u50.lmdb"
        model_dim = 768
        num_layers = 3
        pooling = None

    elif model_name == "rapppid":
        chkpt_paths = [
            data_folder
            / "chkpts/sars_cov2/mH2BqxhcWlsaAsswXzt94t6DKWQ=/epoch=25-step=7982.ckpt",
            data_folder
            / "chkpts/sars_cov2/FANkW5KVGh84Y2pUX9mlT-3q9yc=/epoch=14-step=4605.ckpt",
            data_folder
            / "chkpts/sars_cov2/5wkTQvBTAI-VUmJ5fZoMQ6dW5vk=/epoch=29-step=9210.ckpt",
        ]
        chkpt_path = chkpt_paths[seed - 1]

        db_path = data_folder / "embeddings/sars-cov-2/rapppid.lmdb"
        model_dim = 64
        num_layers = 2
        pooling = None

    ppi_path = data_folder / "ppi/sars-cov-2/covid_ppi.csv"

    net = PPINet(0, 0, model_dim, pooling=pooling, num_layers=num_layers).cuda()
    weights = torch.load(chkpt_path, weights_only=True)
    net.load_state_dict(weights["state_dict"])
    net.eval()

    rows = []

    for idx, (bait_vec, prey_vec, ppi_row) in enumerate(
        get_vec_pairs(db_path, ppi_path)
    ):
        if bait_vec is None or prey_vec is None:
            continue

        yhat = torch.sigmoid(net(bait_vec, prey_vec)).detach().cpu().numpy()[0]

        rows.append(
            {
                "id": idx,
                "prey_id": ppi_row.Preys,
                "bait_id": ppi_row.Bait_RefSeq,
                "yhat": yhat,
                "mist_score": ppi_row["MIST"],
                "saint_bfdr_score": ppi_row["Saint_BFDR"],
                "avg_spec_score": ppi_row["AvgSpec"],
                "saint_score": ppi_row["SaintScore"],
                "fold_change": ppi_row["FoldChange"],
            }
        )

    return pd.DataFrame(rows)


def metrics_from_df(df: pd.DataFrame):
    metrics = dict()

    metrics["mcc"] = m.matthews_corrcoef(df["pos"], df["yhat"] > 0.50)
    metrics["b_acc"] = m.balanced_accuracy_score(df["pos"], df["yhat"] > 0.50)
    metrics["f1"] = m.f1_score(df["pos"], df["yhat"] > 0.50)
    metrics["ap"] = m.average_precision_score(df["pos"], df["yhat"])
    metrics["precision"] = m.precision_score(df["pos"], df["yhat"] > 0.50)
    metrics["recall"] = m.recall_score(df["pos"], df["yhat"] > 0.50)
    metrics["spearman_corr"], metrics["spearman_p"] = spearmanr(df["pos"], df["yhat"])

    return metrics


def sars_cov2(
    output_folder: Optional[Union[Path, str]] = None,
    data_folder: Optional[Union[Path, str]] = None,
):
    output_folder, data_folder = default_paths(output_folder, data_folder)

    plot_style()

    model_names = [
        "prottrans_bert",
        "prottrans_t5",
        "esm",
        "squeezeprot_sp_strict",
        "squeezeprot_sp_nonstrict",
        "proteinbert",
        "prose",
        "rapppid",
    ]

    # if not isfile(output_folder / f"tables/sars_cov2_metrics.csv") and \
    #        len(glob(output_folder / f"tables/sars_cov2_scores*")) != len(model_names) * 3:
    if not isfile(output_folder / f"tables/sars_cov2_metrics.csv"):
        metrics_rows = []

        for model_name in tqdm(model_names):
            for seed in [1, 2, 3]:
                df = get_scores(model_name, data_folder, seed)
                df = df.fillna(0)
                df["pos"] = (
                    (df["mist_score"] >= 0.7)
                    & (df["saint_bfdr_score"] <= 0.05)
                    & (df["avg_spec_score"] >= 2)
                )
                df.to_csv(
                    output_folder / f"tables/sars_cov2_scores_{model_name}_{seed}.csv"
                )

                metrics = metrics_from_df(df)
                metrics["model_name"] = model_name
                metrics["seed"] = seed

                metrics_rows.append(metrics)

        df_metrics = pd.DataFrame(metrics_rows)
        df_metrics.to_csv(output_folder / f"tables/sars_cov2_metrics.csv")

    f, ax = plt.subplots(1, 1, figsize=(5.5, 5))

    for seed in [1, 2, 3]:
        df = pd.read_csv(output_folder / f"tables/scores_s{seed}.csv")
        for model_name in tqdm(model_names):
            if model_name != "rapppid":
                model_df = df[df.model_name == model_name]
                label = [float(x) for x in model_df["label"].tolist()]
                yhat = [float(x) for x in model_df["y_hat"].tolist()]
                fpr, tpr, _ = m.roc_curve(label, yhat)
                if seed == 1 and model_names[0] == model_name:
                    kwarg = {"label": "Human Test Set"}
                else:
                    kwarg = {}
            else:
                paths = [
                    data_folder / "results/traditional/rapppid_C3_S3.json",
                    data_folder / "results/traditional/rapppid_C3_S5.json",
                    data_folder / "results/traditional/rapppid_C3_S8.json",
                ]

                model_df = parse_rapppid(paths[seed - 1])
                fpr, tpr, _ = m.roc_curve(model_df["label"], model_df["y_hat"])

            ax.plot(fpr, tpr, color=colours[4], **kwarg)

    summary_metrics_row = []

    for model_name in tqdm(model_names):
        for seed in [1, 2, 3]:
            df = pd.read_csv(
                output_folder / f"tables/sars_cov2_scores_{model_name}_{seed}.csv"
            )
            fpr, tpr, _ = m.roc_curve(df["pos"], df["yhat"])
            mcc = m.matthews_corrcoef(df["pos"], df["yhat"] > 0.5)
            bacc = m.balanced_accuracy_score(df["pos"], df["yhat"] > 0.5)
            summary_metrics_row.append(
                {
                    "model_name": model_name,
                    "mcc": mcc,
                    "bacc": bacc,
                }
            )

            if seed == 1 and model_names[0] == model_name:
                kwarg = {"label": "SARS-CoV-2 Test Set"}
            else:
                kwarg = {}

            ax.plot(fpr, tpr, color=colours[5], **kwarg)

    summary_metrics_df = pd.DataFrame(summary_metrics_row)
    summary_metrics_df.to_csv(output_folder / f"tables/sars_cov2_metrics.csv")

    ax.grid()
    ax.set_xlabel("False-Positive Rate")
    ax.set_ylabel("True-Positive Rate")
    plt.legend(edgecolor="k", fancybox=False)
    plt.savefig(output_folder / f"figures/sars_cov2_roc.svg")

    # HIGHLIGHT

    f, axs = plt.subplots(1, 2, figsize=(11, 5))

    # for seed in [1, 2, 3]:
    for seed in [1, 2, 3]:
        df = pd.read_csv(output_folder / f"tables/scores_s{seed}.csv")
        for model_name in tqdm(model_names):
            if model_name != "rapppid":
                model_df = df[df.model_name == model_name]
                label = [float(x) for x in model_df["label"].tolist()]
                yhat = [float(x) for x in model_df["y_hat"].tolist()]
                fpr, tpr, _ = m.roc_curve(label, yhat)
                if seed == 2 and model_names[0] == model_name:
                    kwarg = {"label": "Human Test Set"}
                else:
                    kwarg = {}
            else:
                paths = [
                    data_folder / "results/traditional/rapppid_C3_S3.json",
                    data_folder / "results/traditional/rapppid_C3_S5.json",
                    data_folder / "results/traditional/rapppid_C3_S8.json",
                ]

                model_df = parse_rapppid(paths[seed - 1])
                fpr, tpr, _ = m.roc_curve(model_df["label"], model_df["y_hat"])

            if model_name == "squeezeprot_sp_strict":
                highlight_color = colours[1]
                kwarg["zorder"] = 10
            elif model_name == "squeezeprot_sp_nonstrict":
                highlight_color = colours[0]
                kwarg["zorder"] = 10
            elif model_name == "squeezeprot_sp_nonstrict_unlearn":
                highlight_color = colours[3]
                kwarg["zorder"] = 10
            else:
                highlight_color = "#ccc"
                kwarg["zorder"] = 1

            ls = ":"
            axs[0].plot(fpr, tpr, color=colours[4], ls=ls, **kwarg)
            axs[1].plot(fpr, tpr, color=highlight_color, ls=ls, **kwarg)

    for model_name in tqdm(model_names):
        # for seed in [1, 2, 3]:
        for seed in [1, 2, 3]:
            df = pd.read_csv(
                output_folder / f"tables/sars_cov2_scores_{model_name}_{seed}.csv"
            )
            fpr, tpr, _ = m.roc_curve(df["pos"], df["yhat"])

            if seed == 2 and model_names[0] == model_name:
                kwarg = {"label": "SARS-CoV-2 Test Set"}
                highlight_kwarg = {"label": "SARS-CoV-2 Test Set"}
            else:
                kwarg = {}
                highlight_kwarg = {}

            if model_name == "squeezeprot_sp_strict":
                highlight_color = colours[1]

                if seed == 2:
                    highlight_kwarg = {"label": "SqueezeProt-SP (Strict)"}
                highlight_kwarg["zorder"] = 10

            elif model_name == "squeezeprot_sp_nonstrict":
                highlight_color = colours[0]

                if seed == 2:
                    highlight_kwarg = {"label": "SqueezeProt-SP (Non-Strict)"}
                highlight_kwarg["zorder"] = 10

            elif model_name == "squeezeprot_sp_nonstrict_unlearn":
                highlight_color = colours[0]

                if seed == 2:
                    highlight_kwarg = {"label": "SqueezeProt-SP (Non-Strict*)"}
                highlight_kwarg["zorder"] = 10

            else:
                highlight_color = "#ccc"
                highlight_kwarg["zorder"] = 1

            axs[0].plot(fpr, tpr, color=colours[5], **kwarg)
            axs[1].plot(fpr, tpr, color=highlight_color, **highlight_kwarg)

    for i in [0, 1]:
        axs[i].grid()
        axs[i].set_xlabel("False-Positive Rate")
        axs[i].set_ylabel("True-Positive Rate")
        axs[i].legend(edgecolor="k", fancybox=False)
    plt.savefig(output_folder / f"figures/sars_cov2_roc_highlight.svg")

    # BREAKOUT

    f, axs = plt.subplots(4, 4, figsize=(26, 24))

    plt.rcParams["font.size"] = 16

    letters = [
        ["A)", "B)", "C)", "D)"],
        ["E)", "F)", "G)", "H)"],
        ["I)", "J)", "K)", "L)"],
        ["M)", "N)", "O)", "P)"],
    ]

    for model_idx, model_name in tqdm(enumerate(model_names)):
        for seed in [1, 2, 3]:
            df = pd.read_csv(
                output_folder / f"tables/sars_cov2_scores_{model_name}_{seed}.csv"
            )
            fpr, tpr, _ = m.roc_curve(df["pos"], df["yhat"])

            for i in range(4):
                for j in range(4):
                    # if (i == 1 and j == 3) or (i == 3 and j == 3):
                    #    axs[i, j].set_axis_off()
                    #    continue

                    if (i == model_idx // 4) & (j == model_idx % 4):
                        color = colours[5]
                        z = 2
                        if seed == 1:
                            kwarg = {"label": fancy_model_names[model_name]}
                        else:
                            kwarg = {}
                    else:
                        color = "#ccc"
                        z = 1
                        kwarg = {}

                    axs[i, j].plot(fpr, tpr, color=color, zorder=z, **kwarg)
                    axs[i, j].set_xlabel("False-Positive Rate")
                    axs[i, j].set_ylabel("True-Positive Rate")
                    axs[i, j].grid(True)
                    axs[i, j].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
                    axs[i, j].set_title(
                        f"{letters[i][j]}", loc="left", fontweight="bold"
                    )

    for seed in [1, 2, 3]:
        df = pd.read_csv(output_folder / f"tables/scores_s{seed}.csv")
        for model_idx, model_name in tqdm(enumerate(model_names)):
            for i in range(4):
                for j in range(4):
                    # if (i == 1 and j == 3) or (i == 3 and j == 3):
                    #    axs[i, j].set_axis_off()
                    #    continue

                    if (i == (model_idx + len(model_names)) // 4) & (
                        j == (model_idx + len(model_names)) % 4
                    ):
                        color = colours[4]
                        z = 2
                        if seed == 1:
                            kwarg = {"label": fancy_model_names[model_name]}
                        else:
                            kwarg = {}
                    else:
                        color = "#ccc"
                        z = 1
                        kwarg = {}

                    if model_name != "rapppid":
                        model_df = df[df.model_name == model_name]
                        label = [float(x) for x in model_df["label"].tolist()]
                        yhat = [float(x) for x in model_df["y_hat"].tolist()]
                        fpr, tpr, _ = m.roc_curve(label, yhat)

                    else:
                        paths = [
                            data_folder / "results/traditional/rapppid_C3_S3.json",
                            data_folder / "results/traditional/rapppid_C3_S5.json",
                            data_folder / "results/traditional/rapppid_C3_S8.json",
                        ]

                        model_df = parse_rapppid(paths[seed - 1])
                        fpr, tpr, _ = m.roc_curve(model_df["label"], model_df["y_hat"])

                    axs[i, j].plot(fpr, tpr, color=color, zorder=z, **kwarg)

    for i in range(4):
        for j in range(4):
            # if (i == 1 and j == 3) or (i == 3 and j == 3):
            #    continue
            axs[i, j].legend(edgecolor="k", fancybox=False, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_folder / f"figures/sars_cov2_breakdown.svg")
