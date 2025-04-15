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
import numpy as np
import torch
import os.path
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Union
from autofigures.fc_net import PPINet
from autofigures.utils import default_paths, colours, plot_style, fancy_model_names


def hydrate_vec(vec):
    if vec:
        return torch.tensor(json.loads(vec.decode("utf8")))
    else:
        return None


def get_vec(row, env):
    with env.begin() as txn:
        wt_vec = hydrate_vec(txn.get(f"{row.id}_wt".encode("utf8")))
        mut_vec = hydrate_vec(txn.get(f"{row.id}_mt".encode("utf8")))
        ligand_vec = hydrate_vec(txn.get(f"{row.id}_ligand".encode("utf8")))

    return row, wt_vec, mut_vec, ligand_vec


def get_vec_triplets(data_folder: Path, model_name: str):
    elaspic_path = data_folder / "mutation/elaspic-trainin-set-interface-ids.csv"
    db_path = data_folder / f"embeddings/mutation/{model_name}.db"

    env = lmdb.open(str(db_path))

    muts_df = pd.read_csv(elaspic_path)

    for row_idx, row in tqdm(muts_df.iterrows(), total=len(muts_df)):
        yield get_vec(row, env)


def make_cmap(cmap_colours, name):
    c = [
        (idx / (len(cmap_colours) - 1), colour)
        for idx, colour in enumerate(cmap_colours)
    ]
    return LinearSegmentedColormap.from_list(name, c, N=256)


def scatter_hist(x, y, ax, ax_histx, ax_histy):
    cmap = "turbo"

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    print("calculating gaussian...")
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    print("sorting points by density")
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    print("scattering...")
    print(np.max(z), np.min(z), np.mean(z), np.std(z))
    scatter = ax.scatter(x, y, c=z, s=8, cmap=cmap)
    # plt.colorbar(scatter, ax=ax, cax=ax_cbar, location="bottom", label="Density")

    # now determine nice limits by hand:
    x_binwidth = 0.003  # 0.001
    x_xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    x_lim = (int(x_xymax / x_binwidth) + 1) * x_binwidth

    y_binwidth = 0.6  # 0.2
    y_xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    y_lim = (int(y_xymax / y_binwidth) + 1) * y_binwidth

    x_bins = np.arange(-x_lim, x_lim + x_binwidth, x_binwidth)
    y_bins = np.arange(-y_lim, y_lim + y_binwidth, y_binwidth)
    print("histing (x)...")
    ax_histx.hist(x, bins=x_bins, color="#1E88E5")
    ax_histx.spines[["right", "top", "left"]].set_visible(False)
    print("histing (y)...")
    ax_histy.hist(y, bins=y_bins, orientation="horizontal", color="#1E88E5")
    ax_histy.spines[["right", "top", "bottom"]].set_visible(False)


def mutation(
    output_folder: Optional[Union[Path, str]] = None,
    data_folder: Optional[Union[Path, str]] = None,
):
    output_folder, data_folder = default_paths(output_folder, data_folder)

    plot_style()

    model_names = [
        "esm",
        "squeezeprot_sp_nonstrict",
        "squeezeprot_sp_strict",
        "prottrans_t5",
        "prottrans_bert",
        "squeezeprot_u50",
        "prose",
        "proteinbert",
    ]

    def predict(embed_0, embed_1, net):
        return torch.sigmoid(net(embed_0, embed_1)).detach().cpu().numpy()[0]

    for seed in [1, 2, 3]:
        for model_name in model_names:
            df_path = output_folder / f"tables/mutations_{model_name}_{seed}.csv"
            if os.path.isfile(df_path):
                continue

            if model_name == "esm":
                chkpt_paths = [
                    data_folder
                    / "chkpts/ppi/iRcKUvufAzeh9CuJAtFMgjRq8Yo=/epoch=66-step=20569.ckpt",
                    data_folder
                    / "chkpts/ppi/aLK3NQAPP9uHfEDhrBCaFXTjE9I=/epoch=92-step=28551.ckpt",
                    data_folder
                    / "chkpts/ppi/x5i3FQ9dA3iI6IoHUyCzgydVWgM=/epoch=88-step=27323.ckpt",
                ]
                chkpt_path = chkpt_paths[seed - 1]

                model_dim = 6165
                num_layers = 3
                pooling = "average"
            elif model_name == "prottrans_bert":
                chkpt_paths = [
                    data_folder
                    / "chkpts/ppi/DoetbgdKMxDuKdfRWOUHSGAXTf0=/epoch=80-step=24867-v1.ckpt",
                    data_folder
                    / "chkpts/ppi/jjvmU1efEjj1VWqJ8frthcAuiK4=/epoch=90-step=27937-v1.ckpt",
                    data_folder
                    / "chkpts/ppi/nHLZTP7wTAmtJSlpLtAYOH6gAj0=/epoch=80-step=24867-v1.ckpt",
                ]
                chkpt_path = chkpt_paths[seed - 1]

                model_dim = 1024
                num_layers = 3
                pooling = "average"
            elif model_name == "prottrans_t5":
                chkpt_paths = [
                    data_folder
                    / "chkpts/ppi/Dvy3BYf00JQ18SHKOSdCu9zWojs=/epoch=97-step=30086.ckpt",
                    data_folder
                    / "chkpts/ppi/XAnrmSjztabYNXRBbdrU2uV8XRM=/epoch=86-step=26709.ckpt",
                    data_folder
                    / "chkpts/ppi/PUoe1MerYJArEl8h4d-P40V06zA=/epoch=89-step=27630.ckpt",
                ]
                chkpt_path = chkpt_paths[seed - 1]

                model_dim = 1024
                num_layers = 3
                pooling = "average"
            elif model_name == "prose":
                chkpt_paths = [
                    data_folder
                    / "chkpts/ppi/xSIEaL28YQW14K0UhNoWmwX-_HU=/epoch=89-step=27630.ckpt",
                    data_folder
                    / "chkpts/ppi/7KwnY62-2a7UBaKz3jplmvRqSWk=/epoch=99-step=30700.ckpt",
                    data_folder
                    / "chkpts/ppi/3VHXtNHqEashvmxcWsXwCxgPqs0=/epoch=97-step=30086.ckpt",
                ]
                chkpt_path = chkpt_paths[seed - 1]

                model_dim = 6165
                num_layers = 3
                pooling = "average"
            elif model_name == "proteinbert":
                chkpt_paths = [
                    data_folder
                    / "chkpts/ppi/048O3lE7pCo4Y_qpQAZLfxYrz6Q=/epoch=46-step=14429.ckpt",
                    data_folder
                    / "chkpts/ppi/Ir_BXqrPDOutyG20qLfc2Sn4qoE=/epoch=91-step=28244.ckpt",
                    data_folder
                    / "chkpts/ppi/7x1W0IhBtFMoEdgUYTJCY6yuhoc=/epoch=79-step=24560.ckpt",
                ]
                chkpt_path = chkpt_paths[seed - 1]

                model_dim = 1562
                num_layers = 3
                pooling = "average"
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
                chkpt_path = chkpt_paths[seed - 1]
                pooling = None
                num_layers = 3
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
                chkpt_path = chkpt_paths[seed - 1]
                pooling = None
                num_layers = 3
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
                chkpt_path = chkpt_paths[seed - 1]

                model_dim = 768
                num_layers = 3
                pooling = None
            elif model_name == "rapppid":
                chkpt_paths = [
                    data_folder
                    / "chkpts/ppi/mH2BqxhcWlsaAsswXzt94t6DKWQ=/epoch=25-step=7982.ckpt",
                    data_folder
                    / "chkpts/ppi/FANkW5KVGh84Y2pUX9mlT-3q9yc=/epoch=14-step=4605.ckpt",
                    data_folder
                    / "chkpts/ppi/5wkTQvBTAI-VUmJ5fZoMQ6dW5vk=/epoch=29-step=9210.ckpt",
                ]
                chkpt_path = chkpt_paths[seed - 1]

                model_dim = 64
                num_layers = 2
                pooling = None

            net = PPINet(0, 0, model_dim, pooling=pooling, num_layers=num_layers).cuda()
            weights = torch.load(chkpt_path, weights_only=True)
            net.load_state_dict(weights["state_dict"])
            net.eval()

            rows = []

            for row, wt_vec, mut_vec, ligand_vec in get_vec_triplets(
                data_folder, model_name
            ):
                if None in [wt_vec, mut_vec, ligand_vec]:
                    print("missing", model_name, row.id)
                    continue

                y_wt = predict(wt_vec.unsqueeze(0), ligand_vec.unsqueeze(0), net)
                y_mut = predict(mut_vec.unsqueeze(0), ligand_vec.unsqueeze(0), net)

                rows.append(
                    {
                        "id": row.id,
                        "effect": row.effect,
                        "effect_type": row.effect_type,
                        "wt": y_wt,
                        "mut": y_mut,
                        "diff": y_mut - y_wt,
                        "model_name": model_name,
                        "seed": seed,
                    }
                )

            df = pd.DataFrame(rows)
            df.to_csv(df_path)

    for model_name in model_names:
        print(model_name)
        df = pd.read_csv(output_folder / f"tables/mutations_{model_name}_1.csv")
        df = df[df.effect_type == "ΔΔG"]

        fig = plt.figure(figsize=(6, 6), dpi=300)

        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=(4, 0.5),
            height_ratios=(0.5, 4),
            left=0.1,
            right=0.9,
            bottom=0.1,
            top=0.9,
            wspace=0.05,
            hspace=0.05,
        )

        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

        diff = df["diff"].to_numpy()
        effect = df["effect"].to_numpy()
        scatter_hist(diff, effect, ax, ax_histx, ax_histy)

        ax_histx.set_yticks([])
        ax_histy.set_xticks([])

        ax.set_ylim(-5, 9)
        ax.set_xlim(-0.040, 0.040)
        ax.set_ylabel("Change in Binding Free Energy (ΔΔG)")
        ax.set_xlabel("Change in Predicted Interaction Probability (Δŷ)")

        res = spearmanr(diff, effect)
        print(model_name, "spearman", res.statistic, res.pvalue)
        plt.savefig(output_folder / f"figures/mutation_{model_name}.png")

    plt.figure()
    sns.set_palette(sns.color_palette(colours))

    df_summary = pd.DataFrame()

    for model_name in model_names:
        print(model_name)
        df = pd.read_csv(output_folder / f"tables/mutations_{model_name}_1.csv")
        df["model_name"] = fancy_model_names[model_name]
        df_summary = pd.concat([df, df_summary], axis=0, ignore_index=True)

    df_summary = df_summary[df_summary.effect_type == "ΔΔG"]
    sns.jointplot(data=df_summary, x="diff", y="effect", hue="model_name", kind="kde")
    plt.ylim(-15, 15)
    plt.xlim(-0.15, 0.15)
    plt.ylabel("Change in Binding Free Energy (ΔΔG)")
    plt.xlabel("Change in Predicted Interaction Probability (Δ$\hat{y}$)")
    plt.savefig(output_folder / f"figures/mutation.svg")
