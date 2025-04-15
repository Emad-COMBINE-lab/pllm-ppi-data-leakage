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
import os

import pandas as pd
import numpy as np
from sklearn import metrics as m
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from tqdm import tqdm
import pickle
from typing import Union, Optional
from pathlib import Path
from autofigures.utils import plot_style, default_paths, colours


def pre_k(ys, yhats, k):
    idx = np.argsort(yhats)
    top_yhat = np.round(np.take_along_axis(yhats, idx, -1)[:, -k:])
    top_y = np.take_along_axis(ys, idx, -1)[:, -k:]

    tp = np.sum(np.logical_and(top_yhat, top_y), axis=1).astype("float64")

    not_y = np.logical_not(top_y)
    fp = np.sum(np.logical_and(top_yhat, not_y), axis=1).astype("float64")

    denom = tp + fp

    pres = np.divide(tp, denom, out=np.zeros_like(tp), where=denom != 0)

    return np.mean(pres)


def f1_k(y_true, y_pred_proba, k):
    # Get top k predictions
    top_k_indices = np.argsort(-y_pred_proba, axis=1)[:, :k]
    y_pred_k = np.zeros_like(y_pred_proba)

    # Fill in 1s for top k predictions
    for i in range(len(y_pred_k)):
        y_pred_k[i, top_k_indices[i]] = 1

    return m.f1_score(y_true, y_pred_k, average="samples")


def acc_k(ys, yhats, k):
    accs = []

    for y, yhat in zip(ys, yhats):
        idx = np.argsort(yhat)[::-1]

        acc = np.sum(y[idx][:k] == np.round(yhat)[idx][:k]) / k
        accs.append(acc)

    return np.mean(accs)


def lrap_k(ys, yhats, k):
    idx = np.argsort(yhats)
    top_yhat = np.round(np.take_along_axis(yhats, idx, -1)[:, -k:])
    top_y = np.take_along_axis(ys, idx, -1)[:, -k:]

    return m.label_ranking_average_precision_score(top_y, top_yhat)


def kw(
    output_folder: Optional[Union[Path, str]] = None,
    data_folder: Optional[Union[Path, str]] = None,
):
    plot_style()

    output_folder, data_folder = default_paths(output_folder, data_folder)

    metric_type = "ndcg"
    score_fn = m.ndcg_score
    filename = "kw_k_ndcg"

    df = pd.read_csv(data_folder / "kw/results.csv")

    nonstrict_ids = df[
        (df["label_weighting"] == "log")
        & (df["loss_type"] == "asl")
        & (df["nl_type"] == "mish")
        & (df["strict"] == False)
    ].idx.tolist()

    strict_ids = df[
        (df["label_weighting"] == "log")
        & (df["loss_type"] == "asl")
        & (df["nl_type"] == "mish")
        & (df["strict"] == True)
    ].idx.tolist()

    ks = np.arange(1085) + 1

    scores = {
        model_type: {int(k): [] for k in ks}
        for model_type in ["strict", "nonstrict", "rand"]
    }

    k_scores_path = data_folder / f"kw/k_{metric_type}.pickle"

    if not os.path.isfile(k_scores_path):
        for strict, ids in enumerate([nonstrict_ids, strict_ids]):
            for idx in tqdm(ids):
                with open(data_folder / f"kw/out/{idx}.json") as f:
                    x = json.load(f)
                    ys = np.array(x["ys"])
                    yhats = np.array(x["yhats"])
                    yrand = np.random.rand(yhats.shape[0], yhats.shape[1])

                    for k in tqdm(ks, leave=False):
                        k = int(k)

                        score = score_fn(ys, yhats, k=k)

                        if strict == 0:
                            scores["nonstrict"][k].append(score)
                        elif strict == 1:
                            scores["strict"][k].append(score)
                            score_rand = score_fn(ys, yrand, k)
                            scores["rand"][k].append(score_rand)

        with open(k_scores_path, "wb") as f:
            pickle.dump(scores, f)
    else:
        with open(k_scores_path, "rb") as f:
            scores = pickle.load(f)

    mean_scores = {}
    std_scores = {}

    for score_type in ["strict", "nonstrict", "rand"]:
        mean_scores[score_type] = np.array(
            [np.mean(scores[score_type][i]) for i in np.arange(1085) + 1]
        )
        std_scores[score_type] = np.array(
            [np.std(scores[score_type][i]) for i in np.arange(1085) + 1]
        )

    for score_type in ["strict", "nonstrict", "rand"]:
        if score_type == "strict":
            ls = ":"
            lw = 3
            zorder = 10
            colour = colours[0]
        elif score_type == "nonstrict":
            ls = "solid"
            lw = 2
            zorder = 0
            colour = colours[1]
        elif score_type == "rand":
            ls = "--"
            lw = 2
            zorder = 0
            colour = colours[2]

        plt.plot(
            mean_scores[score_type],
            label=score_type,
            ls=ls,
            zorder=zorder,
            lw=lw,
            color=colour,
        )
        # plt.fill_between(np.arange(1085)+1, mean_scores[score_type] + std_scores[score_type], mean_scores[score_type] - std_scores[score_type])

    # plt.axvline(8.38, color='#ccc', ls=':')
    plt.legend()
    plt.xlim(8.38, 1086)

    plt.savefig(output_folder / f"figures/{filename}.svg")

    categories = [
        "Ligand.",
        "Technical term.",
        "Biological process.",
        "PTM.",
        "Molecular function.",
        "Disease.",
        "Coding sequence diversity.",
        "Cellular component.",
        "Domain.",
        "Misc.",
        "Developmental stage.",
    ]

    with open(data_folder / "kw/kw_ids.json") as f:
        category_ids = json.load(f)

    scores = {
        model_type: {category: [] for category in categories}
        for model_type in ["strict", "nonstrict", "rand"]
    }

    for strict, ids in enumerate([nonstrict_ids, strict_ids]):
        for idx in tqdm(ids):
            with open(data_folder / f"kw/out/{idx}.json") as f:
                x = json.load(f)
                ys = np.array(x["ys"])
                yhats = np.array(x["yhats"])
                yrand = np.random.rand(yhats.shape[0], yhats.shape[1])

            for category in tqdm(categories, leave=False):
                cat_ids = np.array(category_ids[category])[
                    ~np.isnan(category_ids[category])
                ].astype("int")
                model_type = "strict" if strict == 1 else "nonstrict"

                # print(f"{len(cat_ids)} cat_ids for {model_type} {category} {idx}")
                if len(cat_ids) == 0:
                    continue

                if metric_type == "ndcg":
                    score_fn = m.ndcg_score
                elif metric_type == "acc":
                    score_fn = lambda y, yhat: acc_k(y, yhat, 10)
                elif metric_type == "lrap":
                    score_fn = m.label_ranking_average_precision_score
                else:
                    return None

                score = score_fn(ys[:, cat_ids], yhats[:, cat_ids])

                scores[model_type][category].append(score)

                if strict == 1:
                    score = score_fn(ys[:, cat_ids], yrand[:, cat_ids])
                    scores["rand"][category].append(score)

    nonstrict_means = []
    nonstrict_stds = []

    strict_means = []
    strict_stds = []

    rand_means = []
    rand_stds = []

    for cat in scores["nonstrict"]:
        if cat == "Misc.":
            continue
        mu = np.mean(scores["nonstrict"][cat])
        sigma = np.std(scores["nonstrict"][cat])
        nonstrict_means.append(mu)
        nonstrict_stds.append(sigma)

        mu = np.mean(scores["strict"][cat])
        sigma = np.std(scores["strict"][cat])
        strict_means.append(mu)
        strict_stds.append(sigma)

        mu = np.mean(scores["rand"][cat])
        sigma = np.std(scores["rand"][cat])
        rand_means.append(mu)
        rand_stds.append(sigma)

    fig, ax = plt.subplots(layout="constrained", figsize=(8, 6))

    if metric_type == "ndcg":
        ax.set_ylabel("Normalized Discounted Cumulative Gain (NDCG)")
        order_idx = np.array([7, 0, 3, 4, 2, 8])
    elif metric_type == "acc":
        ax.set_ylabel("Top-10 Accuracy (Acc@10)")
        order_idx = np.array([2, 0, 4, 7, 8, 3])
    elif metric_type == "lrap":
        ax.set_ylabel("Label Ranking Average Precision (LRAP)")
        order_idx = np.array([3, 8, 0, 7, 2, 4])

    labels = list(categories)
    labels.remove("Misc.")

    sorted(labels)
    labels = [label[:-1] for label in labels]
    labels = np.array(labels)[order_idx]

    means = {
        "Strict": (strict_means, scores["strict"]),
        "Non-Strict": (nonstrict_means, scores["nonstrict"]),
        "Random": (rand_means, scores["rand"]),
    }

    xs = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    for c_idx, (attribute, (measurement, points)) in enumerate(means.items()):
        offset = width * multiplier

        if c_idx == 0:
            hatch = None
        elif c_idx == 1:
            hatch = "//"
        elif c_idx == 2:
            hatch = "."

        measurement = np.array(measurement)[order_idx]

        rects = ax.bar(
            xs + offset,
            measurement,
            width,
            label=attribute,
            color=colours[c_idx],
            hatch=hatch,
            edgecolor="k",
            lw=1,
        )

        for x in xs:
            category = str(labels[x]) + "."
            ys = points[category]

            ax.scatter(len(points[category]) * [x + offset], ys, ec="k", fc="none")

        # ax.errorbar(x + offset, measurement, yerr=yerr, color='black', capsize=7, fmt='none')

        ax.bar_label(
            rects,
            padding=18,
            fmt="{:.3}",
            rotation=90,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
        )
        multiplier += 1

    ax.set_xticks(xs + width, np.array(labels), rotation=-25, ha="left")
    ax.legend(loc="upper left", ncols=3, edgecolor="k", fancybox=False)
    ax.set_ylim(0, 1)

    ax.set_axisbelow(True)
    plt.grid()

    plt.savefig(output_folder / f"figures/kw_{metric_type}_markers.svg")
