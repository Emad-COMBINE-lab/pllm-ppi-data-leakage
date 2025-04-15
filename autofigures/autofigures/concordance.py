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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

from pathlib import Path
from typing import Optional, Union, Set

from autofigures.utils import (
    merge_scores,
    default_paths,
    traditional_scores,
    plot_style,
    colours,
)


def make_hit_miss(vec):
    return {
        "hits": set(np.where(vec == 1)[0].tolist()),
        "misses": set(np.where(vec == 0)[0].tolist()),
    }


def positive_concordance(x, y):
    x = make_hit_miss(x)
    y = make_hit_miss(y)

    if len(x["hits"]) > len(y["hits"]):
        best = x
        worst = y
    else:
        best = y
        worst = x

    common_hits = len(best["hits"].intersection(worst["hits"]))
    worst_hits = len(worst["hits"])

    if worst_hits == 0:
        return 0.0

    return common_hits / worst_hits


def negative_concordance(x, y):
    x = make_hit_miss(x)
    y = make_hit_miss(y)

    if len(x["hits"]) > len(y["hits"]):
        best = x
        worst = y
    else:
        best = y
        worst = x

    common_hits = len(best["misses"].intersection(worst["misses"]))
    worst_hits = len(best["misses"])

    if worst_hits == 0:
        return 0.0

    return common_hits / worst_hits


def get_concordance(x, y):
    x = make_hit_miss(x)
    y = make_hit_miss(y)

    if len(x["hits"]) > len(y["hits"]):
        best = x
        worst = y
    else:
        best = y
        worst = x

    best_size = len(best["hits"]) + len(best["misses"])
    worst_size = len(worst["hits"]) + len(worst["misses"])

    assert best_size == worst_size

    hb = best["hits"]
    mb = best["misses"]
    hw = worst["hits"]
    mw = worst["misses"]
    n = len(best["hits"]) + len(best["misses"])

    m = len(mb) + len(mw)
    h = len(hb) + len(hw)

    beta_h = (n - m) if m < n else 0
    beta_m = (n - h) if h < n else 0

    h_int = len(hb.intersection(hw))
    m_int = len(mb.intersection(mw))

    lambda_h = (h_int - beta_h) / (len(hw) - beta_h)
    lambda_m = (m_int - beta_m) / (len(mb) - beta_m)
    _lambda = (lambda_h + lambda_m) / 2

    print(
        "lambda_h",
        lambda_h,
        "lambda_m",
        lambda_m,
        "_lambda",
        _lambda,
        "h_int",
        h_int,
        "m_int",
        m_int,
        "beta_h",
        beta_h,
        "beta_m",
        beta_m,
        "len(hw)",
        len(hw),
        "len(mb)",
        len(mb),
    )

    return {
        "concordance": _lambda,
        "positive_concordance": lambda_h,
        "negative_concordance": lambda_m,
    }


def get_concordances(df, model_names1, model_names2=None):
    if model_names2 is None:
        model_names2 = model_names1.copy()
        diag = True
    else:
        diag = False

    df["correct"] = df.apply(
        lambda x: 1
        if ((x.y_hat >= 0.5) & (x.label == 1)) | ((x.y_hat < 0.5) & (x.label == 0))
        else 0,
        axis=1,
    )

    results = {
        "concordance": [],
        "positive_concordance": [],
        "negative_concordance": [],
        "labels": [],
    }

    for model_idx1, model_name1 in enumerate(model_names1):
        for model_idx2, model_name2 in enumerate(model_names2):
            print(f"{model_name1} vs. {model_name2}")

            if diag and (model_idx1 == model_idx2 or model_idx1 > model_idx2):
                continue

            result = get_concordance(
                df[df.model_name == model_name1].correct,
                df[df.model_name == model_name2].correct,
            )

            for t in ["concordance", "positive_concordance", "negative_concordance"]:
                results[t].append(result[t])

            results["labels"].append(f"{model_name1}|{model_name2}")

    return results


def jaccard_index(x: Set, y: Set):
    intersection = x.intersection(y)
    union = x.union(y)
    return len(intersection) / len(union)


def get_jaccard(x, y):
    x = make_hit_miss(x)
    y = make_hit_miss(y)

    ji_hits = jaccard_index(x["hits"], y["hits"])
    ji_misses = jaccard_index(x["misses"], y["misses"])

    return {"jaccard_index_hits": ji_hits, "jaccard_index_misses": ji_misses}


def get_jaccards(df, model_names1, model_names2=None):
    if model_names2 is None:
        model_names2 = model_names1.copy()
        diag = True
    else:
        diag = False

    df["correct"] = df.apply(
        lambda x: 1
        if ((x.y_hat >= 0.5) & (x.label == 1)) | ((x.y_hat < 0.5) & (x.label == 0))
        else 0,
        axis=1,
    )

    results = {"jaccard_index_hits": [], "jaccard_index_misses": [], "labels": []}

    for model_idx1, model_name1 in enumerate(model_names1):
        for model_idx2, model_name2 in enumerate(model_names2):
            if diag and (model_idx1 == model_idx2 or model_idx1 > model_idx2):
                continue

            result = get_jaccard(
                df[df.model_name == model_name1].correct,
                df[df.model_name == model_name2].correct,
            )

            for t in ["jaccard_index_hits", "jaccard_index_misses"]:
                results[t].append(result[t])

            results["labels"].append(f"{model_name1}|{model_name2}")

    return results


def get_kappas(df, model_names1, model_names2=None):
    if model_names2 is None:
        model_names2 = model_names1.copy()
        diag = True
    else:
        diag = False

    results = {"kappa": [], "labels": []}

    for model_idx1, model_name1 in enumerate(model_names1):
        for model_idx2, model_name2 in enumerate(model_names2):
            if diag and (model_idx1 == model_idx2 or model_idx1 > model_idx2):
                continue

            model1_yhat_bin = (
                df[(df.model_name == model_name1) & (df.seed == 1)].y_hat > 0.5
            ).astype(int)
            model2_yhat_bin = (
                df[(df.model_name == model_name2) & (df.seed == 1)].y_hat > 0.5
            ).astype(int)

            result = cohen_kappa_score(model1_yhat_bin, model2_yhat_bin)

            results["kappa"].append(result)
            results["labels"].append(f"{model_name1}|{model_name2}")

    return results


# https://www.flerlagetwins.com/2020/11/beeswarm.html
def simple_beeswarm(y, nbins=None):
    """
    Returns x coordinates for the points in ``y``, so that plotting ``x`` and
    ``y`` results in a bee swarm plot.
    """
    y = np.asarray(y)

    if nbins is None:
        nbins = len(y) // 6

    # Get upper bounds of bins
    x = np.zeros(len(y))
    ylo = np.min(y)
    yhi = np.max(y)
    dy = (yhi - ylo) / nbins
    ybins = np.linspace(ylo + dy, yhi - dy, nbins - 1)

    # Divide indices into bins
    i = np.arange(len(y))
    ibs = [0] * nbins
    ybs = [0] * nbins
    nmax = 0
    for j, ybin in enumerate(ybins):
        f = y <= ybin
        ibs[j], ybs[j] = i[f], y[f]
        nmax = max(nmax, len(ibs[j]))
        f = ~f
        i, y = i[f], y[f]
    ibs[-1], ybs[-1] = i, y
    nmax = max(nmax, len(ibs[-1]))

    # Assign x indices
    dx = 1 / (nmax // 2)
    for i, y in zip(ibs, ybs):
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(y)]
            a = i[j::2]
            b = i[j + 1 :: 2]
            x[a] = (0.7 + j / 3 + np.arange(len(b))) * dx
            x[b] = (0.7 + j / 3 + np.arange(len(b))) * -dx

    return x


def concordance(
    output_folder: Optional[Union[Path, str]] = None,
    data_folder: Optional[Union[Path, str]] = None,
    cohen_kappa: bool = False,
):
    jaccard = False

    plot_style()

    output_folder, data_folder = default_paths(output_folder, data_folder)

    df1 = merge_scores(output_folder, seeds=[1])
    df2 = traditional_scores(data_folder)

    df = pd.concat([df1, df2])

    if jaccard:
        fn = get_jaccards
    elif cohen_kappa:
        fn = get_kappas
    else:
        fn = get_concordances

    pllm_names = ["prottrans_t5", "esm", "prottrans_bert", "prose", "proteinbert"]
    trad_names = ["rapppid", "dscript", "pipr", "richoux", "sprint"]

    pllm_nonstrict_results = fn(df, pllm_names, ["squeezeprot_sp_nonstrict"])
    trad_nonstrict_results = fn(df, trad_names, ["squeezeprot_sp_nonstrict"])
    pllm_strict_results = fn(df, pllm_names, ["squeezeprot_sp_strict"])
    trad_strict_results = fn(df, trad_names, ["squeezeprot_sp_strict"])

    if jaccard:
        measures = ["jaccard_index_hits", "jaccard_index_misses"]
        ylabels = ["Jaccard Index\n(Hits)", "Jaccard Index\n(Misses)"]
        ncols = 2
        width = 15
        loc = "upper right"
        yticks_major = np.arange(0, 1.1, 0.1)
        yticks_minor = np.arange(0, 1.05, 0.05)
    elif cohen_kappa:
        measures = ["kappa"]
        ylabels = ["Cohen's κ", ""]
        ncols = 2
        width = 15
        loc = "lower left"
        yticks_major = np.arange(-1, 1.2, 0.2)
        yticks_minor = np.arange(-1, 1.1, 0.1)
    else:
        measures = ["concordance", "positive_concordance", "negative_concordance"]
        ylabels = [
            "Skill-Normalized Concordance\n(Hits & Misses)",
            "Skill-Normalized Concordance\n(Hits)",
            "Skill-Normalized Concordance\n(Misses)",
        ]
        ncols = 3
        width = 22
        loc = "lower left"
        yticks_major = np.arange(0, 1.1, 0.1)
        yticks_minor = np.arange(0, 1.05, 0.05)

    f, axs = plt.subplots(1, ncols, figsize=(width, 5))

    for idx in range(ncols):
        axs[idx].grid(axis="y")
        axs[idx].grid(which="minor", linestyle=":", axis="y")
        axs[idx].set_axisbelow(True)

    for measure_idx, measure in enumerate(measures):
        for result_idx, results in enumerate(
            [
                pllm_nonstrict_results,
                pllm_strict_results,
                trad_nonstrict_results,
                trad_strict_results,
            ]
        ):
            ylabel = ylabels[measure_idx]

            if result_idx == 0:
                marker = "^"
                label = "pLLM v. SqueezeProt-SP (non-strict)"
            elif result_idx == 1:
                marker = "v"
                label = "pLLM v. SqueezeProt-SP (strict)"
            elif result_idx == 2:
                marker = "s"
                label = "non-pLLM v. SqueezeProt-SP (non-strict)"
            elif result_idx == 3:
                marker = "D"
                label = "non-pLLM v. SqueezeProt-SP (strict)"

            if len(results[measure]) == 3:
                x = np.ones(len(results[measure])) * (result_idx * 2.5) - 0.25
                x = x + (np.arange(3) * 0.25)
            else:
                x = simple_beeswarm(results[measure], 1)
                x += result_idx * 3.5

            print(measure, label, results[measure])

            axs[measure_idx].scatter(
                x,
                results[measure],
                c=colours[result_idx % 2],
                ec="k",
                s=75,
                marker=marker,
                label=label,
            )
            # axs[measure_idx].boxplot(results[measure], positions=[result_idx * 2.5], widths=[2.5])
            axs[measure_idx].set_yticks(yticks_major, minor=False)
            axs[measure_idx].set_yticks(yticks_minor, minor=True)
            axs[measure_idx].set_xticks([])
            axs[measure_idx].set_ylabel(ylabel)

    axs[0].legend(loc=loc, edgecolor="k", fancybox=False)

    plt.savefig(output_folder / "figures/concordance.svg")

    # print(pllm_results)
    # print('---')
    # print(trad_results)
    # print('---')
    # print(pllm_trad_results)
