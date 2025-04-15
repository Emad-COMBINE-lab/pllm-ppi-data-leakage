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
from data import KeywordDataset
from net import Protein2KeywordNet
from glob import glob
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import (
    ndcg_score,
    coverage_error,
    label_ranking_average_precision_score,
    label_ranking_loss,
)
from tqdm import tqdm
import gzip
import csv
import numpy as np


def precision_at_k(y_true, y_pred, k):
    """
    Calculate Precision@K for multi-label classification.

    Parameters:
    - y_true (ndarray): Binary ground truth labels, shape (num_samples, num_classes).
    - y_pred (ndarray): Predicted probabilities, shape (num_samples, num_classes).
    - k (int): Number of top predictions to consider.

    Returns:
    - float: Precision@K value.
    """
    num_samples = y_true.shape[0]
    precisions = []
    for i in range(num_samples):
        # Get top K predicted indices
        top_k_indices = np.argsort(-y_pred[i])[:k]
        # Count relevant labels in top K
        relevant = y_true[i][top_k_indices].sum()
        # Precision = relevant / K
        precisions.append(relevant / k)
    return np.mean(precisions)


def recall_at_k(y_true, y_pred, k):
    """
    Calculate Recall@K for multi-label classification.

    Parameters:
    - y_true (ndarray): Binary ground truth labels, shape (num_samples, num_classes).
    - y_pred (ndarray): Predicted probabilities, shape (num_samples, num_classes).
    - k (int): Number of top predictions to consider.

    Returns:
    - float: Recall@K value.
    """
    num_samples = y_true.shape[0]
    recalls = []
    for i in range(num_samples):
        # Get top K predicted indices
        top_k_indices = np.argsort(-y_pred[i])[:k]
        # Count relevant labels in top K
        relevant = y_true[i][top_k_indices].sum()
        # Recall = relevant / total positives
        # total_positives capped by k
        total_positives = min(y_true[i].sum(), k)
        if total_positives > 0:
            recalls.append(relevant / total_positives)
        else:
            recalls.append(0.0)  # Avoid division by zero
    return np.mean(recalls)


def accuracy_at_k(y_true, y_pred, k):
    """
    Calculate Accuracy@K for multi-label classification.

    Parameters:
    - y_true (ndarray): Binary ground truth labels, shape (num_samples, num_classes).
    - y_pred (ndarray): Predicted probabilities, shape (num_samples, num_classes).
    - k (int): Number of top predictions to consider.

    Returns:
    - float: Accuracy@K value.
    """
    num_samples = y_true.shape[0]
    accuracies = []
    for i in range(num_samples):
        # Get top K predicted indices
        top_k_indices = np.argsort(-y_pred[i])[:k]
        # Check if any of the top K predictions are in the true labels
        relevant = y_true[i][top_k_indices].sum()
        accuracies.append(1.0 if relevant > 0 else 0.0)
    return np.mean(accuracies)


# PATHS

test_path = "../../../data/kw/kw_test_vecs.csv.gz"
meta_path = "../../../data/kw/kw_meta.json"
batch_size = 100

# DATA
print("LOADING DATA")
num_test = 0

with gzip.open(test_path, "rt") as f:
    reader = csv.reader(f)
    for line in reader:
        num_test += 1

strict_dataset = KeywordDataset(test_path, True, offset=num_test // 2)
strict_loader = DataLoader(strict_dataset, batch_size=batch_size)

nonstrict_dataset = KeywordDataset(test_path, False, offset=num_test // 2)
nonstrict_loader = DataLoader(nonstrict_dataset, batch_size=batch_size)

# HPARAMS
rows = []
for path in glob("../../../data/kw/hparams/*.json"):
    with open(path) as f:
        rows.append(json.load(f))

df = pd.DataFrame(rows)

print(df)

# INFER
print("INFERRING")

rows = []

for idx in tqdm(range(0, 20), total=20):
    log_path_pattern = (
        Path("../../../data/chkpts/kw/logs/KeywordNet")
        / f"{idx + 1}"
        / Path("checkpoints/*.ckpt")
    )
    log_path = glob(str(log_path_pattern))[0]

    hparams = df.iloc[idx]

    weights = torch.load(log_path, weights_only=True, map_location="cuda")

    net = Protein2KeywordNet(
        meta_path,
        label_weighting=hparams.label_weighting
        if hparams.label_weighting != "None"
        else None,
        loss_type=hparams.loss_type,
        nl_type=hparams.nl_type,
    )

    net.load_state_dict(weights["state_dict"])
    net.eval()

    if hparams.strict == True:
        loader = strict_loader
    else:
        loader = nonstrict_loader

    yhats = None
    ys = None

    for batch in loader:
        y, x = batch

        yhat = net(x).detach()

        if yhats is None:
            yhats = yhat
            ys = y
        else:
            yhats = torch.cat((yhats, yhat), 0)
            ys = torch.cat((ys, y), 0)

    with open(f"../../../data/kw/out/{idx + 1}.json", "w") as f:
        json.dump({"yhats": yhats.numpy().tolist(), "ys": ys.numpy().tolist()}, f)

    null_classes = torch.sum(ys, axis=0) == 0

    ndcg = ndcg_score(ys[:, ~null_classes], yhats[:, ~null_classes])
    ndcg_3 = ndcg_score(ys[:, ~null_classes], yhats[:, ~null_classes], k=3)
    ndcg_5 = ndcg_score(ys[:, ~null_classes], yhats[:, ~null_classes], k=5)
    ndcg_10 = ndcg_score(ys[:, ~null_classes], yhats[:, ~null_classes], k=10)

    pre_3 = precision_at_k(ys[:, ~null_classes], yhats[:, ~null_classes], 3)
    pre_5 = precision_at_k(ys[:, ~null_classes], yhats[:, ~null_classes], 5)
    pre_10 = precision_at_k(ys[:, ~null_classes], yhats[:, ~null_classes], 10)

    rec_3 = recall_at_k(ys[:, ~null_classes], yhats[:, ~null_classes], 3)
    rec_5 = recall_at_k(ys[:, ~null_classes], yhats[:, ~null_classes], 5)
    rec_10 = recall_at_k(ys[:, ~null_classes], yhats[:, ~null_classes], 10)

    acc_3 = accuracy_at_k(ys[:, ~null_classes], yhats[:, ~null_classes], 3)
    acc_5 = accuracy_at_k(ys[:, ~null_classes], yhats[:, ~null_classes], 5)
    acc_10 = accuracy_at_k(ys[:, ~null_classes], yhats[:, ~null_classes], 10)

    cov_err = coverage_error(ys[:, ~null_classes], yhats[:, ~null_classes])
    lrap = label_ranking_average_precision_score(
        ys[:, ~null_classes], yhats[:, ~null_classes]
    )
    label_ranking = label_ranking_loss(ys, yhats)

    row = {
        "idx": idx,
        "seed": hparams.seed,
        "strict": hparams.strict,
        "label_weighting": hparams.label_weighting,
        "loss_type": hparams.loss_type,
        "nl_type": hparams.nl_type,
        "cov_err": cov_err,
        "lrap": lrap,
        "label_ranking": label_ranking,
        "ndcg": ndcg,
        "ndcg_3": ndcg_3,
        "ndcg_5": ndcg_5,
        "ndcg_10": ndcg_10,
        "acc_3": acc_3,
        "acc_5": acc_5,
        "acc_10": acc_10,
        "pre_3": pre_3,
        "pre_5": pre_5,
        "pre_10": pre_10,
        "rec_3": rec_3,
        "rec_5": rec_5,
        "rec_10": rec_10,
    }
    rows.append(row)

results_df = pd.DataFrame(rows)
results_df.to_csv("../../../data/kw/results.csv")
