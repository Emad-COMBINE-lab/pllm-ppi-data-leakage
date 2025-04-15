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
import pickle
import pandas as pd
from pathlib import Path


def parse_dscript(file_path: Path):
    with open(file_path) as f:
        # ,y_hat,label,model_name
        rows = []

        for line in f:
            _, _, label, y_hat = line.split("\t")
            rows.append(
                {
                    "y_hat": float(y_hat),
                    "label": float(label),
                    "seed": 1,
                    "model_name": "dscript",
                }
            )

        return pd.DataFrame(rows)


def parse_rapppid(file_path: Path):
    with open(file_path) as f:
        x = json.load(f)

    y = x["y"]["test"]["9606"]
    yhat = x["yhat"]["test"]["9606"]

    rows = []

    for y_i, y_hat_i in zip(y, yhat):
        rows.append(
            {
                "y_hat": float(y_hat_i),
                "label": float(y_i),
                "seed": 1,
                "model_name": "rapppid",
            }
        )

    return pd.DataFrame(rows)


def parse_pipr(file_path: Path):
    with open(file_path) as f:
        x = json.load(f)

    rows = []

    for row in x:
        rows.append(
            {
                "y_hat": float(row["score_0"]),
                "label": float(row["label"]),
                "seed": 1,
                "model_name": "pipr",
            }
        )

    return pd.DataFrame(rows)


def parse_sprint(file_path: Path):
    with open(file_path) as f:
        # ,y_hat,label,model_name
        rows = []

        for line in f:
            y_hat, label = line.split(" ")
            rows.append(
                {
                    "y_hat": float(y_hat),
                    "label": float(label),
                    "seed": 1,
                    "model_name": "sprint",
                }
            )

        return pd.DataFrame(rows)


def parse_intrepppid(file_path: Path):
    with open(file_path) as f:
        x = json.load(f)

    y = x["test"]["labels"]
    yhat = x["test"]["scores"]

    rows = []

    for y_i, y_hat_i in zip(y, yhat):
        rows.append(
            {
                "y_hat": float(y_hat_i),
                "label": float(y_i),
                "seed": 1,
                "model_name": "intrepppid",
            }
        )

    return pd.DataFrame(rows)


def parse_richoux(pred_path: Path, label_path: Path):
    with open(pred_path, "rb") as f:
        yhat = pickle.load(f).flatten().tolist()

    with open(label_path, "rb") as f:
        y = pickle.load(f).flatten().tolist()

    rows = []

    for y_i, y_hat_i in zip(y, yhat):
        rows.append(
            {
                "y_hat": float(y_hat_i),
                "label": float(y_i),
                "seed": 1,
                "model_name": "richoux",
            }
        )

    return pd.DataFrame(rows)
