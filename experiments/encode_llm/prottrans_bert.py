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
from transformers import AutoTokenizer, AutoModel, pipeline
import re
import numpy as np

model_name = "Rostlab/prot_bert"
device = "cpu"

tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = AutoModel.from_pretrained("Rostlab/prot_bert")


def encode_batch(batch, device="cpu"):
    global model

    if device == "cuda":
        model = model.cuda()
    else:
        model = model.cpu()
        model.to(torch.float32)

    model.eval()
    fe = pipeline("feature-extraction", model=model, tokenizer=tokenizer, device=0)

    sequence_examples = [
        " ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in batch
    ]

    output = fe(sequence_examples)

    avgs = []

    for o in output:
        mu = np.mean(o[0], axis=0).tolist()
        avgs.append(mu)

    return avgs


def encode(sequence, device="cpu"):
    return encode_batch([sequence], device=device)[0]
