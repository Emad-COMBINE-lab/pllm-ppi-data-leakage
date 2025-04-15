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
from transformers import T5Tokenizer, T5EncoderModel
import re

model_name = "Rostlab/prot_t5_xl_half_uniref50-enc"
device = "cpu"

tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
model = T5EncoderModel.from_pretrained(model_name).to(device)


def encode_batch(batch, device="cpu"):
    global model

    if device == "cuda":
        model = model.cuda()
    else:
        model = model.cpu()
        model.to(torch.float32)

    model.eval()

    sequence_examples = [
        " ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in batch
    ]

    ids = tokenizer(sequence_examples, add_special_tokens=True, padding="longest")

    input_ids = torch.tensor(ids["input_ids"]).to(device)
    attention_mask = torch.tensor(ids["attention_mask"]).to(device)

    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

    emb_0 = embedding_repr.last_hidden_state

    emb_0_per_protein = emb_0.mean(dim=1)

    return emb_0_per_protein.tolist()


def encode(sequence, device="cpu"):
    return encode_batch([sequence], device=device)[0]
