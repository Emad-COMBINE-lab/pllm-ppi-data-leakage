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

model = "esm2_t33_650M_UR50D"
batch_size = 5

model, alphabet = torch.hub.load("facebookresearch/esm:main", model)
batch_converter = alphabet.get_batch_converter()
model = model.eval()


def encode_batch(batch, device="cpu"):
    global model

    batch = [("x", seq) for seq in batch]
    batch_labels, batch_strs, batch_tokens = batch_converter(batch)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    if device == "cuda":
        model = model.cuda()
        batch_tokens = batch_tokens.cuda()
    else:
        model = model.cpu()
        batch_tokens = batch_tokens.cpu()

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)

    token_representations = results["representations"][33]

    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(
            token_representations[i, 1 : tokens_len - 1].mean(0).tolist()
        )

    return sequence_representations


def encode(sequence, device="cpu"):
    return encode_batch([sequence], device)[0]
