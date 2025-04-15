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

from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import (
    get_model_with_hidden_layers_as_outputs,
)
import numpy as np

pool = "avg"
seq_len = 1024

pretrained_model_generator, input_encoder = load_pretrained_model()
model = get_model_with_hidden_layers_as_outputs(
    pretrained_model_generator.create_model(seq_len)
)


def encode_batch(batch, pool="avg", device="cpu"):
    global model

    batch = [x[: seq_len - 3] for x in batch]

    encoded_x = input_encoder.encode_X(batch, seq_len)

    local_representations, global_representations = model.predict(
        encoded_x, batch_size=len(batch)
    )
    if pool == "avg":
        return np.mean(local_representations, axis=1).tolist()
    elif pool == "global":
        return global_representations
    else:
        raise ValueError("Unexpected value for pool")


def encode(sequence, pool="avg", device="cpu"):
    return encode_batch([sequence], pool="avg")[0]
