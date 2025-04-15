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

from prose.models.lstm import SkipLSTM
from embed_sequences import embed_sequence

MODEL_PATH = "../../data/chkpts/prose/prose_dlm_3x1024.sav"

model = SkipLSTM.load_pretrained(MODEL_PATH)
model = model.eval()


def encode(sequence, pool="avg", device="cuda"):
    global model

    if device == "cuda":
        model = model.cuda()
    else:
        model = model.cpu()

    return embed_sequence(
        model, sequence.encode("utf8"), pool="avg", use_cuda=True
    ).tolist()


def encode_batch(sequences, pool="avg", device="cpu"):
    return [encode(seq, device) for seq in sequences]
