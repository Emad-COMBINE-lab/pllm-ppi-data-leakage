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

import sentencepiece as sp
import sys

sys.path.insert(1, "/whale/projects/phd/rapppid/rapppid")
import infer

model = infer.load_chkpt("../../data/chkpts/rapppid/1690837077.519848_red-dreamy.ckpt")
spm_file = "../../data/tokenizer/rapppid/spm.model"
spp = sp.SentencePieceProcessor(model_file=spm_file)


def encode_batch(batch, device="cuda"):
    global model

    toks = infer.process_seqs(spp, batch, 1500)

    if device == "cuda":
        model = model.cuda()
        toks = toks.cuda()
    else:
        model = model.cpu()
        toks = toks.cpu()

    out = model(toks)
    return out.tolist()


def encode(sequence, device="cuda"):
    return encode_batch([sequence], device=device)[0]


def predict(embed_0, embed_1):
    return infer.predict(model, embed_0, embed_1).item()
