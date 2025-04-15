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
from transformers import SqueezeBertModel, PreTrainedTokenizerFast


max_length = 512


def encode(
    seq,
    device="cpu",
    weights_path="../../data/chkpts/squeezeprot-sp.strict/checkpoint-1383824",
    tokenizer_path="../../data/tokenizer/bert-base-cased/tokenizer.t0.s8675309",
):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        tokenizer_path,
        model_max_length=max_length,
        mask_token="<mask>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    model = SqueezeBertModel.from_pretrained(weights_path)

    if device == "cuda":
        model = model.cuda()
    else:
        model = model.cpu()
        model.to(torch.float32)

    model.eval()

    toks = tokenizer(seq, truncation=True, return_tensors="pt", padding="max_length")
    vec = (
        model(
            input_ids=toks.input_ids.cuda(), attention_mask=toks.attention_mask.cuda()
        )
        .pooler_output.detach()
        .squeeze()
        .cpu()
        .numpy()
        .tolist()
    )

    return vec


def encode_batch(
    seqs,
    device="cpu",
    weights_path="../../data/chkpts/squeezeprot-sp.strict/checkpoint-1383824",
    tokenizer_path="../../data/tokenizer/bert-base-cased/tokenizer.t0.s8675309",
):
    return encode(seqs, device, weights_path, tokenizer_path)
