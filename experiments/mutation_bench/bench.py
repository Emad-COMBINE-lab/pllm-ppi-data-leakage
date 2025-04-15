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

import lmdb
import json
import pandas as pd
import fire
from tqdm import tqdm
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import sys

sys.path.insert(1, "../encode_llm")
sys.path.insert(1, "../ppi_bench")

plt.rcParams["font.family"] = "Inter"


class Mutation_bench(object):
    def embed(
        self,
        model_name: str,
        batch_size: int = 5,
        muts_path: Path = Path(
            "../../data/mutation/elaspic-trainin-set-interface-ids.csv"
        ),
        output_path=None,
    ):
        if model_name == "esm":
            print("Loading ESM")
            import esmer

            encoder = esmer.encode_batch
        elif model_name == "prose":
            print("Loading PROSE")
            import proser

            encoder = proser.encode_batch
        elif model_name == "proteinbert":
            print("Loading ProteinBERT")
            import proteinberter

            encoder = proteinberter.encode_batch
        elif model_name == "prottrans_bert":
            print("Loading ProtTrans BERT")
            import prottrans_bert

            encoder = prottrans_bert.encode_batch
        elif model_name == "prottrans_t5":
            print("Loading ProtTrans T5")
            import prottrans_t5

            encoder = prottrans_t5.encode_batch
        elif model_name == "squeezeprot_sp_strict":
            print("Loading SqueezeBERT-SP (Strict)")
            import squeezebert

            encoder = lambda x: squeezebert.encode_batch(
                x,
                weights_path="../../data/chkpts/squeezeprot-sp.strict/checkpoint-1383824",
            )
        elif model_name == "squeezeprot_sp_nonstrict":
            print("Loading SqueezeBERT-SP (Non-strict)")
            import squeezebert

            encoder = lambda x: squeezebert.encode_batch(
                x,
                weights_path="../../data/chkpts/squeezeprot-sp.non-strict/checkpoint-1390896",
            )
        elif model_name == "squeezeprot_u50":
            print("Loading SqueezeBERT-U50")
            import squeezebert

            encoder = lambda x: squeezebert.encode_batch(
                x, weights_path="../../data/chkpts/squeezeprot-u50/checkpoint-2477920"
            )
        elif model_name == "rapppid":
            print("Loading RAPPPID")
            import rapppid

            encoder = rapppid.encode_batch
        else:
            print("Unexpected model.")

        print("Reading Mutations")
        muts_df = pd.read_csv(muts_path)

        if output_path is None:
            output_path = f"../../data/embeddings/mutation/{model_name}.db"

        env = lmdb.open(output_path)
        env.set_mapsize(1024 * 1024 * 1024 * 1024)

        with env.begin(write=True) as txn:
            for row_idx, row in tqdm(muts_df.iterrows(), total=len(muts_df)):
                wt_seq = row.protein_wt_sequence
                mt_seq = row.protein_mut_sequence
                ligand_seq = row.ligand_sequence

                if 0 in [len(wt_seq), len(mt_seq), len(ligand_seq)]:
                    print("At least one of the three sequences is zero.")
                    continue

                wt_vec, mt_vec, ligand_vec = encoder([wt_seq, mt_seq, ligand_seq])

                txn.put(
                    f"{row_idx}_wt".encode("utf8"), json.dumps(wt_vec).encode("utf8")
                )
                txn.put(
                    f"{row_idx}_mt".encode("utf8"), json.dumps(mt_vec).encode("utf8")
                )
                txn.put(
                    f"{row_idx}_ligand".encode("utf8"),
                    json.dumps(ligand_vec).encode("utf8"),
                )

    def infer(
        self,
        model_name: str,
        checkpoint_path: Path,
        db_path: Path,
        muts_path: Path = Path(
            "../../data/mutation/elaspic-trainin-set-interface-ids.csv"
        ),
    ):
        if model_name == "esm":
            input_dim = 6165
            pooling = "average"
        elif model_name == "prose":
            input_dim = 6165
            pooling = "average"
        elif model_name == "proteinbert":
            input_dim = 1562
            pooling = "average"
        elif model_name == "prottrans_bert":
            input_dim = 1024
            pooling = "average"
        elif model_name == "prottrans_t5":
            input_dim = 1024
            pooling = "average"
        elif model_name in [
            "squeezeprot_sp_strict",
            "squeezeprot_sp_nonstrict",
            "squeezeprot_u50",
        ]:
            input_dim = 768
            pooling = None
        elif model_name == "rapppid":
            input_dim = 64
            pooling = None
        else:
            print("Unexpected model.")

        print("Input Dim:", input_dim)
        print("Pooling:", pooling)

        env = lmdb.open(db_path)

        with env.begin() as txn:
            myList = [key for key, _ in txn.cursor()]
            print(myList[:10])

        def _get_vec(row_id: int):
            with env.begin() as txn:
                wt_vec = torch.tensor(
                    json.loads(txn.get(f"{row.id}_wt".encode("utf8")).decode("utf8"))
                )
                mut_vec = torch.tensor(
                    json.loads(txn.get(f"{row.id}_mt".encode("utf8")).decode("utf8"))
                )
                ligand_vec = torch.tensor(
                    json.loads(txn.get(f"{row.id}_mt".encode("utf8")).decode("utf8"))
                )

            return wt_vec, mut_vec, ligand_vec

        muts_df = pd.read_csv(muts_path)

        if model_name == "rapppid":
            from rapppid import predict
        else:
            from fc_net import PPINet

            state_dict = torch.load(checkpoint_path)["state_dict"]

            net = PPINet(
                steps_per_epoch=0,
                num_epochs=0,
                input_dim=input_dim,
                pooling=pooling,
                num_layers=3,
            ).cuda()
            net.load_state_dict(state_dict)
            net.eval()

            def predict(embed_0, embed_1):
                return torch.sigmoid(net(embed_0, embed_1)).detach().cpu().numpy()[0]

        rows = []

        for row_idx, row in tqdm(muts_df.iterrows(), total=len(muts_df)):
            wt_vec, mut_vec, ligand_vec = _get_vec(row.id)
            y_wt = predict(wt_vec.unsqueeze(0), ligand_vec.unsqueeze(0))
            y_mut = predict(mut_vec.unsqueeze(0), ligand_vec.unsqueeze(0))

            rows.append(
                {
                    "id": row.id,
                    f"{model_name}_wt": y_wt,
                    f"{model_name}_mut": y_mut,
                    f"{model_name}_diff": y_mut - y_wt,
                }
            )

        ys_df = pd.DataFrame(rows).set_index("id")

        df_joined = muts_df.join(ys_df)

        df_joined.to_csv(f"../../data/mutation/{model_name}_results.csv")


if __name__ == "__main__":
    fire.Fire(Mutation_bench(), name="Mutation Bench")
