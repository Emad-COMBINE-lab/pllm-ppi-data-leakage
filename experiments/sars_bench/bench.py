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
from sklearn import metrics
import torch
from scipy.stats import spearmanr
import sys

sys.path.insert(1, "../encode_llm")
sys.path.insert(1, "../ppi_bench")


class Sars_bench(object):
    def _get_sequences(self, path: Path):
        sequences = {}

        protein = None
        sequence = ""

        for line in open(path):
            if line.strip().startswith(">"):
                protein = line.strip()[1:]
                sequences[protein] = sequence
                sequence = ""
            else:
                sequence += line.strip()

        return sequences

    def _get_vec(self, env, bait_id: str, prey_id: str):
        with env.begin() as txn:
            try:
                bait_vec = torch.tensor(
                    json.loads(txn.get(bait_id.encode("utf8")).decode("utf8"))
                )
            except Exception as e:
                print(e)
                print(bait_id)
                bait_vec = None

            try:
                prey_vec = torch.tensor(
                    json.loads(txn.get(prey_id.encode("utf8")).decode("utf8"))
                )
            except Exception as e:
                print(e)
                print(prey_id)
                prey_vec = None

        return bait_vec, prey_vec

    def embed(
        self,
        model_name: str,
        batch_size: int = 5,
        ppi_path: Path = Path("../../data/ppi/sars-cov-2/covid_ppi.csv"),
        baits_path: Path = Path("../../data/ppi/sars-cov-2/baits.fasta"),
        preys_path: Path = Path("../../data/ppi/sars-cov-2/preys.fasta"),
    ):
        if model_name == "esm2":
            print("Loading ESM2-650M")
            import esmer

            encoder = esmer.encode_batch
        elif model_name == "prose":
            print("Loading ProSE")
            import proser

            encoder = proser.encode_batch
        elif model_name == "proteinbert":
            print("Loading ProteinBERT")
            import proteinberter

            encoder = proteinberter.encode_batch
        elif model_name == "prottrans_bert":
            print("Loading ProtBERT")
            import prottrans_bert

            encoder = prottrans_bert.encode_batch
        elif model_name == "prottrans_t5":
            print("Loading ProtT5")
            import prottrans_t5

            encoder = prottrans_t5.encode_batch
        elif model_name == "squeezeprot_u50":
            import squeezebert

            encoder = lambda x: squeezebert.encode_batch(
                x, weights_path="../../data/chkpts/squeezebert-u50/checkpoint-2477920"
            )
        elif model_name == "squeezeprot_sp_strict":
            print("Loading SqueezeBERT-SP (Strict)")
            import squeezebert

            encoder = lambda x: squeezebert.encode_batch(
                x,
                weights_path="../../data/chkpts/squeezeprot-sp.strict/checkpoint-1383824",
            )
        elif model_name == "squeezeprot_sp_nonstrict":
            print("Loading SqueezeBERT-SP (Non-Strict)")
            import squeezebert

            encoder = lambda x: squeezebert.encode_batch(
                x,
                weights_path="../../data/chkpts/squeezeprot-sp.nonstrict/checkpoint-1390896",
            )
        elif model_name == "rapppid":
            print("Loading RAPPPID")
            import rapppid

            encoder = rapppid.encode_batch
        else:
            print("Unexpected model.")

        print(f"Embedding with {model_name}")

        sequences = self._get_sequences(baits_path)
        sequences.update(self._get_sequences(preys_path))

        ppi_df = pd.read_csv(ppi_path).dropna()

        output_path = f"../../data/embeddings/sars-cov-2/{model_name}.lmdb"

        db_env = lmdb.open(output_path)
        db_env.set_mapsize(1024 * 1024 * 1024 * 1024)  # 1 TB

        bait_batch = []
        prey_batch = []
        bait_ids = []
        prey_ids = []

        with db_env.begin(write=True) as txn:
            for row_idx, row in tqdm(ppi_df.iterrows(), total=len(ppi_df)):
                bait_id, prey_id = row.Bait_RefSeq, row.Preys
                bait_seq = sequences[bait_id][:512]
                prey_seq = sequences[prey_id][:512]

                if len(bait_seq) == 0 or len(prey_seq) == 0:
                    print("Skipping pair with a zero length sequence")
                    continue

                bait_batch.append(bait_seq)
                prey_batch.append(prey_seq)
                bait_ids.append(bait_id)
                prey_ids.append(prey_id)

                if len(bait_batch) % batch_size == 0:
                    bait_vec_batch = encoder(bait_batch)
                    prey_vec_batch = encoder(prey_batch)

                    for bait_id, prey_id, bait_vec, prey_vec in zip(
                        bait_ids, prey_ids, bait_vec_batch, prey_vec_batch
                    ):
                        txn.put(
                            bait_id.encode("utf8"), json.dumps(bait_vec).encode("utf8")
                        )
                        txn.put(
                            prey_id.encode("utf8"), json.dumps(prey_vec).encode("utf8")
                        )

                    bait_batch = []
                    prey_batch = []
                    bait_ids = []
                    prey_ids = []

    def infer(
        self,
        model_name,
        checkpoint_path,
        out_dir=Path("../../data/ppi/sars-cov-2"),
        ppi_path="../../data/ppi/sars-cov-2/covid_ppi.csv",
        db_path=None,
    ):
        if model_name in [
            "squeezeprot-sp.strict",
            "squeezeprot-sp.nonstrict",
            "squeezebert-u50",
        ]:
            input_dim = 768
            pooling = None
        elif model_name == "prottrans_bert":
            input_dim = 1024
            pooling = "average"
        elif model_name == "proteinbert":
            input_dim = 1024
            pooling = "average"
        elif model_name == "prottrans_t5":
            input_dim = 1024
            pooling = "average"
        elif model_name == "esm2":
            input_dim = 6165
            pooling = "average"
        elif model_name == "rapppid":
            input_dim = 64
            pooling = None
        elif model_name == "prose":
            input_dim = 6165
            pooling = "average"
        else:
            input_dim = 768
            pooling = None

        print(f"Infer with {model_name}")

        if model_name == "rapppid":
            from rapppid import predict
        else:
            from fc_net import PPINet

            # Load model
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

            def predict(embedding_0, embedding_1):
                return (
                    torch.sigmoid(net(embedding_0, embedding_1))
                    .detach()
                    .cpu()
                    .numpy()[0]
                )

        ppi_df = pd.read_csv(ppi_path).dropna()

        if db_path is None:
            db_path = f"../../data/embeddings/sars-cov-2/{model_name}.lmdb"

        env = lmdb.open(db_path)

        rows = []

        for row_idx, row in tqdm(ppi_df.iterrows(), total=len(ppi_df)):
            bait_id, prey_id = row.Bait_RefSeq, row.Preys
            bait_vec, prey_vec = self._get_vec(env, bait_id, prey_id)

            if None in [bait_vec, prey_vec]:
                continue

            mist_score = row["MIST"]
            saint_bfdr_score = row["Saint_BFDR"]
            avg_spec_score = row["AvgSpec"]
            saint_score = row["SaintScore"]
            fold_change = row["FoldChange"]
            y = predict(bait_vec.unsqueeze(0), prey_vec.unsqueeze(0))

            rows.append(
                {
                    "id": row_idx,
                    "prey_id": prey_id,
                    "bait_id": bait_id,
                    "y": y,
                    "mist_score": mist_score,
                    "saint_bfdr_score": saint_bfdr_score,
                    "avg_spec_score": avg_spec_score,
                    "saint_score": saint_score,
                    "fold_change": fold_change,
                }
            )

        out_name = f"probs_{model_name}.csv"

        probs_df = pd.DataFrame(rows).set_index("id")
        probs_df.to_csv(out_dir / out_name)

        probs_df.fillna(0)
        probs_df["pos"] = (
            (probs_df["mist_score"] >= 0.7)
            & (probs_df["saint_bfdr_score"] <= 0.05)
            & (probs_df["avg_spec_score"] >= 2)
        )

        computed_metrics = {}

        computed_metrics["mcc"] = metrics.matthews_corrcoef(
            probs_df["pos"], probs_df["y"] > 0.50
        )
        computed_metrics["b_acc"] = metrics.balanced_accuracy_score(
            probs_df["pos"], probs_df["y"] > 0.50
        )
        computed_metrics["f1"] = metrics.f1_score(probs_df["pos"], probs_df["y"] > 0.50)
        computed_metrics["ap"] = metrics.average_precision_score(
            probs_df["pos"], probs_df["y"]
        )
        computed_metrics["precision"] = metrics.precision_score(
            probs_df["pos"], probs_df["y"] > 0.50
        )
        computed_metrics["recall"] = metrics.recall_score(
            probs_df["pos"], probs_df["y"] > 0.50
        )
        computed_metrics["spearman_corr"], computed_metrics["spearman_p"] = spearmanr(
            probs_df["pos"], probs_df["y"]
        )

        json_metrics = json.dumps(computed_metrics, indent=3)

        print(json_metrics)

        with open(out_dir / f"metrics_{out_name.split('.')[0]}.json", "w") as f:
            f.write(json_metrics)


if __name__ == "__main__":
    fire.Fire(Sars_bench(), name="SARS Bench")
