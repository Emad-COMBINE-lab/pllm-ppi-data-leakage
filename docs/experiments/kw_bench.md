# kw_bench

This folder contains the scripts required to train the SqueezeProt UniProt keyword annotation models found in the manuscript.

## make_dataset.py

A pre-computed dataset is already present in the `data` folder, you can also re-compute it if you so desire by using the `make_dataset.py` CLI tool in the `scripts` folder.

!!! info

    Before re-computing the dataset for the kw_bench experiment, one must either run the embeddings_dbs experiment on _all_ SWISS-PROT proteins or use the pre-computed vectors found in the data/embeddings folder (the default).

```
python make_dataset.py <flags>
```

### Arguments

#### Flags

|Short Flag |Long Flag |Default |Description |
|-----------|----------|--------|------------|
||--kw_path|"../../../data/kw/kw.json.gz"|Path to the UniProt keyword annotation data.|
||--kw_train_path|"../../../data/kw/kw_train.csv.gz"|Where to save the generated training annotation dataset.|
||--kw_test_path|"../../../data/kw/kw_test.csv.gz"|Where to save the generated testing annotation dataset.|
||--kw_test_vecs_path|"../../../data/kw/kw_test_vecs.csv.gz"|Where to store the pre-computed vectors for the testing proteins.|
||--kw_train_vecs_path|"../../../data/kw/kw_train_vecs.csv.gz"|Where to store the pre-computed vectors for the testing proteins.|
||--kw_features_path|"../../../data/kw/subloc_kw_feature.csv.gz"|Path to a CSV file with annotation data. We have prpovided this file, which was derived from the UniProt source.|
|-e|--eval_seqs_path|"../../../data/sequences/uniprot_sprot.strict.eval.fasta.gz"|Path to protein sequences found in the evaluation set.|
|-s|--strict_database_path|"../../../data/embeddings/squeezeprot-sp.strict.lmdb"|Path to the LMDB database with SqueezeProt-SP (Strict) embeddings for all UniProt proteins.|
|-n|--nonstrict_database_path|"../../../data/embeddings/squeezeprot-sp.nonstrict.lmdb"|Path to the LMDB database with SqueezeProt-SP (Strict) embeddings for all UniProt proteins.|

## train.py

To train the UniProt keyword model from the manuscript, a CLI tool (`train.py`) was created. You can find it in the `scripts` folder. Here's how to use it:

```
python train.py <flags>
```

### Arguments

#### Flags

|Short Flag |Long Flag |Default |Description |
|-----------|----------|--------|------------|
|-n|--nl_type|"mish"|Activation function to use. Must be one of "[mish](https://pytorch.org/docs/stable/generated/torch.nn.Mish.html)", "[relu6](https://pytorch.org/docs/stable/generated/torch.nn.ReLU6.html)", or "[leaky_relu](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html)".|
||--loss_type|"asl"|What loss function to use. Must be one of "asl" for an [Assymetric Loss](https://arxiv.org/abs/2009.14119), or "bce" for [Binary Cross-Entropy](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) loss.|
||--label_weighting|"log"|What label weighting function to use. Must be one of "log" for log-scaled weighting, "linear" for linearly-scaled weighting, "root" for weighting scaled by its square root, or "None" for no label weighting.|
||--train_path|"../../../data/kw/kw_train_vecs.csv.gz"|Location of the precomputed train sequence vectors.|
||--test_path|"../../../data/kw/kw_test_vecs.csv.gz"|Location of the precomputed test sequence vectors.|
|-m|--meta_path|"../../../data/kw/kw_meta.json"|Protein keyword dataset.|
|-h|--hparams_dir|"../../../data/kw/hparams/"|Directory where hyper-parameters are stored.|
|-c|--chkpts_dir|"../../../chkpts/kw/"|Directory where checkpoints are stored.|
||--logs_dir|"../../../chkpts/kw/logs/"|Directory where logs are stored.|


## test.py

To infer on the testing set, you can run `test.py` like so:

```
python test.py
```

## Requirements

One can install the requirements for this experiment using the `requirements.txt` file in the `experiments/kw_bench` folder.
