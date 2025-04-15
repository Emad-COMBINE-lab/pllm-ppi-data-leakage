# ppi_bench

!!! info

    Before running the ppi_bench experiment, one must either run the embeddings_dbs experiment on the PPI dataset you are training on or use the pre-computed vectors found in the data/embeddings folder (the default).

This folder holds the code required to train the pLLM-based PPI inference models used in the manuscript.

## train.py

To train the pLLM-based PPI inference models from the manuscript, a CLI tool (`train.py`) was created. Here's how to use it:

```
python train.py MAX_EPOCHS NUM_LAYERS DATASET_FILE DATABASE_PATH BATCH_SIZE INPUT_DIM C_LEVEL <flags>
```
### Arguments

#### Positional Arguments

|Argument  |Description                   |Manuscript Value |
|----------|------------------------------|-----------------|
|MAX_EPOCHS|How many epochs to train for. |100              |
|NUM_LAYERS|How many fully-connected layers in the network.|3|
|DATASET_FILE|Path to the PPI dataset to train/test on. Must be in the RAPPPID/INTREPPPID format.|`../../data/ppi/rapppid_[common_string_9606.protein.links.detailed.v12.0_upkb.csv]_Mz70T9t-4Y-i6jWD9sEtcjOr0X8=.h5`|
|DATABASE_PATH|Path to the LMDB database with corresponding protein embeddings.|Various, all from the [embeddings folder](/data#embeddings).|
|DATABASE_PATH|Path to the LMDB database with corresponding protein embeddings. This solely determines which pLLM the PPI model is based on.|Various, all from the [embeddings folder](/data#embeddings).|
|BATCH_SIZE|The number of pairs to train on at once. Increasing this value increases RAM/VRAM use.|128|
|INPUT_DIM|The number of elements in the embeddings inputted into the network. This is determined by the embeddings in DATABASE_PATH.|Corresponds to pLLM|
|C_LEVEL|Which type of dataset (C1, C2, or C3) to use from the RAPPPID PPI dataset. See [Park & Marcotte](https://doi.org/10.1038/nmeth.2259) for details.|3|

#### Flags

|Short Flag |Long Flag |Default |Description |
|-----------|----------|--------|------------|
|-w|--workers|16|Number of CPU threads to use. Set this to fewer threads than your CPU affords you.|
|-p|--pooling|None|What pooling function (if any) to apply to the embedding before inputting to the neural network. Valid values are 'None' (no pooling), 'average' ([AdaptiveAvgPool1d](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool1d.html)), or 'max' ([AdaptiveMaxPool1d](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveMaxPool1d.html)).|
|-s|--seed|8675309|The random seed to use.|

### Example

To train a PPI inference model based on <term:ProtT5> as we did in the manuscript, you could run the following [Bash](https://en.wikipedia.org/wiki/Bash_(Unix_shell)) script:

```bash
DATASET_FILE="../../data/ppi/rapppid_[common_string_9606.protein.links.detailed.v12.0_upkb.csv]_Mz70T9t-4Y-i6jWD9sEtcjOr0X8=.h5"
PROT_T5_DB="../../data/embeddings/prottrans_t5.lmdb"
PROT_T5_DIM=1024

python train.py 100 3 $DATASET_FILE $PROT_T5_DB 128 $PROT_T5_DIM 3 -s 1
```

### Output

When we train a model, it is assigned a model name based on the hash of its hyperparameters. Checkpoints are saved to the folder `../../data/chkpts/<MODEL_ID>/` in a file that ends in `.ckpt`. Also save to that folder are the hyperparameters for the model (end in `.json.gz`) and the inferred probabilities of the testing pairs (`.csv`).

## Requirements

One can install the requirements for this module using the `requirements.txt` file in the `experiments/ppi_bench` folder.
