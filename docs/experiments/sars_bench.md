# sars_bench

This folder contains the scripts required to infer the probabilities of interaction between SARS-CoV-2 proteins and _Homo sapiens_ proteins.

## Embedding SARS-CoV-2 Proteins

Provided in the data folder are pre-computed embeddings for SARS-CoV-2 proteins. If you wish to re-compute them, you'll need to use the `python bench.py embed` command.

```
python bench.py embed MODEL_NAME <flags>
```

### Arguments

#### Positional Arguments

|Argument  |Description                                   |
|----------|----------------------------------------------|
|MODEL_NAME|Name of the LLM to use to embed the proteins. |

#### Flags

|Long Flag   |Default |Description |
|------------|--------|------------|
|--batch_size|5       |The size of the batches used to embed the proteins.|
|--ppi_path|"../../data/ppi/sars-cov-2/covid_ppi.csv"|File that cotains the SARS-CoV-2 v. Human PPIs.|
|--baits_path|"../../data/ppi/sars-cov-2/baits.fasta"|The sequences of the SARS-CoV-2 proteins.|
|--preys_path|""../../data/ppi/sars-cov-2/preys.fasta"|The sequences of the Human proteins.|

## Infer SARS-CoV-2 v. Human PPIs

To infer SARS-CoV-2 interactions with human proteins, you'll need to use the `python bench.py infer` command.

```
python bench.py infer MODEL_NAME CHECKPOINT_PATH <flags>
```

#### Positional Arguments

|Argument  |Description                                   |
|----------|----------------------------------------------|
|MODEL_NAME|Name of the LLM to use to infer the PPIs.     |
|CHECKPOINT_PATH|Path to the pLLM-based PPI inference model checkpoint. |


|Short Flag|Long Flag   |Default |Description |
|---|------------|--------|------------|
|-o|--out_dir|"../../data/ppi/sars-cov-2"|The size of the batches used to embed the proteins.|
|-p|--ppi_path|"../../data/ppi/sars-cov-2/covid_ppi.csv"|File that cotains the SARS-CoV-2 v. Human PPIs.|
|-d|--db_path|`None`|Path to the database that contains the protein embeddings.|


## Requirements

One can install the requirements for this experiment using the `requirements.txt` file in the `experiments/sars_bench` folder.
