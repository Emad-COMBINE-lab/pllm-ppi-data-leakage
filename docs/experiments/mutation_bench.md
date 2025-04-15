# mutation_bench

This folder contains the scripts required to infer the probabilities of interaction between mutatant proteins and their wild-types proteins.

## Embedding Mutated Proteins

Provided in the data folder are pre-computed embeddings for mutated proteins. If you wish to re-compute them, you'll need to use the `python bench.py embed` command.

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
|----------------|--------|------------|
|--batch_size|5       |The size of the batches used to embed the proteins.|
|--muts_path|"../../data/mutation/elaspic-trainin-set-interface-ids.csv"|Data on the binding affinity of mutated and wild-type protein pairs from [ELASPIC2](https://www.sciencedirect.com/science/article/pii/S0022283621000048) |
|--output_path|None|Where to output the embedding database.|

## Requirements

One can install the requirements for this module using the `requirements.txt` file in the `experiments/mutation_bench` folder.
