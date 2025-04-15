# speed_bench

This folder contains the scripts required to measure the speed of the various pLLMs.

## Measuring Speed

To run the speed benchmark you must use the `python bench.py run` command.

### Arguments

#### Positional Arguments

|Argument  |Description                                   |
|----------|----------------------------------------------|
|MODEL_NAME|Name of the LLM to benchmark.                 |
|FASTA_PATH|Path to the sequences which are used to test. |

#### Flags

|Long Flag   |Default |Description |
|------------|--------|------------|
|--batch_size|None    |The size of the batch to use in the benchmark. If left as None, the largest possible batch size that fits in memory will be used.|

## Requirements

This script only requires three external libraries:

- [more_itertools](https://pypi.org/project/more-itertools/)
- [tqdm](https://pypi.org/project/tqdm/)
- [fire](https://pypi.org/project/fire/)