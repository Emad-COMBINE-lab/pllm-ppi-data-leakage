# encode_llm

This module exposes a common API for inferring the embeddings used in this manuscript from amino acid sequences using the pLLMs, as well as <term:RAPPPID>, which is not a language model but from which embeddings can be had.

There are sub-modules for one of each of the pLLMs and <term:RAPPPID>, which are:

| Sub-module        | pLLM             |
|-------------------|------------------|
|`esmer.py`         |<term:ESM>        |
|`proser.py`        |<term:ProSE>      |
|`proteinberter.py` |<term:ProteinBERT>|
|`prottrans_bert.py`|<term:ProtBERT>   |
|`prottrans_t5.py`  |<term:ProtT5>     |
|`rapppid.py`       |<term:RAPPPID>    |
|`squeezebert.py`   |<term:SqueezeProt-SP (Strict)>,<br/> <term:SqueezeProt-SP (Non-Strict)>,<br/><term:SqueezeProt-U50>|

## Common API

All modules contain two functions, **encode** and **encode_batch** for calculating the embeddings from one single sequence or many, respectively.

### encode

```python
encode(sequence: str, device: str = "cpu") -> List[float]
```

|Argument  |Default |Type  |Description                  |
|----------|--------|------|-----------------------------|
|`sequence`| None   |String|Amino-acid sequence to embed.|
|`device`  | "cpu"  |String|Which device to run this on. Must be a valid [PyTorch device string](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device).|

### encode_batch

```python
encode_batch(batch: List[str], device: str = 'cpu') -> List[List[float]]
```

|Argument  |Default |Type     |Description                  |
|----------|--------|---------|-----------------------------|
|`batch`   | None   |List[str]|A list of amino-acid sequences to embed.|
|`device`  | "cpu"  |String   |Which device to run this on. Must be a valid [PyTorch device string](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device).|


## Selecting SqueezeProt Variants

There is only one module for all the SqueezeProt model, but there are three variants. You can use two additional arguments to specify the weights and tokenizer used to embed sequences. To embed with a specific variant, simply point those arguments to the paths of the corresponding weights for those variants.

### encode

```python
encode(seq: str, device: str = "cpu", weights_path: str = "../../data/chkpts/squeezeprot-sp.strict/checkpoint-1383824", tokenizer_path: str = "../../data/tokenizer/bert-base-cased/tokenizer.t0.s8675309")
```

|Argument  |Default |Type  |Description                  |
|----------|--------|------|-----------------------------|
|`sequence`| None   |String|Amino-acid sequence to embed.|
|`device`  | "cpu"  |String|Which device to run this on. Must be a valid [PyTorch device string](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device).|
|`weights_path`| `"../../data/chkpts/squeezeprot-sp.strict/checkpoint-1383824"` |String|Path to the SqueezeProt weights to load. These are specific to the SqueezeProt variant. See the [Data](/data/) section for more.|
|`tokenizer_path`| `"../../data/tokenizer/bert-base-cased/tokenizer.t0.s8675309"`   |String|Path to the tokenizer to load. This is the same value for all the variants. See the [Data](/data/) section for more.|

### encode_batch

```python
encode_batch(batch: List[str], device: str = 'cpu') -> List[List[float]]
```

|Argument  |Default |Type     |Description                  |
|----------|--------|---------|-----------------------------|
|`batch`   | None   |List[str]|A list of amino-acid sequences to embed.|
|`device`  | "cpu"  |String|Which device to run this on. Must be a valid [PyTorch device string](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device).|
|`weights_path`| `"../../data/chkpts/squeezeprot-sp.strict/checkpoint-1383824"` |String|Path to the SqueezeProt weights to load. These are specific to the SqueezeProt variant. See the [Data](/data/) section for more.|
|`tokenizer_path`| `"../../data/tokenizer/bert-base-cased/tokenizer.t0.s8675309"`   |String|Path to the tokenizer to load. This is the same value for all the variants. See the [Data](/data/) section for more.|

## Requirements

One can install the requirements for this module using the `requirements.txt` file in the `experiments/encode_llm` folder.
