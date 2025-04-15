# Data

The data folder contains data files used in the manuscript. It is subdivided into the following folders:

| Section                  | Description                         |
|--------------------------|-------------------------------------|
| [chkpts](#chkpts)        | Where model checkpoints are stored. |
| [embeddings](#embeddings)| Where protein embedding databases are stored. |
| [kw](#kw)| Where protein embedding databases are stored. |
| [lengths](#lengths)| Where data for the length analysis. |
| [mutation](#mutation)| Where data for the mutation analysis. |
| [pllm](#pllm)              | Where PPI datasets are stored.     |
| [ppi](#ppi)              | Where PPI datasets are stored.     |
| [tokenizer](#tokenizer)  | Where tokenizers are stored.        |


## chkpts

This folder contains model checkpoints for the SqueezeProt variants,  <term:RAPPPID>, and <term:ProSE>.

| Folder | Description |
|--------|-------------|
|`squeezeprot-sp.non-strict/checkpoint-1390896`|Weights for <term:SqueezeProt-SP (Non-strict)>.|
|`squeezeprot-sp.strict/checkpoint-1383824`|Weights for <term:SqueezeProt-SP (Strict)>.|
|`squeezeprot-sp.u50/checkpoint-2477920`|Weights for <term:SqueezeProt-U50>.|
|`sars_cov2`|Various weights for pLLM-based PPI models used in the SARS-CoV-2 analysis.|
|`rapppid/1690837077.519848_red-dreamy.ckpt`|Weights for <term:RAPPPID>. These weights were previously published in [Szymborski _et al._](https://doi.org/10.1093/bioinformatics/btac429).|
|`prose/prose_dlm_3x1024.sav`|Weights for <term:ProSE>. These weights were previously published in [Bepler _et al._](https://doi.org/10.1016/j.cels.2021.05.017).|
|`ppi`|Various weights for pLLM-based PPI models.|
|`kw/logs/KeywordNet`|SqueezeProt-SP keyword annotation models.|


This folder also contains sub-folders `ppi`, `sars_cov2` where checkpoints of the pLLM-based PPI inference models are stored. Checkpoints are stored according to their model names, which are hashes of the model hyperparameters. You can see all the hyper-parameters in the `hparams.json` file within the model's folder.

??? info "Mappings from Model IDs to their Hyperparameters"

    | Model ID                     | pLLM     | Seed | Input Dimension|
    |------------------------------|----------|------|----------------|
    |`iRcKUvufAzeh9CuJAtFMgjRq8Yo=`|<term:ESM>|1     |6165            |
    |`aLK3NQAPP9uHfEDhrBCaFXTjE9I=`|<term:ESM>|2     |6165            |
    |`x5i3FQ9dA3iI6IoHUyCzgydVWgM=`|<term:ESM>|3     |6165            |
    |`DoetbgdKMxDuKdfRWOUHSGAXTf0=`|<term:ProtBERT>|1|1024            |
    |`jjvmU1efEjj1VWqJ8frthcAuiK4=`|<term:ProtBERT>|2|1024            |
    |`nHLZTP7wTAmtJSlpLtAYOH6gAj0=`|<term:ProtBERT>|3|1024            |
    |`Dvy3BYf00JQ18SHKOSdCu9zWojs=`|<term:ProtT5>|1  |1024            |
    |`XAnrmSjztabYNXRBbdrU2uV8XRM=`|<term:ProtT5>|2  |1024            |
    |`PUoe1MerYJArEl8h4d-P40V06zA=`|<term:ProtT5>|3  |1024            |
    |`xSIEaL28YQW14K0UhNoWmwX-_HU=`|<term:ProSE>|1   |6165            |
    |`7KwnY62-2a7UBaKz3jplmvRqSWk=`|<term:ProSE>|2   |6165            |
    |`3VHXtNHqEashvmxcWsXwCxgPqs0=`|<term:ProSE>|3   |6165            |
    |`048O3lE7pCo4Y_qpQAZLfxYrz6Q=`|<term:ProteinBERT>|1|1562         |
    |`Ir_BXqrPDOutyG20qLfc2Sn4qoE=`|<term:ProteinBERT>|2|1562         |
    |`7x1W0IhBtFMoEdgUYTJCY6yuhoc=`|<term:ProteinBERT>|3|1562         |
    |`c-HcTcy0NwO7JY8U3lu8Nva7d7o=`|<term:SqueezeProt-SP (Strict)>|1|768|
    |`hdhFlXCydBsLZrIicCM4LMrNWoE=`|<term:SqueezeProt-SP (Strict)>|2|768|
    |`r8TQa4FnoUdDvr2LPGZqZk4z6y4=`|<term:SqueezeProt-SP (Strict)>|3|768|
    |`vFjXUGbR0vMEu8Bu0j2C2445J2A=`|<term:SqueezeProt-SP (Strict)>|4|768|
    |`5yEzfPl2E2eT54OJXLF0K75z_3I=`|<term:SqueezeProt-SP (Strict)>|5|768|
    |`W9DvdTJ7bW1GaCV3TyJ7HZznwZw=`|<term:SqueezeProt-SP (Strict)>|6|768|
    |`ngRDPuiaLHeCXQJvigyMuE1tsmw=`|<term:SqueezeProt-SP (Strict)>|7|768|
    |`MFwIN5YJ_98AUYYGORMlCvgv8j4=`|<term:SqueezeProt-SP (Strict)>|8|768|
    |`yGeHPlv6AviEUwC4jssZy-FuTBM=`|<term:SqueezeProt-SP (Strict)>|9|768|
    |`s3SMyXYEAmNECJgqSfhKutisFLc=`|<term:SqueezeProt-SP (Strict)>|10|768|
    |`0KCyeB_K-ZCIlNwpl3rRopZyM0I=`|<term:SqueezeProt-SP (Non-strict)>|1|768|
    |`1c-2iUOfs3Ye_pV0WGtCApV1Rgg=`|<term:SqueezeProt-SP (Non-strict)>|2|768|
    |`8F6cxSJIb1syPfiZDK04DgCaP90=`|<term:SqueezeProt-SP (Non-strict)>|3|768|
    |`SsTLBBEPiGnpX91hunP9-E8w6jY=`|<term:SqueezeProt-SP (Non-strict)>|4|768|
    |`EdJtO7pKkYBjeeGajJ33-hySWfE=`|<term:SqueezeProt-SP (Non-strict)>|5|768|
    |`K51CpKE1he98de-DV7Pq67DE4ok=`|<term:SqueezeProt-SP (Non-strict)>|6|768|
    |`6U8njyd40iXoaBCx4WdL57FOX9o=`|<term:SqueezeProt-SP (Non-strict)>|7|768|
    |`7YxAquqSeLtB1I85ItoXQvtGsg8=`|<term:SqueezeProt-SP (Non-strict)>|8|768|
    |`qI4P2gmDPMvY6epwa6cIgG9PQlE=`|<term:SqueezeProt-SP (Non-strict)>|9|768|
    |`MmCcff1PHh0lOvl84LiytfrznMw=`|<term:SqueezeProt-SP (Non-strict)>|10|768|
    |`uARWusUXomAV5qa0X9e2V4C5wwI=`|<term:SqueezeProt-U50>|1|768|
    |`cpns6_wtKs93e6ekMXxkBilEGv4=`|<term:SqueezeProt-U50>|2|768|
    |`OcZKrg-rYN5RsfC6TH9btqfdVV8=`|<term:SqueezeProt-U50>|3|768|


## embeddings

This folder holds pre-computed UniProt protein embeddings for each model. Embeddings are stored in [LMDB databases](https://lmdb.readthedocs.io/en/release/).

| Database | pLLM |
|----------|------|
|`esm.lmdb`  |<term:ESM> |
|`prose.lmdb`|<term:ProSE>|
|`proteinbert.lmdb`|<term:ProteinBERT>|
|`prottrans_bert.lmdb`|<term:ProtBERT>|
|`prottrans_t5.lmdb`|<term:ProtT5>|
|`rapppid.lmdb`|<term:RAPPPID>|
|`squeezeprot-sp.nonstrict.lmdb`|<term:SqueezeProt-SP (Non-strict)>|
|`squeezeprot_sp_strict.lmdb`|<term:SqueezeProt-SP (Strict)>|
|`squeezeprot_u50.lmdb`|<term:SqueezeProt-U50>|

Sub-folders `mutation` and `sars-cov-2` include embeddings for mutated proteins from ELASPIC and SARS-CoV-2 proteins, respectively.

## kw
This folder contains data for the UniProt Keyword annotations.

| Folder | Description |
|--------|-------------|
|`hparams`|Log of hyperparameters for the keyword models trains.|
|`out`|Inferred probabilities on a testing dataset. Filename corresponds to model number.|
|`kw.json.gz`|List of UniProt Keywords.|
|`kw_ids.json`|Matching UniProt Keywords to their categories.|
|`kw_meta.json`|Number of instances for each UniProt Keyword.|
|`kw_train.csv`|Contains UniProt accession numbers and the associated keywords and sequences for training the annotation model.|
|`kw_test.csv`|Contains UniProt accession numbers and the associated keywords and sequences for testing the annotation model.|
|`kw_train_vecs.csv`|Contains UniProt accession numbers and the associated keywords, sequences, and embeddings for training the annotation model.|
|`kw_test_vecs.csv`|Contains UniProt accession numbers and the associated keywords, sequences, and embeddings for testing the annotation model.|

## lengths
This folder contains data for the protein length analyses.

| Folder | Description |
|--------|-------------|
|`length_histogram.csv.gz`|Data for the length histogram.|

## mutation
This folder contains data for the mutation analyses.

| Folder | Description |
|--------|-------------|
|`elaspic-trainin-set-interface-ids.csv`|Data from the [ELASPIC2 manuscript](https://doi.org/10.1016/j.jmb.2021.166810).|

## pllm
This folder contains data used to train the SqueezeProt-SP pLLMs.

| Folder | Description |
|--------|-------------|
|`uniprot_sprot.nonstrict.eval.fasta`| FASTA file containing sequences of proteins which are assigned to the evaluation split of the non-strict SqueezeProt-SP model.|
|`uniprot_sprot.nonstrict.eval.txt`| Text file containing sequences of proteins which are assigned to the evaluation split of the non-strict SqueezeProt-SP model.|
|`uniprot_sprot.nonstrict.train.fasta`| FASTA file containing sequences of proteins which are assigned to the training split of the non-strict SqueezeProt-SP model.|
|`uniprot_sprot.nonstrict.train.txt`| Text file containing sequences of proteins which are assigned to the training split of the non-strict SqueezeProt-SP model.|

## ppi

The PPI datasets used in the manuscript are present in this folder. They are:

| File     | Description |
|----------|-------------|
|`rapppid_[common_string_9606.protein.links.detailed.v12.0_upkb.csv]_Mz70T9t-4Y-i6jWD9sEtcjOr0X8=.h5`  | Human PPI dataset from [Szymborski _et al._](https://doi.org/10.1093/bioinformatics/btac429). |
|`sars-cov-2/baits.fasta`  | Sequences of bait proteins from [Gordon _et al._](https://doi.org/10.1038/s41586-020-2286-9). |
|`sars-cov-2/preys.fasta`  | Sequences of prey proteins from [Gordon _et al._](https://doi.org/10.1038/s41586-020-2286-9). |
|`sars-cov-2/covid_ppi.csv`  | Human/SARS-CoV-2 PPI pairs from [Gordon _et al._](https://doi.org/10.1038/s41586-020-2286-9). |


## tokenizer
This folder contains the tokenizer used for the SqueezeProt variants.

| Folder | Description |
|--------|-------------|
|`tokenizer/bert-based-cased`|A tokenizer for the SqueezeProt variants.|
|`tokenizer/rapppid`|A tokenizer for the RAPPPID model.|

