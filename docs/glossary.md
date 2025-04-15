# Definitions & Glossary

## Definitions

### pLLMs

term:ESM
:   Evolutionary Scale Modeling (ESM) is a protein large language model
    released by Meta and described in [Verkuil _et al._](https://www.biorxiv.org/content/10.1101/2022.12.21.521521v1). We use the V2 650M weights released by Meta and retrieved by the PyTorch Hub. It is referred to as `esm` by the code for this manuscript.

term:ProtBERT
:   ProtBERT is one of the many protein large language models described
    by [Elnaggar _et al._](https://doi.org/10.1109/tpami.2021.3095381), and is based on BERT. Weights are retrieved from the HuggingFace repository [Rostlab/prot_bert](https://huggingface.co/Rostlab/prot_bert). It is referred to as `prottrans_bert` by the code for the manuscript.

term:ProtT5
:   ProtBERT is one of the many protein large language models described
    by [Elnaggar _et al._](https://doi.org/10.1109/tpami.2021.3095381), and is based on BERT. Weights are retrieved from the HuggingFace repository [Rostlab/prot_t5_xl_half_uniref50-enc](https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc). It is referred to as `prottrans_t5` by the code for the manuscript.

term:ProSE
:   ProSE is a protein large language models described
    by [Bepler _et al._](https://doi.org/10.1016/j.cels.2021.05.017), and is based on an recurrent neural network architecture. Weights are retrieved from the [GitHub repository](https://github.com/tbepler/prose). It is referred to as `prose` by the code for the manuscript.


term:ProteinBERT
:   ProteinBERT is a protein large language models described
    by [Brandes _et al._](https://doi.org/10.1093/bioinformatics/btac020), and is based on the BERT architecture. It uses an additional Gene Ontology (GO) term annotation prediction task. Weights and code were retrieved from the [GitHub repository](https://github.com/nadavbra/protein_bert). It is referred to as `proteinbert` by the code for the manuscript.

term:SqueezeProt-SP (Strict)
:   SqueezeProt-SP (Strict) is a novel protein large language model introduced in this manuscript. It is trained on a strict SWISS-PROT dataset which excludes proteins from the downstream PPI testing dataset. See the manuscript for more details. It is referred to as `squeezeprot_sp_strict` by the code for the manuscript.

term:SqueezeProt-SP (Non-strict)
:   SqueezeProt-SP (Non-strict) is a novel protein large language model introduced in this manuscript. It is trained on a non-strict SWISS-PROT dataset which includes proteins from the downstream PPI testing dataset. See the manuscript for more details. It is referred to as `squeezeprot_sp_nonstrict` by the code for the manuscript.

term:SqueezeProt-U50
:   SqueezeProt-U50 is a novel protein large language model introduced in this manuscript. It is trained on a non-strict UniRef50 dataset which includes proteins from the downstream PPI testing dataset. See the manuscript for more details. It is referred to as `squeezeprot_u50` by the code for the manuscript.

### Non-pLLM-based PPI Inference Methods

term:RAPPPID
:   RAPPPID is a non-pLLM-based PPI inference method described in [Szymborski _et al._](https://doi.org/10.1093/bioinformatics/btac429) and is based on an AWD-LSTM regularized neural network.

## Glossary

<glossary::term>
