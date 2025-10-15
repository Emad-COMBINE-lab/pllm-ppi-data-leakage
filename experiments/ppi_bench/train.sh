#!/bin/bash

DATASET_FILE="/media/jszym/fast/rapppid_[common_string_9606.protein.links.detailed.v12.0_upkb.csv]_Mz70T9t-4Y-i6jWD9sEtcjOr0X8=.h5"

SQUEEZEBER_U50_RANDOM_DB="/whale/projects/phd/llm/public/data/embeddings/squeezebert-u50.random.lmdb"
SQUEEZEBER_SP_NS_RANDOM_DB="/whale/projects/phd/llm/public/data/embeddings/squeezeprot-sp.nonstrict.random.lmdb"
SQUEEZEBER_SP_S_RANDOM_DB="/whale/projects/phd/llm/public/data/embeddings/squeezeprot-sp.strict.random.lmdb"
PROTBERT_RANDOM_DB="/whale/projects/phd/llm/public/data/embeddings/prottrans_bert.random.lmdb"
PROTT5_RANDOM_DB="/whale/projects/phd/llm/public/data/embeddings/prottrans_t5.random.lmdb"
PROTEINBERT_RANDOM_DB="/whale/projects/phd/llm/public/data/embeddings/proteinbert.random.lmdb"

SQUEEZEBERT_DIM=768
PROTBERT_DIM=1024
PROTT5_DIM=1024
PROTEINBERT_DIM=1022

python train.py 100 3 $DATASET_FILE $PROTEINBERT_RANDOM_DB 128 $PROTEINBERT_DIM 3 -s 1 -p "average"
python train.py 100 3 $DATASET_FILE $PROTT5_RANDOM_DB 128 $PROTT5_DIM 3 -s 1 -p "average"
python train.py 100 3 $DATASET_FILE $SQUEEZEBER_SP_NS_RANDOM_DB 128 $SQUEEZEBERT_DIM 3 -s 1 -p "average"
python train.py 100 3 $DATASET_FILE $SQUEEZEBER_SP_S_RANDOM_DB 128 $SQUEEZEBERT_DIM 3 -s 1 -p "average"
python train.py 100 3 $DATASET_FILE $SQUEEZEBER_U50_RANDOM_DB 128 $SQUEEZEBERT_DIM 3 -s 1 -p "average"
python train.py 100 3 $DATASET_FILE $PROTBERT_RANDOM_DB 128 $PROTBERT_DIM 3 -s 1 -p "average"


python train.py 100 3 $DATASET_FILE $PROTT5_RANDOM_DB 128 $PROTT5_DIM 3 -s 2 -p "average"
python train.py 100 3 $DATASET_FILE $SQUEEZEBER_SP_NS_RANDOM_DB 128 $SQUEEZEBERT_DIM 3 -s 2 -p "average"
python train.py 100 3 $DATASET_FILE $SQUEEZEBER_SP_S_RANDOM_DB 128 $SQUEEZEBERT_DIM 3 -s 2 -p "average"
python train.py 100 3 $DATASET_FILE $PROTBERT_RANDOM_DB 128 $PROTBERT_DIM 3 -s 2 -p "average"
python train.py 100 3 $DATASET_FILE $SQUEEZEBER_U50_RANDOM_DB 128 $SQUEEZEBERT_DIM 3 -s 2 -p "average"
python train.py 100 3 $DATASET_FILE $PROTEINBERT_RANDOM_DB 128 $PROTEINBERT_DIM 3 -s 2 -p "average"


python train.py 100 3 $DATASET_FILE $PROTT5_RANDOM_DB 128 $PROTT5_DIM 3 -s 3 -p "average"
python train.py 100 3 $DATASET_FILE $SQUEEZEBER_SP_NS_RANDOM_DB 128 $SQUEEZEBERT_DIM 3 -s 3 -p "average"
python train.py 100 3 $DATASET_FILE $SQUEEZEBER_SP_S_RANDOM_DB 128 $SQUEEZEBERT_DIM 3 -s 3 -p "average"
python train.py 100 3 $DATASET_FILE $PROTBERT_RANDOM_DB 128 $PROTBERT_DIM 3 -s 3 -p "average"
python train.py 100 3 $DATASET_FILE $SQUEEZEBER_U50_RANDOM_DB 128 $SQUEEZEBERT_DIM 3 -s 3 -p "average"
python train.py 100 3 $DATASET_FILE $PROTEINBERT_RANDOM_DB 128 $PROTEINBERT_DIM 3 -s 3 -p "average"
