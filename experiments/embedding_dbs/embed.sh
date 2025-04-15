#python embed_batch.py squeezeprot_sp_strict /whale/projects/phd/llm/subloc_kw/out/kw_train.csv /whale/projects/phd/llm/subloc_kw/out/squeezeprot_sp_strict.lmdb
#python embed_batch.py squeezeprot_sp_strict /whale/projects/phd/llm/subloc_kw/out/kw_test.csv /whale/projects/phd/llm/subloc_kw/out/squeezeprot_sp_strict.lmdb
#python embed_batch.py squeezeprot_sp_nonstrict /whale/projects/phd/llm/subloc_kw/out/kw_train.csv /whale/projects/phd/llm/subloc_kw/out/squeezeprot_sp_nonstrict.lmdb
#python embed_batch.py squeezeprot_sp_nonstrict /whale/projects/phd/llm/subloc_kw/out/kw_test.csv /whale/projects/phd/llm/subloc_kw/out/squeezeprot_sp_nonstrict.lmdb
DATASET_FILE="/whale/projects/phd/ppi_origami_data/processed/rapppid_900/seed_8675309/rapppid_[common_string_9606.protein.links.detailed.v12.0_upkb.csv]_Mz70T9t-4Y-i6jWD9sEtcjOr0X8=.h5"

python embed_batch.py squeezeprot_sp_nonstrict_unlearn $DATASET_FILE squeezeprot-sp.nonstrict.unlearn.lmdb
