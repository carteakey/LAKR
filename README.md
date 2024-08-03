## Tasks

## Getting Started

### Prerequisites

- Python 3 (>=3.6)
- [DuckDB](https://duckdb.org/)
- [Neo4j](https://neo4j.com/)
- [PyTorch](https://pytorch.org/)
- [Pandas](https://pandas.pydata.org/)
- [Numpy](https://numpy.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Scipy](https://www.scipy.org/)
- [PyTorch](https://pytorch.org/)


### Installation

```bash
pip install -r requirements.txt
```
If using CUDA, install the following packages:
```bash
CUDACXX=/usr/local/cuda-12/bin/nvcc CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=native" FORCE_CMAKE=1 pip install pandas numpy scikit-learn scipy implicit
```

## Data Preprocessing

### Raw Data -> rating_only (K-core Filtering)

`rating_only` contains review records, containing user, item, rating, timestamp in each line, but without text and other attributes.

```bash
python -m src/preprocessing/rec/k_core_filtering -k 10 --input_path ~/llm-enhanced-kg-rec/data/raw --output_path  ~/llm-enhanced-kg-rec/data
```

### Last out split

For each user, the latest review will be used for testing, the second latest review will be used for validation, and all the remaining reviews are used for training.

```bash
python -m src/preprocessing/rec/last_out_split  --input_path ~/llm-enhanced-kg-rec/data/15core/rating_only --output_path ~/llm-enhanced-kg-rec/data/last_out_split --seq_path ~/llm-enhanced-kg-rec/data/last_out_split
```

See the [benchmark_scripts](https://github.com/hyp1231/AmazonReviews2023/blob/main/benchmark_scripts/README.md) from [amazon-reviews-2023](https://github.com/hyp1231/AmazonReviews2023) for more details.

## Data Loading

### Load Data to a DuckDB instance

#### Load the K-core ratings

```bash
python -m src/preprocess/kg/01_load_kcore_ratings_duckdb.py
```

#### Load the metadata and reviews

```bash
python -m src/preprocess/kg/02_load_metadata_reviews_duckdb.py
```

## Knowledge Graph Construction

### Load metadata as a baseline KG

```bash
python -m src/preprocess/kg/03_load_metadata_neo4j.py
```

## Reset the database

### Reset the duckdb database

```bash
rm -rf db/duckdb/amazon-reviews.db
```

### Reset the neo4j database

```bash
cd db/neo4j
sudo reset.sh 
```

# Training

## BPRMF

```bash
python -m main_bprmf --data_name baseline-kg --data_dir ~/llm-enhanced-kg-rec/data/kg --n_epoch 100 --test_batch_size=1000 --use_pretrain 0 --train_batch_size=20000
```
To resume training from a pre-trained model, use the `pretrain_model_path` argument.
```bash
python -m main_bprmf --data_name baseline-kg --data_dir ~/llm-enhanced-kg-rec/data/kg --n_epoch 100 --test_batch_size=1000 --use_pretrain 2 --train_batch_size=100000 --pretrain_model_path ~/llm-enhanced-kg-rec/src/trained_model/BPRMF/baseline-kg/embed-dim64_lr0.0001_pretrain2/model_epoch30.pth  --pretrain_embedding_dir ~/llm-enhanced-kg-rec/data/kg/baseline-kg/pretrain
```

## KGAT

```bash
python -m main_kgat --data_name baseline-kg --data_dir ~/llm-enhanced-kg-rec/data/kg --n_epoch 100 --test_batch_size=1000 --use_pretrain 0 --cf_batch_size=10000 --kg_batch_size 10000 --pretrain_embedding_dir ~/llm-enhanced-kg-rec/data/kg/baseline-kg/pretrain
```








```
CUDACXX=/usr/local/cuda-12/bin/nvcc CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=native" FORCE_CMAKE=1 pip install pandas numpy scikit-learn scipy implicit
```

python -m main_kgat --data_name amazon-book --data_dir ~/repos/mds-tmu-mrp/datasets --use_pretrain 0  --test_batch_size=500   --n_epoch 1 --cf_batch_size 8192 --kg_batch_size 8192 

python -m main_bprmf --data_name amazon-book --data_dir ~/repos/mds-tmu-mrp/datasets --n_epoch 1 --test_batch_size=150 --use_pretrain 0

nohup python -m main_kgat --data_name baseline-kg --data_dir ~/mds-tmu-mrp/datasets --use_pretrain 0  --test_batch_size=500   --n_epoch 100 --cf_batch_size 8192 --kg_batch_size 8192 1> training.log 2> training.err &

python -m main_bprmf --data_name baseline-kg --data_dir ~/repos/mds-tmu-mrp/datasets --n_epoch 100 --test_batch_size=1000 --use_pretrain 0 --train_batch_size=20000


python -m main_kgat --data_name baseline-kg --data_dir ~/repos/mds-tmu-mrp/datasets --n_epoch 100 --test_batch_size=1000 --use_pretrain 1 --cf_batch_size=10000 --kg_batch_size 10000 --pretrain_embedding_dir /home/kchauhan/repos/mds-tmu-mrp/datasets/baseline-kg/pretrain 

python -m main_kgat --data_name baseline-kg --data_dir ~/repos/mds-tmu-mrp/datasets --n_epoch 100 --test_batch_size=1000 --use_pretrain 1 --cf_batch_size=20000 --kg_batch_size 20000 --pretrain_embedding_dir /home/kchauhan/repos/mds-tmu-mrp/datasets/baseline-kg/pretrain

python -m main_bprmf --data_name amazon-book --data_dir ~/repos/mds-tmu-mrp/datasets --n_epoch 5 --test_batch_size=150 --use_pretrain 0 --train_batch_size=32768  


nohup python -m main_bprmf --data_name baseline-kg --data_dir ~/repos/mds-tmu-mrp/datasets --n_epoch 100 --train_batch_size=32768  --test_batch_size=500 --use_pretrain 0 1> training.log 2> training.err &

nohup python -m main_bprmf --data_name baseline-kg --data_dir ~/mds-tmu-mrp/datasets --n_epoch 100 --train_batch_size=32768 --test_batch_size=500 --use_pretrain 0 1> training.log 2> training.err &

2024-07-31 00:21:03,713 - INFO - Total items processed: 29475453
2024-07-31 00:21:03,713 - INFO - Total items inserted: 9543016
2024-07-31 00:21:03,714 - INFO - Reviews for Books loaded
2024-07-31 00:21:03,714 - INFO - Data insertion complete
2024-07-31 00:21:03,795 - INFO - Table 'rating_only' has 9488297 rows
2024-07-31 00:21:03,796 - INFO - Table 'rating_only_positive' has 8038735 rows
2024-07-31 00:21:03,797 - INFO - Table 'raw_meta_Books' has 494691 rows
2024-07-31 00:21:03,798 - INFO - Table 'raw_review_Books' has 9543016 rows 

python main_kgat.py 

K-cores (i.e., dense subsets): These data have been reduced to extract the k-core, such that each of the remaining users and items have k reviews each.

Ratings only: These datasets include no metadata or reviews, but only (user,item,rating,timestamp) tuples. Thus they are suitable for use with mymedialite (or similar) packages.

HAS_SIMILAR_ELEMENTS
DEVELOPED_BY
PUBLISHED_BY
HAS_CHARACTERISTICS
COMPARES_TO
HAS_FEATURE


Label atleast 200
llama - 3.1 8b 
annotate 
Try few shot learning





higher core filtering