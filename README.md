# LLM Enhanced Knowledge Graph-Based Recommendation System

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

### Installation

Install the necessary packages:

```bash
pip install -r requirements.txt
```

For CUDA support:

```bash
CUDACXX=/usr/local/cuda-12/bin/nvcc CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=native" FORCE_CMAKE=1 pip install pandas numpy scikit-learn scipy implicit
```

## Data Preprocessing

### K-core Filtering

Filter the dataset to retain users and items with at least 15 interactions:

```bash
BASE_DIR=~/repos/mds-tmu-mrp
INPUT_PATH=${BASE_DIR}/data/raw
OUTPUT_PATH=${BASE_DIR}/data/processed/k_core_filtering
cd $BASE_DIR/src/preprocess/rec
python -m 01_k_core_filtering.py -k 15 --input_path $INPUT_PATH --output_path $OUTPUT_PATH 
```

### Splitting the Data

#### Last Out Split

```bash
INPUT_PATH=${BASE_DIR}/data/processed/k_core_filtered/15core/rating_only
OUTPUT_PATH=${BASE_DIR}/data/processed/last_out_split
SEQ_PATH=${OUTPUT_PATH}/seq
python -m 02_last_out_split --input_path $INPUT_PATH --output_path $OUTPUT_PATH --seq_path $SEQ_PATH
```

#### Timestamp Split

```bash
OUTPUT_PATH=${BASE_DIR}/data/processed/timestamp_split
python -m 02_timestamp_split --input_path $INPUT_PATH --output_path $OUTPUT_PATH --seq_path $SEQ_PATH
```

#### Random Split

```bash
OUTPUT_PATH=${BASE_DIR}/data/processed/random_split
python -m 03_random_80_20_split --input_path $INPUT_PATH --output_path $OUTPUT_PATH --seq_path $SEQ_PATH
```

## Data Loading

### Loading Data to DuckDB

#### Load K-core Ratings

```bash
python -m src/preprocess/kg/01_load_kcore_ratings_duckdb.py
```

#### Load Metadata and Reviews

```bash
python -m src/preprocess/kg/02_load_metadata_reviews_duckdb.py
```

## Knowledge Graph Construction

### Load Metadata as Baseline KG

```bash
python -m src/preprocess/kg/03_load_metadata_neo4j.py
```

## Resetting the Databases

### Reset DuckDB

```bash
rm -rf db/duckdb/amazon-reviews.db
```

### Reset Neo4j

```bash
cd db/neo4j
sudo reset.sh
```

## Model Training

### BPRMF

Train the BPRMF model:

```bash
DATA_DIR=${BASE_DIR}/data/kg
DATA_NAME=baseline-kg
python -m main_bprmf --data_name $DATA_NAME --data_dir $DATA_DIR --n_epoch 100 --test_batch_size=1000 --use_pretrain 0 --train_batch_size=20000
```

To resume training from a pre-trained model:

```bash
PRETRAIN_MODEL_PATH=${BASE_DIR}/src/trained_model/BPRMF/baseline-kg/embed-dim64_lr0.0001_pretrain0/model_epoch100.pth
PRETRAIN_EMBEDDING_DIR=${BASE_DIR}/data/kg/baseline-kg/pretrain
python -m main_bprmf --data_name $DATA_NAME --data_dir $DATA_DIR --n_epoch 100 --test_batch_size=1000 --use_pretrain 2 --train_batch_size=20000 --pretrain_model_path $PRETRAIN_MODEL_PATH --pretrain_embedding_dir $PRETRAIN_EMBEDDING_DIR
```

### KGAT

```bash
python -m main_kgat --data_name baseline-kg --data_dir $DATA_DIR --n_epoch 100 --test_batch_size=1000 --use_pretrain 0 --cf_batch_size=10000 --kg_batch_size 10000 --pretrain_embedding_dir $PRETRAIN_EMBEDDING_DIR
```

To use pre-trained embeddings:

```bash
python -m main_kgat --data_name $DATA_NAME --data_dir $DATA_DIR --n_epoch 100 --test_batch_size=1000 --use_pretrain 1 --cf_batch_size=20000 --kg_batch_size 20000 --pretrain_embedding_dir $PRETRAIN_EMBEDDING_DIR
```

## Extracting Relationships Using LLM

### Extract Relationships (e.g., SIMILAR_TO_BOOK, RELATED_AUTHOR)

```bash
cd ${BASE_DIR}/src/kg-extract 
python -m 01_kg_review_extraction --relationship SIMILAR_TO_BOOK --model gpt-4o-mini
```

### Evaluate Extracted Relationships

```bash
python -m 02_kg_extraction_rating.py --relationship SIMILAR_TO_BOOK --model llama3
```

### Update KG with Extracted Relationships

```bash
python -m 03_neo4j_update_kg.py --relationship SIMILAR_TO_BOOK
```
