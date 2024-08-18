# LAKR (LLM-Augmented Knowledge-Graph-Based Recommendation)

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
- [Docker Compose](https://docs.docker.com/compose/)

### Installation

Install the necessary packages:

```bash
pip install -r requirements.txt
```

For CUDA support:

```bash
CUDACXX=/usr/local/cuda-12/bin/nvcc CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=native" FORCE_CMAKE=1 pip install pandas numpy scikit-learn scipy implicit
```

Start the databases:

Neo4j:
```bash
cd db/neo4j
docker compose up -d
```
Please note that the Neo4j database is started with the default password `neo4j`. The password is changed after the first run.

Postgres:
```bash
cd db/postgres
docker compose up -d
```

### Running the Code
Update the variables inside the run.sh  and config/env.sh files to point to the correct directories.

## Data Preprocessing 

### K-core Filtering
#### Download the Amazon Reviews dataset 
(Note that the initial download will take some time.)

```bash
./run.sh download_amazon_reviews
```

Filter the dataset to retain users and items with at least 15 interactions :

```bash
./run.sh k_core_filtering
```

### Splitting the Data

#### Last Out Split

```bash
./run.sh last_out_split
```

#### Timestamp Split

```bash
./run.sh timestamp_split
```

#### Random Split

```bash
./run.sh random_split
```

## Data Loading

### Loading Data to DuckDB

#### Load K-core Ratings

```bash
./run.sh load_kcore_ratings_duckdb
```

#### Load Metadata and Reviews

This takes some time as the metadata and reviews are downloaded in huggingface format and then loaded into DuckDB.

```bash
./run.sh load_metadata_reviews_duckdb
```

#### Load Metadata as Baseline KG

```bash
./run.sh load_metadata_neo4j
```

## Knowledge Graph Augmentation using LLM

### Extract Relationships (e.g., SIMILAR_TO_BOOK, RELATED_AUTHOR)

```bash
./run.sh extract_relationships --relationship SIMILAR_TO_BOOK --max_batches 10
```

### Evaluate Extracted Relationships

```bash
./run.sh evaluate_relationships --relationship SIMILAR_TO_BOOK
```

### Update KG with Extracted Relationships

```bash
./run.sh update_kg --relationship SIMILAR_TO_BOOK
```


## Model Training

### BPRMF

Train the BPRMF model:

```bash
./run.sh train_bprmf
```

To resume training from a pre-trained model:

```bash
./run.sh train_bprmf_pretrained
```

### KGAT

```bash
./run.sh train_kgat
```

To use pre-trained embeddings:

```bash
./run.sh train_kgat_pretrained
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

### Reset Postgres

```bash
cd db/postgres
sudo reset.sh
```

