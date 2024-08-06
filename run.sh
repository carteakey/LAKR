#!/bin/bash

# Set environment variables
export DATE=$(date '+%Y-%m-%d-%H-%M-%S')
export ROOT=/home/kchauhan/repos/mds-tmu-mrp
export SRC=$ROOT/src
export LOGS=$ROOT/logs
export DB=$ROOT/db
export DATA_NAME=baseline-kg
export DATA_DIR=$ROOT/data/kg
export PRETRAIN_EMBEDDING_DIR=$ROOT/data/kg/baseline-kg/pretrain


# Create logs directory if it doesn't exist
mkdir -p $LOGS

# Stop on error
set -e

# Activate virtual environment
. .venv/bin/activate

# Backup the DuckDB database
if [ "$1" == "backup_duckdb" ]; then
  echo "Backing up DuckDB database..."
  mv $DB/duckdb/amazon_reviews.duckdb $DB/duckdb/amazon_reviews.duckdb.bkp
  echo "Backup completed." | tee -a $LOGS/backup_duckdb_$DATE.log
fi

# Preprocess data with k-core filtering
if [ "$1" == "k_core_filtering" ]; then
  echo "Running k-core filtering..."
  INPUT_PATH=${ROOT}/data/raw
  OUTPUT_PATH=${ROOT}/data/processed/k_core_filtering
  SEQ_PATH=${OUTPUT_PATH}/seq
  cd $SRC/preprocess/rec
  python -m 01_k_core_filtering.py -k 15 --input_path $INPUT_PATH --output_path $OUTPUT_PATH | tee -a $LOGS/k_core_filtering_$DATE.log
  echo "K-core filtering completed." | tee -a $LOGS/k_core_filtering_$DATE.log
fi

# Split data (Last out split)
if [ "$1" == "last_out_split" ]; then
  echo "Running last out split..."
  INPUT_PATH=${ROOT}/data/processed/k_core_filtered/15core/rating_only
  OUTPUT_PATH=${ROOT}/data/processed/last_out_split
  SEQ_PATH=${OUTPUT_PATH}/seq
  cd $SRC/preprocess/rec
  python -m 02_last_out_split --input_path $INPUT_PATH --output_path $OUTPUT_PATH --seq_path $SEQ_PATH | tee -a $LOGS/last_out_split_$DATE.log
  echo "Last out split completed." | tee -a $LOGS/last_out_split_$DATE.log
fi

# Split data (Timestamp split)
if [ "$1" == "timestamp_split" ]; then
  echo "Running timestamp split..."
  INPUT_PATH=${ROOT}/data/processed/k_core_filtered/15core/rating_only
  OUTPUT_PATH=${ROOT}/data/processed/timestamp_split
  SEQ_PATH=${OUTPUT_PATH}/seq
  cd $SRC/preprocess/rec
  python -m 02_timestamp_split --input_path $INPUT_PATH --output_path $OUTPUT_PATH --seq_path $SEQ_PATH | tee -a $LOGS/timestamp_split_$DATE.log
  echo "Timestamp split completed." | tee -a $LOGS/timestamp_split_$DATE.log
fi

# Split data (Random split)
if [ "$1" == "random_split" ]; then
  echo "Running random split..."
  INPUT_PATH=${ROOT}/data/processed/k_core_filtered/15core/rating_only
  OUTPUT_PATH=${ROOT}/data/processed/random_split
  SEQ_PATH=${OUTPUT_PATH}/seq
  cd $SRC/preprocess/rec
  python -m 03_random_80_20_split --input_path $INPUT_PATH --output_path $OUTPUT_PATH --seq_path $SEQ_PATH | tee -a $LOGS/random_split_$DATE.log
  echo "Random split completed." | tee -a $LOGS/random_split_$DATE.log
fi

# Load K-core ratings to DuckDB
if [ "$1" == "load_kcore_ratings_duckdb" ]; then
  echo "Loading K-core ratings to DuckDB..."
  cd $ROOT
  python -m src/preprocess/kg/01_load_kcore_ratings_duckdb.py | tee -a $LOGS/load_kcore_ratings_duckdb_$DATE.log
  echo "K-core ratings loaded to DuckDB." | tee -a $LOGS/load_kcore_ratings_duckdb_$DATE.log
fi

# Load metadata and reviews to DuckDB
if [ "$1" == "load_metadata_reviews_duckdb" ]; then
  echo "Loading metadata and reviews to DuckDB..."
  python -m src/preprocess/kg/02_load_metadata_reviews_duckdb.py | tee -a $LOGS/load_metadata_reviews_duckdb_$DATE.log
  echo "Metadata and reviews loaded to DuckDB." | tee -a $LOGS/load_metadata_reviews_duckdb_$DATE.log
fi

# Load metadata as a baseline KG to Neo4j
if [ "$1" == "load_metadata_neo4j" ]; then
  echo "Loading metadata as a baseline KG to Neo4j..."
  python -m src/preprocess/kg/03_load_metadata_neo4j.py | tee -a $LOGS/load_metadata_neo4j_$DATE.log
  echo "Metadata loaded as a baseline KG to Neo4j." | tee -a $LOGS/load_metadata_neo4j_$DATE.log
fi

# Reset the DuckDB database
if [ "$1" == "reset_duckdb" ]; then
  echo "Resetting the DuckDB database..."
  rm -rf $DB/duckdb/amazon_reviews.db
  echo "DuckDB database reset." | tee -a $LOGS/reset_duckdb_$DATE.log
fi

# Reset the Neo4j database
if [ "$1" == "reset_neo4j" ]; then
  echo "Resetting the Neo4j database..."
  cd $DB/neo4j
  sudo reset.sh | tee -a $LOGS/reset_neo4j_$DATE.log
  echo "Neo4j database reset." | tee -a $LOGS/reset_neo4j_$DATE.log
fi

# Train BPRMF model
if [ "$1" == "train_bprmf" ]; then
  echo "Training BPRMF model..."
  cd $SRC
  python -m main_bprmf --data_name $DATA_NAME --data_dir $DATA_DIR --n_epoch 100 --test_batch_size=1000 --use_pretrain 0 --train_batch_size=20000 | tee -a $LOGS/train_bprmf_$DATE.log
  echo "BPRMF model training completed." | tee -a $LOGS/train_bprmf_$DATE.log
fi

# Train BPRMF model from pre-trained
if [ "$1" == "train_bprmf_pretrained" ]; then
  echo "Training BPRMF model from pre-trained..."
  cd $SRC
  python -m main_bprmf --data_name $DATA_NAME --data_dir $DATA_DIR --n_epoch 100 --test_batch_size=1000 --use_pretrain 2 --train_batch_size=20000 --pretrain_model_path $PRETRAIN_MODEL_PATH --pretrain_embedding_dir $PRETRAIN_EMBEDDING_DIR | tee -a $LOGS/train_bprmf_pretrained_$DATE.log
  echo "BPRMF model training from pre-trained completed." | tee -a $LOGS/train_bprmf_pretrained_$DATE.log
fi

# Train KGAT model
if [ "$1" == "train_kgat" ]; then
  echo "Training KGAT model..."
  cd $SRC
  python -m main_kgat --data_name $DATA_NAME --data_dir $DATA_DIR --n_epoch 100 --test_batch_size=1000 --use_pretrain 0 --cf_batch_size=10000 --kg_batch_size 10000 --pretrain_embedding_dir $PRETRAIN_EMBEDDING_DIR | tee -a $LOGS/train_kgat_$DATE.log
  echo "KGAT model training completed." | tee -a $LOGS/train_kgat_$DATE.log
fi

# Train KGAT model from pre-trained
if [ "$1" == "train_kgat_pretrained" ]; then
  echo "Training KGAT model from pre-trained..."
  cd $SRC
  python -m main_kgat --data_name $DATA_NAME --data_dir $DATA_DIR --n_epoch 100 --test_batch_size=1000 --use_pretrain 1 --cf_batch_size=10000 --kg_batch_size 10000 --pretrain_model_path $PRETRAIN_MODEL_PATH --pretrain_embedding_dir $PRETRAIN_EMBEDDING_DIR | tee -a $LOGS/train_kgat_pretrained_$DATE.log
  echo "KGAT model training from pre-trained completed." | tee -a $LOGS/train_kgat_pretrained_$DATE.log
fi

# Extract relationships using LLM
if [ "$1" == "extract_relationships" ]; then
  echo "Extracting relationships using LLM..."
  cd $SRC/kg-extract 
  python -m 01_kg_review_extraction --relationship DEALS_WITH_CONCEPTS --model gpt-4o-mini | tee -a $LOGS/extract_relationships_$DATE.log
  echo "Relationships extracted." | tee -a $LOGS/extract_relationships_$DATE.log
fi

# Evaluate extracted relationships
if [ "$1" == "evaluate_relationships" ]; then
  echo "Evaluating extracted relationships..."
  cd $SRC/kg-extract
  python -m 02_kg_extraction_rating --relationship ALL --model llama3 | tee -a $LOGS/evaluate_relationships_$DATE.log
  echo "Relationships evaluated." | tee -a $LOGS/evaluate_relationships_$DATE.log
fi

# Update the KG with extracted relationships
if [ "$1" == "update_kg" ]; then
  echo "Updating the KG with extracted relationships..."
  cd $SRC/kg-extract
  python -m 03_neo4j_update_kg --relationship SIMILAR_TO_BOOK | tee -a $LOGS/update_kg_$DATE.log
  echo "KG updated with extracted relationships." | tee -a $LOGS/update_kg_$DATE.log
fi

# Deactivate virtual environment
deactivate
