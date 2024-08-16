#!/bin/bash

# Set environment variables
export DATE=$(date '+%Y-%m-%d-%H-%M-%S')
# import env.sh
. env.sh

# Create logs directory if it doesn't exist
mkdir -p $LOGS

# Stop on error
set -e

# Activate virtual environment
. .venv/bin/activate

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$2"
}

# Downlad the dataset
if [ "$1" == "download_amazon_reviews" ]; then
  echo "Downloading the dataset..."
  cd $ROOT/data/raw
  wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Books.jsonl.gz | tee -a $LOGS/download_dataset_$DATE.log
  gunzip Books.jsonl.gz | tee -a $LOGS/download_dataset_$DATE.log
  echo "Dataset downloaded." | tee -a $LOGS/download_dataset_$DATE.log
fi

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
  cd $ROOT/src/preprocess/kg
  python -m 01_load_kcore_ratings_duckdb.py | tee -a $LOGS/load_kcore_ratings_duckdb_$DATE.log
  echo "K-core ratings loaded to DuckDB." | tee -a $LOGS/load_kcore_ratings_duckdb_$DATE.log
fi

# Load metadata and reviews to DuckDB
if [ "$1" == "load_metadata_reviews_duckdb" ]; then
  echo "Loading metadata and reviews to DuckDB..."
  cd $ROOT/src/preprocess/kg
  python -m kg/02_load_metadata_reviews_duckdb.py | tee -a $LOGS/load_metadata_reviews_duckdb_$DATE.log
  echo "Metadata and reviews loaded to DuckDB." | tee -a $LOGS/load_metadata_reviews_duckdb_$DATE.log
fi

# Load metadata as a baseline KG to Neo4j
if [ "$1" == "load_metadata_neo4j" ]; then
  echo "Loading metadata as a baseline KG to Neo4j..."
  cd $ROOT/src/preprocess/kg
  python -m 03_load_metadata_neo4j.py | tee -a $LOGS/load_metadata_neo4j_$DATE.log
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
  python -u -m main_kgat --data_name $DATA_NAME --data_dir $DATA_DIR --n_epoch 100 --test_batch_size=1000 --use_pretrain 1 --cf_batch_size=10000 --kg_batch_size 10000 --pretrain_model_path $PRETRAIN_MODEL_PATH --pretrain_embedding_dir $PRETRAIN_EMBEDDING_DIR | tee -a $LOGS/train_kgat_pretrained_$DATE.log
  echo "KGAT model training from pre-trained completed." | tee -a $LOGS/train_kgat_pretrained_$DATE.log
fi

# Extract relationships using LLM
if [ "$1" == "extract_relationships" ]; then
    LOGFILE="$LOGS/extract_relationships_$DATE.log"
    log_message "Extracting relationships using LLM..." "$LOGFILE"
    cd $SRC/kg-extract 
  
    # Check if a relationship is provided
    if [ "$2" != "--relationship" ] || [ -z "$3" ]; then
        log_message "Error: Relationship not specified. Usage: ./run.sh extract_relationships --relationship RELATIONSHIP_TYPE" "$LOGFILE"
        exit 1
    fi
  
    # Extract the relationship from the command line argument
    RELATIONSHIP="$3"
  
    log_message "Starting extraction for relationship: $RELATIONSHIP" "$LOGFILE"
    python -m 01_kg_review_extraction --relationship "$RELATIONSHIP" --model llama3 --max_batches 2 2>&1 | tee -a "$LOGFILE"
    log_message "Relationships of type $RELATIONSHIP extracted." "$LOGFILE"
fi

# Evaluate extracted relationships
if [ "$1" == "evaluate_relationships" ]; then
    LOGFILE="$LOGS/evaluate_relationships_$DATE.log"
    log_message "Evaluating extracted relationships..." "$LOGFILE"
    cd $SRC/kg-extract
    
    # Check if a relationship is provided
    if [ "$2" != "--relationship" ] || [ -z "$3" ]; then
        log_message "Error: Relationship not specified. Usage: ./run.sh evaluate_relationships --relationship RELATIONSHIP_TYPE" "$LOGFILE"
        exit 1
    fi
  
    # Extract the relationship from the command line argument
    RELATIONSHIP="$3"
    
    log_message "Starting evaluation for relationship: $RELATIONSHIP" "$LOGFILE"
    python -m 02_kg_extraction_rating --relationship "$RELATIONSHIP" --model llama3 2>&1 | tee -a "$LOGFILE"
    log_message "Relationships of type $RELATIONSHIP evaluated." "$LOGFILE"
fi

# Update the KG with extracted relationships
if [ "$1" == "update_kg" ]; then
    LOGFILE="$LOGS/update_kg_$DATE.log"
    log_message "Updating the KG with extracted relationships..." "$LOGFILE"
    cd $SRC/kg-extract
  
    # Check if a relationship is provided
    if [ "$2" != "--relationship" ] || [ -z "$3" ]; then
        log_message "Error: Relationship not specified. Usage: ./run.sh update_kg --relationship RELATIONSHIP_TYPE" "$LOGFILE"
        exit 1
    fi
  
    # Extract the relationship from the command line argument
    RELATIONSHIP="$3"
  
    log_message "Starting KG update for relationship: $RELATIONSHIP" "$LOGFILE"
    python -m 03_neo4j_update_kg --relationship "$RELATIONSHIP" 2>&1 | tee -a "$LOGFILE"
    log_message "KG updated with extracted relationships of type $RELATIONSHIP." "$LOGFILE"
fi

# Cleanup KG using 04_neo4j_cleanup script
if [ "$1" == "cleanup_kg" ]; then
  LOGFILE="$LOGS/cleanup_kg_$DATE.log"

  log_message "Cleaning up KG using 04_neo4j_cleanup script..." "$LOGFILE"
  cd $SRC/kg-extract
  log_message "Starting KG cleanup..." "$LOGFILE"
  python -m 04_neo4j_cleanup 2>&1 | tee -a "$LOGFILE"
  log_message "KG cleanup completed." "$LOGFILE"
fi

# Deactivate virtual environment
deactivate
