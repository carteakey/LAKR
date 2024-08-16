# Description: Environment variables for the project

# DuckDB
DUCKDB_PATH=/home/kchauhan/repos/mds-tmu-mrp/db/duckdb/amazon_reviews.duckdb

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=tmu-2024

# OpenAI and Google API keys
OPENAI_API_KEY=XX
GOOGLE_API_KEY=XX

# PostgreSQL
PGPASSWORD=adminpassword
PGHOST=localhost
PGUSER=admin
PGDATBASE=amazon_reviews

# Paths
ROOT=/home/kchauhan/repos/mds-tmu-mrp # Change this to the root directory of the project
SRC=$ROOT/src
LOGS=$ROOT/logs
DB=$ROOT/db
DATA_NAME=baseline-kg
DATA_DIR=$ROOT/data/kg
PRETRAIN_MODEL_PATH=$ROOT/src/trained_model/BPRMF/baseline-kg/embed-dim64_lr0.0001_pretrain0/model_epoch100.pth
PRETRAIN_EMBEDDING_DIR=$ROOT/data/kg/baseline-kg/pretrain
