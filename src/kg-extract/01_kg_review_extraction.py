import os
import duckdb
import json
import time
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from utils.llm_custom import create_unstructured_prompt
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from langchain_community.callbacks import get_openai_callback
import signal
from dotenv import load_dotenv
import pandas as pd
import argparse

# Load environment variables
load_dotenv('/home/kchauhan/repos/mds-tmu-mrp/config/env.sh')

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process reviews for a specific relationship type.")
parser.add_argument("--relationship", type=str, help="Relationship type to process (e.g., SIMILAR_TO_BOOK, RELATED_AUTHOR)")
parser.add_argument("--model", type=str, default="llama3", help="Language model to use (gpt-4o-mini, llama3, phi3-mini)")
args = parser.parse_args()

# Constants
BATCH_SIZE = 10
TIMEOUT_SECONDS = 10
allowed_relationships = [args.relationship]  # Only process the specified relationship
allowed_nodes = ["Book"]
model = args.model

# Initialize Neo4j graph and language model (unchanged)
graph = Neo4jGraph()

if model == "gpt-4o-mini":
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
elif model == "llama3":
    llm = ChatOllama(model="llama3.1:8b-instruct-q4_0", temperature=0)
elif model == 'phi3-mini':
    llm = ChatOllama(model='phi3:3.8b-mini-4k-instruct-q6_K', temperature=0)

chat_template = create_unstructured_prompt(allowed_nodes, allowed_relationships)
llm_transformer = LLMGraphTransformer(
    llm=llm, allowed_nodes=allowed_nodes, allowed_relationships=allowed_relationships
)

# Connect to DuckDB
con = duckdb.connect("/home/kchauhan/repos/mds-tmu-mrp/db/duckdb/amazon_reviews.duckdb")

# Create tables if they don't exist (modified to include relationship type)
con.execute("""
    CREATE TABLE IF NOT EXISTS review_processing_status (
        user_id VARCHAR,
        item_id VARCHAR,
        relationship_type VARCHAR,
        status VARCHAR,
        json_data JSON,
        rating INTEGER,
        PRIMARY KEY (user_id, item_id, relationship_type)
    )
""")

con.execute("""
    CREATE TABLE IF NOT EXISTS skipped_reviews (
        user_id VARCHAR,
        item_id VARCHAR,
        relationship_type VARCHAR,
        reason VARCHAR,
        review_data JSON,
        PRIMARY KEY (user_id, item_id, relationship_type)
    )
""")

# Load test set into a temporary table (unchanged)
test_csv_path = '/home/kchauhan/repos/mds-tmu-mrp/data/processed/random_split/Books.test.csv'
test_df = pd.read_csv(test_csv_path, header=None, names=['user', 'item', 'rating', 'timestamp'])
con.register('test_df', test_df)
con.execute("""
    CREATE TEMPORARY TABLE test_set AS
    SELECT DISTINCT item FROM test_df
""")

# Modified function to get total number of unprocessed reviews
def get_total_unprocessed_reviews():
    result = con.execute(f"""
        SELECT COUNT(*)
        FROM raw_review_Books a
        INNER JOIN raw_meta_Books b ON a.parent_asin = b.parent_asin
        LEFT JOIN test_set t ON a.parent_asin = t.item
        WHERE CAST(a.helpful_vote AS INT) > 0
        AND NOT EXISTS (
            SELECT 1 FROM review_processing_status
            WHERE user_id = a.user_id AND item_id = a.parent_asin AND relationship_type = '{args.relationship}'
        )
        AND NOT EXISTS (
            SELECT 1 FROM skipped_reviews
            WHERE user_id = a.user_id AND item_id = a.parent_asin AND relationship_type = '{args.relationship}'
        )
        AND LENGTH(a.text) > 100
    """).fetchone()
    return result[0] if result else 0

# Function to check if a review has been processed (modified to include relationship type)
def is_review_processed(user_id, item_id):
    result = con.execute(
        "SELECT status FROM review_processing_status WHERE user_id = ? AND item_id = ? AND relationship_type = ?",
        [user_id, item_id, args.relationship],
    ).fetchone()
    return result is not None and result[0] == "processed"

# Function to update review status and store JSON data (modified to include relationship type)
def update_review_status(user_id, item_id, json_data):
    con.execute(
        "INSERT INTO review_processing_status (user_id, item_id, relationship_type, status, json_data) VALUES (?, ?, ?, 'processed', ?)",
        [user_id, item_id, args.relationship, json.dumps(json_data)],
    )

# Function to store skipped review (modified to include relationship type)
def store_skipped_review(user_id, item_id, reason, review_data):
    try:
        con.execute(
            "INSERT INTO skipped_reviews (user_id, item_id, relationship_type, reason, review_data) VALUES (?, ?, ?, ?, ?)",
            [user_id, item_id, args.relationship, reason, json.dumps(review_data)],
        )
    except Exception as e:
        print(f"Error storing skipped review: {e}")

# Function to process a single review (modified to include relationship type in output file name)
def process_review(row):
    user_id = row[5]
    item_id = row[1]

    if is_review_processed(user_id, item_id):
        return "already_processed"

    text = f"Book: {row[0]}\nReview Title: {row[2]}\nReview Text: {row[3]}"
    documents = [Document(page_content=text)]

    try:
        if model == 'gpt-4o-mini':
            graph_documents = llm_transformer.convert_to_graph_documents(documents)
        else:
            graph_documents = llm_transformer.convert_to_graph_documents(documents)
    except Exception as e:
        error_message = f"Error: type validation failed for row: {row}\n, error: {e}\n"
        with open(f"output/{model}/{args.relationship}_errors.txt", "a") as f:
            f.write(error_message)
        store_skipped_review(
            user_id,
            item_id,
            "Type validation error",
            {
                "user_id": user_id,
                "title": row[0],
                "parent_asin": row[1],
                "review_title": row[2],
                "review_text": row[3],
                "helpful_vote": row[4],
                "error": str(e),
            },
        )
        return "skipped"

    nodes = [{"id": node.id, "type": node.type} for node in graph_documents[0].nodes]
    relationships = [
        {
            "source": {"id": rel.source.id, "type": rel.source.type},
            "target": {"id": rel.target.id, "type": rel.target.type},
            "type": rel.type,
        }
        for rel in graph_documents[0].relationships
    ]

    if not nodes or not relationships:
        error_message = f"Error: No nodes or relationships found for row: {row}\n"
        with open(f"output/{model}/{args.relationship}_errors.txt", "a") as f:
            f.write(error_message)
        store_skipped_review(
            user_id,
            item_id,
            "No nodes or relationships",
            {
                "user_id": user_id,
                "title": row[0],
                "parent_asin": row[1],
                "review_title": row[2],
                "review_text": row[3],
                "helpful_vote": row[4],
            },
        )
        return "skipped"

    json_data = {
        "user_id": user_id,
        "title": row[0],
        "parent_asin": row[1],
        "review_title": row[2],
        "review_text": row[3],
        "helpful_vote": row[4],
        "nodes": nodes,
        "relationships": relationships,
    }

    os.makedirs(f"output/{model}/{args.relationship}", exist_ok=True)
    timestamp = int(time.time())
    with open(f"output/{model}/{args.relationship}/{timestamp}_{user_id}_{row[1]}.json", "w") as f:
        json.dump(json_data, f, indent=2)

    update_review_status(user_id, item_id, json_data)
    return "processed"

# Timeout handler function (unchanged)
def timeout_handler(signum, frame):
    raise TimeoutError("Review processing timed out")

# Modified main processing loop
total_reviews = get_total_unprocessed_reviews()
processed_reviews = 0
skipped_reviews = 0

print(f"Total unprocessed reviews for {args.relationship}: {total_reviews}")

offset = 0
while True:
    # Fetch a batch of unprocessed reviews, prioritizing test set items
    batch = con.execute(f"""
        SELECT b.title, b.parent_asin, a.title as review_title, a.text as review_text, a.helpful_vote, a.user_id,
               CASE WHEN t.item IS NOT NULL THEN 1 ELSE 0 END as is_test_item
        FROM raw_review_Books a
        INNER JOIN raw_meta_Books b ON a.parent_asin = b.parent_asin
        LEFT JOIN test_set t ON a.parent_asin = t.item
        WHERE CAST(a.helpful_vote AS INT) > 0
        AND NOT EXISTS (
            SELECT 1 FROM review_processing_status
            WHERE user_id = a.user_id AND item_id = a.parent_asin AND relationship_type = '{args.relationship}'
        )
        AND NOT EXISTS (
            SELECT 1 FROM skipped_reviews
            WHERE user_id = a.user_id AND item_id = a.parent_asin AND relationship_type = '{args.relationship}'
        )
        AND LENGTH(a.text) > 100
        ORDER BY is_test_item DESC, CAST(a.helpful_vote AS INT) DESC
        LIMIT {BATCH_SIZE} OFFSET {offset}
    """).fetchall()

    if not batch:
        break  # No more reviews to process

    for row in batch:
        # Process review (unchanged)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT_SECONDS)

        try:
            result = process_review(row)
            if result == "processed":
                processed_reviews += 1
            elif result == "skipped":
                skipped_reviews += 1
        except TimeoutError:
            print(f"\nTimeout occurred while processing review: {row[1]}")
            store_skipped_review(
                row[5],
                row[1],
                "Processing timeout",
                {
                    "title": row[0],
                    "parent_asin": row[1],
                    "review_title": row[2],
                    "review_text": row[3],
                    "helpful_vote": row[4],
                },
            )
            skipped_reviews += 1
        finally:
            signal.alarm(0)

        # Update progress
        progress = (processed_reviews / total_reviews) * 100
        print(f"\rProgress: {progress:.2f}% ({processed_reviews}/{total_reviews}), Skipped: {skipped_reviews}", end="", flush=True)

    offset += BATCH_SIZE

print(f"\nProcessing complete for {args.relationship}! Total processed: {processed_reviews}, Skipped: {skipped_reviews}")

# Close the DuckDB connection
con.close()