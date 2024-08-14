import os
import duckdb
import json
import time
import re
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from utils.llm_custom import create_unstructured_prompt
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
from dotenv import load_dotenv
import pandas as pd
import argparse
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv('/home/kchauhan/repos/mds-tmu-mrp/config/env.sh')

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process reviews for a specific relationship type.")
parser.add_argument("--relationship", type=str, help="Relationship type to process (e.g., RELATED_TO_AUTHOR)")
parser.add_argument("--model", type=str, default="llama3", help="Language model to use (gpt-4o-mini, llama3, phi3-mini, gemini)")
parser.add_argument("--max_batches", type=int, default=None, help="Maximum number of batches to process")
args = parser.parse_args()

# Constants
BATCH_SIZE = 10
TIMEOUT_SECONDS = 10
MAX_WORKERS = 1  # Number of threads

# Relationship and node configurations
RELATIONSHIP_CONFIGS = {
    'ALL': {
        'relationships': ['SIMILAR_TO_BOOK', 'DEALS_WITH_CONCEPTS', 'PART_OF_SERIES', 'RELATED_AUTHOR'],
        'nodes': ['Book', 'Author', 'Concept', 'Series']
    },
    'SIMILAR_TO_BOOK': {'nodes': ['Book']},
    'SIMILAR_TO_AUTHOR': {'nodes': ['Author']},
    'DEALS_WITH_CONCEPTS': {'nodes': ['Book', 'Concept']},
    'PART_OF_SERIES': {'nodes': ['Book', 'Series']},
    'RELATED_AUTHOR': {'nodes': ['Book', 'Author']}
}

config = RELATIONSHIP_CONFIGS.get(args.relationship)
if not config:
    raise ValueError(f"Invalid relationship type: {args.relationship}")

allowed_relationships = config.get('relationships', [args.relationship])
allowed_nodes = config['nodes']

# Initialize Neo4j graph and language model
graph = Neo4jGraph()

MODEL_CONFIGS = {
    "gpt-4o-mini": lambda: ChatOpenAI(temperature=0, model_name="gpt-4o-mini"),
    "llama3": lambda: ChatOllama(model="llama3.1:8b-instruct-q4_0", temperature=0),
    'phi3-mini': lambda: ChatOllama(model='phi3:3.8b-mini-4k-instruct-q6_K', temperature=0),
    'gemini': lambda: ChatGoogleGenerativeAI(model="gemini-1.5-flash")
}

llm = MODEL_CONFIGS.get(args.model, MODEL_CONFIGS["llama3"])()

chat_template = create_unstructured_prompt(allowed_nodes, allowed_relationships)
llm_transformer = LLMGraphTransformer(
    llm=llm, 
    allowed_nodes=allowed_nodes, 
    allowed_relationships=allowed_relationships
)

# Connect to DuckDB
# TODO: Update the path to the DuckDB database file
con = duckdb.connect("/home/kchauhan/repos/mds-tmu-mrp/db/duckdb/amazon_reviews.duckdb")

# Initialize Neo4j driver
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USER")
neo4j_password = os.getenv("NEO4J_PASSWORD")
neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

def asin_exists_in_neo4j(asin):
    with neo4j_driver.session() as session:
        result = session.run("MATCH (b:Book {parent_asin: $asin}) RETURN COUNT(b) > 0 AS exists", asin=asin)
        return result.single()["exists"]

# Create tables if they don't exist
def create_tables():
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

# Load test set into a temporary table
def load_test_set():
    test_csv_path = '/home/kchauhan/repos/mds-tmu-mrp/data/processed/random_split/Books.test.csv'
    test_df = pd.read_csv(test_csv_path, header=None, names=['user', 'item', 'rating', 'timestamp'])
    con.register('test_df', test_df)
    con.execute("""
        CREATE TEMPORARY TABLE test_set AS
        SELECT DISTINCT item FROM test_df
    """)

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

def is_review_processed(user_id, item_id):
    result = con.execute(
        "SELECT status FROM review_processing_status WHERE user_id = ? AND item_id = ? AND relationship_type = ?",
        [user_id, item_id, args.relationship],
    ).fetchone()
    return result is not None 
    # and result[0] == "processed"

def update_review_status(user_id, item_id, json_data):
    con.execute(
        "INSERT INTO review_processing_status (user_id, item_id, relationship_type, status, json_data) VALUES (?, ?, ?, 'processed', ?)",
        [user_id, item_id, args.relationship, json.dumps(json_data)],
    )

def store_skipped_review(user_id, item_id, reason, review_data):
    try:
        con.execute(
            "INSERT INTO skipped_reviews (user_id, item_id, relationship_type, reason, review_data) VALUES (?, ?, ?, ?, ?)",
            [user_id, item_id, args.relationship, reason, json.dumps(review_data)],
        )
    except Exception as e:
        print(f"Error storing skipped review: {e}")

def extract_author_name(author_str):
    # Regex to extract the name value
    match = re.search(r"'name': '([^']+)'", author_str)
    if match:
        return match.group(1)
    return None

def process_review(row):
    user_id, item_id,author_str = row[5], row[1], row[7]

    # Extract the author's name
    author_name = extract_author_name(author_str)
    
    if is_review_processed(user_id, item_id):
        return "already_processed"

    if not asin_exists_in_neo4j(item_id):
        store_skipped_review(
            user_id, item_id, "ASIN not found in Neo4j",
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
    
    text = f"Book: {row[0]}\nAuthor: {author_name}\nReview Title: {row[3]}\nReview Text: {row[4]}"
    documents = [Document(page_content=text)]

    try:
        graph_documents = llm_transformer.convert_to_graph_documents(documents)
    except Exception as e:
        error_message = f"Error: type validation failed for row: {row}\n, error: {e}\n"
        with open(f"output/{args.model}/{args.relationship}_errors.txt", "a") as f:
            f.write(error_message)
        store_skipped_review(
            user_id, item_id, "Type validation error",
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
        with open(f"output/{args.model}/{args.relationship}_errors.txt", "a") as f:
            f.write(error_message)
        store_skipped_review(
            user_id, item_id, "No nodes or relationships",
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

    os.makedirs(f"output/{args.model}/{args.relationship}", exist_ok=True)
    timestamp = int(time.time())
    with open(f"output/{args.model}/{args.relationship}/{timestamp}_{user_id}_{row[1]}.json", "w") as f:
        json.dump(json_data, f, indent=2)

    update_review_status(user_id, item_id, json_data)
    return "processed"

def worker(row):
    try:
        with ThreadPoolExecutor() as executor:
            future = executor.submit(process_review, row)
            result = future.result(timeout=TIMEOUT_SECONDS)
        return result
    except concurrent.futures.TimeoutError:
        print(f"Timeout occurred while processing review: {row[1]}")
        store_skipped_review(
            row[5], row[1], "Processing timeout",
            {
                "title": row[0],
                "parent_asin": row[1],
                "review_title": row[2],
                "review_text": row[3],
                "helpful_vote": row[4],
            },
        )
        return "skipped"
    except Exception as e:
        print(f"An error occurred: {e}")
        return "skipped"

def main():
    create_tables()
    load_test_set()

    total_reviews = get_total_unprocessed_reviews()
    processed_reviews = 0
    skipped_reviews = 0

    print(f"Total unprocessed reviews for {args.relationship}: {total_reviews}")

    offset = 0
    batch_count = 0
    while True:
        if args.max_batches and batch_count >= args.max_batches:
            print(f"Reached maximum number of batches ({args.max_batches}). Stopping.")
            break

        batch = con.execute(f"""
            SELECT b.title, b.parent_asin, a.title as review_title, a.text as review_text, a.helpful_vote, a.user_id,
                   CASE WHEN t.item IS NOT NULL THEN 1 ELSE 0 END as is_test_item, b.author
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
    # Using ThreadPoolExecutor to parallelize the processing
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(worker, row) for row in batch]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result == "processed":
                        processed_reviews += 1
                    elif result == "skipped":
                        skipped_reviews += 1
                except concurrent.futures.TimeoutError:
                    print(f"Timeout occurred for a review in batch")
                    skipped_reviews += 1
        # Update progress
        progress = (processed_reviews / total_reviews) * 100
        print(f"\rProgress: {progress:.2f}% ({processed_reviews}/{total_reviews}), Skipped: {skipped_reviews}", end="", flush=True)

        offset += BATCH_SIZE
        batch_count += 1

    print(f"\nProcessing complete for {args.relationship}! Total processed: {processed_reviews}, Skipped: {skipped_reviews}")

if __name__ == "__main__":
    main()

# Close the DuckDB connection
con.close()