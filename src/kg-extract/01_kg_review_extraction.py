from neo4j import GraphDatabase
import os
import duckdb
import json
import langchain
import time
from langchain_community.graphs import Neo4jGraph
from llm_custom import create_unstructured_prompt
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from typing import List, Optional

BATCH_SIZE = 10  # Number of reviews to process in each batch

allowed_relationships = [
    # "OTHER_BOOKS_RECOMMENDED",
    # "HAS_FEATURES",
    # "DEALS_WITH_CONCEPTS",
    "SIMILAR_TO_BOOK"
    # "MENTIONED_IN_REVIEW",
]

allowed_nodes = [
    "Book"
    #  , "Concept", "Feature"]
]
model = "llama3"  # "gpt-4o-mini" or "llama3"

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "tmu-2024"

graph = Neo4jGraph()

os.environ["OPENAI_API_KEY"] = (
    "sk-proj-D0yZxT7GzCISYBvPvoubT3BlbkFJVbZeySVcTH50HMV1CqSI"
)

if model == "gpt-4o-mini":
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
elif model == "llama3":
    llm = ChatOllama(model="llama3.1:8b-instruct-q8_0", temperature=0)

chat_template = create_unstructured_prompt(allowed_nodes, allowed_relationships)

llm_transformer = LLMGraphTransformer(
    llm=llm, allowed_nodes=allowed_nodes, allowed_relationships=allowed_relationships
)

# Connect to DuckDB
con = duckdb.connect("/home/kchauhan/repos/mds-tmu-mrp/db/duckdb/amazon_reviews.duckdb")

# Create a new table to track review processing status and store JSON data
con.execute(
    """
    CREATE TABLE IF NOT EXISTS review_processing_status (
        user_id VARCHAR,
        item_id VARCHAR,
        status VARCHAR,
        json_data JSON,
        rating INTEGER,
        PRIMARY KEY (user_id, item_id)
    )
"""
)

# Create a new table to track skipped reviews
con.execute(
    """
    CREATE TABLE IF NOT EXISTS skipped_reviews (
        user_id VARCHAR,
        item_id VARCHAR,
        reason VARCHAR,
        review_data JSON,
        PRIMARY KEY (user_id, item_id)
    )
"""
)


# Function to check if a review has been processed
def is_review_processed(user_id, item_id):
    result = con.execute(
        "SELECT status FROM review_processing_status WHERE user_id = ? AND item_id = ?",
        [user_id, item_id],
    ).fetchone()
    return result is not None and result[0] == "processed"


# Function to update review status and store JSON data
def update_review_status(user_id, item_id, json_data):
    con.execute(
        "INSERT INTO review_processing_status (user_id, item_id, status, json_data) VALUES (?, ?, 'processed', ?)",
        [user_id, item_id, json.dumps(json_data)],
    )


# Function to store skipped review
def store_skipped_review(user_id, item_id, reason, review_data):
    try:
        con.execute(
            "INSERT INTO skipped_reviews (user_id, item_id, reason, review_data) VALUES (?, ?, ?, ?)",
            [user_id, item_id, reason, json.dumps(review_data)],
        )
    except Exception as e:
        print(f"Error storing skipped review: {e}")


# Function to get total number of unprocessed reviews
def get_total_unprocessed_reviews():
    result = con.execute(
        """
        SELECT COUNT(*)
        FROM raw_review_Books a
        INNER JOIN raw_meta_Books b ON a.parent_asin = b.parent_asin
        WHERE CAST(a.helpful_vote AS INT) > 0
        AND NOT EXISTS (
            SELECT 1 FROM review_processing_status
            WHERE user_id = a.user_id AND item_id = a.parent_asin
        )
        AND NOT EXISTS (
            SELECT 1 FROM skipped_reviews
            WHERE user_id = a.user_id AND item_id = a.parent_asin
        )
        AND LENGTH(a.text) > 100 -- Filter out short reviews
        
    """
    ).fetchone()
    return result[0] if result else 0

def reset_processing_status():
    con = duckdb.connect(
        "/home/kchauhan/repos/mds-tmu-mrp/db/duckdb/amazon_reviews.duckdb"
    )
    con.execute ("DROP TABLE review_processing_status")
    con.execute("DROP TABLE skipped_reviews")
    con.execute("DELETE FROM review_processing_status")
    con.execute("DELETE FROM skipped_reviews")
    con.close()
    print("Review processing status and skipped reviews tables have been reset.")

# Uncomment the following line to reset the tables
# reset_processing_status()

# Clear all json files in the output directory
for file in os.listdir(f"output/{model}"):
    if file.endswith(".json"):
        os.remove(f"output/{model}/{file}")

# Get total number of unprocessed reviews
total_reviews = get_total_unprocessed_reviews()
processed_reviews = 0
skipped_reviews = 0

print(f"Total unprocessed reviews: {total_reviews}")

# Process reviews in batches
offset = 0
while True:
    # Fetch a batch of unprocessed reviews
    batch = con.execute(
        f"""
        SELECT b.title, b.parent_asin, a.title as review_title, a.text as review_text, a.helpful_vote, a.user_id
        FROM raw_review_Books a
        INNER JOIN raw_meta_Books b ON a.parent_asin = b.parent_asin
        WHERE CAST(a.helpful_vote AS INT) > 0
        AND NOT EXISTS (
            SELECT 1 FROM review_processing_status
            WHERE user_id = a.user_id AND item_id = a.parent_asin
        )
        AND NOT EXISTS (
            SELECT 1 FROM skipped_reviews
            WHERE user_id = a.user_id AND item_id = a.parent_asin
        )
        AND LENGTH(a.text) > 100 -- Filter out short reviews
        ORDER BY CAST(a.helpful_vote AS INT) DESC
        LIMIT {BATCH_SIZE} OFFSET {offset}
    """
    ).fetchall()

    if not batch:
        break  # No more reviews to process

    for row in batch:
        user_id = row[5]
        item_id = row[1]

        if is_review_processed(user_id, item_id):
            continue

        text = f"Book: {row[0]}\nReview Title: {row[2]}\nReview Text: {row[3]}"
        documents = [Document(page_content=text)]

        try:
            graph_documents = llm_transformer.convert_to_graph_documents(documents)
        except Exception as e:
            error_message = (
                f"Error: type validation failed for row: {row}\n, error: {e}\n"
            )
            with open(f"output/{model}/errors.txt", "a") as f:
                f.write(error_message)
            store_skipped_review(
                user_id,
                item_id,
                "Type validation error",
                {
                    "title": row[0],
                    "parent_asin": row[1],
                    "review_title": row[2],
                    "review_text": row[3],
                    "helpful_vote": row[4],
                    "error": str(e),
                },
            )
            skipped_reviews += 1
            processed_reviews += 1
            continue

        nodes = [
            {"id": node.id, "type": node.type} for node in graph_documents[0].nodes
        ]
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
            with open(f"output/{model}/errors.txt", "a") as f:
                f.write(error_message)
            store_skipped_review(
                user_id,
                item_id,
                "No nodes or relationships",
                {
                    "title": row[0],
                    "parent_asin": row[1],
                    "review_title": row[2],
                    "review_text": row[3],
                    "helpful_vote": row[4],
                },
            )
            skipped_reviews += 1
            processed_reviews += 1
            continue

        json_data = {
            "title": row[0],
            "parent_asin": row[1],
            "review_title": row[2],
            "review_text": row[3],
            "helpful_vote": row[4],
            "nodes": nodes,
            "relationships": relationships,
        }

        os.makedirs(f"output/{model}", exist_ok=True)
        with open(f"output/{model}/{time.time()}_{row[1]}.json", "w") as f:
            json.dump(json_data, f, indent=2)

        # Update review status and store JSON data in DuckDB
        update_review_status(user_id, item_id, json_data)

        processed_reviews += 1
        # Update progress
        progress = (processed_reviews / total_reviews) * 100
        print(
            f"\rProgress: {progress:.2f}% ({processed_reviews}/{total_reviews}), Skipped: {skipped_reviews}",
            end="",
            flush=True,
        )

    offset += BATCH_SIZE

print(
    f"\nProcessing complete! Total processed: {processed_reviews}, Skipped: {skipped_reviews}"
)

# Close the DuckDB connection
con.close()



