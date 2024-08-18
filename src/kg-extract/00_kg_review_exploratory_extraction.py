import os
import json
import time
import duckdb
import argparse
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from dotenv import load_dotenv

#Load environment variables
load_dotenv(os.getenv('BASE_DIR')+'/env.sh')

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process reviews from DuckDB and extract relationships and nodes.")
parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Language model to use (gpt-4o-mini, llama3, phi3-mini)")
parser.add_argument("--batch_size", type=int, default=10, help="Number of reviews to process in each batch")
args = parser.parse_args()

# Constants
OUTPUT_DIR = "output"
DUCKDB_PATH = "/home/kchauhan/repos/mds/LAKR/db/duckdb/amazon_reviews.duckdb"
BATCH_SIZE = args.batch_size

# Initialize DuckDB connection
con = duckdb.connect(DUCKDB_PATH)

# Initialize language model
if args.model == "gpt-4o-mini":
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
elif args.model == "llama3":
    llm = ChatOllama(model="llama3.1:8b-instruct-q4_0", temperature=0)
elif args.model == 'phi3-mini':
    llm = ChatOllama(model='phi3:3.8b-mini-4k-instruct-q6_K', temperature=0)
else:
    raise ValueError("Invalid model specified")

# Initialize LLMGraphTransformer
llm_transformer = LLMGraphTransformer(llm=llm)

def process_review(row):
    text = f"Book: {row[0]}\nReview Title: {row[2]}\nReview Text: {row[3]}"
    documents = [Document(page_content=text)]

    try:
        graph_documents = llm_transformer.convert_to_graph_documents(documents)
    except Exception as e:
        print(f"Error processing review: {e}")
        return None

    nodes = [{"id": node.id, "type": node.type} for node in graph_documents[0].nodes]
    relationships = [
        {
            "source": {"id": rel.source.id, "type": rel.source.type},
            "target": {"id": rel.target.id, "type": rel.target.type},
            "type": rel.type,
        }
        for rel in graph_documents[0].relationships
    ]

    return {
        "user_id": row[5],
        "title": row[0],
        "parent_asin": row[1],
        "review_title": row[2],
        "review_text": row[3],
        "helpful_vote": row[4],
        "nodes": nodes,
        "relationships": relationships,
    }

def save_to_json(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def get_total_reviews():
    result = con.execute("""
        SELECT COUNT(*)
        FROM raw_review_Books a
        INNER JOIN raw_meta_Books b ON a.parent_asin = b.parent_asin
        WHERE CAST(a.helpful_vote AS INT) > 0
        AND LENGTH(a.text) > 100
    """).fetchone()
    return result[0] if result else 0

def main():
    total_reviews = get_total_reviews()
    processed_reviews = 0
    
    print(f"Total reviews to process: {total_reviews}")

    offset = 0
    while True:
        # Fetch a batch of reviews from DuckDB
        batch = con.execute(f"""
            SELECT b.title, b.parent_asin, a.title as review_title, a.text as review_text, a.helpful_vote, a.user_id
            FROM raw_review_Books a
            INNER JOIN raw_meta_Books b ON a.parent_asin = b.parent_asin
            WHERE CAST(a.helpful_vote AS INT) > 0
            AND LENGTH(a.text) > 100
            ORDER BY CAST(a.helpful_vote AS INT) DESC
            LIMIT {BATCH_SIZE} OFFSET {offset}
        """).fetchall()

        if not batch:
            break  # No more reviews to process

        for row in batch:
            processed_data = process_review(row)
            if processed_data:
                filename = f"{OUTPUT_DIR}/{args.model}/{processed_data['user_id']}_{processed_data['parent_asin']}.json"
                save_to_json(processed_data, filename)
                processed_reviews += 1

            # Update progress
            progress = (processed_reviews / total_reviews) * 100
            print(f"\rProgress: {progress:.2f}% ({processed_reviews}/{total_reviews})", end="", flush=True)

        offset += BATCH_SIZE

    print(f"\nProcessing complete! Total processed: {processed_reviews}")

    # Close the DuckDB connection
    con.close()

if __name__ == "__main__":
    main()