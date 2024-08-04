import os
import duckdb
import json
import time
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from llm_custom import create_unstructured_prompt
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from langchain_community.callbacks import get_openai_callback
import signal
from dotenv import load_dotenv

# ... (previous code remains the same)

# Function to process a single review
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
        with open(f"output/{model}/errors.txt", "a") as f:
            f.write(error_message)
        store_skipped_review(
            user_id,
            item_id,
            "Type validation error",
            {
                "user_id": user_id,  # Added user_id here
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
        with open(f"output/{model}/errors.txt", "a") as f:
            f.write(error_message)
        store_skipped_review(
            user_id,
            item_id,
            "No nodes or relationships",
            {
                "user_id": user_id,  # Added user_id here
                "title": row[0],
                "parent_asin": row[1],
                "review_title": row[2],
                "review_text": row[3],
                "helpful_vote": row[4],
            },
        )
        return "skipped"

    json_data = {
        "user_id": user_id,  # Added user_id here
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

    update_review_status(user_id, item_id, json_data)
    return "processed"

# ... (rest of the code remains the same)