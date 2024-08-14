import os
import duckdb
import json
import argparse
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/home/kchauhan/repos/mds-tmu-mrp/config/env.sh')

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Rate extractions for a specific relationship type.")
parser.add_argument("--relationship", type=str, help="Relationship type to rate (e.g., SIMILAR_TO_BOOK, RELATED_AUTHOR, 'ALL')")
parser.add_argument("--model", type=str, default="llama3", help="Language model to use (gpt-4o-mini, llama3, phi3-mini)")
args = parser.parse_args()

model =  args.model

if model == "gpt-4o-mini":
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
elif model == "llama3":
    llm = ChatOllama(model="llama3.1:8b-instruct-q4_0", temperature=0, format="json")

# Connect to DuckDB
con = duckdb.connect("/home/kchauhan/repos/mds-tmu-mrp/db/duckdb/amazon_reviews.duckdb")

def reset_ratings():
    con.execute(f"UPDATE review_processing_status SET rating = NULL WHERE relationship_type = '{args.relationship}' and status = 'processed'")

# reset_ratings()

def get_kg_extraction_rating_prompt(input):
    rate_kg_extraction = f"""
        You are a top level annotator.
        You are asked to rate the quality of the knowledge graph extraction.
        You are given a set of reviews and the extracted nodes and relationships.
        You are asked to rate the extraction quality.
        Rate the extraction quality on a scale of 1 to 5.
        1 being the worst and 5 being the best.
        Just provide the rating.
        If the source and target books are the same, rate the extraction as 1.
        Do not provide any feedback.
        Do not return any information about the source or target books.
        Based on your knowledge, rate the accuracy of the extracted knowledge graph.
        The Allowed nodes and relationships are:
        - Nodes: {input['nodes']}
        - Relationships: {input['relationships']}
        Here is the extracted knowledge graph json:
        
    """
    return {
        "rate_kg_extraction": rate_kg_extraction,
    }

def rate_extraction(llm, input_data):
    prompt = get_kg_extraction_rating_prompt(input_data)

    messages = [
        (
            "system",
            f"{prompt['rate_kg_extraction']}",
        ),
        (
            "human", 
            str(input_data.get("kg_json"))
        ),
    ]
    response = llm.invoke(messages)
    try:
        print(response)
        if model == "gpt-4o-mini":
            rating = int(response.content)
        else:
            rating = json.loads(response.content).get("rating")
        if rating is None:
            raise ValueError
        return max(1, min(5, rating))  # Ensure rating is between 1 and 5
    except ValueError:
        print(f"Error parsing rating: {response.content}")
        return None

# Get all processed reviews without ratings for the specified relationship
unrated_reviews = con.execute(
    f"""
    SELECT user_id, item_id, json_data
    FROM review_processing_status
    WHERE rating IS NULL AND relationship_type = '{args.relationship}'
"""
).fetchall()

total_reviews = len(unrated_reviews)
processed_reviews = 0

print(f"Total unrated reviews for {args.relationship}: {total_reviews}")

for user_id, item_id, json_data in unrated_reviews:
    data = json.loads(json_data)

    rating_input = {
        "nodes": ["Book"],
        "relationships": [args.relationship],
        "kg_json": json_data,
    }

    rating = rate_extraction(llm, rating_input)

    if rating is not None:
        # Update the rating in the database
        con.execute(
            """
            UPDATE review_processing_status
            SET rating = ?
            WHERE user_id = ? AND item_id = ? AND relationship_type = ?
        """,
            [rating, user_id, item_id, args.relationship],
        )

    processed_reviews += 1
    progress = (processed_reviews / total_reviews) * 100
    print(
        f"\rProgress: {progress:.2f}% ({processed_reviews}/{total_reviews})",
        end="",
        flush=True,
    )

print(f"\nRating process complete for {args.relationship}!")

def get_good_extractions(min_rating=4):
    return con.execute(
        f"""
        SELECT json_data
        FROM review_processing_status
        WHERE rating >= {min_rating} AND relationship_type = '{args.relationship}'
    """
    ).fetchall()

# Example usage:
good_extractions = get_good_extractions()
print(f"Number of good extractions for {args.relationship}: {len(good_extractions)}")

con.close()