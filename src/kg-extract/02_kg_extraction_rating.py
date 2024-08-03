import os
import duckdb
import json
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage


model = "llama3"  # "gpt-4o-mini" or "llama3"

# Set up environment variables
os.environ["OPENAI_API_KEY"] = (
    "sk-proj-D0yZxT7GzCISYBvPvoubT3BlbkFJVbZeySVcTH50HMV1CqSI"
)

if model == "gpt-4o-mini":
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
elif model == "llama3":
    llm = ChatOllama(model="llama3.1:8b-instruct-q8_0", temperature=0, format="json")

# Connect to DuckDB
con = duckdb.connect("/home/kchauhan/repos/mds-tmu-mrp/db/duckdb/amazon_reviews.duckdb")


def get_kg_extraction_rating_prompt(input):
    rate_kg_extraction = f"""
        You are a top level annotator.
        You are asked to rate the quality of the knowledge graph extraction.
        You are given a set of reviews and the extracted nodes and relationships.
        You are asked to rate the extraction quality.
        Rate the extraction quality on a scale of 1 to 5.
        1 being the worst and 5 being the best.
        Just provide the rating.
        Do not provide any feedback.
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
        rating = json.loads(response.content).get("rating")
        return max(1, min(5, rating))  # Ensure rating is between 1 and 5
    except ValueError:
        print(f"Error parsing rating: {response.content}")
        return None


# Get all processed reviews without ratings
unrated_reviews = con.execute(
    """
    SELECT user_id, item_id, json_data
    FROM review_processing_status
    WHERE rating IS NULL
"""
).fetchall()

total_reviews = len(unrated_reviews)
processed_reviews = 0

print(f"Total unrated reviews: {total_reviews}")

for user_id, item_id, json_data in unrated_reviews:
    data = json.loads(json_data)

    rating_input = {
        "nodes": ["Book"],
        "relationships": ["SIMILAR_TO_BOOK"],
        "kg_json": json_data,
    }

    rating = rate_extraction(llm, rating_input)

    if rating is not None:
        # Update the rating in the database
        con.execute(
            """
            UPDATE review_processing_status
            SET rating = ?
            WHERE user_id = ? AND item_id = ?
        """,
            [rating, user_id, item_id],
        )

        # Update the JSON file with the rating
        data["rating"] = rating
        with open(f"output/{model}/{item_id}.json", "w") as f:
            json.dump(data, f, indent=2)

    processed_reviews += 1
    progress = (processed_reviews / total_reviews) * 100
    print(
        f"\rProgress: {progress:.2f}% ({processed_reviews}/{total_reviews})",
        end="",
        flush=True,
    )

print("\nRating process complete!")


def get_good_extractions(min_rating=4):
    return con.execute(
        f"""
        SELECT json_data
        FROM review_processing_status
        WHERE rating >= {min_rating}
    """
    ).fetchall()


# Example usage:
good_extractions = get_good_extractions()
print(f"Number of good extractions: {len(good_extractions)}")

con.close()
