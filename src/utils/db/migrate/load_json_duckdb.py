import os
import json
import duckdb
from tqdm import tqdm
import argparse

def load_json_to_duckdb(json_directory, database_path, relationship):
    # Connect to DuckDB
    con = duckdb.connect(database_path)

    # Ensure the review_processing_status table exists
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

    # Get list of JSON files
    json_files = [f for f in os.listdir(json_directory) if f.endswith('.json')]

    # Process each JSON file
    for filename in tqdm(json_files, desc=f"Processing {relationship} JSON files"):
        file_path = os.path.join(json_directory, filename)
        
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Insert or update data in the table
        con.execute("""
            INSERT OR REPLACE INTO review_processing_status 
            (user_id, item_id, relationship_type, status, json_data)
            VALUES (?, ?, ?, 'processed', ?)
        """, (
            data['user_id'],
            data['parent_asin'],
            relationship,
            json.dumps(data),
              # Use helpful_vote as rating, or None if not present
        ))

    # Commit changes and close connection
    con.commit()
    con.close()

    print(f"All JSON files for {relationship} have been loaded into the review_processing_status table in {database_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load JSON files for a specific relationship into DuckDB review_processing_status table")
    parser.add_argument("--relationship", type=str, required=True, help="Relationship type to process (e.g., RELATED_TO_AUTHOR)")
    parser.add_argument("--model", type=str, required=True, help="Language model used (e.g., gpt-4o-mini, llama3, phi3-mini)")
    parser.add_argument("--db_path", type=str, required=True, help="Path to the DuckDB database")
    args = parser.parse_args()

    # Construct the JSON directory path based on the model and relationship
    json_directory = f"output/{args.model}/{args.relationship}"
    
    # Check if the directory exists
    if not os.path.exists(json_directory):
        print(f"Error: Directory {json_directory} does not exist.")
        exit(1)

    load_json_to_duckdb(json_directory, args.db_path, args.relationship)