import os
import json
import duckdb
from tqdm import tqdm

def load_json_to_duckdb(json_directory, database_path, table_name):
    # Connect to DuckDB
    con = duckdb.connect(database_path)

    # Create the table if it doesn't exist
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            user_id VARCHAR,
            title VARCHAR,
            parent_asin VARCHAR,
            review_title VARCHAR,
            review_text VARCHAR,
            helpful_vote INTEGER,
            nodes JSON,
            relationships JSON
        )
    """)

    # Get list of JSON files
    json_files = [f for f in os.listdir(json_directory) if f.endswith('.json')]

    # Process each JSON file
    for filename in tqdm(json_files, desc="Processing JSON files"):
        file_path = os.path.join(json_directory, filename)
        
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Insert data into the table
        con.execute(f"""
            INSERT INTO {table_name} (user_id, title, parent_asin, review_title, review_text, helpful_vote, nodes, relationships)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data['user_id'],
            data['title'],
            data['parent_asin'],
            data['review_title'],
            data['review_text'],
            data['helpful_vote'],
            json.dumps(data['nodes']),
            json.dumps(data['relationships'])
        ))

    # Commit changes and close connection
    con.commit()
    con.close()

    print(f"All JSON files have been loaded into the {table_name} table in {database_path}")

# Usage example
if __name__ == "__main__":
    json_directory = "/path/to/your/json/files"
    database_path = "/path/to/your/duckdb/database.duckdb"
    table_name = "processed_reviews"

    load_json_to_duckdb(json_directory, database_path, table_name)