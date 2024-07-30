import pandas as pd
import duckdb
from huggingface_hub import hf_hub_download
from datasets import load_dataset
import json
from unidecode import unidecode
from tqdm import tqdm
import pyarrow as pa

# DuckDB configuration
DB_FILE = "/home/kchauhan/repos/mds-tmu-mrp/db/duckdb/amazon_reviews.duckdb"
con = duckdb.connect(DB_FILE)

# Load rating_only_positive data once
rating_only_positive = con.execute("SELECT * FROM rating_only_positive").fetchdf()
rating_only_positive_item_ids = set(rating_only_positive["item_id"].values)

def load_all_categories():
    category_filepath = hf_hub_download(
        repo_id="McAuley-Lab/Amazon-Reviews-2023",
        filename="all_categories.txt",
        repo_type="dataset",
    )
    with open(category_filepath, "r") as file:
        return [_.strip() for _ in file.readlines()]

def create_table(table_name, columns):
    column_definitions = [
        f"{column} {'JSON' if column in ['details', 'images', 'videos', 'features'] else 'VARCHAR'}"
        for column in columns
    ]
    column_definitions_str = ", ".join(column_definitions)
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(f"CREATE TABLE {table_name} ({column_definitions_str})")

def clean_text(text):
    return unidecode(text) if isinstance(text, str) else text

def clean_nested_dict(d):
    if isinstance(d, dict):
        return {k: clean_nested_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [clean_nested_dict(v) for v in d]
    elif isinstance(d, str):
        return clean_text(d)
    else:
        return d

def process_item(item, columns):
    processed_item = {}
    for k, v in item.items():
        if k in columns:
            if k in ["details", "images", "videos", "features"]:
                try:
                    parsed_json = json.loads(v)
                    cleaned_json = json.dumps(clean_nested_dict(parsed_json))
                    processed_item[k] = cleaned_json
                except (json.JSONDecodeError, TypeError):
                    processed_item[k] = json.dumps(clean_text(v))
            else:
                processed_item[k] = clean_text(str(v))
    return processed_item

def process_dataset(dataset, table_name, columns, batch_size=10000):
    create_table(table_name, columns)
    
    processed_data = []
    for item in tqdm(dataset, desc=f"Processing {table_name}"):
        if item["parent_asin"] in rating_only_positive_item_ids:
            processed_item = process_item(item, columns)
            processed_data.append(processed_item)
            
            if len(processed_data) >= batch_size:
                df = pd.DataFrame(processed_data, columns=columns)
                con.execute(f"INSERT INTO {table_name} SELECT * FROM df")
                processed_data = []

    if processed_data:
        df = pd.DataFrame(processed_data, columns=columns)
        con.execute(f"INSERT INTO {table_name} SELECT * FROM df")

if __name__ == "__main__":
    all_categories = ["Books"]  # Load other categories as needed

    metadata_columns = [
        "main_category", "title", "average_rating", "rating_number", "features",
        "description", "price", "images", "videos", "store", "categories",
        "details", "parent_asin", "bought_together", "subtitle", "author",
    ]

    review_columns = [
        "rating", "title", "text", "images", "asin", "parent_asin", "user_id",
        "timestamp", "helpful_vote", "verified_purchase",
    ]

    for category in all_categories:
        print(f"Loading metadata for category: {category}")
        meta_dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{category}",
            split="full",
            trust_remote_code=True,
        )
        process_dataset(meta_dataset, f"raw_meta_{category}", metadata_columns)
        print(f"Metadata for {category} loaded")

        print(f"Loading reviews for category: {category}")
        review_dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_review_{category}",
            split="full",
            trust_remote_code=True,
        )
        process_dataset(review_dataset, f"raw_review_{category}", review_columns)
        print(f"Reviews for {category} loaded")

    print("Data insertion complete")

    tables = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    for table in tables:
        table_name = table[0]
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"Table '{table_name}' has {count} rows")

    con.close()