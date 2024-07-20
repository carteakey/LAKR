import pandas as pd
import duckdb
from huggingface_hub import hf_hub_download
from datasets import load_dataset
import json
from unidecode import unidecode
from tqdm import tqdm

# DuckDB configuration
DB_FILE = "/home/kchauhan/repos/mds-tmu-mrp/db/duckdb/amazon_reviews.duckdb"
con = duckdb.connect(DB_FILE)

def load_all_categories():
    category_filepath = hf_hub_download(
        repo_id="McAuley-Lab/Amazon-Reviews-2023",
        filename="all_categories.txt",
        repo_type="dataset",
    )
    with open(category_filepath, "r") as file:
        all_categories = [_.strip() for _ in file.readlines()]
    return all_categories

def create_table(table_name, columns):
    column_definitions = [
        f"{column} {'JSON' if column in ['details', 'images', 'videos', 'features'] else 'VARCHAR'}"
        for column in columns
    ]
    column_definitions_str = ", ".join(column_definitions)
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(f"CREATE TABLE {table_name} ({column_definitions_str})")

def clean_text(text):
    if isinstance(text, str):
        return unidecode(text)
    return text

def process_item(item, columns):
    processed_item = {}
    for k, v in item.items():
        if k in columns:
            if k in ["details", "images", "videos", "features"]:
                try:
                    parsed_json = json.loads(v)
                    # Clean text in JSON
                    cleaned_json = json.dumps(clean_nested_dict(parsed_json))
                    processed_item[k] = cleaned_json
                except (json.JSONDecodeError, TypeError):
                    processed_item[k] = json.dumps(clean_text(v))
            else:
                processed_item[k] = clean_text(str(v))
    return processed_item

def clean_nested_dict(d):
    if isinstance(d, dict):
        return {k: clean_nested_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [clean_nested_dict(v) for v in d]
    elif isinstance(d, str):
        return clean_text(d)
    else:
        return d

def process_dataset(dataset, table_name, columns, batch_size=1000):
    create_table(table_name, columns)
    total_items = len(dataset)
    
    data_batch = []
    for item in tqdm(dataset, total=total_items, desc=f"Processing {table_name}"):
        data_batch.append(process_item(item, columns))
        if len(data_batch) >= batch_size:
            df = pd.DataFrame(data_batch, columns=columns)
            con.execute(f"INSERT INTO {table_name} SELECT * FROM df")
            data_batch = []

    if data_batch:
        df = pd.DataFrame(data_batch, columns=columns)
        con.execute(f"INSERT INTO {table_name} SELECT * FROM df")

if __name__ == "__main__":
    all_categories = load_all_categories()

    all_categories = ["Video_Games"]

    # Load item metadata
    for category in all_categories:
        columns = [
            "main_category", "title", "average_rating", "rating_number", "features",
            "description", "price", "images", "videos", "store", "categories",
            "details", "parent_asin", "bought_together", "subtitle", "author"
        ]

        print(f"Loading metadata for category: {category}")
        meta_dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{category}",
            split="full",
            trust_remote_code=True,
        )
        process_dataset(meta_dataset, f"raw_meta_{category}", columns)
        print(f"Metadata for {category} loaded")

    # Load reviews
    for category in all_categories:
        columns = [
            "rating", "title", "text", "images", "asin", "parent_asin", "user_id",
            "timestamp", "helpful_vote", "verified_purchase"
        ]

        print(f"Loading reviews for category: {category}")
        review_dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_review_{category}",
            split="full",
            trust_remote_code=True,
        )
        process_dataset(review_dataset, f"raw_review_{category}", columns)
        print(f"Reviews for {category} loaded")

    print("Data insertion complete")

    # Query the database to retrieve all tables
    tables = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()

    # Print the number of rows in each table
    for table in tables:
        table_name = table[0]
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"Table '{table_name}' has {count} rows")

    con.close()