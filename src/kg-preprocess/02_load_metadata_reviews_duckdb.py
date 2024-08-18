import pandas as pd
import duckdb
from huggingface_hub import hf_hub_download
from datasets import load_dataset
import json
from unidecode import unidecode
from tqdm import tqdm
import logging
from datetime import datetime

# DuckDB configuration
DB_FILE = "/home/kchauhan/repos/mds/LAKR/db/duckdb/amazon_reviews.duckdb"
con = duckdb.connect(DB_FILE)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load rating_only_positive data
logger.info("Loading rating_only_positive data")
rating_only_positive = con.execute("SELECT * FROM rating_only_positive").fetchdf()
rating_only_positive_item_ids = set(rating_only_positive["item_id"])
rating_only_positive_user_ids = set(rating_only_positive["user_id"])
logger.info(f"Loaded {len(rating_only_positive_item_ids)} unique item IDs and {len(rating_only_positive_user_ids)} unique user IDs")

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

def process_dataset(dataset, table_name, columns, is_review_data=False, batch_size=10000):
    create_table(table_name, columns)
    
    start_time = datetime.now()
    total_items = len(dataset)
    items_processed = 0
    items_inserted = 0
    
    processed_data = []
    for item in tqdm(dataset, desc=f"Processing {table_name}"):
        if is_review_data:
            if item["parent_asin"] in rating_only_positive_item_ids and item["user_id"] in rating_only_positive_user_ids:
                processed_item = process_item(item, columns)
                processed_data.append(processed_item)
                items_inserted += 1
        else:
            if item["parent_asin"] in rating_only_positive_item_ids:
                processed_item = process_item(item, columns)
                processed_data.append(processed_item)
                items_inserted += 1
        
        items_processed += 1
        
        if len(processed_data) >= batch_size:
            df = pd.DataFrame(processed_data, columns=columns)
            con.execute(f"INSERT INTO {table_name} SELECT * FROM df")
            processed_data = []
        
        if items_processed % batch_size == 0:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            items_per_second = items_processed / elapsed_time if elapsed_time > 0 else 0
            estimated_time_left = (total_items - items_processed) / items_per_second if items_per_second > 0 else 0
            
            logger.info(f"Processed {items_processed}/{total_items} items. "
                        f"Inserted {items_inserted} items. "
                        f"Speed: {items_per_second:.2f} items/second. "
                        f"Estimated time left: {estimated_time_left/60:.2f} minutes.")

    if processed_data:
        df = pd.DataFrame(processed_data, columns=columns)
        con.execute(f"INSERT INTO {table_name} SELECT * FROM df")

    logger.info(f"Total items processed: {items_processed}")
    logger.info(f"Total items inserted: {items_inserted}")

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
        logger.info(f"Loading metadata for category: {category}")
        meta_dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{category}",
            split="full",
            trust_remote_code=True,
        )
        process_dataset(meta_dataset, f"raw_meta_{category}", metadata_columns)
        logger.info(f"Metadata for {category} loaded")

        logger.info(f"Loading reviews for category: {category}")
        review_dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_review_{category}",
            split="full",
            trust_remote_code=True,
        )
        process_dataset(review_dataset, f"raw_review_{category}", review_columns, is_review_data=True)
        logger.info(f"Reviews for {category} loaded")

    logger.info("Data insertion complete")

    tables = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    for table in tables:
        table_name = table[0]
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        logger.info(f"Table '{table_name}' has {count} rows")

    con.close()