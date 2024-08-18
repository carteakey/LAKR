import gc
import json
import logging
from datetime import datetime

import duckdb
import polars as pl
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from neo4j import GraphDatabase
from tqdm import tqdm
from unidecode import unidecode

# Neo4j configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "tmu-2024"

# DuckDB configuration
DB_FILE = "/home/kchauhan/repos/mds/LAKR/db/duckdb/amazon_reviews.duckdb"
con = duckdb.connect(DB_FILE)

# Load rating_only_positive data once
rating_only_positive = con.execute("SELECT * FROM rating_only_positive").fetchdf()
rating_only_positive_item_ids = set(rating_only_positive["item_id"].values)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_all_categories():
    category_filepath = hf_hub_download(
        repo_id="McAuley-Lab/Amazon-Reviews-2023",
        filename="all_categories.txt",
        repo_type="dataset",
    )
    with open(category_filepath, "r") as file:
        all_categories = [_.strip() for _ in file.readlines()]
    return all_categories

def clean_text(text):
    if isinstance(text, str):
        return unidecode(text)
    return text

def clean_nested_dict(d):
    if isinstance(d, dict):
        return {k: clean_nested_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [clean_nested_dict(v) for v in d]
    elif isinstance(d, str):
        return clean_text(d)
    else:
        return d

def parse_json_field(field_value):
    if isinstance(field_value, str):
        field_value = field_value.replace("'", '"')
        try:
            parsed_json = json.loads(field_value)
            return clean_nested_dict(parsed_json)
        except (json.JSONDecodeError, TypeError):
            return clean_text(field_value)
    return field_value

def process_item(item, columns):
    processed_item = {}
    for k, v in item.items():
        if k in columns:
            if k in ["details", "images", "videos", "features", "author", "categories"]:
                processed_item[k] = parse_json_field(v)
            else:
                processed_item[k] = clean_text(str(v))
    return processed_item

def create_constraints_and_indexes(driver):
    with driver.session() as session:
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (b:Book) REQUIRE b.title IS UNIQUE")
        session.run("CREATE INDEX IF NOT EXISTS FOR (b:Book) ON (b.parent_asin)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (a:Author) ON (a.name)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (p:Publisher) ON (p.name)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (c:Category) ON (c.name)")

def create_graph_batch(tx, records):
    query = """
    UNWIND $records AS record
    MERGE (b:Book {title: record.book.title})
    SET b += record.book
    WITH b, record
    FOREACH (authorName IN CASE WHEN record.author IS NOT NULL THEN [record.author] ELSE [] END |
        MERGE (a:Author {name: authorName})
        MERGE (b)-[:WRITTEN_BY]->(a)
    )
    FOREACH (publisherName IN CASE WHEN record.publisher IS NOT NULL THEN [record.publisher] ELSE [] END |
        MERGE (p:Publisher {name: publisherName})
        MERGE (b)-[:PUBLISHED_BY]->(p)
    )
    FOREACH (categoryName IN record.categories |
        MERGE (c:Category {name: categoryName})
        MERGE (b)-[:CATEGORIZED_UNDER]->(c)
    )
    """
    tx.run(query, records=records)

def process_batch(batch, columns, rating_only_positive_item_ids):
    records = []
    for item in batch:
        if item['parent_asin'] in rating_only_positive_item_ids:
            processed_item = process_item(item, columns)
            book = {
                'title': processed_item.get('title'),
                'parent_asin': processed_item.get('parent_asin')
            }
            
            author = processed_item['author']['name'] if isinstance(processed_item.get('author'), dict) and 'name' in processed_item['author'] else None
            
            publisher = None
            if isinstance(processed_item.get('details'), dict) and 'Publisher' in processed_item['details']:
                publisher = processed_item['details']['Publisher'].split(';')[0].strip()
            
            categories = processed_item.get('categories', [])
            
            records.append({
                'book': book,
                'author': author,
                'publisher': publisher,
                'categories': categories
            })
    return records

def process_dataset(dataset, driver, columns, rating_only_positive_item_ids, batch_size=1000):
    total_items = len(dataset)
    total_batches = (total_items + batch_size - 1) // batch_size
    
    start_time = datetime.now()
    items_processed = 0
    
    with driver.session() as session:
        for i in tqdm(range(0, total_items, batch_size), total=total_batches, desc="Processing batches"):
            batch = [dataset[j] for j in range(i, min(i + batch_size, total_items))]
            records = process_batch(batch, columns, rating_only_positive_item_ids)
            session.execute_write(create_graph_batch, records)
            
            items_processed += len(batch)
            elapsed_time = (datetime.now() - start_time).total_seconds()
            items_per_second = items_processed / elapsed_time if elapsed_time > 0 else 0
            estimated_time_left = (total_items - items_processed) / items_per_second if items_per_second > 0 else 0
            
            logger.info(f"Processed {items_processed}/{total_items} items. "
                        f"Speed: {items_per_second:.2f} items/second. "
                        f"Estimated time left: {estimated_time_left/60:.2f} minutes.")
            
            gc.collect()

if __name__ == "__main__":
    all_categories = ["Books"]  # You can expand this list as needed

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Create constraints and indexes
    create_constraints_and_indexes(driver)

    # Load item metadata
    for category in all_categories:
        columns = [
            "title",
            "average_rating",
            "rating_number",
            "description",
            "price",
            "store",
            "categories",
            "details",
            "parent_asin",
            "subtitle",
            "author",
        ]

        logger.info(f"Loading metadata for category: {category}")
        meta_dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{category}",
            split="full",
            trust_remote_code=True,
        )
        process_dataset(meta_dataset, driver, columns, rating_only_positive_item_ids)
        logger.info(f"Metadata for {category} loaded")

    # Close the Neo4j connection
    driver.close()

    logger.info("Data insertion complete")