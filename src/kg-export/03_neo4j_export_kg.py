import csv
from neo4j import GraphDatabase
from collections import defaultdict
import yaml
from tqdm import tqdm
import logging
import duckdb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# DuckDB configuration
DB_FILE = "/home/kchauhan/repos/mds-tmu-mrp/db/duckdb/amazon_reviews.duckdb"

def process_kg_and_user_data(neo4j_uri, neo4j_username, neo4j_password, output_dir, train_csv, valid_csv, test_csv):
    neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
    
    # Load rating_only_positive data from DuckDB
    logger.info("Loading rating_only_positive data from DuckDB")
    con = duckdb.connect(DB_FILE)
    rating_only_positive = con.execute("SELECT * FROM rating_only_positive").fetchdf()
    rating_only_positive_item_ids = set(rating_only_positive["item_id"])
    rating_only_positive_user_ids = set(rating_only_positive["user_id"])
    logger.info(f"Loaded {len(rating_only_positive_item_ids)} unique item IDs and {len(rating_only_positive_user_ids)} unique user IDs")
    con.close()

    # Step 1: Process Knowledge Graph
    entity_dict = {}
    relation_dict = {}
    item_dict = {}
    entity_counter = 0
    relation_counter = 0

    with neo4j_driver.session() as session:
        # Get all nodes and relationships
        result = session.run("""
            MATCH (a)-[r]->(b)
            RETURN id(a) AS head_id, id(r) AS relation_id, id(b) AS tail_id, 
                   type(r) AS relation_type, a.parent_asin AS head_asin, b.parent_asin AS tail_asin
        """)
        
        with open(f"{output_dir}/kg_final.txt", "w") as kg_file:
            for record in tqdm(result, desc="Processing Knowledge Graph"):
                head_id = record["head_id"]
                relation_id = record["relation_id"]
                tail_id = record["tail_id"]
                relation_type = record["relation_type"]
                head_asin = record["head_asin"]
                tail_asin = record["tail_asin"]
                
                # Only process items that are in rating_only_positive_item_ids
                if (head_asin in rating_only_positive_item_ids or 
                    tail_asin in rating_only_positive_item_ids):
                    # Map entities and relations to integer IDs
                    if head_id not in entity_dict:
                        entity_dict[head_id] = entity_counter
                        if head_asin:
                            item_dict[head_asin] = entity_counter
                        entity_counter += 1
                    if tail_id not in entity_dict:
                        entity_dict[tail_id] = entity_counter
                        if tail_asin:
                            item_dict[tail_asin] = entity_counter
                        entity_counter += 1
                    if relation_type not in relation_dict:
                        relation_dict[relation_type] = relation_counter
                        relation_counter += 1
                    
                    # Write KG triple with remapped IDs
                    kg_file.write(f"{entity_dict[head_id]}\t{relation_dict[relation_type]}\t{entity_dict[tail_id]}\n")

    # Write entity list
    with open(f"{output_dir}/entity_list.txt", "w") as entity_file:
        entity_file.write("org_id\tremap_id\n")
        for org_id, remap_id in entity_dict.items():
            entity_file.write(f"{org_id}\t{remap_id}\n")

    # Write item list
    with open(f"{output_dir}/item_list.txt", "w") as item_file:
        item_file.write("org_id\tremap_id\n")
        for org_id, remap_id in item_dict.items():
            item_file.write(f"{org_id}\t{remap_id}\n")

    # Write relation list
    with open(f"{output_dir}/relation_list.txt", "w") as relation_file:
        relation_file.write("org_id\tremap_id\n")
        for org_id, remap_id in relation_dict.items():
            relation_file.write(f"{org_id}\t{remap_id}\n")

    # Step 2: Process User Interaction Data
    def process_csv(file_path, output_file):
        user_dict = {}
        user_items = defaultdict(set)
        user_counter = len(user_dict)

        with open(file_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in tqdm(csvreader, desc=f"Processing {file_path}"):
                user_id, parent_asin, rating, timestamp, _ = row
                
                # Only process users and items that are in rating_only_positive dataset
                if (user_id in rating_only_positive_user_ids and 
                    parent_asin in rating_only_positive_item_ids):
                    if user_id not in user_dict:
                        user_dict[user_id] = user_counter
                        user_counter += 1
                    
                    if parent_asin in item_dict:
                        item_id = item_dict[parent_asin]
                        user_items[user_dict[user_id]].add(item_id)

        with open(output_file, "w") as out_file:
            for user_remap_id, item_set in user_items.items():
                out_file.write(f"{user_remap_id} {' '.join(map(str, item_set))}\n")

        return user_dict

    # Process train, valid, and test CSV files
    train_users = process_csv(train_csv, f"{output_dir}/train.txt")
    process_csv(valid_csv, f"{output_dir}/valid.txt")
    process_csv(test_csv, f"{output_dir}/test.txt")

    # Write user_list.txt
    with open(f"{output_dir}/user_list.txt", "w") as user_file:
        user_file.write("org_id\tremap_id\n")
        for org_id, remap_id in train_users.items():
            user_file.write(f"{org_id}\t{remap_id}\n")

    neo4j_driver.close()

# Read config from main.yaml
conf = yaml.safe_load(open("/home/kchauhan/repos/mds-tmu-mrp/config/main.yaml"))
neo4j_uri = conf["neo4j"]["uri"]
neo4j_username = conf["neo4j"]["user"]
neo4j_password = conf["neo4j"]["password"]

output_dir = "/home/kchauhan/repos/mds-tmu-mrp/datasets/baseline-kg"
train_csv = "/home/kchauhan/repos/mds-tmu-mrp/datasets/last_out_split/Books.train.csv"
valid_csv = "/home/kchauhan/repos/mds-tmu-mrp/datasets/last_out_split/Books.valid.csv"
test_csv = "/home/kchauhan/repos/mds-tmu-mrp/datasets/last_out_split/Books.test.csv"

process_kg_and_user_data(neo4j_uri, neo4j_username, neo4j_password, output_dir, train_csv, valid_csv, test_csv)