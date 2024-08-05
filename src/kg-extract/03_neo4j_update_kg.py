import duckdb
from neo4j import GraphDatabase
import json
from fuzzywuzzy import fuzz
from dotenv import load_dotenv
import os
import tqdm
import logging
from datetime import datetime

# Load environment variables
load_dotenv('/home/kchauhan/repos/mds-tmu-mrp/config/env.sh')

# Initialize Neo4j driver
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
DUCKDB_PATH = os.getenv("DUCKDB_PATH")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def reset_processing_status():
    con = duckdb.connect(DUCKDB_PATH)
    try:
        con.execute("UPDATE review_processing_status SET status = 'processed' WHERE status = 'KG_updated'")
        logging.info("Review processing status has been reset for KG-updated records.")
    except Exception as e:
        logging.error(f"Error resetting processing status: {e}")
    finally:
        con.close()

    try:
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with neo4j_driver.session() as session:
            session.run("MATCH (s:Book)-[r:SIMILAR_TO_BOOK]->(t:Book) DELETE r")
            logging.info("SIMILAR_TO_BOOK relationships have been deleted.")
    except Exception as e:
        logging.error(f"Error deleting SIMILAR_TO_BOOK relationships: {e}")
    finally:
        neo4j_driver.close()

# Uncomment the following line to reset the table
reset_processing_status()

class Neo4jUpdater:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def close(self):
        self.driver.close()

    def update_graph(self, json_data):
        with self.driver.session() as session:
            return session.execute_write(self._update_graph_tx, json_data)

    @staticmethod
    def _update_graph_tx(tx, json_data):
        main_book_asin = json_data['parent_asin']
        main_book_title = json_data['title']
        logging.info(f"Processing main book with ASIN: {main_book_asin}, title: {main_book_title}")

        main_book = tx.run("""
            MATCH (b:Book {parent_asin: $asin})
            RETURN b
        """, asin=main_book_asin).single()

        if main_book is None:
            logging.warning(f"Book with ASIN {main_book_asin} does not exist, skipping.")
            return False

        for node in json_data['nodes']:
            node_name = node['id']
            node_type = node['type']
            if node_type in ['Feature', 'Concept']:
                tx.run(f"""
                    MERGE (n:{node_type} {{name: $name}})
                """, name=node_name)
                logging.info(f"Node created or matched: {node_type} - {node_name}")

        for rel in json_data['relationships']:
            source = rel['source']
            target = rel['target']
            rel_type = rel['type']

            source_id = source['id']
            target_id = target['id']
            target_type = target['type']

            if rel_type == 'SIMILAR_TO_BOOK':
                exact_match = tx.run("""
                    MATCH (b:Book)
                    WHERE toLower(b.title) = toLower($title)
                    RETURN b.parent_asin AS asin
                """, title=target_id).single()

                if exact_match:
                    tx.run("""
                        MATCH (s:Book {parent_asin: $main_asin})
                        MATCH (t:Book {parent_asin: $target_asin})
                        MERGE (s)-[r:SIMILAR_TO_BOOK]->(t)
                    """, main_asin=main_book_asin, target_asin=exact_match['asin'])
                    logging.info(f"Exact match found for book title: {target_id}")
                else:
                    result = tx.run("""
                        MATCH (b:Book)
                        RETURN b.title AS title, b.parent_asin AS asin
                    """)
                    books = [(record["title"], record["asin"]) for record in result]

                    best_match = max(books, key=lambda x: fuzz.ratio(x[0].lower(), target_id.lower()))
                    match_score = fuzz.ratio(best_match[0].lower(), target_id.lower())
                    if match_score > 90:  # Set a threshold
                        tx.run("""
                            MATCH (s:Book {parent_asin: $main_asin})
                            MATCH (t:Book {parent_asin: $target_asin})
                            MERGE (s)-[r:SIMILAR_TO_BOOK]->(t)
                        """, main_asin=main_book_asin, target_asin=best_match[1])
                        logging.info(f"Fuzzy match found for book title: {target_id} -> {best_match[0]} with score {match_score}")
                    else:
                        logging.warning(f"No close match found for book title: {target_id}")
            else:
                tx.run(f"""
                    MATCH (s:Book {{parent_asin: $main_asin}})
                    MATCH (t:{target_type} {{name: $target_name}})
                    MERGE (s)-[r:{rel_type}]->(t)
                """, main_asin=main_book_asin, target_name=target_id)
                logging.info(f"Relationship created: {main_book_asin} -[{rel_type}]-> {target_id}")

        return True

def process_duckdb_records():
    updater = Neo4jUpdater(NEO4J_URI, (NEO4J_USER, NEO4J_PASSWORD))
    con = duckdb.connect(DUCKDB_PATH)

    try:
        records = con.execute("""
            SELECT user_id, item_id, json_data
            FROM review_processing_status
            WHERE status = 'processed'
            AND rating >= 4
        """).fetchall()

        for user_id, item_id, json_data in tqdm.tqdm(records):
            logging.info(f"Processing review for user {user_id} and item {item_id}")
            json_content = json.loads(json_data)

            if updater.update_graph(json_content):
                con.execute("""
                    UPDATE review_processing_status
                    SET status = 'KG_updated'
                    WHERE user_id = ? AND item_id = ?
                """, [user_id, item_id])
                logging.info(f"Successfully updated KG for user {user_id} and item {item_id}")
            else:
                logging.warning(f"Failed to update KG for user {user_id} and item {item_id}")

    except Exception as e:
        logging.error(f"Error during update process: {e}")
    finally:
        updater.close()
        con.close()

    logging.info("Update process completed.")

if __name__ == "__main__":
    process_duckdb_records()
