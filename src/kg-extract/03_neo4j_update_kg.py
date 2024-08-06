import duckdb
from neo4j import GraphDatabase
import json
from fuzzywuzzy import fuzz
from dotenv import load_dotenv
import os
import tqdm
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    # try:
    #     neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    #     with neo4j_driver.session() as session:
    #         session.run("MATCH (s:Book)-[r:SIMILAR_TO_BOOK]->(t:Book) DELETE r")
    #         logging.info("SIMILAR_TO_BOOK relationships have been deleted.")
    # except Exception as e:
    #     logging.error(f"Error deleting SIMILAR_TO_BOOK relationships: {e}")
    # finally:
    #     neo4j_driver.close()

# Uncomment the following line to reset the table
# reset_processing_status()

class Neo4jUpdater:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.existing_books = {}
        self.existing_concepts = []

    def close(self):
        self.driver.close()

    def cache_existing_data(self):
        with self.driver.session() as session:
            self.existing_books = {
                record["title"].lower(): record["asin"] for record in session.run("""
                    MATCH (b:Book)
                    RETURN b.title AS title, b.parent_asin AS asin
                """)
            }
            self.existing_concepts = [
                record["name"].lower() for record in session.run("""
                    MATCH (n:Concept)
                    RETURN n.name AS name
                """)
            ]

    def update_graph(self, json_data):
        with self.driver.session() as session:
            return session.execute_write(self._update_graph_tx, json_data)

    def _update_graph_tx(self, tx, json_data):
        main_book_asin = json_data['parent_asin']
        main_book_title = json_data['title']
        logging.info(f"Processing main book with ASIN: {main_book_asin}, title: {main_book_title}")

        # Check if main book exists
        if main_book_asin not in self.existing_books.values():
            logging.warning(f"Book with ASIN {main_book_asin} does not exist, skipping.")
            return False

        # Create or match Concept nodes with fuzzy matching
        for node in json_data['nodes']:
            node_name = node['id'].lower()
            node_type = node['type']

            if node_type in ['Feature', 'Concept']:
                best_match = max(self.existing_concepts, key=lambda x: fuzz.ratio(x, node_name), default=None)

                if best_match and fuzz.ratio(best_match, node_name) > 85:
                    matched_name = best_match
                    tx.run(f"""
                        MATCH (n:{node_type} {{name: $matched_name}})
                        SET n.aliases = CASE 
                            WHEN n.aliases IS NULL THEN [$new_name] 
                            WHEN NOT $new_name IN n.aliases THEN n.aliases + $new_name 
                            ELSE n.aliases END
                    """, matched_name=matched_name, new_name=node_name)
                    logging.info(f"Matched existing {node_type}: {matched_name} (Alias: {node_name})")
                else:
                    tx.run(f"""
                        CREATE (n:{node_type} {{name: $name}})
                    """, name=node_name)
                    logging.info(f"Created new {node_type}: {node_name}")

        for rel in json_data['relationships']:
            source = rel['source']
            target = rel['target']
            rel_type = rel['type']

            source_id = source['id']
            target_id = target['id']
            source_type = source['type']
            target_type = target['type']

            if rel_type == 'SIMILAR_TO_BOOK':
                if main_book_title.lower() == target_id.lower():
                    logging.info(f"Skipping self-reference for book: {main_book_title}")
                    continue

                target_asin = self.existing_books.get(target_id.lower())
                if target_asin:
                    tx.run("""
                        MATCH (s:Book {parent_asin: $main_asin})
                        MATCH (t:Book {parent_asin: $target_asin})
                        MERGE (s)-[r:SIMILAR_TO_BOOK]->(t)
                    """, main_asin=main_book_asin, target_asin=target_asin)
                    logging.info(f"Exact match found for book title: {target_id}")
                else:
                    best_match = max(self.existing_books.keys(), key=lambda x: fuzz.ratio(x, target_id.lower()))
                    match_score = fuzz.ratio(best_match, target_id.lower())
                    if match_score > 90:
                        target_asin = self.existing_books[best_match]
                        tx.run("""
                            MATCH (s:Book {parent_asin: $main_asin})
                            MATCH (t:Book {parent_asin: $target_asin})
                            MERGE (s)-[r:SIMILAR_TO_BOOK]->(t)
                        """, main_asin=main_book_asin, target_asin=target_asin)
                        logging.info(f"Fuzzy match found for book title: {target_id} -> {best_match} with score {match_score}")
                    else:
                        logging.warning(f"No close match found for book title: {target_id}")
                        return False

            elif rel_type == 'DEALS_WITH_CONCEPTS':
                best_match = max(self.existing_concepts, key=lambda x: fuzz.ratio(x, target_id.lower()), default=None)

                if best_match and fuzz.ratio(best_match, target_id.lower()) > 85:
                    matched_name = best_match
                    tx.run("""
                        MATCH (s:Book {parent_asin: $main_asin})
                        MATCH (t:Concept {name: $matched_name})
                        MERGE (s)-[r:DEALS_WITH_CONCEPTS]->(t)
                    """, main_asin=main_book_asin, matched_name=matched_name)
                    logging.info(f"Relationship created: {main_book_asin} -[DEALS_WITH_CONCEPTS]-> {matched_name} (Matched: {target_id})")
                else:
                    logging.warning(f"No matching Concept found for: {target_id}")
                    return False

            else:
                tx.run(f"""
                    MATCH (s:{source_type} {{name: $source_name}})
                    MATCH (t:{target_type} {{name: $target_name}})
                    MERGE (s)-[r:{rel_type}]->(t)
                """, source_name=source_id, target_name=target_id)
                logging.info(f"Relationship created: {source_id} -[{rel_type}]-> {target_id}")

        return True

    def cleanup_concepts(self):
        with self.driver.session() as session:
            session.write_transaction(self._cleanup_concepts)
    
    @staticmethod
    def _cleanup_concepts(tx):
        # Remove concepts that are linked to one book or fewer
        result = tx.run("""
            MATCH (c:Concept)
            WHERE COUNT { (c)<-[:DEALS_WITH_CONCEPTS]-() } <= 1
            WITH c, c.name AS name
            DETACH DELETE c
            RETURN count(c) as removed_concepts, collect(name) as removed_names

        """)
        
        cleanup_stats = result.single()
        removed_count = cleanup_stats['removed_concepts']
        removed_names = cleanup_stats['removed_names']

        logging.info(f"Cleaned up {removed_count} concepts that were linked to one book or fewer.")
        logging.debug(f"Removed concepts: {', '.join(removed_names)}")

      # Optionally, you could also log concepts with low connections (e.g., 2-3) for review
        low_connection_result = tx.run("""
            MATCH (c:Concept)
            WHERE 1 < COUNT { (c)<-[:DEALS_WITH_CONCEPTS]-() } <= 3
            RETURN c.name as name, COUNT { (c)<-[:DEALS_WITH_CONCEPTS]-() } as links
            ORDER BY links
        """)
                
        for record in low_connection_result:
            logging.info(f"Low-connection concept: {record['name']} (links: {record['links']})")

def process_duckdb_records():
    updater = Neo4jUpdater(NEO4J_URI, (NEO4J_USER, NEO4J_PASSWORD))
    con = duckdb.connect(DUCKDB_PATH)

    try:
        updater.cache_existing_data()
        records = con.execute("""
            SELECT user_id, item_id, json_data
            FROM review_processing_status
            WHERE status = 'processed'
            AND rating >= 4
        """).fetchall()

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_record, updater, con, user_id, item_id, json_data) for user_id, item_id, json_data in records]

            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                future.result()

    except Exception as e:
        logging.error(f"Error during update process: {e}")
    finally:
        updater.cleanup_concepts()
        updater.close()
        con.close()
        
    logging.info("Update process completed.")

def process_record(updater, con, user_id, item_id, json_data):
    try:
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
        logging.error(f"Error processing record for user {user_id} and item {item_id}: {e}")

if __name__ == "__main__":
    process_duckdb_records()
