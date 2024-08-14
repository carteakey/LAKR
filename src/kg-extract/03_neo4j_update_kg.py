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

# Uncomment the following line to reset the table
# reset_processing_status()

class Neo4jUpdater:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.existing_books = {}
        self.existing_concepts = []
        self.existing_series = []

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
                record["name"] for record in session.run("""
                    MATCH (n:Concept)
                    RETURN n.name AS name
                """)
            ]
            self.existing_series = [
                record["name"] for record in session.run("""
                    MATCH (s:Series)
                    RETURN s.name AS name
                """)
            ]
            logging.info(f"Cached {len(self.existing_books)} books, {len(self.existing_concepts)} concepts, and {len(self.existing_series)} series")

    def update_graph(self, json_data):
        with self.driver.session() as session:
            return session.execute_write(self._update_graph_tx, json_data)

    def _update_graph_tx(self, tx, json_data):
        try:
            main_book_asin = json_data['parent_asin']
            main_book_title = json_data['title']
            logging.info(f"Processing main book with ASIN: {main_book_asin}, title: {main_book_title}")
            # Check if main book exists
            if main_book_asin not in self.existing_books.values():
                logging.warning(f"Book with ASIN {main_book_asin} does not exist, skipping.")
                return False

            # Create or match nodes (Books, Concepts, Series)
            for node in json_data['nodes']:
                node_name = node['id']
                node_type = node['type']

                if node_type in ['Series']:
                    self._create_or_match_series(tx, node_type, node_name)
                elif node_type in ['Feature', 'Concept']:
                    self._create_or_match_concept(tx, node_type, node_name)

            # Create relationships
            for rel in json_data['relationships']:
                source = rel['source']
                target = rel['target']
                rel_type = rel['type']

                source_id = source['id']
                target_id = target['id']
                source_type = source['type']
                target_type = target['type']

                if rel_type == 'SIMILAR_TO_BOOK':
                    self._create_similar_to_book_relationship(tx, main_book_asin, target_id)
                elif rel_type == 'DEALS_WITH_CONCEPTS':
                    self._create_deals_with_concepts_relationship(tx, main_book_asin, target_id)
                elif rel_type == 'PART_OF_SERIES':
                    self._create_part_of_series_relationship(tx, source_id, target_id)
                else:
                    self._create_generic_relationship(tx, source_type, source_id, target_type, target_id, rel_type)

        except Exception as e:
            logging.error(f"Error processing graph update: {e}")
            return False

        return True
        
    def _create_or_match_series(self, tx, node_type, node_name):
        # if node_type == 'Book':
            # if node_name.lower() not in self.existing_books:
            #     tx.run("""
            #         MERGE (n:Book {title: $name, parent_asin: $asin})
            #     """, name=node_name, asin=node_name)  # Using name as ASIN for simplicity; adjust if you have actual ASIN
            #     logging.info(f"Created new Book: {node_name}")
            #     self.existing_books[node_name.lower()] = node_name  # Using title as ASIN for simplicity
        if node_type == 'Series':
            best_match = self._find_best_match(node_name, self.existing_series)
            if best_match:
                logging.info(f"Matched existing Series: {best_match}")
            else:
                tx.run(f"""
                    CREATE (n:Series {{name: $name}})
                """, name=node_name)
                logging.info(f"Created new Series: {node_name}")
                self.existing_series.append(node_name)

    def _create_or_match_concept(self, tx, node_type, node_name):
        best_match = self._find_best_match(node_name, self.existing_concepts)
        if best_match:
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
            self.existing_concepts.append(node_name)

    def _create_similar_to_book_relationship(self, tx, main_book_asin, target_id):
        if main_book_asin == target_id:
            logging.info(f"Skipping self-reference for book: {main_book_asin}")
            return

        target_asin = self._find_book_asin(target_id)
        if target_asin:
            tx.run("""
                MATCH (s:Book {parent_asin: $main_asin})
                MATCH (t:Book {parent_asin: $target_asin})
                MERGE (s)-[r:SIMILAR_TO_BOOK]->(t)
            """, main_asin=main_book_asin, target_asin=target_asin)
            logging.info(f"Created SIMILAR_TO_BOOK relationship: {main_book_asin} -> {target_asin}")
        else:
            logging.warning(f"No match found for book title: {target_id}")

    def _create_deals_with_concepts_relationship(self, tx, main_book_asin, target_id):
        best_match = self._find_best_match(target_id, self.existing_concepts)
        if best_match:
            tx.run("""
                MATCH (s:Book {parent_asin: $main_asin})
                MATCH (t:Concept {name: $matched_name})
                MERGE (s)-[r:DEALS_WITH_CONCEPTS]->(t)
            """, main_asin=main_book_asin, matched_name=best_match)
            logging.info(f"Created DEALS_WITH_CONCEPTS relationship: {main_book_asin} -> {best_match}")
        else:
            logging.warning(f"No matching Concept found for: {target_id}")

    def _create_part_of_series_relationship(self, tx, source_id, target_id):
        book_asin = self._find_book_asin(source_id)
        series_name = self._find_best_match(target_id, self.existing_series)
        
        if book_asin and series_name:
            tx.run("""
                MATCH (b:Book {parent_asin: $book_asin})
                MATCH (s:Series {name: $series_name})
                MERGE (b)-[r:PART_OF_SERIES]->(s)
            """, book_asin=book_asin, series_name=series_name)
            logging.info(f"Created PART_OF_SERIES relationship: {book_asin} -> {series_name}")
        else:
            if not book_asin:
                logging.warning(f"Book not found: {source_id}")
            if not series_name:
                logging.warning(f"Series not found: {target_id}")
            logging.warning(f"Could not create PART_OF_SERIES relationship: Book '{source_id}' or Series '{target_id}' not found")
            
    def _create_generic_relationship(self, tx, source_type, source_id, target_type, target_id, rel_type):
        tx.run(f"""
            MATCH (s:{source_type} {{name: $source_name}})
            MATCH (t:{target_type} {{name: $target_name}})
            MERGE (s)-[r:{rel_type}]->(t)
        """, source_name=source_id, target_name=target_id)
        logging.info(f"Created {rel_type} relationship: {source_id} -> {target_id}")

    def _find_best_match(self, name, existing_names):
        best_match = None
        highest_ratio = 0
        for existing_name in existing_names:
            ratio = fuzz.ratio(existing_name.lower(), name.lower())
            if ratio > highest_ratio and ratio > 85:
                highest_ratio = ratio
                best_match = existing_name
        return best_match

    def _find_book_asin(self, book_title):
        if book_title.lower() in self.existing_books:
            return self.existing_books[book_title.lower()]
        best_match = max(self.existing_books.keys(), key=lambda x: fuzz.ratio(x, book_title.lower()))
        match_score = fuzz.ratio(best_match, book_title.lower())
        if match_score > 85:
            return self.existing_books[best_match]
        return None

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
