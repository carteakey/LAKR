import logging
import os

from dotenv import load_dotenv

from ..utils.db.neo4j import initialize_neo4j_driver

# Load environment variables
load_dotenv(os.getenv('BASE_DIR')+'/env.sh')

neo4j_driver = initialize_neo4j_driver()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class Neo4jCleaner:
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver

    def close(self):
        self.driver.close()

    def cleanup_concepts_and_series(self):
        with self.driver.session() as session:
            session.write_transaction(self._cleanup_concepts)
            session.write_transaction(self._cleanup_series)

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
        
        # Log concepts with low connections (e.g., linked to 2-3 books)
        low_connection_result = tx.run("""
            MATCH (c:Concept)
            WHERE 1 < COUNT { (c)<-[:DEALS_WITH_CONCEPTS]-() } <= 3
            RETURN c.name as name, COUNT { (c)<-[:DEALS_WITH_CONCEPTS]-() } as links
            ORDER BY links
        """)
                
        for record in low_connection_result:
            logging.info(f"Low-connection concept: {record['name']} (links: {record['links']})")

    @staticmethod
    def _cleanup_series(tx):
        # Remove series that are linked to fewer than two books
        result = tx.run("""
            MATCH (s:Series)
            WHERE COUNT { (s)<-[:PART_OF_SERIES]-() } < 2
            WITH s, s.name AS name
            DETACH DELETE s
            RETURN count(s) as removed_series, collect(name) as removed_names
        """)

        cleanup_stats = result.single()
        removed_count = cleanup_stats['removed_series']
        removed_names = cleanup_stats['removed_names']

        logging.info(f"Cleaned up {removed_count} series that were linked to fewer than two books.")
        logging.debug(f"Removed series: {', '.join(removed_names)}")

        # Log series with low connections (e.g., linked to 2-3 books)
        low_connection_result = tx.run("""
            MATCH (s:Series)
            WHERE 1 < COUNT { (s)<-[:PART_OF_SERIES]-() } <= 3
            RETURN s.name as name, COUNT { (s)<-[:PART_OF_SERIES]-() } as links
            ORDER BY links
        """)
                
        for record in low_connection_result:
            logging.info(f"Low-connection series: {record['name']} (links: {record['links']})")


    def delete_relationship(self, rel_type):
        with self.driver.session() as session:
            try:
                result = session.run(f"""
                    MATCH (s)-[r:{rel_type}]->(t)
                    DELETE r
                    RETURN count(r) as deleted_count
                """)
                deleted_count = result.single()["deleted_count"]
                logging.info(f"{deleted_count} '{rel_type}' relationships have been deleted.")
            except Exception as e:
                logging.error(f"Error deleting '{rel_type}' relationships: {e}")
    
cleaner = Neo4jCleaner(neo4j_driver)

try:
    cleaner.cleanup_concepts_and_series()
except Exception as e:
    logging.error(f"Error during cleanup: {e}")
finally:
    cleaner.close()
    logging.info("Cleanup process completed.")

