from neo4j import GraphDatabase
import json
import os

# Neo4j connection details
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "tmu-2024")

class Neo4jUpdater:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def close(self):
        self.driver.close()

    def update_graph(self, json_data):
        with self.driver.session() as session:
            session.execute_write(self._update_graph_tx, json_data)

    @staticmethod
    def _update_graph_tx(tx, json_data):
        # Ensure the main book exists
        main_book_asin = json_data['parent_asin']
        tx.run("""
            MERGE (b:Book {parent_asin: $asin})
            SET b.title = $title
        """, asin=main_book_asin, title=json_data['title'])

        # Create or match other nodes (Feature, Concept)
        for node in json_data['nodes']:
            node_data = eval(node)
            if node_data['type'] in ['Feature', 'Concept']:
                tx.run("""
                    MERGE (n:%s {name: $name})
                """ % node_data['type'], name=node_data['id'])

        # Create relationships
        for rel in json_data['relationships']:
            rel_data = eval(rel)
            source = eval(rel_data['source'])
            target = eval(rel_data['target'])
            rel_type = rel_data['type']

            if rel_type == 'SIMILAR_TO_BOOK':
                # For similar books, we need to search by name
                tx.run("""
                    MATCH (s:Book {parent_asin: $main_asin})
                    MATCH (t:Book)
                    WHERE t.title CONTAINS $target_name OR $target_name CONTAINS t.title
                    MERGE (s)-[r:SIMILAR_TO_BOOK]->(t)
                """, main_asin=main_book_asin, target_name=target['id'])
            elif rel_type in ['HAS_FEATURES', 'DEALS_WITH_CONCEPTS']:
                # For features and concepts, we can use the main book's ASIN
                tx.run("""
                    MATCH (s:Book {parent_asin: $main_asin})
                    MATCH (t:%s {name: $target_name})
                    MERGE (s)-[r:%s]->(t)
                """ % (target['type'], rel_type), main_asin=main_book_asin, target_name=target['id'])
            else:
                print(f"Unhandled relationship type: {rel_type}")

def process_json_files(directory):
    updater = Neo4jUpdater(URI, AUTH)
    
    try:
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                with open(os.path.join(directory, filename), 'r') as file:
                    json_data = json.load(file)
                    updater.update_graph(json_data)
        print("Update process completed successfully.")
    except Exception as e:
        print(f"Error during update process: {e}")
    finally:
        updater.close()

if __name__ == "__main__":
    process_json_files('/home/kchauhan/repos/mds-tmu-mrp/datasets/review_kg_extraction')  # Replace with your actual directory path