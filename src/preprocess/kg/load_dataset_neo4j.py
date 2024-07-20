from neo4j import GraphDatabase
import ast
import duckdb
import re

relevant_columns = ['title', 'average_rating', 'rating_number', 'description', 'price', 'store', 'parent_asin', 'subtitle', 'author', 'features', 'categories', 'details', 'bought_together']

# Define a function to safely parse JSON-like strings
def safe_parse(data):
    try:
        return ast.literal_eval(data)
    except (ValueError, SyntaxError):
        return data

# Function to safely convert values
def safe_convert(value, type_func):
    if value is None or value == 'None' or (isinstance(value, str) and value.strip() == ''):
        return None
    try:
        return type_func(value)
    except (ValueError, TypeError):
        return None

# Function to clean strings
def clean_string(s):
    if not isinstance(s, str):
        return s
    # Remove or replace problematic characters
    s = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', s)
    # Replace other non-UTF8 characters
    return s.encode('utf-8', errors='ignore').decode('utf-8')

# Connect to DuckDB and load the data
conn = duckdb.connect('/home/kchauhan/repos/mds-tmu-mrp/db/duckdb/amazon_reviews.duckdb')
query = f"SELECT {', '.join(relevant_columns)} FROM raw_meta_Video_Games WHERE main_category ='Video Games' AND categories NOT LIKE '%Accessories%' AND categories  LIKE '%''Games''%'"
data_cleaned = conn.execute(query).fetchdf()

# Apply safe_parse to relevant columns
for col in ['features', 'categories', 'details', 'bought_together']:
    data_cleaned[col] = data_cleaned[col].apply(safe_parse)

# Connect to Neo4j
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "invincible"))

def create_knowledge_graph(tx, video_game, properties, relationships):
    # Create the video game node
    query = """
    MERGE (vg:VideoGame {title: $title})
    SET vg += $properties
    """
    tx.run(query, title=video_game, properties=properties)

    # Create relationships
    for rel in relationships:
        query = f"""
        MERGE (n:{rel['type']} {{name: $name}})
        MERGE (vg:VideoGame {{title: $title}})
        MERGE (vg)-[:{rel['relationship']}]->(n)
        """
        tx.run(query, name=clean_string(rel['name']), title=video_game)

# Function to create the graph from the dataframe
def create_graph_from_dataframe(driver, df):
    with driver.session() as session:
        for index, row in df.iterrows():
            video_game = row['title']
            properties = {
                'average_rating': safe_convert(row['average_rating'], float),
                'rating_number': safe_convert(row['rating_number'], int),
                'description': clean_string(str(row['description'])) if row['description'] is not None else None,
                'price': safe_convert(row['price'], float),
                'store': clean_string(str(row['store'])) if row['store'] is not None else None,
                # 'parent_asin': clean_string(str(row['parent_asin'])) if row['parent_asin'] is not None else None,
                # 'subtitle': clean_string(str(row['subtitle'])) if row['subtitle'] is not None else None,
                # 'author': clean_string(str(row['author'])) if row['author'] is not None else None
            }
            properties = {k: v for k, v in properties.items() if v is not None}
            
            relationships = []

            # Add store relationship
            if row['store']:
                relationships.append({'type': 'Store', 'name': row['store'], 'relationship': 'BELONGS_TO'})
            
            # Add feature relationships
            if isinstance(row['features'], list):
                for feature in row['features']:
                    relationships.append({'type': 'Feature', 'name': clean_string(str(feature)), 'relationship': 'HAS_FEATURE'})
            
            # Add category relationships
            if isinstance(row['categories'], list):
                for category in row['categories']:
                    relationships.append({'type': 'Category', 'name': clean_string(str(category)), 'relationship': 'CATEGORIZED_UNDER'})
            
            # # Add detail relationships
            # if isinstance(row['details'], dict):
            #     for detail_key, detail_value in row['details'].items():
            #         detail = f"{clean_string(str(detail_key))}: {clean_string(str(detail_value))}"
            #         relationships.append({'type': 'Detail', 'name': detail, 'relationship': 'HAS_DETAIL'})
            
            # # Add bought together relationships
            # if isinstance(row['bought_together'], list):
            #     for item in row['bought_together']:
            #         relationships.append({'type': 'BoughtTogetherItem', 'name': clean_string(str(item)), 'relationship': 'BOUGHT_TOGETHER_WITH'})
            
            # # Add author relationship
            # if row['author']:
            #     relationships.append({'type': 'Author', 'name': row['author'], 'relationship': 'CREATED_BY'})
            
            session.execute_write(create_knowledge_graph, video_game, properties, relationships)

# Create the graph from the cleaned dataframe
create_graph_from_dataframe(driver, data_cleaned)

# Close the connections
driver.close()
conn.close()