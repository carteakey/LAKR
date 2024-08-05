import duckdb
import json

# Connect to DuckDB
con = duckdb.connect("/home/kchauhan/repos/mds-tmu-mrp/db/duckdb/amazon_reviews.duckdb")

# Create new tables with the updated schema
con.execute("""
    CREATE TABLE IF NOT EXISTS review_processing_status_new (
        user_id VARCHAR,
        item_id VARCHAR,
        relationship_type VARCHAR,
        status VARCHAR,
        json_data JSON,
        rating INTEGER,
        PRIMARY KEY (user_id, item_id, relationship_type)
    )
""")

con.execute("""
    CREATE TABLE IF NOT EXISTS skipped_reviews_new (
        user_id VARCHAR,
        item_id VARCHAR,
        relationship_type VARCHAR,
        reason VARCHAR,
        review_data JSON,
        PRIMARY KEY (user_id, item_id, relationship_type)
    )
""")

# Migrate data from the old tables to the new ones
con.execute("""
    INSERT INTO review_processing_status_new (user_id, item_id, relationship_type, status, json_data, rating)
    SELECT user_id, item_id, 'SIMILAR_TO_BOOK', status, json_data, rating
    FROM review_processing_status
""")

con.execute("""
    INSERT INTO skipped_reviews_new (user_id, item_id, relationship_type, reason, review_data)
    SELECT user_id, item_id, 'SIMILAR_TO_BOOK', reason, review_data
    FROM skipped_reviews
""")

# Rename the new tables to replace the old ones
con.execute("ALTER TABLE review_processing_status RENAME TO review_processing_status_old")
con.execute("ALTER TABLE review_processing_status_new RENAME TO review_processing_status")

con.execute("ALTER TABLE skipped_reviews RENAME TO skipped_reviews_old")
con.execute("ALTER TABLE skipped_reviews_new RENAME TO skipped_reviews")

print("Data migration completed successfully!")

# Close the connection
con.close()