import duckdb
from dotenv import load_dotenv
import os

# DuckDB configuration
load_dotenv("/home/kchauhan/repos/mds-tmu-mrp/config/env.sh")
con = duckdb.connect(os.getenv('DB_FILE'))

DATASET = "/home/kchauhan/repos/mds-tmu-mrp/datasets/10core/rating_only/Books.csv"

con.sql("DROP TABLE IF EXISTS rating_only")
con.sql("CREATE TABLE rating_only (user_id VARCHAR, item_id VARCHAR, rating FLOAT, timestamp BIGINT)")
con.execute("COPY rating_only FROM '{}' (FORMAT CSV, HEADER FALSE)".format(DATASET))

print(con.execute("SELECT * FROM rating_only LIMIT 5").fetchdf())

# Keep positive ratings only
con.sql("DROP TABLE IF EXISTS rating_only_positive")
con.sql("CREATE TABLE rating_only_positive AS SELECT * FROM rating_only WHERE rating > 3")
print(con.execute("SELECT * FROM rating_only_positive LIMIT 5").fetchdf())
