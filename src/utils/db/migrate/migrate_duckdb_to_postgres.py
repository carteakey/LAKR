import duckdb 

# Connect to DuckDB
con = duckdb.connect("test.duckdb")
con.install_extension("postgres")
con.load_extension("postgres")

con.sql("""
        ATTACH 'dbname=amazon_reviews user=admin password=adminpassword host=127.0.0.1' AS postgres_db (TYPE POSTGRES);
        """)
        
con.sql ("""
         ATTACH '/home/kchauhan/repos/mds-tmu-mrp/db/duckdb/amazon_reviews.duckdb' AS db1;
         """
         )

con.sql(""" DROP TABLE IF EXISTS postgres_db.rating_only_positive ;""")
con.sql(""" DROP TABLE IF EXISTS postgres_db.review_processing_status ;""")
con.sql(""" DROP TABLE IF EXISTS postgres_db.skipped_reviews ;""")

con.sql (""" CREATE TABLE postgres_db.rating_only_positive AS SELECT * FROM db1.rating_only_positive;""")
con.sql (""" CREATE TABLE postgres_db.review_processing_status AS SELECT * FROM db1.review_processing_status;""")
con.sql (""" CREATE TABLE postgres_db.skipped_reviews AS SELECT * FROM db1.skipped_reviews;""")

# con.sql("""COPY FROM DATABASE db1 TO postgres_db;""")

