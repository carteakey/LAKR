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
con.sql("""COPY FROM DATABASE db1 TO postgres_db;""")

