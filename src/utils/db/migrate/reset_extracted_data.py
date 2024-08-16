# Reset all processed data to the original state
import os
import logging
from neo4j import GraphDatabase
import duckdb

con = duckdb.connect("/home/kchauhan/repos/mds-tmu-mrp/db/duckdb/amazon_reviews.duckdb")

def reset_ratings():
    con.execute("DELETE FROM review_processing_status")
    con.execute("DELETE FROM skipped_reviews")
    
    
reset_ratings()