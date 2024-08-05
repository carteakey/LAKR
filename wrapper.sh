export date=$(date '+%Y-%m-%d-%H-%M-%S')
export root=/home/kchauhan/repos/mds-tmu-mrp
export src=$root/src
export logs=$root/logs
export db=$root/db

# stop on error
set -e

# mv $db/duckdb/amazon_reviews.duckdb $db/duckdb/amazon_reviews.duckdb.bkp
. .venv/bin/activate

# python -m k_core_filtering -k 15 --input_path /home/kchauhan/repos/mds-tmu-mrp/datasets/raw --output_path /home/kchauhan/repos/mds-tmu-mrp/datasets
# python -m last_out_split  --input_path /home/kchauhan/repos/mds-tmu-mrp/datasets/15core/rating_only --output_path /home/kchauhan/repos/mds-tmu-mrp/datasets/last_out_split --seq_path /home/kchauhan/repos/mds-tmu-mrp/datasets/last_out_split

# cd $src/preprocess/kg/
# python -m load_kcore_ratings_duckdb 1>$logs/load_kcore_ratings_duckdb.$date.log 
# python -m load_metadata_review_duckdb >$logs/load_metadata_review_duckdb.$date.log 

# increase memory limit for Neo4j
# docker exec neo4j-apoc sed -i 's/dbms.memory.transaction.total.max=2.7g/dbms.memory.transaction.total.max=3g/' /var/lib/neo4j/conf/neo4j.conf

# restart Neo4j for the changes to take effect
# docker restart neo4j-apoc

# reset neo4j database
# docker exec -it neo4j-apoc cypher-shell -u neo4j -p tmu-2024 'MATCH (n) DETACH DELETE n'

# #set neo4j initial password
# docker exec -it  neo4j-apoc cypher-shell -u neo4j -p neo4j 'CALL dbms.security.changePassword("tmu-2024")'

cd $src/preprocess/kg/
python -m 03_load_metadata_neo4j 1>$logs/load_metadata_neo4j.$date.log

#export data from neo4j
cd $src/kg-export/
python -m 03_neo4j_export_kg  1>$logs/neo4j_export_kg.$date.log
