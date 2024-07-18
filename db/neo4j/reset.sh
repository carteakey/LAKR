
cd /home/kchauhan/repos/mds-tmu-mrp/db/neo4j
docker compose down
rm -Rf data/databases/* data/transactions/*
docker compose up -d