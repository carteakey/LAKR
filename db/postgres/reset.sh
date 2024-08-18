cd $BASE_DIR/db/neo4j/db/postgres
# stop the database
docker compose down
# delete the volume
docker volume rm postgres_data
# start the database
docker compose up -d
