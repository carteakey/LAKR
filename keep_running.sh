#!/bin/bash

# Array of relationships to process
RELATIONSHIPS=(
    "SIMILAR_TO_BOOK"
    "PART_OF_SERIES"
    "SIMILAR_TO_AUTHOR"
    "DEALS_WITH_CONCEPTS"
)

LOGFILE="keep_running.log"
BASE_COMMAND="./run.sh"

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOGFILE"
}

run_command() {
    local command="$1"
    local relationship="$2"
    log_message "Starting $command for relationship: $relationship"
    $BASE_COMMAND $command --relationship "$relationship"
    if [ $? -ne 0 ]; then
        log_message "Error occurred during $command for $relationship"
        return 1
    else
        log_message "Finished $command for relationship: $relationship"
        return 0
    fi
}

process_step() {
    local step="$1"
    log_message "Starting $step step for all relationships"
    for relationship in "${RELATIONSHIPS[@]}"
    do
        run_command "$step" "$relationship"
        # Optional: add a small delay between relationships
        sleep 2
    done
    log_message "Completed $step step for all relationships"
    log_message "-------------------------------------------"
}

while true
do
    log_message "Starting new processing cycle"

    # Extract relationships for all types
    process_step "extract_relationships"

    # Evaluate relationships for all types
    process_step "evaluate_relationships"

    # Update KG for all relationship types
    process_step "update_kg"

    log_message "Completed one full cycle of processing"
    log_message "=========================================="
    
    # Optional: add a delay before starting the next cycle
    sleep 60
done