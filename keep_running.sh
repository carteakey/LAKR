#!/bin/bash

# bash script to keep the script running and retrigger it if it fails
LOGFILE="keep_running.log"
LAUNCH="./run.sh extract_relationships"

while :
do
    echo "New launch at `date`" >> "${LOGFILE}"
    ${LAUNCH} >> "${LOGFILE}" 2>&1 &
    wait
done