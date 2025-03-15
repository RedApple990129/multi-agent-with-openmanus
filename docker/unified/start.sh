#!/bin/bash

# Start Flask server in background
FLASK_APP=src/server.py FLASK_ENV=development PYTHONPATH=/app:/app/src python3 -m flask run --host=0.0.0.0 &

# Start tools server in background
PYTHONPATH=/app:/app/src python3 src/tools/server.py &

# Wait for all background processes to complete
wait

# Exit with the status of the last command
exit $?