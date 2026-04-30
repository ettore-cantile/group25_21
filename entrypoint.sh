#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting Data Preprocessing..."
# Run the preprocessing script to generate JSON files from CSVs
python3 src/py/run_preprocessing.py

echo "Preprocessing completed successfully."

echo "Starting Microservice and Web Server..."
# Option 1: Start the microservice in the background if it's a Flask/FastAPI app
# Uncomment the line below if you need to run the python microservice
python3 src/py/ms_compute_fp_local.py &

# Option 2: Start a simple HTTP server to serve the frontend
# We use 0.0.0.0 to make it accessible outside the container
python3 -m http.server 8000 --bind 0.0.0.0