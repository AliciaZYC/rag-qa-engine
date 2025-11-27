#!/bin/bash
# Run data ingestion script

echo "Starting data ingestion..."
echo "This will load data, chunk it, generate embeddings, and store in PostgreSQL"
echo ""

# Wait for database to be ready
echo "Waiting for database to be ready..."
sleep 5

# Run ingestion
python /app/db/ingest_data.py

echo ""
echo "Ingestion complete!"

