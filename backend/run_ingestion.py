#!/usr/bin/env python3
"""
Convenience script to run data ingestion
Can be run directly: python run_ingestion.py
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from db.ingest_data import DataIngestion

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("STARTING DATA INGESTION")
    print("=" * 60)
    print("This will:")
    print("  1. Load parquet data")
    print("  2. Chunk text into ~1000 character segments")
    print("  3. Generate embeddings using sentence-transformers")
    print("  4. Store in PostgreSQL with pgvector")
    print("  5. Create vector similarity indexes")
    print(f"  Note: Limited to 100 chunks for testing")
    print("=" * 60 + "\n")
    
    ingestion = DataIngestion()
    ingestion.run()

