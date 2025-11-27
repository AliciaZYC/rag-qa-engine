# Database Module

## Structure

- `connection.py` - Database connection handler with pgvector support
- `init.sql` - Database schema initialization
- `ingest_data.py` - Data ingestion pipeline (loads, chunks, embeds, stores)
- `inspect_data.py` - Utility to inspect parquet data structure
- `run_ingestion.sh` - Shell script to run ingestion

## Running Data Ingestion

### Option 1: From Docker Container (Recommended)

```bash
# Start services
docker-compose up -d

# Run ingestion in the backend container
docker-compose exec backend python db/ingest_data.py
```

### Option 2: Using Shell Script

```bash
docker-compose exec backend bash db/run_ingestion.sh
```

### Option 3: Via API Endpoint

```bash
# Trigger ingestion via HTTP
curl -X POST http://localhost:5000/api/db/ingest
```

## What the Ingestion Does

1. **Loads** parquet data from `/app/data/train.parquet`
2. **Chunks** text into ~500 character segments (breaks at sentence boundaries)
3. **Generates** embeddings using `sentence-transformers` (all-MiniLM-L6-v2, 384 dimensions)
4. **Stores** chunks with embeddings in PostgreSQL with pgvector
5. **Creates** vector similarity indexes (HNSW or IVFFlat based on data size)

## Configuration

Edit `ingest_data.py` to adjust:
- `CHUNK_SIZE = 500` - Characters per chunk
- `BATCH_SIZE = 100` - Database insert batch size
- Embedding model: Change `'all-MiniLM-L6-v2'` to another sentence-transformers model

## Checking Ingestion Status

```bash
# Check document count
docker-compose exec backend python -c "from db import db; import psycopg2.extras; cur = db.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor); cur.execute('SELECT COUNT(*) as count FROM documents'); print(cur.fetchone())"

# Or via API
curl http://localhost:5000/api/db/test
```

