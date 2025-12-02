# RAG QA Engine

## Tech Stack

**Frontend:** React + Vite  
**Backend:** Flask (Python)  
**Database:** PostgreSQL with pgvector  
**Containerization:** Docker + Docker Compose

## How to Run

Download Ollama

```bash
ollama serve
ollama pull qwen2.5:0.5b
docker-compose up --build
```

- Frontend: http://localhost:5173
- Backend API: http://localhost:5001
- pgAdmin (Database UI): http://localhost:8080

**Note:** If localhost doesn't work, use http://127.0.0.1:5173
alternative pgAdmin UI: http://127.0.0.1:8080/browser/

## Data Ingestion

Load data into PostgreSQL with vector embeddings:

```bash
docker-compose exec backend python run_ingestion.py
```

**What it does:**

1. Loads `backend/data/train.parquet`
2. Chunks text into 1000-character segments
3. Generates 384-dim embeddings using `sentence-transformers`
4. Stores in PostgreSQL with pgvector
5. Creates HNSW index for similarity search

**Configuration:** Edit `backend/db/ingest_data.py`

- `CHUNK_SIZE = 1000` - Characters per chunk
- `MAX_CHUNKS = 100` - Limit chunks (set to `None` for all data)

**Verify ingestion:**

```bash
curl http://localhost:5001/api/db/test
```

**db UI:**

password：rag_password

go to http://localhost:8080, click add new server
<img width="1276" height="999" alt="image" src="https://github.com/user-attachments/assets/d68620ec-faeb-4aef-b07a-241e9b823bfb" />
<img width="1220" height="957" alt="image" src="https://github.com/user-attachments/assets/2e246454-dd55-49ba-be04-0a30836406e8" />
then save it

## Stop

```bash
docker-compose down
```

---

Project proposal: https://docs.google.com/document/d/13tYaCE_t6VKL4LSJCatbF2SKsk8L3r1HHC_RbCLDvpc/edit?tab=t.0
