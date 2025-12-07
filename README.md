# RAG QA Engine

## Tech Stack

**Frontend:** React + Vite  
**Backend:** Flask (Python)  
**Database:** PostgreSQL with pgvector  
**Containerization:** Docker + Docker Compose

## How to Run

### Prerequisites

1. **Install Ollama**
   - Download from https://ollama.com/download
   - Or use: `curl -fsSL https://ollama.com/install.sh | sh`

2. **Start Ollama and Download Model**
```bash
# Start Ollama service (keep this running)
ollama serve

# In a new terminal, pull the model
ollama pull qwen2.5:0.5b
```

### Setup and Run
```bash
# 1. Build and start Docker containers
docker-compose up --build -d

# 2. Download embedding models and NLTK data (first time only)
docker-compose exec backend python download_models.py

# 3. Run data ingestion with upgraded pipeline (LegalBERT + Hybrid Chunking)
docker-compose exec backend python db/ingest_data.py

# 4. Verify system is ready
docker-compose exec backend python -c "from embeddings import DualEmbedder, ModelType; print('✓ Embeddings working')"
```

### Access Points

- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:5001
- **pgAdmin (Database UI):** http://localhost:8080

**Note:** If localhost doesn't work, use http://127.0.0.1:5173

### Quick Verification (PowerShell)

Test all components with a single command:
```powershell
.\verify_functionality.ps1
```

This will check:
- Frontend accessibility
- Backend health (Ollama, Database, Embedder)
- Database document count
- LegalBERT embeddings (768-dim)

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