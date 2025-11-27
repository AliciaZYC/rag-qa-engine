from flask import Flask, jsonify, request
from flask_cors import CORS
from db import db
import subprocess
import threading

app = Flask(__name__)
CORS(app)

# Track ingestion status
ingestion_status = {
    "running": False,
    "last_run": None,
    "status": "not_started"
}

@app.route('/')
def home():
    return jsonify({"message": "RAG QA Engine API"})

@app.route('/api/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/api/db/test')
def test_db():
    """Test database connection and pgvector extension"""
    try:
        is_connected = db.test_connection()
        
        # Get document count
        with db.get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM documents;")
            result = cursor.fetchone()
            doc_count = result['count']
            
            # Check if index exists
            cursor.execute("""
                SELECT indexname FROM pg_indexes 
                WHERE tablename = 'documents' AND indexname LIKE '%embedding%'
            """)
            index_info = cursor.fetchone()
            has_index = index_info is not None
        
        return jsonify({
            "status": "connected" if is_connected else "error",
            "database": "PostgreSQL with pgvector",
            "documents_count": doc_count,
            "has_vector_index": has_index
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/db/ingest', methods=['POST'])
def ingest_data():
    """Trigger data ingestion (runs in background)"""
    global ingestion_status
    
    if ingestion_status["running"]:
        return jsonify({
            "status": "already_running",
            "message": "Ingestion is already in progress"
        }), 409
    
    def run_ingestion():
        global ingestion_status
        try:
            ingestion_status["running"] = True
            ingestion_status["status"] = "running"
            
            # Run ingestion script
            result = subprocess.run(
                ["python", "db/ingest_data.py"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                ingestion_status["status"] = "completed"
            else:
                ingestion_status["status"] = f"failed: {result.stderr}"
        except Exception as e:
            ingestion_status["status"] = f"error: {str(e)}"
        finally:
            ingestion_status["running"] = False
            import datetime
            ingestion_status["last_run"] = datetime.datetime.now().isoformat()
    
    # Start ingestion in background thread
    thread = threading.Thread(target=run_ingestion)
    thread.start()
    
    return jsonify({
        "status": "started",
        "message": "Data ingestion started in background. Check /api/db/ingest/status for progress."
    })

@app.route('/api/db/ingest/status')
def ingest_status():
    """Get ingestion status"""
    return jsonify(ingestion_status)

@app.route('/api/query', methods=['POST'])
def query():
    return jsonify({"response": "Query endpoint ready"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

