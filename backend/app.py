from flask import Flask, jsonify
from flask_cors import CORS
from db import db

app = Flask(__name__)
CORS(app)

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
        
        return jsonify({
            "status": "connected" if is_connected else "error",
            "database": "PostgreSQL with pgvector",
            "documents_count": doc_count
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/query', methods=['POST'])
def query():
    return jsonify({"response": "Query endpoint ready"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

