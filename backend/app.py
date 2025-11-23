from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({"message": "RAG QA Engine API"})

@app.route('/api/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/api/query', methods=['POST'])
def query():
    return jsonify({"response": "Query endpoint ready"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

