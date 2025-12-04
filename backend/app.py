from flask import Flask, jsonify, request
from flask_cors import CORS
from db import db
import subprocess
import threading
import requests
import json
import os
from typing import Dict, List

app = Flask(__name__)
CORS(app)

# Ollama configuration
# Read from environment variables, use default values if not set
# In Docker containers, use host.docker.internal to access host services
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")  # Smallest model

# Track ingestion status
ingestion_status = {
    "running": False,
    "last_run": None,
    "status": "not_started"
}

# Store conversation history (Redis recommended for production)
conversations = {}

class OllamaClient:
    """Simple Ollama client"""
    
    @staticmethod
    def check_connection():
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
            return response.status_code == 200
        except:
            return False
    
    @staticmethod
    def chat(messages: List[Dict], model: str = OLLAMA_MODEL):
        """
        Send chat request to Ollama
        
        Args:
            messages: List of message history
            model: Model to use
            
        Returns:
            AI response text
        """
        url = f"{OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 1000,  # Maximum generation length
            }
        }
        
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                result = response.json()
                return result.get('message', {}).get('content', '')
            else:
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}"

# Initialize Ollama client
ollama = OllamaClient()

@app.route('/')
def home():
    return jsonify({
        "message": "RAG QA Engine API with Ollama",
        "endpoints": {
            "database": {
                "/api/db/test": "Test database connection",
                "/api/db/ingest": "Trigger data import (POST)",
                "/api/db/ingest/status": "View import status"
            },
            "ollama": {
                "/api/chat": "AI chat (POST)",
                "/api/chat/simple": "Simple Q&A (POST)",
                "/api/chat/clear": "Clear session (POST)",
                "/api/chat/history": "View history (GET)",
                "/api/models": "View available models"
            },
            "general": {
                "/api/health": "Health check",
                "/api/query": "Query endpoint (POST)"
            }
        }
    })

@app.route('/api/health')
def health():
    """Health check - Check database and Ollama status"""
    db_status = False
    try:
        db_status = db.test_connection()
    except:
        pass
    
    ollama_status = ollama.check_connection()
    
    return jsonify({
        "status": "healthy" if (ollama_status or db_status) else "error",
        "services": {
            "database": "connected" if db_status else "disconnected",
            "ollama": "connected" if ollama_status else "disconnected",
            "ollama_model": OLLAMA_MODEL if ollama_status else "N/A"
        }
    })

# ========== Database related endpoints ==========

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

# ========== Ollama related endpoints ==========

@app.route('/api/models')
def list_models():
    """List available Ollama models"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            return jsonify({
                "current_model": OLLAMA_MODEL,
                "available_models": [m['name'] for m in models],
                "recommended_small_models": [
                    "qwen2.5:0.5b",  # 500M parameters, fastest
                    "gemma2:2b",     # 2B parameters, better quality
                    "tinyllama",     # 1.1B parameters
                    "phi3:mini"      # Microsoft's small model
                ]
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint - Supports contextual conversation
    
    Request body:
    {
        "message": "User's question",
        "session_id": "Optional session ID for maintaining context",
        "system_prompt": "Optional system prompt"
    }
    """
    try:
        data = request.json
        user_message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        system_prompt = data.get('system_prompt', 
            "You are a professional legal advisor assistant. Please provide legal advice in easy-to-understand language. "
            "Note: You provide general legal information and advice, which does not constitute formal legal opinion. "
            "For complex legal issues, users should consult a professional lawyer."
        )
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        # Get or create conversation history
        if session_id not in conversations:
            conversations[session_id] = [
                {"role": "system", "content": system_prompt}
            ]
        
        messages = conversations[session_id]
        
        # Add user message
        messages.append({"role": "user", "content": user_message})
        
        # Call Ollama
        ai_response = ollama.chat(messages)
        
        # Add AI response to history
        messages.append({"role": "assistant", "content": ai_response})
        
        # Limit history length (keep system prompt + last 10 rounds of conversation)
        if len(messages) > 21:  # 1 system + 20 messages (10 rounds)
            conversations[session_id] = [messages[0]] + messages[-20:]
        else:
            conversations[session_id] = messages
        
        return jsonify({
            "response": ai_response,
            "session_id": session_id,
            "message_count": len(messages) - 1  # Excluding system prompt
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/simple', methods=['POST'])
def simple_chat():
    """
    Simple chat endpoint (no context)
    
    Request body:
    {
        "message": "User's question"
    }
    """
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        messages = [
            {
                "role": "system", 
                "content": "You are a professional legal advisor assistant. Please answer user's legal questions concisely."
            },
            {
                "role": "user", 
                "content": user_message
            }
        ]
        
        # Call Ollama
        ai_response = ollama.chat(messages)
        
        return jsonify({
            "response": ai_response
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    """
    Clear conversation history
    
    Request body:
    {
        "session_id": "Session ID to clear"
    }
    """
    data = request.json
    session_id = data.get('session_id', 'default')
    
    if session_id in conversations:
        del conversations[session_id]
        return jsonify({
            "message": f"Session '{session_id}' cleared successfully"
        })
    else:
        return jsonify({
            "message": f"No session found with id '{session_id}'"
        }), 404

@app.route('/api/chat/history', methods=['GET'])
def get_history():
    """
    Get conversation history (for debugging)
    
    Query params:
    - session_id: Session ID
    """
    session_id = request.args.get('session_id', 'default')
    
    if session_id in conversations:
        # Filter out system messages, only return user and assistant conversation
        messages = [
            msg for msg in conversations[session_id] 
            if msg['role'] != 'system'
        ]
        return jsonify({
            "session_id": session_id,
            "messages": messages,
            "total_count": len(messages)
        })
    else:
        return jsonify({
            "session_id": session_id,
            "messages": [],
            "total_count": 0
        })

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List all active sessions"""
    sessions = []
    for sid, messages in conversations.items():
        # Count user messages
        user_messages = [m for m in messages if m['role'] == 'user']
        sessions.append({
            "session_id": sid,
            "message_count": len(user_messages),
            "last_message": user_messages[-1]['content'][:50] + "..." 
                          if user_messages else "No messages"
        })
    
    return jsonify({
        "sessions": sessions,
        "total": len(sessions)
    })

# ========== Original query endpoint ==========

@app.route('/api/query', methods=['POST'])
def query():
    """
    General query endpoint - Can be extended to combine database and Ollama queries
    
    Request body:
    {
        "query": "Query content",
        "use_ai": true/false,
        "session_id": "Optional session ID"
    }
    """
    try:
        data = request.json
        query_text = data.get('query', '')
        use_ai = data.get('use_ai', True)
        session_id = data.get('session_id', 'default')
        
        if not query_text:
            return jsonify({"error": "Query is required"}), 400
        
        response_data = {
            "query": query_text
        }
        
        # If AI is enabled, use Ollama to generate response
        if use_ai and ollama.check_connection():
            # Get or create conversation history
            if session_id not in conversations:
                conversations[session_id] = [
                    {"role": "system", "content": "You are a professional assistant. Please help users based on their questions."}
                ]
            
            messages = conversations[session_id]
            messages.append({"role": "user", "content": query_text})
            
            # Call Ollama
            ai_response = ollama.chat(messages)
            messages.append({"role": "assistant", "content": ai_response})
            
            # Save conversation history
            if len(messages) > 21:
                conversations[session_id] = [messages[0]] + messages[-20:]
            else:
                conversations[session_id] = messages
            
            response_data["response"] = ai_response
            response_data["ai_used"] = True
        else:
            response_data["response"] = "Query endpoint ready (AI disabled or unavailable)"
            response_data["ai_used"] = False
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Check service status on startup
    print("=" * 50)
    print("Starting RAG QA Engine with Ollama")
    print("=" * 50)
    
    # Check Ollama
    print(f"   Attempting to connect: {OLLAMA_BASE_URL}")
    if not ollama.check_connection():
        print("⚠️  Warning: Ollama service is not running!")
        print("   Please run in another terminal: ollama serve")
        print("   Then pull the model: ollama pull qwen2.5:0.5b")
        print(f"   Current Ollama URL configuration: {OLLAMA_BASE_URL}")
    else:
        print("✅ Ollama service connected")
        print(f"   Ollama URL: {OLLAMA_BASE_URL}")
        print(f"   Using model: {OLLAMA_MODEL}")
    
    # Check database
    try:
        if db.test_connection():
            print("✅ Database connected")
        else:
            print("⚠️  Warning: Database connection failed")
    except Exception as e:
        print(f"⚠️  Warning: Database unavailable - {e}")
    
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)