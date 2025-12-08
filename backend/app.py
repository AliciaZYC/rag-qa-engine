from flask import Flask, jsonify, request
from flask_cors import CORS
from db import db
import subprocess
import threading
import requests
import json
import os
import sys
from typing import Dict, List, Optional
import traceback

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# LangChain imports
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# RAG imports - NEW
from embeddings import DualEmbedder, ModelType

app = Flask(__name__)
CORS(app)

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")

# Track ingestion status
ingestion_status = {
    "running": False,
    "last_run": None,
    "status": "not_started"
}

# Store conversation history (Redis recommended for production)
conversations = {}

class LangChainOllamaClient:
    """LangChain-based Ollama client for LLM interactions"""
    
    def __init__(self):
        """Initialize the ChatOllama model"""
        self.llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=0.7,
            num_predict=1000,
        )
    
    def check_connection(self):
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            print(f"[DEBUG] Ollama connection check: {OLLAMA_BASE_URL}/api/tags -> Status: {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            print(f"[DEBUG] Ollama connection failed: {OLLAMA_BASE_URL} -> Error: {str(e)}")
            return False
    
    def chat(self, messages: List[Dict], model: str = None):
        """
        Send chat request to Ollama using LangChain
        
        Args:
            messages: List of message history in format [{"role": "user", "content": "..."}, ...]
            model: Model to use (optional, uses default if not specified)
            
        Returns:
            AI response text
        """
        try:
            # Convert message format to LangChain format
            langchain_messages = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if role == "system":
                    langchain_messages.append(SystemMessage(content=content))
                elif role == "user":
                    langchain_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    langchain_messages.append(AIMessage(content=content))
            
            # Use specific model if provided
            if model and model != OLLAMA_MODEL:
                llm = ChatOllama(
                    base_url=OLLAMA_BASE_URL,
                    model=model,
                    temperature=0.7,
                    num_predict=1000,
                )
                response = llm.invoke(langchain_messages)
            else:
                response = self.llm.invoke(langchain_messages)
            
            return response.content
            
        except Exception as e:
            print(f"[ERROR] LangChain chat error: {str(e)}")
            return f"Error connecting to Ollama: {str(e)}"

# Initialize LangChain Ollama client
ollama = LangChainOllamaClient()

# Initialize RAG Embedder - NEW
print("\n" + "=" * 60)
print("INITIALIZING RAG COMPONENTS")
print("=" * 60)
embedder = None
try:
    print("Loading embedding model for retrieval...")
    embedder = DualEmbedder(
        primary_model=ModelType.LEGAL_BERT,
        use_fallback=False  # Set to True for faster MiniLM fallback
    )
    print(f"✅ Embedder loaded: {embedder.get_model_name()}")
    print(f"   Dimension: {embedder.get_dimension()}")
    print(f"   Legal optimized: {embedder.is_legal_optimized()}")
except Exception as e:
    print(f"⚠️  Failed to load embedder: {e}")
    print("   RAG endpoints will not be available")
    traceback.print_exc()
print("=" * 60 + "\n")

# Add this helper function after the imports section (around line 40)
def needs_retrieval(message: str) -> bool:
    """
    Determine if a user message requires document retrieval.
    Returns False for greetings, casual messages, and meta questions.
    Returns True for actual legal/contract questions.
    """
    message_lower = message.lower().strip()
    
    # Common greetings and casual messages that don't need retrieval
    non_retrieval_patterns = [
        # Greetings
        'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening',
        # Farewells
        'bye', 'goodbye', 'see you', 'thanks', 'thank you',
        # Meta questions
        'how are you', 'what can you do', 'who are you', 'help',
        # Very short messages
    ]
    
    # Check if message is very short (likely a greeting)
    if len(message_lower) < 3:
        return False
    
    # Check against non-retrieval patterns
    for pattern in non_retrieval_patterns:
        if message_lower == pattern or message_lower.startswith(pattern + ' ') or message_lower.startswith(pattern + '?'):
            return False
    
    # If message has question indicators, it likely needs retrieval
    question_indicators = ['what', 'how', 'when', 'where', 'who', 'why', 'which', 'can', 'does', 'is', 'are']
    has_question_word = any(word in message_lower.split()[:3] for word in question_indicators)
    
    # Has question mark or question word = likely needs retrieval
    if '?' in message or has_question_word:
        return True
    
    # Default: if message is substantive (>10 chars), assume it needs retrieval
    return len(message) > 10

@app.route('/')
def home():
    return jsonify({
        "message": "RAG QA Engine API - Full Implementation",
        "version": "3.0 - Complete RAG with LegalBERT",
        "endpoints": {
            "rag": {
                "/api/chat/rag": "RAG-powered chat with retrieval (POST)",
                "/api/chat/rag/test": "Test RAG endpoint (GET)",
                "/api/retrieve": "Direct retrieval without LLM (POST)"
            },
            "database": {
                "/api/db/test": "Test database connection",
                "/api/db/ingest": "Trigger data import (POST)",
                "/api/db/ingest/status": "View import status",
                "/api/db/stats": "Database statistics (GET)"
            },
            "ollama": {
                "/api/chat": "AI chat with context (POST)",
                "/api/chat/simple": "Simple Q&A (POST)",
                "/api/chat/clear": "Clear session (POST)",
                "/api/chat/history": "View history (GET)",
                "/api/models": "View available models"
            },
            "general": {
                "/api/health": "Health check",
                "/api/query": "Query endpoint (POST)"
            }
        },
        "features": {
            "langchain": "Integrated for LLM orchestration",
            "rag": "Full RAG pipeline with LegalBERT embeddings",
            "retrieval": "Hybrid structural-semantic chunking",
            "citations": "Source tracking and similarity scores"
        }
    })

@app.route('/api/health')
def health():
    """Health check - Check all services"""
    db_status = False
    try:
        db_status = db.test_connection()
    except Exception as e:
        print(f"[DEBUG] Database health check failed: {str(e)}")
    
    ollama_status = ollama.check_connection()
    embedder_status = embedder is not None
    
    response_data = {
        "status": "healthy" if (ollama_status and db_status and embedder_status) else "partial",
        "services": {
            "database": "connected" if db_status else "disconnected",
            "ollama": "connected" if ollama_status else "disconnected",
            "ollama_model": OLLAMA_MODEL if ollama_status else "N/A",
            "embedder": "loaded" if embedder_status else "not loaded",
            "embedding_model": embedder.get_model_name() if embedder_status else "N/A",
            "langchain": "integrated"
        }
    }
    
    print(f"[DEBUG] Health check response: {response_data}")
    return jsonify(response_data)

# ========== NEW: RAG Endpoints ==========

@app.route('/api/chat/rag', methods=['POST'])
def rag_chat():
    """
    Full RAG pipeline: Retrieve relevant chunks + Generate answer with citations
    
    Request body:
    {
        "message": "What are the indemnification terms?",
        "top_k": 5,  // Optional, default 5
        "session_id": "optional_session_id",  // For conversation context
        "include_sources": true  // Optional, default true
    }
    
    Response:
    {
        "query": "original query",
        "response": "LLM generated answer with citations",
        "sources": [
            {
                "id": 1,
                "content": "chunk text...",
                "provision_label": "indemnification",
                "section_number": "Article 5",
                "similarity": 0.856,
                "rank": 1
            }
        ],
        "model_info": {...},
        "session_id": "session_id"
    }
    """
    try:
        data = request.json
        user_query = data.get('message', '')
        top_k = data.get('top_k', 5)
        session_id = data.get('session_id', 'default')
        include_sources = data.get('include_sources', True)
        
        if not user_query:
            return jsonify({"error": "Message is required"}), 400
        
        if not embedder:
            return jsonify({
                "error": "Embedder not initialized. RAG is unavailable.",
                "fallback": "Use /api/chat/simple for non-RAG chat"
            }), 503
        
        # NEW: Check if retrieval is needed
        if not needs_retrieval(user_query):
            print(f"[RAG] Query doesn't require retrieval: {user_query}")
            
            # Get or create conversation context
            if session_id not in conversations:
                conversations[session_id] = []
            
            # Simple system prompt for greetings/casual messages
            system_prompt = """You are an expert legal document assistant. Respond naturally to greetings and casual messages. 
When the user asks questions about contracts, you will retrieve and reference relevant documents."""
            
            # Build messages
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add recent history
            if conversations[session_id]:
                messages.extend(conversations[session_id][-6:])
            
            # Add current message
            messages.append({"role": "user", "content": user_query})
            
            # Get LLM response without retrieval
            print(f"[RAG] Generating response without retrieval...")
            ai_response = ollama.chat(messages)
            
            # Update conversation history
            conversations[session_id].append({"role": "user", "content": user_query})
            conversations[session_id].append({"role": "assistant", "content": ai_response})
            
            # Limit history
            if len(conversations[session_id]) > 20:
                conversations[session_id] = conversations[session_id][-20:]
            
            # Return response WITHOUT sources
            return jsonify({
                "query": user_query,
                "response": ai_response,
                "sources": [],  # Empty sources for non-retrieval responses
                "model_info": {
                    "llm_model": OLLAMA_MODEL,
                    "retrieval_used": False
                },
                "session_id": session_id
            })
        
        # EXISTING: Continue with full retrieval for actual questions
        print(f"[RAG] Query: {user_query}")
        query_embedding = embedder.encode_query(user_query)
        
        # Step 2: Retrieve relevant chunks from database
        print(f"[RAG] Retrieving top {top_k} chunks...")
        with db.get_cursor() as cursor:
            cursor.execute("""
                SELECT 
                    id,
                    content,
                    provision_label,
                    section_number,
                    chunk_method,
                    token_count,
                    1 - (embedding <=> %s::vector) as similarity
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding.tolist(), query_embedding.tolist(), top_k))
            
            retrieved_chunks = cursor.fetchall()
        
        if not retrieved_chunks:
            return jsonify({
                "error": "No relevant documents found in database",
                "suggestion": "Please run data ingestion first: POST /api/db/ingest"
            }), 404
        
        # Step 3: Build context from retrieved chunks
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            # Format each chunk with metadata for LLM
            chunk_header = f"[Document {i}"
            if chunk['provision_label']:
                chunk_header += f" - {chunk['provision_label']}"
            if chunk['section_number']:
                chunk_header += f" - {chunk['section_number']}"
            chunk_header += f" | Relevance: {chunk['similarity']:.2f}]"
            
            context_parts.append(f"{chunk_header}\n{chunk['content']}")
            
            # Store source info for response
            if include_sources:
                sources.append({
                    "id": chunk['id'],
                    "content": chunk['content'],
                    "provision_label": chunk['provision_label'],
                    "section_number": chunk['section_number'],
                    "chunk_method": chunk['chunk_method'],
                    "token_count": chunk['token_count'],
                    "similarity": float(chunk['similarity']),
                    "rank": i
                })
        
        context_text = "\n\n".join(context_parts)
        
        print(f"[RAG] Retrieved {len(retrieved_chunks)} chunks")
        print(f"[RAG] Top similarity: {retrieved_chunks[0]['similarity']:.3f}")
        
        # Step 4: Get or create conversation context
        if session_id not in conversations:
            conversations[session_id] = []
        
        # Step 5: Create system prompt with retrieved context
        system_prompt = f"""You are an expert legal document assistant. Your task is to answer the user's question based ONLY on the contract provisions provided below.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the provided documents
2. When referencing information, cite the document number (e.g., "According to Document 1...")
3. If documents contain conflicting information, mention both and cite sources
4. If the provided documents don't contain enough information, explicitly state: "The provided documents do not contain sufficient information to answer this question."
5. Do not make assumptions or use external knowledge
6. Be concise but comprehensive
7. Use legal terminology appropriately

CONTRACT PROVISIONS:
{context_text}

Remember: Your answer must be grounded in these documents. Always cite document numbers when making claims."""

        # Build messages including conversation history
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add recent conversation history (last 3 turns)
        if conversations[session_id]:
            messages.extend(conversations[session_id][-6:])  # Last 3 turns (6 messages)
        
        # Add current user message
        messages.append({"role": "user", "content": user_query})
        
        # Step 6: Get LLM response
        print(f"[RAG] Generating response with {OLLAMA_MODEL}...")
        ai_response = ollama.chat(messages)
        
        # Step 7: Update conversation history
        conversations[session_id].append({"role": "user", "content": user_query})
        conversations[session_id].append({"role": "assistant", "content": ai_response})
        
        # Limit history (keep last 10 turns = 20 messages)
        if len(conversations[session_id]) > 20:
            conversations[session_id] = conversations[session_id][-20:]
        
        # Step 8: Return response with sources
        response_data = {
            "query": user_query,
            "response": ai_response,
            "sources": sources if include_sources else None,
            "model_info": {
                "embedding_model": embedder.get_model_name(),
                "embedding_dimension": embedder.get_dimension(),
                "llm_model": OLLAMA_MODEL,
                "chunks_retrieved": len(sources),
                "top_similarity": float(retrieved_chunks[0]['similarity'])
            },
            "session_id": session_id
        }
        
        print(f"[RAG] Response generated successfully")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"[ERROR] RAG endpoint error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }), 500


@app.route('/api/retrieve', methods=['POST'])
def retrieve_only():
    """
    Direct retrieval endpoint without LLM generation
    Useful for debugging and testing retrieval quality
    
    Request body:
    {
        "query": "search query",
        "top_k": 5  // Optional
    }
    """
    try:
        data = request.json
        query = data.get('query', '')
        top_k = data.get('top_k', 5)
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        if not embedder:
            return jsonify({"error": "Embedder not initialized"}), 503
        
        # Embed query
        query_embedding = embedder.encode_query(query)
        
        # Retrieve chunks
        with db.get_cursor() as cursor:
            cursor.execute("""
                SELECT 
                    id,
                    content,
                    provision_label,
                    section_number,
                    chunk_method,
                    token_count,
                    1 - (embedding <=> %s::vector) as similarity
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding.tolist(), query_embedding.tolist(), top_k))
            
            results = cursor.fetchall()
        
        return jsonify({
            "query": query,
            "results": [
                {
                    "id": r['id'],
                    "content": r['content'],
                    "provision_label": r['provision_label'],
                    "section_number": r['section_number'],
                    "similarity": float(r['similarity']),
                    "rank": i
                }
                for i, r in enumerate(results, 1)
            ],
            "count": len(results)
        })
        
    except Exception as e:
        print(f"[ERROR] Retrieve endpoint error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat/rag/test', methods=['GET'])
def test_rag():
    """
    Test endpoint to verify RAG components are working
    """
    # Check embedder
    embedder_status = {
        "loaded": embedder is not None,
        "info": embedder.get_info() if embedder else None
    }
    
    # Check database
    db_status = {"connected": False, "document_count": 0}
    try:
        db_status["connected"] = db.test_connection()
        if db_status["connected"]:
            with db.get_cursor() as cursor:
                cursor.execute("SELECT COUNT(*) as count FROM documents")
                result = cursor.fetchone()
                db_status["document_count"] = result['count']
    except Exception as e:
        db_status["error"] = str(e)
    
    # Check Ollama
    ollama_status = {
        "connected": ollama.check_connection(),
        "model": OLLAMA_MODEL
    }
    
    # Sample test queries
    test_queries = [
        "What are the indemnification provisions?",
        "Which law governs this agreement?",
        "What are the confidentiality requirements?",
        "How can this contract be terminated?",
        "What happens in case of force majeure?"
    ]
    
    # Overall status
    all_ready = (
        embedder_status["loaded"] and 
        db_status["connected"] and 
        db_status["document_count"] > 0 and
        ollama_status["connected"]
    )
    
    return jsonify({
        "status": "ready" if all_ready else "not_ready",
        "components": {
            "embedder": embedder_status,
            "database": db_status,
            "ollama": ollama_status
        },
        "test_queries": test_queries,
        "usage": {
            "endpoint": "/api/chat/rag",
            "method": "POST",
            "example": {
                "message": "What are the indemnification terms?",
                "top_k": 5,
                "include_sources": True
            }
        }
    })

# ========== Database Endpoints ==========

@app.route('/api/db/test')
def test_db():
    """Test database connection and pgvector extension"""
    try:
        is_connected = db.test_connection()
        
        with db.get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM documents;")
            result = cursor.fetchone()
            doc_count = result['count']
            
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


@app.route('/api/db/stats')
def db_stats():
    """Get database statistics"""
    try:
        with db.get_cursor() as cursor:
            # Basic counts
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_docs,
                    COUNT(DISTINCT provision_label) as unique_labels,
                    AVG(token_count) as avg_tokens,
                    MAX(token_count) as max_tokens,
                    MIN(token_count) as min_tokens
                FROM documents
            """)
            stats = cursor.fetchone()
            
            # Provision distribution
            cursor.execute("""
                SELECT provision_label, COUNT(*) as count
                FROM documents
                WHERE provision_label IS NOT NULL
                GROUP BY provision_label
                ORDER BY count DESC
                LIMIT 10
            """)
            provision_dist = cursor.fetchall()
            
            # Chunking methods
            cursor.execute("""
                SELECT chunk_method, COUNT(*) as count
                FROM documents
                WHERE chunk_method IS NOT NULL
                GROUP BY chunk_method
            """)
            chunking_methods = cursor.fetchall()
            
        return jsonify({
            "total_documents": stats['total_docs'],
            "unique_provision_types": stats['unique_labels'],
            "token_statistics": {
                "average": float(stats['avg_tokens']) if stats['avg_tokens'] else 0,
                "min": stats['min_tokens'],
                "max": stats['max_tokens']
            },
            "provision_distribution": [
                {"label": p['provision_label'], "count": p['count']}
                for p in provision_dist
            ],
            "chunking_methods": [
                {"method": m['chunk_method'], "count": m['count']}
                for m in chunking_methods
            ]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/db/ingest', methods=['POST'])
def ingest_data():
    """Trigger data ingestion"""
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
            
            # Run NEW ingestion script with hybrid chunking
            result = subprocess.run(
                ["python", "db/ingest_data_v2.py"],
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
    
    thread = threading.Thread(target=run_ingestion)
    thread.start()
    
    return jsonify({
        "status": "started",
        "message": "Data ingestion started. Check /api/db/ingest/status for progress."
    })


@app.route('/api/db/ingest/status')
def ingest_status_endpoint():
    """Get ingestion status"""
    return jsonify(ingestion_status)

# ========== Original Ollama Endpoints (Keep for backward compatibility) ==========

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
                    "qwen2.5:0.5b",
                    "gemma2:2b",
                    "tinyllama",
                    "phi3:mini"
                ]
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint with conversation context"""
    try:
        data = request.json
        user_message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        system_prompt = data.get('system_prompt', 
            "You are a professional legal advisor assistant."
        )
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        if session_id not in conversations:
            conversations[session_id] = [
                {"role": "system", "content": system_prompt}
            ]
        
        messages = conversations[session_id]
        messages.append({"role": "user", "content": user_message})
        
        ai_response = ollama.chat(messages)
        messages.append({"role": "assistant", "content": ai_response})
        
        if len(messages) > 21:
            conversations[session_id] = [messages[0]] + messages[-20:]
        else:
            conversations[session_id] = messages
        
        return jsonify({
            "response": ai_response,
            "session_id": session_id,
            "message_count": len(messages) - 1
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat/simple', methods=['POST'])
def simple_chat():
    """Simple chat without context"""
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        messages = [
            {"role": "system", "content": "You are a professional legal advisor assistant."},
            {"role": "user", "content": user_message}
        ]
        
        ai_response = ollama.chat(messages)
        
        return jsonify({"response": ai_response})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    """Clear conversation history"""
    data = request.json
    session_id = data.get('session_id', 'default')
    
    if session_id in conversations:
        del conversations[session_id]
        return jsonify({"message": f"Session '{session_id}' cleared"})
    else:
        return jsonify({"message": f"No session found with id '{session_id}'"}), 404


@app.route('/api/chat/history', methods=['GET'])
def get_history():
    """Get conversation history"""
    session_id = request.args.get('session_id', 'default')
    
    if session_id in conversations:
        messages = [msg for msg in conversations[session_id] if msg['role'] != 'system']
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


@app.route('/api/query', methods=['POST'])
def query():
    """General query endpoint"""
    try:
        data = request.json
        query_text = data.get('query', '')
        use_ai = data.get('use_ai', True)
        session_id = data.get('session_id', 'default')
        
        if not query_text:
            return jsonify({"error": "Query is required"}), 400
        
        response_data = {"query": query_text}
        
        if use_ai and ollama.check_connection():
            if session_id not in conversations:
                conversations[session_id] = [
                    {"role": "system", "content": "You are a professional assistant."}
                ]
            
            messages = conversations[session_id]
            messages.append({"role": "user", "content": query_text})
            
            ai_response = ollama.chat(messages)
            messages.append({"role": "assistant", "content": ai_response})
            
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
    # Service startup checks
    print("\n" + "=" * 60)
    print("STARTING RAG QA ENGINE v3.0")
    print("=" * 60)
    
    # Check Ollama
    print(f"\n[1/3] Checking Ollama service...")
    print(f"   URL: {OLLAMA_BASE_URL}")
    if ollama.check_connection():
        print(f"   ✅ Connected | Model: {OLLAMA_MODEL}")
    else:
        print(f"   ⚠️  Not connected")
        print(f"   Run: ollama serve && ollama pull {OLLAMA_MODEL}")
    
    # Check Database
    print(f"\n[2/3] Checking database...")
    try:
        if db.test_connection():
            with db.get_cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM documents")
                count = cursor.fetchone()[0]
            print(f"   ✅ Connected | Documents: {count}")
        else:
            print(f"   ⚠️  Connection failed")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    # Check Embedder
    print(f"\n[3/3] Checking embedder...")
    if embedder:
        print(f"   ✅ Loaded | Model: {embedder.get_model_name()}")
        print(f"   Dimension: {embedder.get_dimension()}")
    else:
        print(f"   ⚠️  Not loaded - RAG unavailable")
    
    print("\n" + "=" * 60)
    print("API READY")
    print("=" * 60)
    print(f"Access API at: http://localhost:5000")
    print(f"API Documentation: http://localhost:5000/")
    print(f"Test RAG: http://localhost:5000/api/chat/rag/test")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)