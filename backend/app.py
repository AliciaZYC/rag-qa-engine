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

# Ollama 配置
# 从环境变量读取，如果未设置则使用默认值
# 在 Docker 容器中，使用 host.docker.internal 访问宿主机服务
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")  # 最小的模型

# Track ingestion status
ingestion_status = {
    "running": False,
    "last_run": None,
    "status": "not_started"
}

# 存储会话历史（生产环境建议使用 Redis）
conversations = {}

class OllamaClient:
    """简单的 Ollama 客户端"""
    
    @staticmethod
    def check_connection():
        """检查 Ollama 服务是否运行"""
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
            return response.status_code == 200
        except:
            return False
    
    @staticmethod
    def chat(messages: List[Dict], model: str = OLLAMA_MODEL):
        """
        发送聊天请求到 Ollama
        
        Args:
            messages: 消息历史列表
            model: 使用的模型
            
        Returns:
            AI 的回复文本
        """
        url = f"{OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 1000,  # 最大生成长度
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

# 初始化 Ollama 客户端
ollama = OllamaClient()

@app.route('/')
def home():
    return jsonify({
        "message": "RAG QA Engine API with Ollama",
        "endpoints": {
            "database": {
                "/api/db/test": "测试数据库连接",
                "/api/db/ingest": "触发数据导入（POST）",
                "/api/db/ingest/status": "查看导入状态"
            },
            "ollama": {
                "/api/chat": "AI 聊天（POST）",
                "/api/chat/simple": "简单问答（POST）",
                "/api/chat/clear": "清除会话（POST）",
                "/api/chat/history": "查看历史（GET）",
                "/api/models": "查看可用模型"
            },
            "general": {
                "/api/health": "健康检查",
                "/api/query": "查询端点（POST）"
            }
        }
    })

@app.route('/api/health')
def health():
    """健康检查 - 检查数据库和 Ollama 状态"""
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

# ========== 数据库相关端点 ==========

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

# ========== Ollama 相关端点 ==========

@app.route('/api/models')
def list_models():
    """列出可用的 Ollama 模型"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            return jsonify({
                "current_model": OLLAMA_MODEL,
                "available_models": [m['name'] for m in models],
                "recommended_small_models": [
                    "qwen2.5:0.5b",  # 500M 参数，最快
                    "gemma2:2b",     # 2B 参数，效果更好
                    "tinyllama",     # 1.1B 参数
                    "phi3:mini"      # 微软的小模型
                ]
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    主要聊天端点 - 支持上下文对话
    
    Request body:
    {
        "message": "用户的问题",
        "session_id": "可选的会话ID，用于保持上下文",
        "system_prompt": "可选的系统提示词"
    }
    """
    try:
        data = request.json
        user_message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        system_prompt = data.get('system_prompt', 
            "你是一位专业的法律顾问助手。请用通俗易懂的语言为用户提供法律建议。"
            "注意：你提供的是一般性法律信息和建议，不构成正式的法律意见。"
            "对于复杂的法律问题，建议用户咨询专业律师。"
        )
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        # 获取或创建会话历史
        if session_id not in conversations:
            conversations[session_id] = [
                {"role": "system", "content": system_prompt}
            ]
        
        messages = conversations[session_id]
        
        # 添加用户消息
        messages.append({"role": "user", "content": user_message})
        
        # 调用 Ollama
        ai_response = ollama.chat(messages)
        
        # 添加 AI 回复到历史
        messages.append({"role": "assistant", "content": ai_response})
        
        # 限制历史长度（保留系统提示 + 最近10轮对话）
        if len(messages) > 21:  # 1 system + 20 messages (10 rounds)
            conversations[session_id] = [messages[0]] + messages[-20:]
        else:
            conversations[session_id] = messages
        
        return jsonify({
            "response": ai_response,
            "session_id": session_id,
            "message_count": len(messages) - 1  # 不计算系统提示
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/simple', methods=['POST'])
def simple_chat():
    """
    简单聊天端点（无上下文）
    
    Request body:
    {
        "message": "用户的问题"
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
                "content": "你是一位专业的法律顾问助手。请简洁地回答用户的法律问题。"
            },
            {
                "role": "user", 
                "content": user_message
            }
        ]
        
        # 调用 Ollama
        ai_response = ollama.chat(messages)
        
        return jsonify({
            "response": ai_response
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    """
    清除会话历史
    
    Request body:
    {
        "session_id": "要清除的会话ID"
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
    获取会话历史（用于调试）
    
    Query params:
    - session_id: 会话ID
    """
    session_id = request.args.get('session_id', 'default')
    
    if session_id in conversations:
        # 过滤掉系统消息，只返回用户和助手的对话
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
    """列出所有活跃的会话"""
    sessions = []
    for sid, messages in conversations.items():
        # 计算用户消息数量
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

# ========== 原有的查询端点 ==========

@app.route('/api/query', methods=['POST'])
def query():
    """
    通用查询端点 - 可以扩展为结合数据库和 Ollama 的查询
    
    Request body:
    {
        "query": "查询内容",
        "use_ai": true/false,
        "session_id": "可选的会话ID"
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
        
        # 如果启用 AI，使用 Ollama 生成回答
        if use_ai and ollama.check_connection():
            # 获取或创建会话历史
            if session_id not in conversations:
                conversations[session_id] = [
                    {"role": "system", "content": "你是一位专业的助手，请根据用户的问题提供帮助。"}
                ]
            
            messages = conversations[session_id]
            messages.append({"role": "user", "content": query_text})
            
            # 调用 Ollama
            ai_response = ollama.chat(messages)
            messages.append({"role": "assistant", "content": ai_response})
            
            # 保存会话历史
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
    # 启动时检查服务状态
    print("=" * 50)
    print("启动 RAG QA Engine with Ollama")
    print("=" * 50)
    
    # 检查 Ollama
    print(f"   尝试连接: {OLLAMA_BASE_URL}")
    if not ollama.check_connection():
        print("⚠️  警告: Ollama 服务未运行!")
        print("   请在另一个终端运行: ollama serve")
        print("   然后拉取模型: ollama pull qwen2.5:0.5b")
        print(f"   当前配置的 Ollama URL: {OLLAMA_BASE_URL}")
    else:
        print("✅ Ollama 服务已连接")
        print(f"   Ollama URL: {OLLAMA_BASE_URL}")
        print(f"   使用模型: {OLLAMA_MODEL}")
    
    # 检查数据库
    try:
        if db.test_connection():
            print("✅ 数据库已连接")
        else:
            print("⚠️  警告: 数据库连接失败")
    except Exception as e:
        print(f"⚠️  警告: 数据库不可用 - {e}")
    
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)