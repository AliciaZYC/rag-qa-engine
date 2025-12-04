# LangChain Integration

## Overview

This application now uses **LangChain** for LLM interactions with Ollama, providing a more robust and extensible framework for AI operations.

## What Changed

### Before (Direct API Calls)
- Used `requests` library to directly call Ollama's REST API
- Manual message formatting and error handling
- Limited abstraction for future enhancements

### After (LangChain Integration)
- Uses `langchain-community` ChatOllama integration
- Structured message types (SystemMessage, HumanMessage, AIMessage)
- Easy to extend with chains, agents, and RAG capabilities later

## Architecture

```
Frontend (React)
    ↓
Backend Flask API
    ↓
LangChain ChatOllama
    ↓
Ollama Service (localhost:11434)
    ↓
LLM Model (qwen2.5:0.5b)
```

## Key Components

### 1. LangChainOllamaClient Class

Located in `backend/app.py`:

```python
class LangChainOllamaClient:
    def __init__(self):
        self.llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=0.7,
            num_predict=1000,
        )
    
    def chat(self, messages: List[Dict]):
        # Converts standard message format to LangChain format
        # Returns AI response
```

### 2. Message Format Conversion

The client automatically converts between:
- **Standard format**: `{"role": "user", "content": "..."}`
- **LangChain format**: `HumanMessage(content="...")`

### 3. API Endpoints

All existing endpoints work the same:
- `/api/chat` - Contextual conversation
- `/api/chat/simple` - Single-turn Q&A
- `/api/query` - General query endpoint

## Testing

Run the test script to verify integration:

```bash
cd backend
python test_langchain.py
```

## Dependencies Added

```
langchain==0.1.0
langchain-community==0.0.10
requests==2.31.0
```

## Future Enhancements (RAG Ready)

With LangChain integrated, we can easily add:

1. **Vector Store Integration**
   - Connect pgvector with LangChain
   - Use `PGVector` from `langchain-community`

2. **Retrieval Chains**
   - `RetrievalQA` chain for RAG
   - Custom prompts for legal domain

3. **Memory Management**
   - `ConversationBufferMemory`
   - `ConversationSummaryMemory`

4. **Advanced Features**
   - Multi-query retrieval
   - Contextual compression
   - Re-ranking

## Configuration

Environment variables (set in `docker-compose.yml`):

```yaml
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=qwen2.5:0.5b
```

## Benefits

1. **Abstraction**: Clean interface for LLM operations
2. **Extensibility**: Easy to add chains, agents, tools
3. **Compatibility**: Works with multiple LLM providers
4. **Community**: Access to LangChain ecosystem
5. **RAG Ready**: Foundation for retrieval-augmented generation

## Next Steps

1. ✅ LangChain integration complete
2. ⏳ Add vector store retrieval (RAG)
3. ⏳ Implement document chunking strategy
4. ⏳ Create retrieval chains
5. ⏳ Add re-ranking and filtering

