import { useState, useRef, useEffect } from "react";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: "assistant",
      content: "Hello! I'm your legal advisor assistant with RAG-powered retrieval. Ask me about legal contracts and I'll provide answers with citations from actual documents.",
      timestamp: new Date(),
      citations: [],
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId] = useState(`session_${Date.now()}`);
  const [backendStatus, setBackendStatus] = useState({
    ollama: false,
    database: false,
    embedder: false,
    checked: false,
  });
  const [useRAG, setUseRAG] = useState(true); // Toggle for RAG vs simple chat
  const [expandedCitations, setExpandedCitations] = useState({}); // Track expanded citations
  const messagesEndRef = useRef(null);

  const API_BASE_URL = import.meta.env.VITE_API_URL;

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    checkBackendStatus();
  }, []);

  const checkBackendStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/health`);
      if (response.ok) {
        const data = await response.json();
        setBackendStatus({
          ollama: data.services?.ollama === "connected",
          database: data.services?.database === "connected",
          embedder: data.services?.embedder === "loaded",
          model: data.services?.ollama_model || "N/A",
          embeddingModel: data.services?.embedding_model || "N/A",
          checked: true,
        });
      } else {
        setBackendStatus({ 
          ollama: false, 
          database: false, 
          embedder: false,
          checked: true 
        });
      }
    } catch (error) {
      console.error("Backend health check failed:", error);
      setBackendStatus({ 
        ollama: false, 
        database: false, 
        embedder: false,
        checked: true 
      });
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      type: "user",
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      // Choose endpoint based on RAG toggle and system availability
      const endpoint = useRAG && backendStatus.embedder && backendStatus.database
        ? `${API_BASE_URL}/api/chat/rag`
        : `${API_BASE_URL}/api/chat`;

      const requestBody = useRAG && backendStatus.embedder && backendStatus.database
        ? {
            message: userMessage.content,
            top_k: 5,
            session_id: sessionId,
            include_sources: true
          }
        : {
            message: userMessage.content,
            session_id: sessionId,
            system_prompt: "You are a professional legal advisor assistant."
          };

      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(
          errorData?.error || `HTTP error! status: ${response.status}`
        );
      }

      const data = await response.json();

      // Build assistant message with sources if available
      const assistantMessage = {
        id: Date.now() + 1,
        type: "assistant",
        content: data.response || "Sorry, unable to get a response.",
        timestamp: new Date(),
        citations: data.sources || [],
        modelInfo: data.model_info || null,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Error sending message:", error);

      const errorMessage = {
        id: Date.now() + 1,
        type: "assistant",
        content: `❌ Error: ${error.message}\n\nPlease ensure:\n1. Backend is running (docker-compose up)\n2. Ollama service is started (ollama serve)\n3. Model is downloaded (ollama pull qwen2.5:0.5b)\n4. Data is ingested (POST /api/db/ingest)`,
        timestamp: new Date(),
        citations: [],
        isError: true,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearChat = async () => {
    try {
      await fetch(`${API_BASE_URL}/api/chat/clear`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: sessionId,
        }),
      });
    } catch (error) {
      console.error("Error clearing chat:", error);
    }

    setMessages([
      {
        id: Date.now(),
        type: "assistant",
        content: "Chat cleared. Do you have any new questions?",
        timestamp: new Date(),
        citations: [],
      },
    ]);
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(e);
    }
  };

  // Helper to format similarity score
  const formatSimilarity = (score) => {
    return (score * 100).toFixed(1);
  };

  // Toggle citation expansion
  const toggleCitation = (messageId, citationIdx) => {
    const key = `${messageId}-${citationIdx}`;
    setExpandedCitations(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="header-title">
            <svg
              className="header-icon"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
            <h1>Legal RAG Assistant</h1>
            {backendStatus.checked && (
              <span
                className={`status-badge ${
                  backendStatus.ollama && backendStatus.embedder
                    ? "connected"
                    : "disconnected"
                }`}
                title={
                  backendStatus.ollama && backendStatus.embedder
                    ? `RAG Ready - ${backendStatus.embeddingModel}`
                    : "System not fully ready"
                }
              >
                {backendStatus.ollama && backendStatus.embedder 
                  ? "● RAG Ready" 
                  : "● Offline"}
              </span>
            )}
          </div>
          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
            {/* RAG Toggle */}
            {backendStatus.embedder && backendStatus.database && (
              <label 
                style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '0.5rem',
                  fontSize: '0.875rem',
                  color: '#9ca3af',
                  cursor: 'pointer'
                }}
              >
                <input
                  type="checkbox"
                  checked={useRAG}
                  onChange={(e) => setUseRAG(e.target.checked)}
                  style={{ cursor: 'pointer' }}
                />
                <span>Use RAG (Retrieval)</span>
              </label>
            )}
            <button
              className="clear-button"
              onClick={handleClearChat}
              title="Clear chat"
            >
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
              </svg>
            </button>
          </div>
        </div>
      </header>

      {/* Messages Container */}
      <main className="messages-container">
        <div className="messages">
          {/* System Status Alert */}
          {backendStatus.checked && (!backendStatus.ollama || !backendStatus.embedder || !backendStatus.database) && (
            <div className="connection-alert">
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                <line x1="12" y1="9" x2="12" y2="13" />
                <line x1="12" y1="17" x2="12.01" y2="17" />
              </svg>
              <div>
                <strong>System Status</strong>
                <p>Some components are not ready:</p>
                <ol>
                  {!backendStatus.ollama && (
                    <li>
                      <code>ollama serve</code> - Start Ollama service
                    </li>
                  )}
                  {!backendStatus.embedder && (
                    <li>
                      <code>Embedder not loaded</code> - Check backend logs
                    </li>
                  )}
                  {!backendStatus.database && (
                    <li>
                      <code>docker-compose exec backend python db/ingest_data_v2.py</code> - Run data ingestion
                    </li>
                  )}
                </ol>
              </div>
            </div>
          )}

          {messages.map((message) => (
            <div key={message.id} className={`message-wrapper ${message.type}`}>
              <div className={`message ${message.isError ? "error" : ""}`}>
                <div className="message-avatar">
                  {message.type === "user" ? (
                    <svg
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                    >
                      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                      <circle cx="12" cy="7" r="4" />
                    </svg>
                  ) : (
                    <svg
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                    >
                      <circle cx="12" cy="12" r="10" />
                      <path d="M8 14s1.5 2 4 2 4-2 4-2" />
                      <line x1="9" y1="9" x2="9.01" y2="9" />
                      <line x1="15" y1="9" x2="15.01" y2="9" />
                    </svg>
                  )}
                </div>
                <div className="message-content">
                  <div className="message-header">
                    <span className="message-sender">
                      {message.type === "user" ? "You" : "AI Assistant"}
                    </span>
                    <span className="message-time">
                      {message.timestamp.toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </span>
                  </div>
                  <p
                    className="message-text"
                    style={{ whiteSpace: "pre-wrap" }}
                  >
                    {message.content}
                  </p>

                  {/* Citations Section - NEW */}
                  {message.citations && message.citations.length > 0 && (
                    <div className="citations">
                      <div className="citations-header">
                        <svg
                          className="citations-icon"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                        >
                          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                          <polyline points="14 2 14 8 20 8" />
                          <line x1="16" y1="13" x2="8" y2="13" />
                          <line x1="16" y1="17" x2="8" y2="17" />
                          <polyline points="10 9 9 9 8 9" />
                        </svg>
                        <span>
                          Referenced Documents ({message.citations.length})
                        </span>
                        {message.modelInfo && (
                          <span style={{ 
                            marginLeft: 'auto', 
                            fontSize: '0.75rem',
                            color: '#10b981'
                          }}>
                            Top Match: {formatSimilarity(message.modelInfo.top_similarity)}%
                          </span>
                        )}
                      </div>
                      <div className="citations-list">
                        {message.citations.slice(0, 5).map((citation, idx) => {
                          const citationKey = `${message.id}-${idx}`;
                          const isExpanded = expandedCitations[citationKey];
                          const shouldTruncate = citation.content.length > 200;
                          
                          return (
                            <div 
                              key={idx} 
                              className="citation-item"
                              onClick={() => shouldTruncate && toggleCitation(message.id, idx)}
                              style={{ 
                                cursor: shouldTruncate ? 'pointer' : 'default',
                                transition: 'all 0.2s ease'
                              }}
                            >
                              <div className="citation-header">
                                <span className="citation-source">
                                  Document {citation.rank}
                                  {citation.provision_label && 
                                    ` - ${citation.provision_label}`}
                                  {citation.section_number && 
                                    ` (${citation.section_number})`}
                                </span>
                                <span className="citation-relevance">
                                  {formatSimilarity(citation.similarity)}% match
                                </span>
                              </div>
                              <p className="citation-excerpt">
                                {isExpanded || !shouldTruncate
                                  ? citation.content
                                  : citation.content.substring(0, 200) + "..."}
                              </p>
                              {citation.token_count && (
                                <div style={{ 
                                  fontSize: '0.75rem', 
                                  color: '#6b7280',
                                  marginTop: '0.5rem',
                                  display: 'flex',
                                  gap: '1rem'
                                }}>
                                  <span>ID: {citation.id}</span>
                                  <span>Tokens: {citation.token_count}</span>
                                  <span>Method: {citation.chunk_method}</span>
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                      {/* Model Info */}
                      {message.modelInfo && (
                        <div style={{ 
                          marginTop: '0.75rem',
                          paddingTop: '0.75rem',
                          borderTop: '1px solid rgba(255, 255, 255, 0.1)',
                          fontSize: '0.75rem',
                          color: '#6b7280',
                          textAlign: 'center'
                        }}>
                          {message.modelInfo.embedding_model} ({message.modelInfo.embedding_dimension}D) + {message.modelInfo.llm_model}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}

          {/* Loading Indicator */}
          {isLoading && (
            <div className="message-wrapper assistant">
              <div className="message">
                <div className="message-avatar">
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                  >
                    <circle cx="12" cy="12" r="10" />
                    <path d="M8 14s1.5 2 4 2 4-2 4-2" />
                    <line x1="9" y1="9" x2="9.01" y2="9" />
                    <line x1="15" y1="9" x2="15.01" y2="9" />
                  </svg>
                </div>
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Input Area */}
      <footer className="input-container">
        <form className="input-form" onSubmit={handleSendMessage}>
          <input
            type="text"
            className="input-field"
            placeholder={
              !backendStatus.checked
                ? "Connecting to server..."
                : !backendStatus.ollama
                ? "Please start Ollama service first..."
                : useRAG && backendStatus.embedder
                ? "Ask about legal contracts (RAG-powered)..."
                : "Enter your legal question..."
            }
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={isLoading || !backendStatus.ollama}
          />
          <button
            type="submit"
            className="send-button"
            disabled={!input.trim() || isLoading || !backendStatus.ollama}
            title={
              !backendStatus.ollama
                ? "Ollama service not connected"
                : useRAG && !backendStatus.embedder
                ? "RAG not available, will use simple chat"
                : "Send message"
            }
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        </form>
      </footer>
    </div>
  );
}

export default App;