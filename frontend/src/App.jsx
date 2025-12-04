import { useState, useRef, useEffect } from "react";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: "assistant",
      content: "Hello! I'm your legal advisor assistant. How can I help you?",
      timestamp: new Date(),
      citations: [],
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId] = useState(`session_${Date.now()}`); // Generate unique session ID
  const [backendStatus, setBackendStatus] = useState({
    ollama: false,
    checked: false,
  });
  const messagesEndRef = useRef(null);

  // API base URL - Configured via Docker environment variable
  // Connects to localhost:5001 (mapped to backend container:5000)
  const API_BASE_URL = import.meta.env.VITE_API_URL;

  // Auto-scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Check backend status
  useEffect(() => {
    checkBackendStatus();
    // Check status every 30 seconds
    const interval = setInterval(checkBackendStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkBackendStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/health`);
      if (response.ok) {
        const data = await response.json();
        setBackendStatus({
          ollama: data.services?.ollama === "connected",
          database: data.services?.database === "connected",
          model: data.services?.ollama_model || "N/A",
          checked: true,
        });

        // If first successful connection, show welcome message
        if (data.services?.ollama === "connected" && !backendStatus.ollama) {
          setMessages((prev) => [
            ...prev,
            {
              id: Date.now(),
              type: "assistant",
              content: `Successfully connected to Ollama service! Using model: ${
                data.services?.ollama_model || "qwen2.5:0.5b"
              }. You can start asking questions now.`,
              timestamp: new Date(),
              citations: [],
            },
          ]);
        }
      } else {
        setBackendStatus({ ollama: false, database: false, checked: true });
      }
    } catch (error) {
      console.error("Backend health check failed:", error);
      setBackendStatus({ ollama: false, database: false, checked: true });
    }
  };

  // Handle sending a message
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
      // Call backend Ollama API
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: userMessage.content,
          session_id: sessionId,
          system_prompt:
            "You are a professional legal advisor assistant. Please provide legal advice in easy-to-understand language. Note: You provide general legal information and advice, which does not constitute formal legal opinion. For complex legal issues, users should consult a professional lawyer.",
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(
          errorData?.error || `HTTP error! status: ${response.status}`
        );
      }

      const data = await response.json();

      // Add AI response
      const assistantMessage = {
        id: Date.now() + 1,
        type: "assistant",
        content: data.response || "Sorry, unable to get a response.",
        timestamp: new Date(),
        citations: [], // Can add citation feature later
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Error sending message:", error);

      // Error message
      const errorMessage = {
        id: Date.now() + 1,
        type: "assistant",
        content: `❌ Error: ${error.message}\n\nPlease ensure:\n1. Ollama service is started on host (ollama serve)\n2. Model is downloaded (ollama pull qwen2.5:0.5b)\n3. Docker services are running (docker-compose up)`,
        timestamp: new Date(),
        citations: [],
        isError: true,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle clearing the chat
  const handleClearChat = async () => {
    // Clear backend session
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

    // Clear frontend messages
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

  // Handle key press events (support Enter to send)
  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(e);
    }
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
            <h1>Legal Advisor Assistant</h1>
            {backendStatus.checked && (
              <span
                className={`status-badge ${
                  backendStatus.ollama ? "connected" : "disconnected"
                }`}
                title={
                  backendStatus.ollama
                    ? `Connected - Model: ${backendStatus.model}`
                    : "Not connected to Ollama"
                }
              >
                {backendStatus.ollama ? "● Online" : "● Offline"}
              </span>
            )}
          </div>
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
      </header>

      {/* Messages Container */}
      <main className="messages-container">
        <div className="messages">
          {/* Show alert if not connected */}
          {backendStatus.checked && !backendStatus.ollama && (
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
                <strong>Not connected to Ollama service</strong>
                <p>Please follow these steps to start the service:</p>
                <ol>
                  <li>
                    Start Ollama on host machine: <code>ollama serve</code>
                  </li>
                  <li>
                    Pull the model: <code>ollama pull qwen2.5:0.5b</code>
                  </li>
                  <li>
                    Start Docker services: <code>docker-compose up</code>
                  </li>
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
            title={!backendStatus.ollama ? "Ollama service not connected" : "Send message"}
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
