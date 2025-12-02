import { useState, useRef, useEffect } from "react";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: "assistant",
      content: "你好！我是你的法律顾问助手。请问有什么可以帮助你的？",
      timestamp: new Date(),
      citations: [],
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId] = useState(`session_${Date.now()}`); // 生成唯一的会话ID
  const [backendStatus, setBackendStatus] = useState({
    ollama: false,
    checked: false,
  });
  const messagesEndRef = useRef(null);

  // API 基础URL
  const API_BASE_URL = "http://localhost:5001";

  // Auto-scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // 检查后端状态
  useEffect(() => {
    checkBackendStatus();
    // 每30秒检查一次状态
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

        // 如果是第一次成功连接，显示欢迎消息
        if (data.services?.ollama === "connected" && !backendStatus.ollama) {
          setMessages((prev) => [
            ...prev,
            {
              id: Date.now(),
              type: "assistant",
              content: `已成功连接到 Ollama 服务！使用模型：${
                data.services?.ollama_model || "qwen2.5:0.5b"
              }。你可以开始提问了。`,
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
      // 调用后端 Ollama API
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: userMessage.content,
          session_id: sessionId,
          system_prompt:
            "你是一位专业的法律顾问助手。请用通俗易懂的语言为用户提供法律建议。注意：你提供的是一般性法律信息和建议，不构成正式的法律意见。对于复杂的法律问题，建议用户咨询专业律师。",
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(
          errorData?.error || `HTTP error! status: ${response.status}`
        );
      }

      const data = await response.json();

      // 添加 AI 回复
      const assistantMessage = {
        id: Date.now() + 1,
        type: "assistant",
        content: data.response || "抱歉，无法获取回复。",
        timestamp: new Date(),
        citations: [], // 可以后续添加引用功能
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Error sending message:", error);

      // 错误消息
      const errorMessage = {
        id: Date.now() + 1,
        type: "assistant",
        content: `❌ 错误：${error.message}\n\n请确保：\n1. 后端服务正在运行 (python app.py)\n2. Ollama 服务已启动 (ollama serve)\n3. 已下载模型 (ollama pull qwen2.5:0.5b)`,
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
    // 清除后端会话
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

    // 清除前端消息
    setMessages([
      {
        id: Date.now(),
        type: "assistant",
        content: "对话已清除。有什么新的问题吗？",
        timestamp: new Date(),
        citations: [],
      },
    ]);
  };

  // 处理按键事件（支持 Ctrl+Enter 发送）
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
            <h1>法律咨询助手</h1>
            {backendStatus.checked && (
              <span
                className={`status-badge ${
                  backendStatus.ollama ? "connected" : "disconnected"
                }`}
                title={
                  backendStatus.ollama
                    ? `已连接 - 模型: ${backendStatus.model}`
                    : "未连接到 Ollama"
                }
              >
                {backendStatus.ollama ? "● 在线" : "● 离线"}
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
          {/* 如果没有连接，显示提示 */}
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
                <strong>未连接到 Ollama 服务</strong>
                <p>请按以下步骤启动服务：</p>
                <ol>
                  <li>
                    终端1：运行 <code>ollama serve</code>
                  </li>
                  <li>
                    终端2：运行 <code>ollama pull qwen2.5:0.5b</code>
                  </li>
                  <li>
                    终端3：运行 <code>python app.py</code>
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
                      {message.type === "user" ? "你" : "AI 助手"}
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
                ? "正在连接服务器..."
                : !backendStatus.ollama
                ? "请先启动 Ollama 服务..."
                : "输入你的法律问题..."
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
            title={!backendStatus.ollama ? "Ollama 服务未连接" : "发送消息"}
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
