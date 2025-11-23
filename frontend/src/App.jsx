import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [count, setCount] = useState(0)
  const [apiMessage, setApiMessage] = useState('')
  const [apiStatus, setApiStatus] = useState('loading')

  useEffect(() => {
    const apiUrl = `http://${window.location.hostname}:5000/api/health`
    fetch(apiUrl)
      .then(res => res.json())
      .then(data => {
        setApiMessage(`Backend: ${data.status}`)
        setApiStatus('connected')
      })
      .catch((error) => {
        setApiMessage('Backend: disconnected')
        setApiStatus('error')
        console.error('API Error:', error)
      })
  }, [])

  return (
    <div className="App">
      <header className="App-header">
        <h1>RAG QA Engine</h1>
        <p className={`status-${apiStatus}`}>
          {apiMessage}
        </p>
        <div className="card">
          <button onClick={() => setCount((count) => count + 1)}>
            count is {count}
          </button>
        </div>
        <p>
          Edit <code>src/App.jsx</code> and save to test HMR
        </p>
      </header>
    </div>
  )
}

export default App

