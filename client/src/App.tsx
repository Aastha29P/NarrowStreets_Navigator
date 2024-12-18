import { useState, useRef, useEffect } from 'react'
import './App.css'

interface Message {
  role: 'user' | 'assistant'
  content: string
  images?: string[]
}

const Logo = () => (
  <div className="flex items-center">
    <div className="flex items-center bg-white/90 px-6 py-3 rounded-xl shadow-md">
      {/* Normal n */}
      <div className="logo-n text-orange-400 text-6xl">
        n
      </div>
      {/* Connected text */}
      <div className="flex flex-col">
        <div className="logo-text text-orange-400 text-3xl leading-tight tracking-wide -ml-1">arrow</div>
        <div className="text-green-600 text-sm font-medium tracking-[0.2em] uppercase">Streets Navigator</div>
      </div>
    </div>
  </div>
)

function App() {
  const [message, setMessage] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!message.trim() || isLoading) return

    const userMessage = message
    setMessage('')
    setMessages(prev => [...prev, { role: 'user', content: userMessage }])
    setIsLoading(true)

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userMessage }),
      })

      if (!response.ok) throw new Error('Network response was not ok')

      const data = await response.json()
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: data.response,
        images: data.images 
      }])
    } catch (error) {
      console.error('Error:', error)
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error. Please try again.' 
      }])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-orange-50">
      <div className="container mx-auto max-w-4xl p-4">
        <div className="bg-white rounded-xl shadow-2xl overflow-hidden">
          {/* Chat Header */}
          <div className="bg-gradient-to-r from-orange-300 to-orange-400 p-8">
            <div className="flex items-center justify-center">
              <Logo />
            </div>
            <p className="text-white text-center mt-4 text-sm font-medium">Your Friendly Travel Companion 🌎</p>
          </div>

          {/* Chat Messages */}
          <div className="h-[500px] overflow-y-auto p-6 bg-gray-50">
            {messages.length === 0 ? (
              <div className="text-center space-y-6 py-8">
                <div className="bg-white rounded-2xl p-8 shadow-md max-w-lg mx-auto">
                  <h2 className="text-xl font-semibold text-gray-800 mb-4">Hi there! I'm your NarrowStreets Navigator!</h2>
                  <p className="text-gray-600 mb-6">I can help plan your perfect trip! Try asking me:</p>
                  <div className="space-y-3">
                    <div className="bg-orange-50 p-4 rounded-lg hover:bg-orange-100 transition-colors cursor-pointer"
                         onClick={() => setMessage("Tell me about Taj Mahal and its history.")}>
                      <p className="text-orange-500">🗾 "Tell me about Taj Mahal and its history."</p>
                    </div>
                    <div className="bg-orange-50 p-4 rounded-lg hover:bg-orange-100 transition-colors cursor-pointer"
                         onClick={() => setMessage("What are the must-visit places in Delhi?")}>
                      <p className="text-orange-500">🗼 "What are the must-visit places in Delhi?"</p>
                    </div>
                    <div className="bg-orange-50 p-4 rounded-lg hover:bg-orange-100 transition-colors cursor-pointer"
                         onClick={() => setMessage("Show me some beautiful temples in South India.")}>
                      <p className="text-orange-500">🏛️ "Show me some beautiful temples in South India."</p>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                {messages.map((msg, index) => (
                  <div
                    key={index}
                    className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[80%] rounded-2xl px-4 py-2 ${
                        msg.role === 'user'
                          ? 'bg-orange-400 text-white rounded-br-none'
                          : 'bg-white shadow-md text-gray-800 rounded-bl-none'
                      }`}
                    >
                      <p className="whitespace-pre-wrap text-sm leading-relaxed">{msg.content}</p>
                      {msg.images && msg.images.length > 0 && (
                        <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-2">
                          {msg.images.map((image, imgIndex) => (
                            <img
                              key={imgIndex}
                              src={image}
                              alt="Place"
                              className="rounded-lg w-full h-48 object-cover"
                              loading="lazy"
                            />
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-white shadow-md rounded-2xl px-4 py-2 rounded-bl-none">
                  <div className="flex space-x-2">
                    <div className="w-2 h-2 bg-orange-400 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-orange-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    <div className="w-2 h-2 bg-orange-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Chat Input */}
          <div className="border-t p-4 bg-white">
            <form onSubmit={handleSubmit} className="flex space-x-4">
              <input
                type="text"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Ask me to create a travel itinerary for you! ✈️"
                className="flex-1 p-4 border rounded-xl focus:outline-none focus:ring-2 focus:ring-orange-400 bg-gray-50"
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={isLoading || !message.trim()}
                className="bg-orange-400 text-white px-6 py-4 rounded-xl hover:bg-orange-500 disabled:bg-orange-200 transition-colors flex items-center space-x-2"
              >
                {isLoading ? (
                  <span>Creating... ✨</span>
                ) : (
                  <>
                    <span>Send</span>
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                      <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" />
                    </svg>
                  </>
                )}
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
