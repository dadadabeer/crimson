import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import OpenAI from 'openai';
import './ChatPage.css';

const ChatPage = () => {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hello! I'm Keynes, your personal value investing AI assistant. How can I help you today?",
      sender: 'ai',
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);

  // Initialize OpenAI client
  const openai = new OpenAI({
    apiKey: process.env.REACT_APP_OPENAI_API_KEY,
    dangerouslyAllowBrowser: true // Note: This is for demo purposes only
  });

  // Handle file upload
  const handleFileUpload = async (files) => {
    if (!files || files.length === 0) return;
    
    setIsUploading(true);
    const formData = new FormData();
    
    // Add files to form data
    Array.from(files).forEach(file => {
      formData.append('files', file);
    });

    try {
      const response = await fetch('http://localhost:4000/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        
        // Add upload success message
        const uploadMessage = {
          id: messages.length + 1,
          text: `Successfully uploaded and processed ${result.indexed} file(s). You can now ask questions about the uploaded content.`,
          sender: 'ai',
          timestamp: new Date(),
          type: 'upload-success'
        };
        
        setMessages(prev => [...prev, uploadMessage]);
        
        // Update uploaded files list
        setUploadedFiles(prev => [...prev, ...Array.from(files)]);
      } else {
        throw new Error('Upload failed');
      }
    } catch (error) {
      console.error('Upload error:', error);
      const errorMessage = {
        id: messages.length + 1,
        text: "Sorry, I couldn't process the uploaded files. Please try again.",
        sender: 'ai',
        timestamp: new Date(),
        type: 'error'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsUploading(false);
    }
  };

  // Handle file input change
  const handleFileInputChange = (e) => {
    const files = e.target.files;
    handleFileUpload(files);
  };

  // Handle drag and drop
  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    handleFileUpload(files);
  };

  // Handle asking questions about uploaded content
  const handleAskQuestion = async (question) => {
    setIsLoading(true);
    
    try {
      const response = await fetch('http://localhost:4000/api/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question }),
      });

      if (response.ok) {
        const result = await response.json();
        
        const aiResponse = {
          id: messages.length + 1,
          text: result.answer,
          sender: 'ai',
          timestamp: new Date(),
          sources: result.sources
        };
        
        setMessages(prev => [...prev, aiResponse]);
      } else {
        throw new Error('Ask failed');
      }
    } catch (error) {
      console.error('Ask error:', error);
      const errorResponse = {
        id: messages.length + 1,
        text: "I'm sorry, I couldn't process your question. Please make sure you've uploaded some documents first.",
        sender: 'ai',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorResponse]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (inputMessage.trim() && !isLoading) {
      const newMessage = {
        id: messages.length + 1,
        text: inputMessage,
        sender: 'user',
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, newMessage]);
      const currentMessage = inputMessage;
      setInputMessage('');
      setIsLoading(true);
      
      try {
        // If we have uploaded files, use the backend API
        if (uploadedFiles.length > 0) {
          await handleAskQuestion(currentMessage);
        } else {
          // Use direct OpenAI API for general conversation
          const conversationHistory = messages.map(msg => ({
            role: msg.sender === 'user' ? 'user' : 'assistant',
            content: msg.text
          }));
          
          conversationHistory.push({
            role: 'user',
            content: currentMessage
          });

          const completion = await openai.chat.completions.create({
            model: "gpt-4",
            messages: [
              {
                role: "system",
                content: "You are Keynes, a personal value investing AI assistant. You help users with investment analysis, market insights, and financial advice. Be knowledgeable about value investing principles, market analysis, and financial planning. Keep responses concise but informative."
              },
              ...conversationHistory
            ],
            max_tokens: 500,
            temperature: 0.7,
          });

          const aiResponse = {
            id: messages.length + 2,
            text: completion.choices[0].message.content,
            sender: 'ai',
            timestamp: new Date()
          };
          
          setMessages(prev => [...prev, aiResponse]);
        }
      } catch (error) {
        console.error('Error calling API:', error);
        const errorResponse = {
          id: messages.length + 2,
          text: "I'm sorry, I'm having trouble connecting to my AI service right now. Please try again later.",
          sender: 'ai',
          timestamp: new Date()
        };
        setMessages(prev => [...prev, errorResponse]);
      } finally {
        setIsLoading(false);
      }
    }
  };

  return (
    <div className="chat-container">
      {/* Sidebar */}
      <div className="chat-sidebar">
        <div className="sidebar-header">
          <h1 className="keynes-logo">Keynes</h1>
          <button
            onClick={() => navigate('/')}
            className="back-button"
          >
            ← Back
          </button>
        </div>
        <div className="sidebar-content">
          <p className="sidebar-description">
            Your personal value investing AI assistant
          </p>
          
          {/* File Upload Section */}
          <div className="upload-section">
            <h3>Upload Documents</h3>
            <div 
              className="upload-area"
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept="image/*,.pdf"
                onChange={handleFileInputChange}
                style={{ display: 'none' }}
              />
              {isUploading ? (
                <p>Processing files...</p>
              ) : (
                <div>
                  <p>📁 Drag & drop files here</p>
                  <p>or click to browse</p>
                  <p className="upload-hint">Supports: Images, PDFs</p>
                </div>
              )}
            </div>
            
            {uploadedFiles.length > 0 && (
              <div className="uploaded-files">
                <h4>Uploaded Files:</h4>
                {uploadedFiles.map((file, index) => (
                  <div key={index} className="uploaded-file">
                    📄 {file.name}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="chat-main">
        {/* Chat Messages */}
        <div className="chat-messages">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`message ${message.sender === 'user' ? 'message-user' : 'message-ai'}`}
            >
              <div className={`message-bubble ${message.sender === 'user' ? 'bubble-user' : 'bubble-ai'}`}>
                <p className="message-text">{message.text}</p>
                {message.sources && (
                  <div className="sources">
                    <strong>Sources:</strong>
                    {message.sources.map((source, idx) => (
                      <div key={idx} className="source-item">
                        {source.meta.filename} (Score: {source.score})
                      </div>
                    ))}
                  </div>
                )}
                <span className="message-time">
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
              </div>
            </div>
          ))}
        </div>

        {/* Input Area */}
        <div className="chat-input-container">
          <form onSubmit={handleSendMessage} className="chat-input-form">
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder={isLoading ? "Keynes is thinking..." : "Type your message here..."}
              className="chat-input"
              disabled={isLoading}
            />
            <button type="submit" className="send-button" disabled={isLoading || !inputMessage.trim()}>
              {isLoading ? "..." : "Send"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default ChatPage; 