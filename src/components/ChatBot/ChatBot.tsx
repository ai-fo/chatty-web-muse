
import React, { useState, useEffect, useRef } from "react";
import Message from "./Message";
import MessageInput from "./MessageInput";
import TypingIndicator from "./TypingIndicator";

type ChatMessage = {
  id: string;
  content: string;
  isBot: boolean;
  timestamp: string;
};

const ChatBot = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "1",
      content: "Hello! How can I help you today?",
      isBot: true,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ]);
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const simulateBotResponse = (userMessage: string) => {
    setIsTyping(true);
    
    // Simulate thinking time
    setTimeout(() => {
      const responses = [
        "That's interesting! Tell me more.",
        "I understand. How can I help you further?",
        "I'm here to assist you with any questions.",
        "Could you provide more details?",
        "That's a great question! Let me help you with that.",
      ];
      
      const randomResponse = responses[Math.floor(Math.random() * responses.length)];
      
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          content: randomResponse,
          isBot: true,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        },
      ]);
      
      setIsTyping(false);
    }, 1500); // Simulate 1.5 seconds delay
  };

  const handleSendMessage = (content: string) => {
    const newMessage: ChatMessage = {
      id: Date.now().toString(),
      content,
      isBot: false,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };
    
    setMessages((prev) => [...prev, newMessage]);
    simulateBotResponse(content);
  };

  return (
    <div className="flex flex-col bg-chatbot-light h-[600px] rounded-xl shadow-lg overflow-hidden border border-gray-200">
      <div className="bg-white border-b px-4 py-3">
        <h2 className="font-semibold text-lg">ModernChat</h2>
        <p className="text-sm text-gray-500">Ask me anything</p>
      </div>
      
      <div className="flex-1 overflow-y-auto p-4 bg-gradient-to-b from-white to-chatbot-light/50">
        {messages.map((message) => (
          <Message
            key={message.id}
            content={message.content}
            isBot={message.isBot}
            timestamp={message.timestamp}
          />
        ))}
        
        {isTyping && <TypingIndicator />}
        <div ref={messagesEndRef} />
      </div>
      
      <MessageInput onSendMessage={handleSendMessage} />
    </div>
  );
};

export default ChatBot;
