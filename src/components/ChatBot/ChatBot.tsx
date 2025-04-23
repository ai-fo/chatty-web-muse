
import React, { useState, useEffect, useRef } from "react";
import { useToast } from "@/hooks/use-toast";
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
  const { toast } = useToast();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleBotResponse = async (userMessage: string) => {
    setIsTyping(true);
    
    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: userMessage }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response from bot');
      }

      const botResponses: string[] = await response.json();
      
      // Add each part of the response as a separate message
      for (const responseContent of botResponses) {
        setMessages((prev) => [
          ...prev,
          {
            id: Date.now().toString() + Math.random(),
            content: responseContent,
            isBot: true,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          },
        ]);
        // Small delay between messages for better UX
        await new Promise(resolve => setTimeout(resolve, 500));
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to get response from the chatbot. Please try again.",
        variant: "destructive",
      });
      console.error("Chat error:", error);
    } finally {
      setIsTyping(false);
    }
  };

  const handleSendMessage = (content: string) => {
    const newMessage: ChatMessage = {
      id: Date.now().toString(),
      content,
      isBot: false,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };
    
    setMessages((prev) => [...prev, newMessage]);
    handleBotResponse(content);
  };

  return (
    <div className="flex flex-col bg-chatbot-light h-[600px] rounded-xl shadow-lg overflow-hidden border border-gray-200">
      <div className="bg-white border-b px-4 py-3">
        <h2 className="font-semibold text-lg">ChatHotline</h2>
        <p className="text-sm text-gray-500">Ask me anything about your documents</p>
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

