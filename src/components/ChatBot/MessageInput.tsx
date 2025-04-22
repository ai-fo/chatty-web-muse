
import React, { useState } from "react";
import { Send } from "lucide-react";

type MessageInputProps = {
  onSendMessage: (message: string) => void;
};

const MessageInput = ({ onSendMessage }: MessageInputProps) => {
  const [message, setMessage] = useState("");

  const handleSendMessage = () => {
    if (message.trim()) {
      onSendMessage(message);
      setMessage("");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="border-t bg-white p-4 flex items-center gap-2">
      <input
        type="text"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Type your message..."
        className="flex-1 border rounded-full px-4 py-2 focus:outline-none focus:ring-2 focus:ring-chatbot-primary/40"
      />
      <button
        onClick={handleSendMessage}
        disabled={!message.trim()}
        className="bg-chatbot-primary hover:bg-chatbot-secondary text-white rounded-full p-2 transition-colors duration-200 disabled:opacity-50"
      >
        <Send size={20} />
      </button>
    </div>
  );
};

export default MessageInput;
