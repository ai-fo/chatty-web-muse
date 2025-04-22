
import React from "react";
import { cn } from "@/lib/utils";
import { Bot } from "lucide-react";

type MessageProps = {
  content: string;
  isBot?: boolean;
  timestamp?: string;
};

const Message = ({ content, isBot = false, timestamp }: MessageProps) => {
  return (
    <div
      className={cn(
        "flex w-full mb-4 chat-message-animation",
        isBot ? "justify-start" : "justify-end"
      )}
    >
      <div
        className={cn(
          "flex max-w-[80%] rounded-2xl px-4 py-3",
          isBot
            ? "bg-white border border-gray-200 shadow-sm text-gray-800"
            : "bg-gradient-to-r from-chatbot-primary to-chatbot-secondary text-white"
        )}
      >
        {isBot && (
          <div className="flex-shrink-0 mr-3">
            <Bot size={20} className="text-chatbot-primary" />
          </div>
        )}
        <div className="flex flex-col">
          <div className="text-sm">{content}</div>
          {timestamp && (
            <div className={cn("text-xs mt-1", isBot ? "text-gray-400" : "text-white/70")}>
              {timestamp}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Message;
