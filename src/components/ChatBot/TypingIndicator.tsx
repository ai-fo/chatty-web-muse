
import React from "react";

const TypingIndicator = () => {
  return (
    <div className="flex w-full mb-4 justify-start">
      <div className="bg-white border border-gray-200 shadow-sm rounded-2xl px-4 py-3">
        <div className="typing-indicator">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>
    </div>
  );
};

export default TypingIndicator;
