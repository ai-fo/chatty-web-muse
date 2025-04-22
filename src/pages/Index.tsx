
import ChatBot from "../components/ChatBot/ChatBot";

const Index = () => {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-white to-chatbot-light p-4">
      <div className="w-full max-w-md">
        <ChatBot />
      </div>
    </div>
  );
};

export default Index;
