
import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import './index.css'

// Using React 18's createRoot API
const root = createRoot(document.getElementById("root")!);
root.render(<App />);
