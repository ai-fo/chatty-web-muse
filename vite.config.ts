
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

// Configuration simplifiée pour minimiser les problèmes de dépendances
export default defineConfig({
  server: {
    host: "localhost",
    port: 8080,
  },
  plugins: [
    react(),
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  // Désactiver les optimisations qui peuvent causer des problèmes
  optimizeDeps: {
    disabled: true
  },
  // Simplifier la construction
  build: {
    sourcemap: false,
    minify: false
  }
});
