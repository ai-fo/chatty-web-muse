
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

// Configuration with componentTagger for Lovable features
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8080,
  },
  plugins: [
    react({
      jsxImportSource: "react"
    }),
    mode === 'development' && componentTagger(),
  ].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  // Enable optimizeDeps for better dependency handling
  optimizeDeps: {
    include: ["react", "react-dom"],
    esbuildOptions: {
      jsx: "automatic",
    }
  },
  // Improved build settings
  build: {
    sourcemap: true,
    minify: "esbuild",
    rollupOptions: {
      external: [],
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom'],
        },
      }
    }
  }
}));
