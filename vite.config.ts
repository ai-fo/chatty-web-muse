
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

// Updated configuration to fix JSX runtime issues
export default defineConfig({
  server: {
    host: "localhost",
    port: 8080,
  },
  plugins: [
    react({
      jsxImportSource: "react", 
      jsxRuntime: "automatic"
    }),
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  // Enable optimizeDeps for better dependency handling
  optimizeDeps: {
    include: ["react", "react-dom"],
    esbuildOptions: {
      // Configure JSX in esbuild to ensure proper transformation
      jsx: "automatic",
    }
  },
  // Improved build settings
  build: {
    sourcemap: true,
    minify: "esbuild",
    rollupOptions: {
      // Ensure proper externalization of React
      external: [],
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom'],
        },
      }
    }
  }
});
