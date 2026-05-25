import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev proxy: vite serves the SPA on :5173 and forwards /api/* to the
// bare-metal FastAPI on :8000. Production builds are served as static
// files by FastAPI itself (DESIGN §4.8), so the proxy is dev-only.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
