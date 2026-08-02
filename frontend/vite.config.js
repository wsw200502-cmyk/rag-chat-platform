import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/chat': 'http://localhost:8080',
      '/add_docs': 'http://localhost:8080',
      '/clear': 'http://localhost:8080',
      '/sessions': 'http://localhost:8080'
    }
  }
})