export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/chat/review': 'http://localhost:8000',  // ★ 新增
      '/chat': 'http://localhost:8000',
      '/chat/stream': 'http://localhost:8000',
      '/add_docs': 'http://localhost:8000',
      '/eval': 'http://localhost:8000'
    }
  }
})