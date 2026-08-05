@echo off
title Miku AI 智能助手

cd /d "%~dp0"

echo ========================================
echo   Miku AI 智能助手 - 一键启动
echo ========================================
echo.

:: ========== 环境变量 ==========
set OLLAMA_HOST=127.0.0.1:11434
set OLLAMA_MAX_LOADED_MODELS=1
set OLLAMA_NUM_PARALLEL=1
set OLLAMA_CONTEXT_LENGTH=8192
set HTTP_PROXY=
set HTTPS_PROXY=
set NO_PROXY=localhost,127.0.0.1,::1
:: 已去掉 OLLAMA_NUM_GPU，让 Ollama 自动管理 GPU/CPU 分层
:: MAX_LOADED_MODELS=1：显存只保留一个模型，避免多模型抢显存导致卡顿
:: CONTEXT_LENGTH=8192：限制上下文长度，KV Cache 显存占用降到 1/16

:: ========== 清理残留进程 ==========
echo [0/3] 清理残留进程...
taskkill /F /IM node.exe >nul 2>&1
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM ollama.exe >nul 2>&1
timeout /t 2 /nobreak >nul

:: ========== 清理 Python 缓存 ==========
del /s /q *.pyc >nul 2>&1
for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d" >nul 2>&1

:: ========== 清理 Chroma 临时文件 ==========
if exist "chroma_db" (
    attrib -R "chroma_db\*.*" /S /D >nul 2>&1
    del /F /Q "chroma_db\*.sqlite3-journal" >nul 2>&1
    del /F /Q "chroma_db\*.sqlite3-wal" >nul 2>&1
    del /F /Q "chroma_db\*.sqlite3-shm" >nul 2>&1
)

:: ========== 彻底删除 Vite 缓存 ==========
echo [0/3] 清理 Vite 缓存...
if exist "frontend\node_modules\.vite" (
    attrib -R "frontend\node_modules\.vite\*.*" /S /D >nul 2>&1
    rmdir /s /q "frontend\node_modules\.vite" >nul 2>&1
)
if exist "frontend\dist" rmdir /s /q "frontend\dist" >nul 2>&1

echo [OK] 清理完成
echo.

:: ========== 启动 Ollama ==========
echo [1/3] 启动 Ollama...
start "Ollama" ollama serve
timeout /t 8 /nobreak >nul

ollama list >nul 2>&1
if errorlevel 1 (
    echo [WARN] Ollama 可能未正常启动，请检查窗口
) else (
    echo [OK] Ollama 已就绪
)

:: ========== 预热 9B 模型并设为常驻 ==========
echo 正在加载主力模型 qwen3.5:9b 并设为常驻...
powershell -Command "Invoke-RestMethod -Uri http://127.0.0.1:11434/api/generate -Method Post -Body '{\"model\": \"qwen3.5:9b\", \"keep_alive\": -1, \"prompt\": \"\"}' | Out-Null"
if %errorlevel% equ 0 (
    echo [OK] 9B 模型已常驻显存
) else (
    echo [WARN] 9B 模型预热失败，将在首次请求时自动加载
)

:: ========== 启动后端 ==========
echo [2/3] 启动后端服务...
start "Miku Backend" cmd /k "cd /d ""%~dp0"" && venv\Scripts\activate.bat && python agent_api.py"
timeout /t 5 /nobreak >nul

:: ========== 启动前端 ==========
echo [3/3] 启动前端...
if not exist "frontend\node_modules" (
    echo [提示] 首次启动，安装依赖...
    cd frontend
    call npm install
    cd ..
)

cd /d "%~dp0\frontend"
:: 二次确认 .vite 已删除
if exist "node_modules\.vite" (
    rmdir /s /q "node_modules\.vite" >nul 2>&1
)
start "Miku Frontend" cmd /k "npm run dev"
cd /d "%~dp0"

echo.
echo ========================================
echo   所有服务已启动！
echo   前端: http://localhost:5173
echo   后端: http://localhost:8080
echo.
echo   建议用【无痕窗口】访问前端
echo   首次打开请按 Ctrl+F5 强制刷新
echo ========================================
pause