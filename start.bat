@echo off
title Miku AI 智能助手

cd /d "%~dp0"

echo ========================================
echo   Miku AI 智能助手 - 一键启动
echo ========================================
echo.

:: ==================== 新增：自动清锁 ====================
echo [0/3] 清理残留进程与缓存...

:: 强制结束可能卡死的后端和前端进程
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM node.exe >nul 2>&1
timeout /t 1 /nobreak >nul

:: 修复 ChromaDB 只读/锁问题：取消只读属性，删除 SQLite 临时锁文件
if exist "chroma_db" (
    attrib -R "chroma_db\*.*" /S /D >nul 2>&1
    del /F /Q "chroma_db\*.sqlite3-journal" >nul 2>&1
    del /F /Q "chroma_db\*.sqlite3-wal" >nul 2>&1
    del /F /Q "chroma_db\*.sqlite3-shm" >nul 2>&1
)

:: 修复 Vite EPERM 问题：强制删除缓存目录
if exist "frontend\node_modules\.vite-temp" (
    rmdir /s /q "frontend\node_modules\.vite-temp" >nul 2>&1
)
if exist "frontend\node_modules\.vite" (
    rmdir /s /q "frontend\node_modules\.vite" >nul 2>&1
)

echo [OK] 环境已清理
echo.
:: =======================================================

:: 检查 Ollama
echo [1/3] 检查 Ollama 服务...
ollama list >nul 2>&1
if %errorlevel% neq 0 (
    echo [提示] Ollama 未运行，正在启动...
    start "Ollama" ollama serve
    timeout /t 5 /nobreak >nul
) else (
    echo [OK] Ollama 已运行
)

:: 检查环境
if not exist "venv\Scripts\activate.bat" (
    echo [错误] 未找到虚拟环境 venv
    pause
    exit /b 1
)
if not exist "agent_api.py" (
    echo [错误] 未找到 agent_api.py，请把本脚本放在项目根目录
    pause
    exit /b 1
)

:: 启动后端
echo [2/3] 启动后端服务...
start "Miku Backend" cmd /k "cd /d ""%~dp0"" && venv\Scripts\activate.bat && python agent_api.py"

timeout /t 3 /nobreak >nul

:: 启动前端
echo [3/3] 启动前端界面...
if not exist "frontend\node_modules" (
    echo [提示] 首次启动，安装前端依赖...
    cd frontend
    call npm install
    cd ..
)
start "Miku Frontend" cmd /k "cd /d ""%~dp0\frontend"" && npm run dev"

echo.
echo ========================================
echo   启动完成！请访问 http://localhost:5173
echo ========================================
echo.
echo [提示] 关闭本窗口不会停止服务
echo [提示] 请手动关闭 Backend 和 Frontend 窗口来停止
pause