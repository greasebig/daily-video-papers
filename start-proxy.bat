@echo off
chcp 65001 >nul
REM 启动留言代理（端口 7897）。需先设置 GITHUB_TOKEN。
if "%GITHUB_TOKEN%"=="" (
  echo [ERROR] 尚未设置 GITHUB_TOKEN
  echo 请先执行:  set GITHUB_TOKEN=ghp_xxx
  pause
  exit /b 1
)
echo 启动留言代理 http://localhost:7897 ...
node comments-proxy.js
pause
