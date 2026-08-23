@echo off
chcp 65001 >nul
REM 一键完成：灌示例留言（幂等）+ 启动代理（新窗口）。
REM 需先设置:  set GITHUB_TOKEN=ghp_xxx
if "%GITHUB_TOKEN%"=="" (
  echo [ERROR] 尚未设置 GITHUB_TOKEN
  echo 请先执行:  set GITHUB_TOKEN=ghp_xxx
  pause
  exit /b 1
)
where python >nul 2>nul && set PY=python
where python3 >nul 2>nul && set PY=python3
where py >nul 2>nul && set PY=py
if "%PY%"=="" (
  echo [ERROR] 找不到 python，请先安装或调整命令。
  pause
  exit /b 1
)
echo === 写入示例留言到 comments/data.json（幂等） ===
%PY% seed_comments.py
echo === 启动代理（新窗口） ===
start "Comments Proxy" cmd /k "node comments-proxy.js"
echo.
echo 代理已在新窗口启动。打开你的 github.io 站点即可看到留言墙并发帖。
echo 留言会写入 comments/data.json（仓库文件，无需 Issue）。
pause
