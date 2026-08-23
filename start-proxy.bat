@echo off
chcp 65001 >nul
REM 启动留言代理（端口 7897）。需先设置 GITHUB_TOKEN。
if "%GITHUB_TOKEN%"=="" (
  echo [ERROR] 尚未设置 GITHUB_TOKEN
  echo 请先执行:  set GITHUB_TOKEN=你的PAT
  pause
  exit /b 1
)
REM 自动定位 node
where node >nul 2>nul
if errorlevel 1 (
  echo [ERROR] 未找到 node，请先安装 Node.js 并加入 PATH，或编辑本文件指定 node 路径
  pause
  exit /b 1
)
REM 若处于「需要代理才能访问 GitHub」的网络，取消下一行注释并改为你的代理地址
REM set HTTPS_PROXY=http://127.0.0.1:7890
echo 启动留言代理 http://localhost:7897 ...
echo 代理启动后，打开 https://greasebig.github.io 即可发帖（token 不会离开本机）
node comments-proxy.js
pause
