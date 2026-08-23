#!/usr/bin/env bash
# 启动留言代理（端口 7897）。需先设置 GITHUB_TOKEN。
set -e
if [ -z "$GITHUB_TOKEN" ]; then
  echo "[ERROR] GITHUB_TOKEN 未设置。请先: export GITHUB_TOKEN=ghp_xxx"
  exit 1
fi
echo "启动留言代理 http://localhost:7897 ..."
exec node comments-proxy.js
