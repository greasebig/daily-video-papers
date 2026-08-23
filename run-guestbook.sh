#!/usr/bin/env bash
# 一键完成：灌示例留言（幂等）+ 启动代理。
# 需先设置: export GITHUB_TOKEN=ghp_xxx
set -e
if [ -z "$GITHUB_TOKEN" ]; then
  echo "[ERROR] GITHUB_TOKEN 未设置。请先: export GITHUB_TOKEN=ghp_xxx"
  exit 1
fi
echo "=== 灌示例留言到 GitHub Issue ==="
python3 seed_comments.py
echo "=== 启动代理（后台） ==="
node comments-proxy.js &
PROXY_PID=$!
echo "代理已启动 (pid $PROXY_PID)。打开你的 github.io 站点即可看到留言墙并发帖。"
echo "issue 编号已自动写回设定档。"
wait $PROXY_PID
