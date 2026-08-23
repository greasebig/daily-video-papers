#!/usr/bin/env python3
# Seed the guestbook's comment store (comments/data.json) in the target repo.
# The guestbook is FILE-BACKED: comments live in the repo as a JSON file
# (the same pattern the paper site uses to commit its content), so it works
# with a token that can write repo contents (no issue-comment permission needed).
#
# Run locally (your machine has the GitHub token + proxy 7890):
#   GITHUB_TOKEN=ghp_xxx python3 seed_comments.py
# Re-running is safe: it won't duplicate the seed file.
#   --force  re-write the sample file even if already seeded
import os, sys, json, base64, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("GITHUB_REPO", "greasebig/greasebig.github.io")
TOKEN = os.environ.get("GITHUB_TOKEN", "")
DATA_PATH = os.environ.get("GITHUB_DATA", "comments/data.json")
API = "https://api.github.com"
SEEDED_FILE = os.path.join(ROOT, ".guestbook_seeded")

SAMPLES = [
    {"user": {"login": "Nova"}, "created_at": "2026-08-20T09:00:00Z",
     "body": "这个每日论文聚合站太好用了，每天早上来刷一下新进展 🚀"},
    {"user": {"login": "阿杰"}, "created_at": "2026-08-21T14:30:00Z",
     "body": "World Model 那一块 updates 很及时，关注很久了！"},
    {"user": {"login": "Mika"}, "created_at": "2026-08-22T20:10:00Z",
     "body": "配色和玻璃拟态风格好看，加载也快。留个言试试滚动～"},
    {"user": {"login": "小鹿"}, "created_at": "2026-08-23T08:05:00Z",
     "body": "希望以后能加搜索功能，按关键词过滤论文就好啦。"},
]


def proxy_handlers():
    handlers = []
    for proto in ("http", "https"):
        p = os.environ.get(proto.upper() + "_PROXY") or os.environ.get(proto + "_proxy")
        if p:
            handlers.append(urllib.request.ProxyHandler({proto: p}))
    if not handlers:
        handlers.append(urllib.request.ProxyHandler())
    return handlers


def api(method, url, data=None):
    req = urllib.request.Request(
        url, data=(json.dumps(data).encode() if data is not None else None), method=method
    )
    req.add_header("Authorization", "Bearer " + TOKEN)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "gh-comments-seed")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    opener = urllib.request.build_opener(*proxy_handlers())
    with opener.open(req, timeout=30) as r:
        return r.status, json.loads(r.read().decode() or "{}")


def main():
    if not TOKEN:
        sys.exit("Set GITHUB_TOKEN first (e.g. GITHUB_TOKEN=ghp_xxx python3 seed_comments.py).")
    if os.path.exists(SEEDED_FILE) and "--force" not in sys.argv:
        print("Already seeded — comments/data.json exists (use --force to rewrite).")
        return
    # does the file already exist in the repo?
    st, info = api("GET", f"{API}/repos/{REPO}/contents/{DATA_PATH}")
    content = base64.b64encode(json.dumps(SAMPLES, ensure_ascii=False, indent=2).encode("utf-8")).decode()
    if st == 200:
        sha = info.get("sha")
        st2, _ = api("PUT", f"{API}/repos/{REPO}/contents/{DATA_PATH}",
                     {"message": "guestbook: (re)seed sample comments", "content": content, "sha": sha})
    else:
        st2, _ = api("PUT", f"{API}/repos/{REPO}/contents/{DATA_PATH}",
                     {"message": "guestbook: seed sample comments", "content": content})
    if st2 in (200, 201):
        with open(SEEDED_FILE, "w") as f:
            f.write("ok")
        print("Seeded %d sample comments into %s/%s" % (len(SAMPLES), REPO, DATA_PATH))
    else:
        print("Seed failed (HTTP %s). Check token has repo contents write access." % st2)
    print("\nThen start the proxy:  node comments-proxy.js  (or run-guestbook.bat)")
    print("Open your github.io site — the guestbook reads/writes comments/data.json.")


if __name__ == "__main__":
    main()
