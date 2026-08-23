#!/usr/bin/env python3
# Seed the GitHub Issue used by the comments widget, then auto-patch the
# issue number into the widget config so everything is wired in one go.
#
# Run locally (your machine has the GitHub token + proxy 7890):
#   GITHUB_TOKEN=ghp_xxx python3 seed_comments.py
# Re-running is safe: it reuses the same issue and won't duplicate comments.
#   -i N     use existing issue N instead of creating/reading one
#   --new    force creating a fresh issue
#   --force  re-post sample comments even if already seeded
import os, sys, json, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("GITHUB_REPO", "greasebig/greasebig.github.io")
TOKEN = os.environ.get("GITHUB_TOKEN", "")
API = "https://api.github.com"
ISSUE_FILE = os.path.join(ROOT, ".guestbook_issue")
SEEDED_FILE = os.path.join(ROOT, ".guestbook_seeded")

SAMPLES = [
    "这个每日论文聚合站太好用了，每天早上来刷一下新进展 🚀",
    "World Model 那一块 updates 很及时，关注很久了！",
    "配色和玻璃拟态风格好看，加载也快。留个言试试滚动～",
    "希望以后能加搜索功能，按关键词过滤论文就好啦。",
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


def get_issue():
    if "-i" in sys.argv:
        return int(sys.argv[sys.argv.index("-i") + 1])
    if "--new" not in sys.argv and os.path.exists(ISSUE_FILE):
        with open(ISSUE_FILE) as f:
            return int(f.read().strip())
    _, iss = api("POST", f"{API}/repos/{REPO}/issues",
                 {"title": "Site Guestbook / 留言墙", "body": "Auto-created for the comments widget."})
    n = iss["number"]
    with open(ISSUE_FILE, "w") as f:
        f.write(str(n))
    return n


def patch_configs(n):
    # Literal replacements (regex \s proved unreliable here); one per file.
    targets = {
        os.path.join(ROOT, "docs", "index.html"): [("issue: 1", "issue: %d" % n)],
        os.path.join(ROOT, "docs", "comments", "index.html"): [("issue: 1", "issue: %d" % n)],
        os.path.join(ROOT, "docs", "comments", "comments.js"): [("CFG.issue || 1", "CFG.issue || %d" % n)],
    }
    for path, subs in targets.items():
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            s = f.read()
        s2 = s
        for old, new in subs:
            if old not in s2:
                print("WARN: '%s' not found in %s" % (old, os.path.relpath(path, ROOT)))
                continue
            s2 = s2.replace(old, new, 1)
        if s2 != s:
            with open(path, "w", encoding="utf-8") as f:
                f.write(s2)
            print("patched", os.path.relpath(path, ROOT))


def main():
    if not TOKEN:
        sys.exit("Set GITHUB_TOKEN first (e.g. GITHUB_TOKEN=ghp_xxx python3 seed_comments.py).")
    issue = get_issue()
    print("Using issue #%d in %s" % (issue, REPO))

    already = os.path.exists(SEEDED_FILE) and "--force" not in sys.argv
    if already:
        print("Already seeded — skipping sample posts (use --force to re-post).")
    else:
        for msg in SAMPLES:
            _, c = api("POST", f"{API}/repos/{REPO}/issues/{issue}/comments", {"body": msg})
            print("Posted comment", c.get("id", "?"))
        with open(SEEDED_FILE, "w") as f:
            f.write(str(issue))

    patch_configs(issue)
    print("\nDone. Now start the proxy:  node comments-proxy.js  (or run-guestbook.bat)")
    print("Then open your github.io site and the guestbook is live on issue #%d." % issue)


if __name__ == "__main__":
    main()
