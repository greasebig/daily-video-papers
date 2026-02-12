#!/usr/bin/env python3
"""
arXiv Papers Fetcher V3 (Topic-based, no AI analysis)
Based on V1 with direct translation sources (Google/MyMemory).
"""

import os
import re
import json
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

# Config
CATEGORIES = [c.strip() for c in os.environ.get("CATEGORIES", "cs.CV,cs.AI,cs.MM,cs.RO,cs.LG").split(",") if c.strip()]
DAYS_TO_CHECK = int(os.environ.get("DAYS_TO_CHECK", "3"))
DAYS_TO_COMPARE = int(os.environ.get("DAYS_TO_COMPARE", "5"))
MAX_PAPERS = int(os.environ.get("MAX_PAPERS", "0"))
REBUILD = int(os.environ.get("REBUILD", "0"))
TOPIC_NAME = os.environ.get("TOPIC_NAME", "Video")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")

# Keyword config
STRICT_KEYWORDS = [k.strip().lower() for k in os.environ.get("KEYWORDS", "").split(",") if k.strip()]
CONTEXT_KEYWORDS = [k.strip().lower() for k in os.environ.get("CONTEXT_KEYWORDS", "").split(",") if k.strip()]
REQUIRED_WORD = os.environ.get("REQUIRED_WORD", "").strip().lower()

# Translation config (free)
MYMEMORY_EMAIL = os.environ.get("MYMEMORY_EMAIL", "lujundaljdljd@163.com")

# Default video keywords if none provided
if not STRICT_KEYWORDS and not CONTEXT_KEYWORDS:
    STRICT_KEYWORDS = [
        "video generation", "video synthesis", "video editing", "video edit",
        "video diffusion", "text-to-video", "image-to-video", "video-to-video",
        "video understanding", "video recognition", "video classification", "video retrieval",
        "video captioning", "video question answering", "video qa", "video summarization",
        "video super-resolution", "video enhancement", "video restoration", "video denoising",
        "video interpolation", "video compression", "video coding", "video segmentation",
        "video object segmentation", "video instance segmentation", "video matting", "video prediction",
    ]
    CONTEXT_KEYWORDS = [
        "action recognition", "action detection", "temporal action", "object tracking",
        "multi-object tracking", "optical flow", "frame interpolation", "future frame prediction",
        "motion generation", "character animation", "talking head", "human motion",
    ]
    if not REQUIRED_WORD:
        REQUIRED_WORD = "video"

STRICT_PATTERN = re.compile(r"\b(" + "|".join(map(re.escape, STRICT_KEYWORDS)) + r")\b") if STRICT_KEYWORDS else None
CONTEXT_PATTERN = re.compile(r"\b(" + "|".join(map(re.escape, CONTEXT_KEYWORDS)) + r")\b") if CONTEXT_KEYWORDS else None


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _info(msg):
    print(f"{_ts()} | INFO | {msg}")


def _warn(msg):
    print(f"{_ts()} | WARN | {msg}")


def _err(msg):
    print(f"{_ts()} | ERROR | {msg}")


# --- Free Translation Providers ---

def translate_google(text):
    try:
        url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=zh-CN&dt=t&q=" + urllib.parse.quote(text)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return "".join([part[0] for part in data[0]])
    except Exception as e:
        _warn(f"Google Translate failed: {e}")
        return None


def translate_mymemory(text):
    try:
        params = {"q": text, "langpair": "en|zh-CN"}
        if MYMEMORY_EMAIL:
            params["de"] = MYMEMORY_EMAIL
        url = "https://api.mymemory.translated.net/get?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("responseStatus") == 200:
                return data.get("responseData", {}).get("translatedText")
        return None
    except Exception as e:
        _warn(f"MyMemory failed: {e}")
        return None


def translate_text(text):
    if not text or not text.strip():
        return ""
    res = translate_google(text)
    if not res:
        time.sleep(1)
        res = translate_mymemory(text)
    return res or text


# --- Core Logic ---

def fetch_arxiv_papers(category, days=3, max_retries=3):
    base_url = "http://export.arxiv.org/api/query?"
    start_date = datetime.now() - timedelta(days=days)
    params = {
        "search_query": f"cat:{category}",
        "start": 0,
        "max_results": 200,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = base_url + urllib.parse.urlencode(params)
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(5)
            with urllib.request.urlopen(url, timeout=30) as response:
                root = ET.fromstring(response.read())
            namespace = {"atom": "http://www.w3.org/2005/Atom"}
            papers = []
            for entry in root.findall("atom:entry", namespace):
                pub_date = datetime.strptime(entry.find("atom:published", namespace).text, "%Y-%m-%dT%H:%M:%SZ")
                if pub_date < start_date:
                    continue
                papers.append({
                    "id": entry.find("atom:id", namespace).text.split("/abs/")[-1],
                    "title": entry.find("atom:title", namespace).text.strip().replace("\n", " "),
                    "summary": entry.find("atom:summary", namespace).text.strip().replace("\n", " "),
                    "authors": [a.find("atom:name", namespace).text for a in entry.findall("atom:author", namespace)],
                    "published": pub_date.strftime("%Y-%m-%d"),
                    "pdf_url": entry.find("atom:id", namespace).text.replace("/abs/", "/pdf/"),
                    "abs_url": entry.find("atom:id", namespace).text,
                    "categories": [c.attrib["term"] for c in entry.findall("atom:category", namespace)],
                })
            _info(f"Fetched {len(papers)} papers for {category}")
            return papers
        except Exception as e:
            _warn(f"Fetch failed: {e}")
    return []


def is_topic_related(paper):
    if not STRICT_PATTERN and not CONTEXT_PATTERN:
        return True
    text = (paper.get("title", "") + " " + paper.get("summary", "")).lower()
    if STRICT_PATTERN and STRICT_PATTERN.search(text):
        return True
    if CONTEXT_PATTERN and CONTEXT_PATTERN.search(text):
        if REQUIRED_WORD:
            return REQUIRED_WORD in text
        return True
    return False


def extract_links(paper):
    text = paper["summary"]
    urls = re.findall(r'https?://[^\s<>"{}\\|\\^`\[\]]+', text)
    links = {"project": None, "code": None}
    for url in urls:
        u = url.lower()
        if "github.com" in u or "gitlab.com" in u:
            links["code"] = url
        elif any(k in u for k in ["project", "page", "site"]):
            links["project"] = url
    return links


def load_recent_papers(days=5, papers_dir=None):
    if papers_dir is None:
        papers_dir = Path(__file__).parent.parent / "papers"
    recent_ids = set()
    if not papers_dir.exists():
        return recent_ids
    start_date = datetime.now() - timedelta(days=days)
    for f in papers_dir.glob("*.md"):
        try:
            if datetime.strptime(f.stem, "%Y-%m-%d") >= start_date:
                recent_ids.update(re.findall(r'arxiv\.org/abs/(\d+\.\d+v\d+|\d+\.\d+)', f.read_text()))
        except Exception:
            continue
    return recent_ids


def generate_markdown(papers, date_str):
    md = f"# arXiv {TOPIC_NAME} Papers - {date_str}\n\n**Paper Count**: {len(papers)}\n\n---\n\n"
    for i, p in enumerate(papers, 1):
        _info(f"Translating {i}/{len(papers)}: {p['id']}")
        t_zh = translate_text(p["title"])
        s_zh = translate_text(p["summary"])
        links = extract_links(p)
        md += f"## {i}. {p['title']} / {t_zh}\n\n"
        md += f"**Date**: {p['published']} | **arXiv**: [{p['id']}]({p['abs_url']}) | **PDF**: [Link]({p['pdf_url']})\n\n"
        md += f"**Categories**: {', '.join(p['categories'])}\n\n"
        if links["project"]:
            md += f"**Project**: {links['project']}  "
        if links["code"]:
            md += f"**Code**: {links['code']}\n\n"
        md += f"<details><summary><b>Abstract / 摘要</b></summary>\n\n{p['summary']}\n\n{s_zh}\n\n</details>\n\n---\n\n"
        time.sleep(1.5)
    return md




def build_docs_site(repo_root, topic_slug, topic_name, papers_dir):
    docs_root = repo_root / "docs" / topic_slug
    docs_root.mkdir(parents=True, exist_ok=True)
    papers_out = docs_root / "papers"
    papers_out.mkdir(parents=True, exist_ok=True)

    # copy markdown papers into docs folder for Pages access
    for md in papers_dir.glob("*.md"):
        target = papers_out / md.name
        try:
            target.write_text(md.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass

    # build simple index that renders markdown via marked.js
    index_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{topic_name} Papers</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');
    :root {{ --bg:#0b0f17; --card:rgba(255,255,255,0.06); --border:rgba(255,255,255,0.16); --ink:#f4f7ff; --muted:#b7c0d8; --accent:#2ec4b6; }}
    *{{box-sizing:border-box}}
    body{{margin:0;font-family:"Space Grotesk",sans-serif;color:var(--ink);background:radial-gradient(1000px 500px at 10% -10%, #1f3b63 0%, transparent 60%),linear-gradient(160deg,#0b0f17,#12243a);min-height:100vh}}
    .wrap{{max-width:1100px;margin:0 auto;padding:48px 24px 80px}}
    h1{{font-family:"Fraunces",serif;margin:0 0 8px;font-size:clamp(32px,4vw,56px)}}
    p{{color:var(--muted)}}
    .layout{{display:grid;grid-template-columns:260px 1fr;gap:20px;margin-top:24px}}
    .panel{{padding:16px;border-radius:16px;background:var(--card);border:1px solid var(--border)}}
    .list a{{display:block;color:var(--ink);text-decoration:none;padding:8px 6px;border-radius:10px}}
    .list a:hover{{background:rgba(255,255,255,0.06)}}
    .content{{padding:20px;border-radius:16px;background:var(--card);border:1px solid var(--border)}}
    @media (max-width: 900px){{.layout{{grid-template-columns:1fr}}}}
  </style>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>
  <div class="wrap">
    <h1>{topic_name} Papers</h1>
    <p>Daily arXiv digests rendered directly on this page.</p>
    <div class="layout">
      <div class="panel list" id="list"></div>
      <div class="content" id="content">Select a date to load papers.</div>
    </div>
  </div>
  <script>
    const files = {sorted([md.name for md in papers_dir.glob('*.md')], reverse=True)};
    const list = document.getElementById('list');
    const content = document.getElementById('content');
    function loadFile(name){{
      fetch('./papers/' + name).then(r=>r.text()).then(md=>{{
        content.innerHTML = marked.parse(md);
      }});
    }}
    files.forEach((f, i)=>{{
      const a = document.createElement('a');
      a.textContent = f.replace('.md','');
      a.href = '#';
      a.onclick = (e)=>{{e.preventDefault(); loadFile(f);}};
      list.appendChild(a);
      if (i===0) loadFile(f);
    }});
  </script>
</body>
</html>
"""
    (docs_root / "index.html").write_text(index_html, encoding="utf-8")


def update_readme_index(base_dir):
    papers_dir = base_dir / "papers"
    readme_path = base_dir / "README.md"
    if not papers_dir.exists() or not readme_path.exists():
        return
    files = sorted(papers_dir.glob("*.md"), reverse=True)
    idx, content = "", ""
    for f in files:
        c = f.read_text()
        cnt = re.search(r'\*\*Paper Count\*\*: (\d+)', c)
        cnt_str = cnt.group(1) if cnt else "0"
        idx += f"- [{f.stem}](papers/{f.name}) - {cnt_str} papers\n"
        content += f"<details><summary><b>{f.stem} ({cnt_str} papers)</b></summary>\n\n{c}\n\n</details>\n\n"
    raw = readme_path.read_text()
    raw = re.sub(r'<!-- PAPERS_INDEX_START -->.*?<!-- PAPERS_INDEX_END -->', lambda m:f'<!-- PAPERS_INDEX_START -->\n{idx}<!-- PAPERS_INDEX_END -->', raw, flags=re.DOTALL)
    raw = re.sub(r'<!-- PAPERS_CONTENT_START -->.*?<!-- PAPERS_CONTENT_END -->',lambda m: f'<!-- PAPERS_CONTENT_START -->\n{content}<!-- PAPERS_CONTENT_END -->', raw, flags=re.DOTALL)
    readme_path.write_text(raw)


def main():
    base_dir = (Path(__file__).parent.parent / OUTPUT_DIR).resolve()
    papers_dir = base_dir / "papers"
    if REBUILD and papers_dir.exists():
        for f in papers_dir.glob("*.md"):
            try:
                f.unlink()
            except Exception:
                pass
    papers_dir.mkdir(parents=True, exist_ok=True)

    recent = load_recent_papers(DAYS_TO_COMPARE, papers_dir)

    all_p = []
    for cat in CATEGORIES:
        all_p.extend(fetch_arxiv_papers(cat, DAYS_TO_CHECK))
        time.sleep(2)
    unique = {p["id"]: p for p in all_p}
    topic = [p for p in unique.values() if is_topic_related(p) and p["id"] not in recent]
    if not topic:
        _info("No new papers.")
        return
    topic.sort(key=lambda x: x["published"], reverse=True)
    if MAX_PAPERS > 0:
        topic = topic[:MAX_PAPERS]
    date_str = datetime.now().strftime("%Y-%m-%d")
    md = generate_markdown(topic, date_str)
    (papers_dir / f"{date_str}.md").write_text(md)
    update_readme_index(base_dir)
    _info("Done.")


if __name__ == "__main__":
    main()
