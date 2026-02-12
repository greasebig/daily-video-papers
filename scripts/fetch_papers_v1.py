#!/usr/bin/env python3
"""
arXiv Papers Fetcher V1 (Extended Video Version)
Fully Free Version: Uses Google/Bing/MyMemory web interfaces for translation.
No API Keys required.
"""

import os
import re
import json
import time
import random
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

# Config
CATEGORIES = ["cs.CV", "cs.AI", "cs.MM", "cs.RO", "cs.LG"]
DAYS_TO_CHECK = int(os.environ.get("DAYS_TO_CHECK", "3"))
DAYS_TO_COMPARE = int(os.environ.get("DAYS_TO_COMPARE", "5"))
MAX_PAPERS = int(os.environ.get("MAX_PAPERS", "0"))

# Extended video-related keywords
VIDEO_KEYWORDS = [
    "video generation", "video synthesis", "video editing", "video edit",
    "video diffusion", "text-to-video", "image-to-video", "video-to-video",
    "motion generation", "character animation", "talking head", "human motion",
    "video understanding", "video recognition", "video classification", "action recognition",
    "action detection", "temporal action", "video retrieval", "video captioning",
    "video question answering", "video QA", "video summarization",
    "video super-resolution", "video enhancement", "video restoration", "video denoising",
    "video interpolation", "frame interpolation", "video compression", "video coding",
    "video segmentation", "video object segmentation", "VOS", "video instance segmentation",
    "VIS", "object tracking", "multi-object tracking", "MOT", "video matting",
    "video model", "temporal modeling", "spatio-temporal", "video transformer",
    "video representation", "optical flow", "video prediction", "future frame prediction",
]
import re

# =========================
# 1️⃣ 强视频任务（必须是视频任务）
# =========================
VIDEO_STRICT = [
    "video generation",
    "video synthesis",
    "video editing",
    "video edit",
    "video diffusion",
    "text-to-video",
    "image-to-video",
    "video-to-video",
    "video understanding",
    "video recognition",
    "video classification",
    "video retrieval",
    "video captioning",
    "video question answering",
    "video qa",
    "video summarization",
    "video super-resolution",
    "video enhancement",
    "video restoration",
    "video denoising",
    "video interpolation",
    "video compression",
    "video coding",
    "video segmentation",
    "video object segmentation",
    "video instance segmentation",
    "video matting",
    "video prediction",
]

# =========================
# 2️⃣ 上下文视频任务（可能不写 video）
# 必须和 video 同时出现
# =========================
VIDEO_CONTEXT = [
    "action recognition",
    "action detection",
    "temporal action",
    "object tracking",
    "multi-object tracking",
    "optical flow",
    "frame interpolation",
    "future frame prediction",
    "motion generation",
    "character animation",
    "talking head",
    "human motion",
]

# =========================
# 3️⃣ 构建单个大正则（性能更好）
# =========================
STRICT_PATTERN = re.compile(
    r"\b(" + "|".join(map(re.escape, VIDEO_STRICT)) + r")\b"
)

CONTEXT_PATTERN = re.compile(
    r"\b(" + "|".join(map(re.escape, VIDEO_CONTEXT)) + r")\b"
)

# =========================
# 4️⃣ 主函数
# =========================
def is_video_related(paper):
    text = (paper.get("title", "") + " " + paper.get("summary", "")).lower()

    # 强视频任务直接匹配
    if STRICT_PATTERN.search(text):
        return True

    # 上下文类必须包含 video
    if "video" in text and CONTEXT_PATTERN.search(text):
        return True

    return False
    
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
    """Free Google Translate web interface."""
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
    """Free MyMemory API."""
    try:
        params = {"q": text, "langpair": "en|zh-CN"}
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
    """Translation chain: Google -> MyMemory -> Original."""
    if not text or not text.strip():
        return ""
    
    # 1. Try Google
    result = translate_google(text)
    if result: return result
    
    # 2. Try MyMemory
    time.sleep(1)
    result = translate_mymemory(text)
    if result: return result
    
    return text

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
            if attempt > 0: time.sleep(5)
            with urllib.request.urlopen(url, timeout=30) as response:
                root = ET.fromstring(response.read())
            namespace = {"atom": "http://www.w3.org/2005/Atom"}
            papers = []
            for entry in root.findall("atom:entry", namespace):
                pub_date = datetime.strptime(entry.find("atom:published", namespace).text, "%Y-%m-%dT%H:%M:%SZ")
                if pub_date < start_date: continue
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

#def is_video_related(paper):
#    text = (paper["title"] + " " + paper["summary"]).lower()
#    return any(k.lower() in text for k in VIDEO_KEYWORDS)

def extract_links(paper):
    text = paper["summary"]
    urls = re.findall(r'https?://[^\s<>"{}\\|\\^`\[\]]+', text)
    links = {"project": None, "code": None}
    for url in urls:
        u = url.lower()
        if "github.com" in u or "gitlab.com" in u: links["code"] = url
        elif any(k in u for k in ["project", "page", "site"]): links["project"] = url
    return links

def load_recent_papers(days=5):
    papers_dir = Path(__file__).parent.parent / "papers"
    recent_ids = set()
    if not papers_dir.exists(): return recent_ids
    start_date = datetime.now() - timedelta(days=days)
    for f in papers_dir.glob("*.md"):
        try:
            if datetime.strptime(f.stem, "%Y-%m-%d") >= start_date:
                recent_ids.update(re.findall(r'arxiv\.org/abs/(\d+\.\d+)', f.read_text()))
        except: continue
    return recent_ids

def generate_markdown(papers, date_str):
    md = f"# arXiv Video Papers - {date_str}\n\n**Paper Count**: {len(papers)}\n\n---\n\n"
    for i, p in enumerate(papers, 1):
        _info(f"Translating {i}/{len(papers)}: {p['id']}")
        t_zh = translate_text(p["title"])
        s_zh = translate_text(p["summary"])
        links = extract_links(p)
        md += f"## {i}. {p['title']}\n\n**中文标题**: {t_zh}\n\n"
        md += f"**Date**: {p['published']} | **arXiv**: [{p['id']}]({p['abs_url']}) | **PDF**: [Link]({p['pdf_url']})\n\n"
        if links["project"]: md += f"**Project**: {links['project']}  "
        if links["code"]: md += f"**Code**: {links['code']}\n\n"
        md += f"<details><summary><b>Abstract</b></summary>\n\n{p['summary']}\n\n</details>\n\n"
        md += f"<details><summary><b>中文摘要</b></summary>\n\n{s_zh}\n\n</details>\n\n---\n\n"
        time.sleep(1.5) # Anti-ban delay
    return md

def update_readme_index():
    base_dir = Path(__file__).parent.parent
    papers_dir = base_dir / "papers"
    readme_path = base_dir / "README.md"
    if not papers_dir.exists(): return
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
    recent = load_recent_papers(DAYS_TO_COMPARE)
    all_p = []
    for cat in CATEGORIES:
        all_p.extend(fetch_arxiv_papers(cat, DAYS_TO_CHECK))
        time.sleep(2)
    unique = {p["id"]: p for p in all_p}
    video = [p for p in unique.values() if is_video_related(p) and p["id"] not in recent]
    if not video:
        _info("No new papers.")
        return
    video.sort(key=lambda x: x["published"], reverse=True)
    if MAX_PAPERS > 0: video = video[:MAX_PAPERS]
    date_str = datetime.now().strftime("%Y-%m-%d")
    md = generate_markdown(video, date_str)
    papers_dir = Path(__file__).parent.parent / "papers"
    papers_dir.mkdir(exist_ok=True)
    (papers_dir / f"{date_str}.md").write_text(md)
    update_readme_index()
    _info("Done.")

if __name__ == "__main__":
    main()
