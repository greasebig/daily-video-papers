#!/usr/bin/env python3
"""
arXiv Papers Fetcher V2 (Extended Video Version with AI Analysis)
Fully Free Translation Version. AI Analysis still requires a key if used.
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
from openai import OpenAI
import tempfile
import subprocess

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

def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _info(msg):
    print(f"{_ts()} | INFO | {msg}")

def _warn(msg):
    print(f"{_ts()} | WARN | {msg}")

# --- Free Translation Providers ---

def translate_google(text):
    try:
        url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=zh-CN&dt=t&q=" + urllib.parse.quote(text)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return "".join([part[0] for part in data[0]])
    except: return None

def translate_mymemory(text):
    try:
        params = {"q": text, "langpair": "en|zh-CN"}
        url = "https://api.mymemory.translated.net/get?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("responseStatus") == 200:
                return data.get("responseData", {}).get("translatedText")
        return None
    except: return None

def translate_text(text):
    if not text or not text.strip(): return ""
    res = translate_google(text)
    if not res:
        time.sleep(1)
        res = translate_mymemory(text)
    return res or text

# --- AI Analysis (Requires Key) ---

DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
client, model_name = None, None

if DEEPSEEK_KEY:
    client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
    model_name = "deepseek-chat"
elif OPENAI_KEY and OPENAI_KEY.startswith("AIza"):
    client = OpenAI(api_key=OPENAI_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    model_name = "gemini-2.0-flash"
elif OPENAI_KEY:
    client = OpenAI(api_key=OPENAI_KEY)
    model_name = "gpt-4o-mini"

def analyze_paper_with_ai(paper, pdf_text):
    if not client: return "AI analysis skipped: no valid API key."
    max_chars = 30000
    if len(pdf_text) > max_chars:
        half = max_chars // 2
        pdf_text = pdf_text[:half] + "\n\n[... Omitted ...]\n\n" + pdf_text[-half:]
    prompt = f"Analyze this paper title: {paper['title']}\nContent: {pdf_text}\nProvide core contributions, technical approach, experimental design, reliability, value, and strengths/weaknesses in concise Chinese."
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": "你是一位严谨的学术研究分析专家。"},{"role": "user", "content": prompt}],
            temperature=0.5, max_tokens=2000
        )
        return resp.choices[0].message.content.strip()
    except Exception as e: return f"AI analysis failed: {e}"

# --- Core Logic ---

def fetch_arxiv_papers(category, days=3):
    base_url = "http://export.arxiv.org/api/query?"
    start_date = datetime.now() - timedelta(days=days)
    params = {"search_query": f"cat:{category}", "max_results": 200, "sortBy": "submittedDate", "sortOrder": "descending"}
    url = base_url + urllib.parse.urlencode(params)
    try:
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
        return papers
    except: return []

def is_video_related(paper):
    text = (paper["title"] + " " + paper["summary"]).lower()
    return any(k.lower() in text for k in VIDEO_KEYWORDS)

def extract_links(paper):
    text = paper["summary"]
    urls = re.findall(r'https?://[^\s<>"{}\\|\\^`\[\]]+', text)
    links = {"project": None, "code": None}
    for url in urls:
        u = url.lower()
        if "github.com" in u or "gitlab.com" in u: links["code"] = url
        elif any(k in u for k in ["project", "page", "site"]): links["project"] = url
    return links

def generate_markdown(papers, date_str):
    md = f"# arXiv Video Papers (V2) - {date_str}\n\n**Paper Count**: {len(papers)}\n\n---\n\n"
    for i, p in enumerate(papers, 1):
        _info(f"Processing {i}/{len(papers)}: {p['id']}")
        t_zh = translate_text(p["title"])
        s_zh = translate_text(p["summary"])
        links = extract_links(p)
        ai_analysis = "AI analysis skipped."
        if client:
            with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
                try:
                    urllib.request.urlretrieve(p["pdf_url"], tmp.name)
                    txt = subprocess.run(["pdftotext", "-layout", tmp.name, "-"], capture_output=True, text=True).stdout
                    if txt: ai_analysis = analyze_paper_with_ai(p, txt)
                except: pass
        md += f"## {i}. {p['title']}\n\n**中文标题**: {t_zh}\n\n"
        md += f"**Date**: {p['published']} | **arXiv**: [{p['id']}]({p['abs_url']}) | **PDF**: [Link]({p['pdf_url']})\n\n"
        if links["project"]: md += f"**Project**: {links['project']}  "
        if links["code"]: md += f"**Code**: {links['code']}\n\n"
        md += f"<details><summary><b>Abstract</b></summary>\n\n{p['summary']}\n\n</details>\n\n"
        md += f"<details><summary><b>中文摘要</b></summary>\n\n{s_zh}\n\n</details>\n\n"
        md += f"### AI 阅读分析\n\n{ai_analysis}\n\n---\n\n"
        time.sleep(1.5)
    return md

def main():
    papers_dir = Path(__file__).parent.parent / "papers"
    recent = set()
    if papers_dir.exists():
        for f in papers_dir.glob("*.md"):
            recent.update(re.findall(r'arxiv\.org/abs/(\d+\.\d+)', f.read_text()))
    all_p = []
    for cat in CATEGORIES:
        all_p.extend(fetch_arxiv_papers(cat))
        time.sleep(2)
    unique = {p["id"]: p for p in all_p}
    video = [p for p in unique.values() if is_video_related(p) and p["id"] not in recent]
    if not video: return
    video.sort(key=lambda x: x["published"], reverse=True)
    if MAX_PAPERS > 0: video = video[:MAX_PAPERS]
    date_str = datetime.now().strftime("%Y-%m-%d")
    md = generate_markdown(video, date_str)
    papers_dir.mkdir(exist_ok=True)
    (papers_dir / f"{date_str}.md").write_text(md)
    _info("Done.")

if __name__ == "__main__": main()
