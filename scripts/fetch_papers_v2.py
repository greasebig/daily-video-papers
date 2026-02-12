#!/usr/bin/env python3
"""
arXiv Papers Fetcher V2 (Extended Video Version with AI Analysis)
Optimized for Gemini and MyMemory (No local LibreTranslate server).
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

# API Keys
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

# Translation config
MYMEMORY_EMAIL = os.environ.get("MYMEMORY_EMAIL", "lujundaljdljd@163.com")
MYMEMORY_URL = "https://api.mymemory.translated.net/get"
MYMEMORY_MAX_BYTES = 450
TRANSLATION_SLEEP = float(os.environ.get("TRANSLATION_SLEEP", "1.0"))
_TRANSLATION_CACHE = {}

def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _info(msg):
    print(f"{_ts()} | INFO | {msg}")

def _warn(msg):
    print(f"{_ts()} | WARN | {msg}")

def _err(msg):
    print(f"{_ts()} | ERROR | {msg}")

# Initialize AI Client
client = None
model_name = None

if DEEPSEEK_KEY:
    _info("Using DeepSeek API for translation and analysis")
    client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
    model_name = "deepseek-chat"
elif OPENAI_KEY and OPENAI_KEY.startswith("AIza"):
    _info("Using Gemini API (via Google AI Studio) for translation and analysis")
    client = OpenAI(
        api_key=OPENAI_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    model_name = "gemini-2.0-flash"
elif OPENAI_KEY:
    _info("Using Standard OpenAI API for translation and analysis")
    client = OpenAI(api_key=OPENAI_KEY)
    model_name = "gpt-4o-mini"
else:
    _warn("No AI API Key found. Will fallback to MyMemory for translation and skip AI analysis.")

def translate_with_ai(text):
    """Use AI to translate text."""
    if not client:
        return None
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a professional translator. Translate the following academic text to Chinese. Keep technical terms in English when appropriate. Only return the translation, no explanations."},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        _warn(f"AI Translation failed: {e}")
        return None

def translate_with_mymemory(text):
    """Use MyMemory API to translate text."""
    try:
        if len(text.encode("utf-8")) > MYMEMORY_MAX_BYTES:
            return text
        params = {"q": text, "langpair": "en|zh-CN"}
        if MYMEMORY_EMAIL:
            params["de"] = MYMEMORY_EMAIL
        url = MYMEMORY_URL + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "daily-video-papers/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("responseStatus") == 200:
            return data.get("responseData", {}).get("translatedText") or text
        return text
    except Exception as e:
        _warn(f"MyMemory Translation failed: {e}")
        return text

def translate_text(text):
    """Translation chain: AI -> MyMemory -> Original."""
    if not text or not text.strip():
        return ""
    if text in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[text]
    result = translate_with_ai(text)
    if not result:
        result = translate_with_mymemory(text)
        time.sleep(TRANSLATION_SLEEP)
    _TRANSLATION_CACHE[text] = result
    return result

def download_pdf(pdf_url, output_path):
    """Download a PDF file."""
    try:
        urllib.request.urlretrieve(pdf_url, output_path)
        return True
    except Exception as e:
        _warn(f"PDF download failed: {e}")
        return False

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF."""
    try:
        result = subprocess.run(["pdftotext", "-layout", pdf_path, "-"], capture_output=True, text=True, timeout=30)
        return result.stdout if result.returncode == 0 else None
    except Exception as e:
        _warn(f"PDF text extraction error: {e}")
        return None

def analyze_paper_with_ai(paper, pdf_text):
    """Analyze the full paper using AI."""
    if not client:
        return "AI analysis skipped: no API key."
    max_chars = 30000
    if len(pdf_text) > max_chars:
        half = max_chars // 2
        pdf_text = pdf_text[:half] + "\n\n[... Omitted ...]\n\n" + pdf_text[-half:]
    
    prompt = f"""
You are a senior computer vision researcher. Read the full paper content and provide a critical analysis.
Paper title: {paper['title']}
Full text:
{pdf_text}

Please cover:
1. Core contributions in 1-2 sentences.
2. Main technical approach and architecture.
3. Experimental design and baselines.
4. Result reliability and potential issues (overfitting, cherry-picking).
5. Practical value and limitations.
6. Strengths, weaknesses, and open problems.

Respond in concise, professional Chinese. Each item should be 2-3 sentences.
"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一位严谨的学术研究分析专家，擅长批判性阅读和评估论文质量。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=2000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        _warn(f"AI analysis failed: {e}")
        return f"AI analysis unavailable: {e}"

def fetch_arxiv_papers(category, days=3, max_retries=3):
    """Fetch papers for a category from arXiv API."""
    base_url = "http://export.arxiv.org/api/query?"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    query = f"cat:{category}"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": 200,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = base_url + urllib.parse.urlencode(params)
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(5 * attempt)
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
            root = ET.fromstring(data)
            namespace = {"atom": "http://www.w3.org/2005/Atom"}
            papers = []
            for entry in root.findall("atom:entry", namespace):
                published = entry.find("atom:published", namespace).text
                pub_date = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ")
                if pub_date < start_date:
                    continue
                paper = {
                    "id": entry.find("atom:id", namespace).text.split("/abs/")[-1],
                    "title": entry.find("atom:title", namespace).text.strip().replace("\n", " "),
                    "summary": entry.find("atom:summary", namespace).text.strip().replace("\n", " "),
                    "authors": [author.find("atom:name", namespace).text for author in entry.findall("atom:author", namespace)],
                    "published": pub_date.strftime("%Y-%m-%d"),
                    "pdf_url": entry.find("atom:id", namespace).text.replace("/abs/", "/pdf/"),
                    "abs_url": entry.find("atom:id", namespace).text,
                    "categories": [cat.attrib["term"] for cat in entry.findall("atom:category", namespace)],
                }
                papers.append(paper)
            _info(f"Fetched {len(papers)} papers for {category}")
            return papers
        except Exception as e:
            _warn(f"Fetch attempt {attempt + 1} failed for {category}: {e}")
            if attempt == max_retries - 1:
                return []
    return []

def is_video_related(paper):
    """Check whether a paper is video-related."""
    text = (paper["title"] + " " + paper["summary"]).lower()
    return any(keyword.lower() in text for keyword in VIDEO_KEYWORDS)

def extract_links(paper):
    """Extract project/code links from abstract."""
    text = paper["summary"]
    url_pattern = r'https?://[^\s<>"{}\\|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    links = {"project": None, "code": None}
    for url in urls:
        url_lower = url.lower()
        if "github.com" in url_lower or "gitlab.com" in url_lower:
            if not links["code"]:
                links["code"] = url
        elif any(k in url_lower for k in ["project", "page", "site"]):
            if not links["project"]:
                links["project"] = url
        elif not links["project"]:
            links["project"] = url
    return links

def load_recent_papers(days=5):
    """Load recent paper IDs for de-duplication."""
    papers_dir = Path(__file__).parent.parent / "papers"
    recent_ids = set()
    if not papers_dir.exists():
        return recent_ids
    start_date = datetime.now() - timedelta(days=days)
    for md_file in papers_dir.glob("*.md"):
        try:
            if datetime.strptime(md_file.stem, "%Y-%m-%d") >= start_date:
                content = md_file.read_text(encoding="utf-8")
                recent_ids.update(re.findall(r'arxiv\.org/abs/(\d+\.\d+)', content))
        except Exception:
            continue
    return recent_ids

def generate_markdown(papers, date_str):
    """Generate Markdown for papers."""
    md_content = f"# arXiv Video Papers (V2) - {date_str}\n\n"
    md_content += f"**Update Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += f"**Paper Count**: {len(papers)}\n\n---\n\n"

    for i, paper in enumerate(papers, 1):
        _info(f"Processing {i}/{len(papers)}: {paper['id']}")
        title_zh = translate_text(paper["title"])
        summary_zh = translate_text(paper["summary"])
        links = extract_links(paper)
        
        # AI Analysis
        ai_analysis = "AI analysis skipped."
        if client:
            with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
                if download_pdf(paper["pdf_url"], tmp.name):
                    pdf_text = extract_text_from_pdf(tmp.name)
                    if pdf_text:
                        ai_analysis = analyze_paper_with_ai(paper, pdf_text)

        md_content += f"## {i}. {paper['title']}\n\n"
        md_content += f"**Chinese Title**: {title_zh}\n\n"
        md_content += f"**Authors**: {', '.join(paper['authors'][:5])}{' et al.' if len(paper['authors']) > 5 else ''}\n\n"
        md_content += f"**Date**: {paper['published']} | **arXiv**: [{paper['id']}]({paper['abs_url']}) | **PDF**: [Link]({paper['pdf_url']})\n\n"
        if links["project"]:
            md_content += f"**Project**: {links['project']}  "
        if links["code"]:
            md_content += f"**Code**: {links['code']}\n\n"
        md_content += f"**Categories**: {', '.join(paper['categories'])}\n\n"
        md_content += f"<details><summary><b>Abstract</b></summary>\n\n{paper['summary']}\n\n</details>\n\n"
        md_content += f"<details><summary><b>Chinese Abstract</b></summary>\n\n{summary_zh}\n\n</details>\n\n"
        md_content += f"### AI 阅读分析\n\n{ai_analysis}\n\n---\n\n"
        time.sleep(1)
    return md_content

def update_readme_index():
    """Update README index and daily content."""
    base_dir = Path(__file__).parent.parent
    papers_dir = base_dir / "papers"
    readme_path = base_dir / "README.md"
    if not papers_dir.exists():
        return
    
    paper_files = sorted(papers_dir.glob("*.md"), reverse=True)
    index_lines = []
    content_blocks = []
    for f in paper_files:
        content = f.read_text(encoding="utf-8")
        count = re.search(r'\*\*Paper Count\*\*: (\d+)', content)
        count_str = count.group(1) if count else "0"
        index_lines.append(f"- [{f.stem}](papers/{f.name}) - {count_str} papers")
        content_blocks.append(
            f"<details><summary><b>{f.stem} ({count_str} papers)</b></summary>\n\n"
            f"{content}\n\n"
            f"</details>"
        )
    
    index_content = "\n" + "\n".join(index_lines) + "\n"
    content_section = "\n" + "\n\n".join(content_blocks) + "\n"

    readme_content = readme_path.read_text(encoding="utf-8")
    readme_content = re.sub(r'<!-- PAPERS_INDEX_START -->.*?<!-- PAPERS_INDEX_END -->', 
                            f'<!-- PAPERS_INDEX_START -->{index_content}<!-- PAPERS_INDEX_END -->', 
                            readme_content, flags=re.DOTALL)
    readme_content = re.sub(r'<!-- PAPERS_CONTENT_START -->.*?<!-- PAPERS_CONTENT_END -->', 
                            f'<!-- PAPERS_CONTENT_START -->{content_section}<!-- PAPERS_CONTENT_END -->', 
                            readme_content, flags=re.DOTALL)
    readme_path.write_text(readme_content, encoding="utf-8")

def main():
    recent_ids = load_recent_papers(DAYS_TO_COMPARE)
    all_papers = []
    for i, cat in enumerate(CATEGORIES):
        _info(f"Fetching {cat}...")
        papers = fetch_arxiv_papers(cat, DAYS_TO_CHECK)
        all_papers.extend(papers)
        if i < len(CATEGORIES) - 1:
            time.sleep(2)

    unique_papers = {p["id"]: p for p in all_papers}
    video_papers = [p for p in unique_papers.values() if is_video_related(p)]
    new_papers = [p for p in video_papers if p["id"] not in recent_ids]

    if not new_papers:
        _info("No new papers found.")
        return

    new_papers.sort(key=lambda x: x["published"], reverse=True)
    if MAX_PAPERS > 0:
        new_papers = new_papers[:MAX_PAPERS]

    date_str = datetime.now().strftime("%Y-%m-%d")
    md_content = generate_markdown(new_papers, date_str)

    papers_dir = Path(__file__).parent.parent / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    (papers_dir / f"{date_str}.md").write_text(md_content, encoding="utf-8")
    update_readme_index()
    _info(f"Completed. Output written to papers/{date_str}.md")

if __name__ == "__main__":
    main()
