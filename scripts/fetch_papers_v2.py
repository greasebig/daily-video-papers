#!/usr/bin/env python3
"""
arXiv Papers Fetcher V2 (Extended Video Version with AI Analysis)
Adds PDF download and AI analysis. Translation is free by default.
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

# Translation config (free by default)
TRANSLATION_CHAIN = os.environ.get("TRANSLATION_CHAIN", "mymemory,libretranslate")
MYMEMORY_EMAIL = os.environ.get("MYMEMORY_EMAIL", "lujundaljdljd@163.com")
LIBRETRANSLATE_URL = os.environ.get("LIBRETRANSLATE_URL")
LIBRETRANSLATE_API_KEY = os.environ.get("LIBRETRANSLATE_API_KEY")
MYMEMORY_URL = "https://api.mymemory.translated.net/get"
MYMEMORY_MAX_BYTES = int(os.environ.get("MYMEMORY_MAX_BYTES", "450"))
TRANSLATION_SLEEP = float(os.environ.get("TRANSLATION_SLEEP", "0.8"))
TRANSLATION_MAX_RETRIES = int(os.environ.get("TRANSLATION_MAX_RETRIES", "5"))
TRANSLATION_BASE_SLEEP = float(os.environ.get("TRANSLATION_BASE_SLEEP", "2.0"))
TRANSLATION_MAX_SLEEP = float(os.environ.get("TRANSLATION_MAX_SLEEP", "60.0"))
_TRANSLATION_CACHE = {}


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _info(msg):
    print(f"{_ts()} | INFO | {msg}")


def _warn(msg):
    print(f"{_ts()} | WARN | {msg}")


def _err(msg):
    print(f"{_ts()} | ERROR | {msg}")


_info(f"Starting V2 fetch. Chain={TRANSLATION_CHAIN}, DaysToCheck={DAYS_TO_CHECK}, DaysToCompare={DAYS_TO_COMPARE}, MaxPapers={MAX_PAPERS}")


def _normalize_chain(chain_str):
    items = [p.strip().lower() for p in chain_str.split(",") if p.strip()]
    seen = set()
    ordered = []
    for p in items:
        if p not in ("mymemory", "libretranslate"):
            continue
        if p in seen:
            continue
        seen.add(p)
        ordered.append(p)
    if not ordered:
        ordered = ["mymemory"]
    return ordered


def _provider_chain():
    chain = _normalize_chain(TRANSLATION_CHAIN)
    if "libretranslate" in chain and not LIBRETRANSLATE_URL:
        _warn("LibreTranslate in chain but LIBRETRANSLATE_URL not set. Skipping LibreTranslate.")
        chain = [p for p in chain if p != "libretranslate"]
    if not chain:
        chain = ["mymemory"]
    return chain

# AI analysis client
deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
openai_key = os.environ.get("OPENAI_API_KEY")

client = None
model_name = None

if deepseek_key:
    _info("Using DeepSeek API for analysis")
    client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
    model_name = "deepseek-chat"
elif openai_key and openai_key.startswith("AIza"):
    _info("Using Gemini API for analysis")
    client = OpenAI(
        api_key=openai_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    model_name = "gemini-2.0-flash"
elif openai_key:
    _info("Using OpenAI API for analysis")
    client = OpenAI(api_key=openai_key)
    model_name = "gpt-4o-mini"
else:
    _info("No AI analysis API key found. Skipping AI analysis.")


def fetch_arxiv_papers(category, days=3, max_retries=3):
    """Fetch papers for a category from arXiv API."""
    base_url = "http://export.arxiv.org/api/query?"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    query = f"cat:{category}"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": 500,
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
    if not client or not model_name:
        return "AI analysis skipped: no API key."

    max_chars = 30000
    if len(pdf_text) > max_chars:
        half = max_chars // 2
        pdf_text = pdf_text[:half] + "\n\n[... Omitted ...]\n\n" + pdf_text[-half:]

    _info(f"AI analysis using model: {model_name}")

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
                {"role": "system", "content": "You are a rigorous academic reviewer with strong critical thinking."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=2000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        _warn(f"AI analysis failed: {e}")
        if "402" in str(e) or "Insufficient Balance" in str(e):
            return "AI analysis failed: insufficient API balance."
        return f"AI analysis unavailable: {e}"


def _looks_chinese(text):
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk >= max(3, int(len(text) * 0.3))


def _hard_split_by_bytes(text, max_bytes):
    buf = ""
    buf_bytes = 0
    for ch in text:
        b = len(ch.encode("utf-8"))
        if buf and buf_bytes + b > max_bytes:
            yield buf
            buf = ch
            buf_bytes = b
        else:
            buf += ch
            buf_bytes += b
    if buf:
        yield buf


def _split_text_by_bytes(text, max_bytes):
    if len(text.encode("utf-8")) <= max_bytes:
        return [text]
    parts = re.split(r"(?<=[.!?。！？])\s+", text)
    chunks = []
    current = ""
    current_bytes = 0
    for part in parts:
        if not part:
            continue
        part_bytes = len(part.encode("utf-8"))
        if part_bytes > max_bytes:
            for piece in _hard_split_by_bytes(part, max_bytes):
                piece_bytes = len(piece.encode("utf-8"))
                if current and current_bytes + piece_bytes + 1 > max_bytes:
                    chunks.append(current)
                    current = ""
                    current_bytes = 0
                if current:
                    current += " " + piece
                    current_bytes += 1 + piece_bytes
                else:
                    current = piece
                    current_bytes = piece_bytes
                chunks.append(current)
                current = ""
                current_bytes = 0
            continue
        if current and current_bytes + part_bytes + 1 > max_bytes:
            chunks.append(current)
            current = part
            current_bytes = part_bytes
        else:
            if current:
                current += " " + part
                current_bytes += 1 + part_bytes
            else:
                current = part
                current_bytes = part_bytes
    if current:
        chunks.append(current)
    return chunks


def _translate_mymemory(text):
    params = {
        "q": text,
        "langpair": "en|zh-CN",
    }
    if MYMEMORY_EMAIL:
        params["de"] = MYMEMORY_EMAIL
    url = MYMEMORY_URL + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    req = urllib.request.Request(url, headers={"User-Agent": "daily-video-papers/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("responseStatus") != 200:
        raise RuntimeError(f"MyMemory error: {data.get('responseStatus')} {data.get('responseDetails')}")
    return data.get("responseData", {}).get("translatedText") or ""


def _translate_libretranslate(text):
    if not LIBRETRANSLATE_URL:
        raise RuntimeError("LIBRETRANSLATE_URL not set")
    payload = {
        "q": text,
        "source": "en",
        "target": "zh",
        "format": "text",
    }
    if LIBRETRANSLATE_API_KEY:
        payload["api_key"] = LIBRETRANSLATE_API_KEY
    data = urllib.parse.urlencode(payload).encode("utf-8")
    req = urllib.request.Request(
        LIBRETRANSLATE_URL.rstrip("/") + "/translate",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "daily-video-papers/1.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result.get("translatedText") or ""


def _translate_with_retries(func, chunk, chunk_bytes, provider_name):
    for attempt in range(1, TRANSLATION_MAX_RETRIES + 1):
        try:
            return func(chunk)
        except urllib.error.HTTPError as e:
            status = getattr(e, "code", None)
            retry_after = e.headers.get("Retry-After") if hasattr(e, "headers") else None
            if status in (429, 500, 502, 503, 504):
                base = TRANSLATION_BASE_SLEEP * (2 ** (attempt - 1))
                sleep_s = min(TRANSLATION_MAX_SLEEP, base)
                if retry_after:
                    try:
                        sleep_s = max(sleep_s, float(retry_after))
                    except ValueError:
                        pass
                sleep_s += random.uniform(0, 1.0)
                _warn(f"{provider_name} HTTP {status} on attempt {attempt} (bytes={chunk_bytes}). Sleep {sleep_s:.1f}s")
                time.sleep(sleep_s)
                continue
            raise
        except Exception as e:
            base = TRANSLATION_BASE_SLEEP * (2 ** (attempt - 1))
            sleep_s = min(TRANSLATION_MAX_SLEEP, base) + random.uniform(0, 1.0)
            _warn(f"{provider_name} error on attempt {attempt} (bytes={chunk_bytes}): {e}. Sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)
            continue
    return ""


def translate_text(text):
    """Translate text to Chinese using a free API chain."""
    if not text or not text.strip():
        return ""
    if _looks_chinese(text):
        return text
    cached = _TRANSLATION_CACHE.get(text)
    if cached is not None:
        return cached

    chain = _provider_chain()
    chunks = _split_text_by_bytes(text, MYMEMORY_MAX_BYTES)
    translated_chunks = []

    for chunk in chunks:
        chunk_bytes = len(chunk.encode("utf-8"))
        translated = ""
        for provider in chain:
            if provider == "libretranslate":
                translated = _translate_with_retries(_translate_libretranslate, chunk, chunk_bytes, "LibreTranslate")
            else:
                translated = _translate_with_retries(_translate_mymemory, chunk, chunk_bytes, "MyMemory")
            if translated:
                break
        if not translated:
            _err(f"Translation failed after retries (bytes={chunk_bytes}). Returning source chunk.")
            translated = chunk
        translated_chunks.append(translated)
        time.sleep(TRANSLATION_SLEEP)

    result = "".join(translated_chunks).strip()
    _TRANSLATION_CACHE[text] = result or text
    return result or text


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


def generate_markdown_v2(papers, date_str):
    """Generate Markdown V2 with AI analysis."""
    md_content = f"# arXiv Video Papers - {date_str} (V2 Enhanced)\n\n"
    md_content += f"**Update Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += f"**Paper Count**: {len(papers)} | **Version**: V2 (AI Analysis)\n\n---\n\n"

    for i, paper in enumerate(papers, 1):
        _info(f"Processing {i}/{len(papers)}: {paper['id']}")
        title_zh = translate_text(paper["title"])
        summary_zh = translate_text(paper["summary"])
        links = extract_links(paper)

        ai_analysis = "PDF download or analysis failed."
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / f"{paper['id']}.pdf"
            if download_pdf(paper["pdf_url"], pdf_path):
                pdf_text = extract_text_from_pdf(pdf_path)
                if pdf_text and len(pdf_text.strip()) > 500:
                    ai_analysis = analyze_paper_with_ai(paper, pdf_text)
                else:
                    ai_analysis = "PDF text extraction failed or too short."

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
        md_content += f"<details><summary><b>AI Analysis</b></summary>\n\n{ai_analysis}\n\n</details>\n\n---\n\n"
        time.sleep(2)
    return md_content


def _build_readme_sections(papers_dir):
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
    return index_content, content_section


def update_readme_index():
    """Update README index and daily content."""
    base_dir = Path(__file__).parent.parent
    papers_dir = base_dir / "papers"
    readme_path = base_dir / "README.md"
    if not papers_dir.exists():
        return
    index_content, content_section = _build_readme_sections(papers_dir)

    readme_content = readme_path.read_text(encoding="utf-8")
    index_pattern = r'<!-- PAPERS_INDEX_START -->.*?<!-- PAPERS_INDEX_END -->'
    index_replacement = f'<!-- PAPERS_INDEX_START -->{index_content}<!-- PAPERS_INDEX_END -->'
    readme_content = re.sub(index_pattern, index_replacement, readme_content, flags=re.DOTALL)

    content_pattern = r'<!-- PAPERS_CONTENT_START -->.*?<!-- PAPERS_CONTENT_END -->'
    content_replacement = f'<!-- PAPERS_CONTENT_START -->{content_section}<!-- PAPERS_CONTENT_END -->'
    readme_content = re.sub(content_pattern, content_replacement, readme_content, flags=re.DOTALL)

    readme_path.write_text(readme_content, encoding="utf-8")


def main():
    recent_ids = load_recent_papers(DAYS_TO_COMPARE)
    all_papers = []
    for i, cat in enumerate(CATEGORIES):
        _info(f"Fetching {cat}...")
        papers = fetch_arxiv_papers(cat, DAYS_TO_CHECK)
        all_papers.extend(papers)
        if i < len(CATEGORIES) - 1:
            time.sleep(3)

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
    md_content = generate_markdown_v2(new_papers, date_str)

    papers_dir = Path(__file__).parent.parent / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    (papers_dir / f"{date_str}.md").write_text(md_content, encoding="utf-8")
    update_readme_index()
    _info(f"Completed. Output written to papers/{date_str}.md")


if __name__ == "__main__":
    main()
