#!/usr/bin/env python3
"""
arXiv Papers Fetcher V2 (Extended Video Version with AI Analysis)
在 V1 基础上增加 PDF 全文下载和 AI 深度分析功能，支持 DeepSeek API
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
from openai import OpenAI
import tempfile
import subprocess

# 配置
CATEGORIES = ["cs.CV", "cs.AI", "cs.MM", "cs.RO", "cs.LG"]
DAYS_TO_CHECK = 3
DAYS_TO_COMPARE = 5

# 扩展后的视频相关关键词
VIDEO_KEYWORDS = [
    # 生成与编辑
    "video generation", "video synthesis", "video editing", "video edit",
    "video diffusion", "text-to-video", "image-to-video", "video-to-video",
    "motion generation", "character animation", "talking head", "human motion",
    # 理解与分析
    "video understanding", "video recognition", "video classification", "action recognition",
    "action detection", "temporal action", "video retrieval", "video captioning",
    "video question answering", "video QA", "video summarization",
    # 处理与增强
    "video super-resolution", "video enhancement", "video restoration", "video denoising",
    "video interpolation", "frame interpolation", "video compression", "video coding",
    # 分割与跟踪
    "video segmentation", "video object segmentation", "VOS", "video instance segmentation",
    "VIS", "object tracking", "multi-object tracking", "MOT", "video matting",
    # 基础模型与时序
    "video model", "temporal modeling", "spatio-temporal", "video transformer",
    "video representation", "optical flow", "video prediction", "future frame prediction"
]

# 初始化客户端
deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
openai_key = os.environ.get("OPENAI_API_KEY")

if deepseek_key:
    print("Using DeepSeek API")
    client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
    model_name = "deepseek-chat"
elif openai_key and openai_key.startswith("AIza"):
    print("Using Gemini API (via Google AI Studio)")
    client = OpenAI(
        api_key=openai_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    model_name = "gemini-2.0-flash"
else:
    print("Using Standard OpenAI API")
    client = OpenAI(api_key=openai_key)
    model_name = "gpt-4o-mini"

def fetch_arxiv_papers(category, days=3, max_retries=3):
    """从 arXiv API 获取指定类别的论文"""
    base_url = "http://export.arxiv.org/api/query?"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    query = f"cat:{category}"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": 500,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    url = base_url + urllib.parse.urlencode(params)
    for attempt in range(max_retries):
        try:
            if attempt > 0: time.sleep(5 * attempt)
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
            root = ET.fromstring(data)
            namespace = {"atom": "http://www.w3.org/2005/Atom"}
            papers = []
            for entry in root.findall("atom:entry", namespace):
                published = entry.find("atom:published", namespace).text
                pub_date = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ")
                if pub_date < start_date: continue
                paper = {
                    "id": entry.find("atom:id", namespace).text.split("/abs/")[-1],
                    "title": entry.find("atom:title", namespace).text.strip().replace("\n", " "),
                    "summary": entry.find("atom:summary", namespace).text.strip().replace("\n", " "),
                    "authors": [author.find("atom:name", namespace).text for author in entry.findall("atom:author", namespace)],
                    "published": pub_date.strftime("%Y-%m-%d"),
                    "pdf_url": entry.find("atom:id", namespace).text.replace("/abs/", "/pdf/"),
                    "abs_url": entry.find("atom:id", namespace).text,
                    "categories": [cat.attrib["term"] for cat in entry.findall("atom:category", namespace)]
                }
                papers.append(paper)
            return papers
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1: return []
    return []

def is_video_related(paper):
    """判断论文是否与视频相关"""
    text = (paper["title"] + " " + paper["summary"]).lower()
    return any(keyword.lower() in text for keyword in VIDEO_KEYWORDS)

def extract_links(paper):
    """从摘要中提取项目和代码链接"""
    text = paper["summary"]
    url_pattern = r'https?://[^\s<>"{}\\|\\^`\\[\\]]+'
    urls = re.findall(url_pattern, text)
    links = {"project": None, "code": None}
    for url in urls:
        url_lower = url.lower()
        if "github.com" in url_lower or "gitlab.com" in url_lower:
            if not links["code"]: links["code"] = url
        elif any(k in url_lower for k in ["project", "page", "site"]):
            if not links["project"]: links["project"] = url
        elif not links["project"]: links["project"] = url
    return links

def download_pdf(pdf_url, output_path):
    """下载 PDF 文件"""
    try:
        urllib.request.urlretrieve(pdf_url, output_path)
        return True
    except Exception as e:
        print(f"  PDF download failed: {e}")
        return False

def extract_text_from_pdf(pdf_path):
    """从 PDF 提取文本"""
    try:
        result = subprocess.run(["pdftotext", "-layout", pdf_path, "-"], capture_output=True, text=True, timeout=30)
        return result.stdout if result.returncode == 0 else None
    except Exception as e:
        print(f"  PDF text extraction error: {e}")
        return None

def analyze_paper_with_ai(paper, pdf_text):
    """使用 AI 分析论文全文"""
    max_chars = 30000
    if len(pdf_text) > max_chars:
        half = max_chars // 2
        pdf_text = pdf_text[:half] + "\n\n[... Omitted ...]\n\n" + pdf_text[-half:]
    
    print(f"  [Debug] Using model: {model_name} for AI analysis...")
    
    prompt = f'''你是一位资深的计算机视觉研究专家。请仔细阅读以下论文的全文内容，并进行深度分析。
论文标题: {paper["title"]}
论文全文:
{pdf_text}

请从以下几个方面进行批判性分析：
1. **核心观点**: 用最简单直白的语言（1-2句话）说明论文的核心创新点
2. **技术方法**: 简要说明采用的主要技术方法和架构
3. **实验验证**: 评估实验设计的合理性、数据集选择、对比方法是否充分
4. **结果可靠性**: 分析实验结果的可信度，是否存在过拟合、cherry-picking 等问题
5. **实用价值**: 评估该研究的实际应用价值和局限性
6. **批判性评价**: 指出论文的优点和不足，以及可能存在的问题

请用简洁、专业但易懂的中文回答，每个方面控制在 2-3 句话以内。'''

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一位严谨的学术研究分析专家，擅长批判性阅读和评估论文质量。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [Error] AI analysis failed: {str(e)}")
        if "402" in str(e) or "Insufficient Balance" in str(e):
            return "AI 分析失败：API 余额不足"
        return f"AI 分析暂时不可用: {e}"

def translate_text(text):
    """使用 AI 翻译文本"""
    if not text or not text.strip():
        return ""
    print(f"  [Debug] Using model: {model_name} for translation...")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a professional translator. Translate the following academic text to Chinese. Keep technical terms in English when appropriate. Only return the translation, no explanations."},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        result = response.choices[0].message.content.strip()
        return result if result else text
    except Exception as e:
        print(f"  [Error] Translation failed: {str(e)}")
        return text

def load_recent_papers(days=5):
    """加载最近几天的论文 ID 用于去重"""
    papers_dir = Path(__file__).parent.parent / "papers"
    recent_ids = set()
    if not papers_dir.exists(): return recent_ids
    start_date = datetime.now() - timedelta(days=days)
    for md_file in papers_dir.glob("*.md"):
        try:
            if datetime.strptime(md_file.stem, "%Y-%m-%d") >= start_date:
                content = md_file.read_text(encoding="utf-8")
                recent_ids.update(re.findall(r'arxiv\.org/abs/(\d+\.\d+)', content))
        except: continue
    return recent_ids

def generate_markdown_v2(papers, date_str):
    """生成 V2 版本的 Markdown（包含 AI 分析）"""
    md_content = f"# arXiv Video Papers - {date_str} (V2 Enhanced)\n\n"
    md_content += f"**Update Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += f"**Paper Count**: {len(papers)} | **Version**: V2 (AI Analysis)\n\n---\n\n"
    
    for i, paper in enumerate(papers, 1):
        print(f"Processing {i}/{len(papers)}: {paper['id']}")
        title_zh = translate_text(paper["title"])
        summary_zh = translate_text(paper["summary"])
        links = extract_links(paper)
        
        ai_analysis = "PDF 下载或分析失败"
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / f'{paper["id"]}.pdf'
            if download_pdf(paper["pdf_url"], pdf_path):
                pdf_text = extract_text_from_pdf(pdf_path)
                if pdf_text and len(pdf_text.strip()) > 500:
                    ai_analysis = analyze_paper_with_ai(paper, pdf_text)
                else:
                    ai_analysis = "PDF 文本提取失败或内容过短"
        
        md_content += f"## {i}. {paper['title']}\n\n"
        md_content += f"**中文标题**: {title_zh}\n\n"
        md_content += f"**Authors**: {', '.join(paper['authors'][:5])}{' et al.' if len(paper['authors']) > 5 else ''}\n\n"
        md_content += f"**Date**: {paper['published']} | **arXiv**: [{paper['id']}]({paper['abs_url']}) | **PDF**: [Link]({paper['pdf_url']})\n\n"
        if links["project"]: md_content += f"**Project**: {links['project']}  "
        if links["code"]: md_content += f"**Code**: {links['code']}\n\n"
        md_content += f"**Categories**: {', '.join(paper['categories'])}\n\n"
        md_content += f'<details><summary><b>Abstract</b></summary>\n\n{paper["summary"]}\n\n</details>\n\n'
        md_content += f'<details><summary><b>中文摘要</b></summary>\n\n{summary_zh}\n\n</details>\n\n'
        md_content += f'<details><summary><b>🤖 AI 阅读分析</b></summary>\n\n{ai_analysis}\n\n</details>\n\n---\n\n'
        time.sleep(2)
    return md_content

def update_readme_index():
    """更新 README 中的论文索引"""
    base_dir = Path(__file__).parent.parent
    papers_dir = base_dir / "papers"
    readme_path = base_dir / "README.md"
    if not papers_dir.exists(): return
    paper_files = sorted(papers_dir.glob("*.md"), reverse=True)
    index_content = "\n"
    for f in paper_files:
        content = f.read_text(encoding="utf-8")
        count = re.search(r'\*\*Paper Count\*\*: (\d+)', content)
        index_content += f"- [{f.stem}](papers/{f.name}) - {count.group(1) if count else '0'} papers\n"
    
    readme_content = readme_path.read_text(encoding="utf-8")
    pattern = r'<!-- PAPERS_INDEX_START -->.*?<!-- PAPERS_INDEX_END -->'
    replacement = f'<!-- PAPERS_INDEX_START -->{index_content}<!-- PAPERS_INDEX_END -->'
    readme_path.write_text(re.sub(pattern, replacement, readme_content, flags=re.DOTALL), encoding="utf-8")

def main():
    recent_ids = load_recent_papers(DAYS_TO_COMPARE)
    all_papers = []
    for i, cat in enumerate(CATEGORIES):
        print(f"Fetching {cat}...")
        papers = fetch_arxiv_papers(cat, DAYS_TO_CHECK)
        all_papers.extend(papers)
        if i < len(CATEGORIES) - 1: time.sleep(3)
    
    unique_papers = {p["id"]: p for p in all_papers}
    video_papers = [p for p in unique_papers.values() if is_video_related(p)]
    new_papers = [p for p in video_papers if p["id"] not in recent_ids]
    
    if not new_papers:
        print("No new papers found.")
        return
    
    new_papers.sort(key=lambda x: x["published"], reverse=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    md_content = generate_markdown_v2(new_papers, date_str)
    
    papers_dir = Path(__file__).parent.parent / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    (papers_dir / f"{date_str}.md").write_text(md_content, encoding="utf-8")
    update_readme_index()

if __name__ == "__main__":
    main()
