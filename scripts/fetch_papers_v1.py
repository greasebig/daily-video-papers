#!/usr/bin/env python3
"""
arXiv Papers Fetcher V1 (Extended Video Version)
每日抓取 arXiv 视频相关论文，扩大筛选范围，支持 DeepSeek API
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

# 配置
CATEGORIES = ["cs.CV", "cs.AI", "cs.MM", "cs.RO", "cs.LG"]
DAYS_TO_CHECK = 3  # 检查最近 3 天的论文
DAYS_TO_COMPARE = 5  # 与最近 5 天对比去重

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

# 初始化 OpenAI 客户端
# 优先使用 DEEPSEEK_API_KEY，如果没有则回退到默认配置
api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
base_url = "https://api.deepseek.com" if os.environ.get("DEEPSEEK_API_KEY") else None
model_name = "deepseek-chat" if os.environ.get("DEEPSEEK_API_KEY") else "gemini-2.5-flash"

client = OpenAI(api_key=api_key, base_url=base_url)

def fetch_arxiv_papers(category, days=3, max_retries=3):
    """从 arXiv API 获取指定类别的论文"""
    base_url = "http://export.arxiv.org/api/query?"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    query = f"cat:{category}"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": 500,  # 扩大范围以确保不遗漏
        "sortBy": "submittedDate",
        "sortOrder": "descending"
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
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
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

def translate_text(text):
    """使用 AI 翻译文本"""
    if not text or not text.strip():
        return ""
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
        # 如果是余额不足 (402)，打印明确提示
        if "402" in str(e) or "Insufficient Balance" in str(e):
            print(f"  [Warning] API 余额不足，跳过翻译。")
        else:
            print(f"  [Warning] 翻译出错: {e}")
        return f"[翻译失败: {text[:50]}...]" if len(text) > 50 else f"[翻译失败: {text}]"

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

def generate_markdown(papers, date_str):
    """生成 Markdown 格式的论文列表"""
    md_content = f"# arXiv Video Papers - {date_str}\n\n"
    md_content += f"**Update Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += f"**Paper Count**: {len(papers)}\n\n---\n\n"
    
    for i, paper in enumerate(papers, 1):
        print(f"Processing {i}/{len(papers)}: {paper['id']}")
        title_zh = translate_text(paper["title"])
        summary_zh = translate_text(paper["summary"])
        links = extract_links(paper)
        
        md_content += f"## {i}. {paper['title']}\n\n"
        md_content += f"**中文标题**: {title_zh}\n\n"
        md_content += f"**Authors**: {', '.join(paper['authors'][:5])}{' et al.' if len(paper['authors']) > 5 else ''}\n\n"
        md_content += f"**Date**: {paper['published']} | **arXiv**: [{paper['id']}]({paper['abs_url']}) | **PDF**: [Link]({paper['pdf_url']})\n\n"
        if links["project"]: md_content += f"**Project**: {links['project']}  "
        if links["code"]: md_content += f"**Code**: {links['code']}\n\n"
        md_content += f"**Categories**: {', '.join(paper['categories'])}\n\n"
        md_content += f"<details><summary><b>Abstract</b></summary>\n\n{paper['summary']}\n\n</details>\n\n"
        md_content += f"<details><summary><b>中文摘要</b></summary>\n\n{summary_zh}\n\n</details>\n\n---\n\n"
        time.sleep(1)
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
    md_content = generate_markdown(new_papers, date_str)
    
    papers_dir = Path(__file__).parent.parent / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    (papers_dir / f"{date_str}.md").write_text(md_content, encoding="utf-8")
    update_readme_index()

if __name__ == "__main__":
    main()
