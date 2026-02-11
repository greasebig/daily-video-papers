# Daily Video Papers 🎥

每日自动更新 arXiv 视频相关研究论文。

## 🌟 功能特性

- **广泛筛选**：涵盖视频生成、编辑、理解、分割、跟踪、增强等全方位视频研究。
- **双语支持**：所有论文标题 and 摘要均配备中文翻译。
- **AI 深度分析**：支持 V2 版本，利用 AI 进行全文阅读和批判性分析。
- **自动去重**：智能对比最近 5 天内容，确保不重复更新。
- **一键访问**：自动提取项目主页和代码仓库链接。

## 📚 论文索引

<!-- PAPERS_INDEX_START -->
- [2026-02-11](papers/2026-02-11.md) - 357 papers
<!-- PAPERS_INDEX_END -->

## 🚀 快速开始

1. **配置 API Key**：在仓库 `Settings -> Secrets and variables -> Actions -> Secrets` 中添加 `DEEPSEEK_API_KEY`。
2. **手动运行**：在 `Actions` 标签页选择 `Daily Video Papers Update` 并点击 `Run workflow`。
3. **切换版本**：在 `Variables` 中设置 `VERSION` 为 `v2` 即可开启 AI 深度分析模式。

## 🛠️ 技术细节

- **数据源**：arXiv API (cs.CV, cs.AI, cs.MM, cs.RO, cs.LG)
- **翻译/分析**：DeepSeek API (优先) / Gemini (备用)
- **自动化**：GitHub Actions

---
*本项目由 Manus 自动生成并维护。*
