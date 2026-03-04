# Daily Video Papers 🎥

?? **Website**: https://greasebig.github.io/daily-video-papers/

![Actions](https://github.com/greasebig/daily-video-papers/actions/workflows/daily-update.yml/badge.svg) ![Pages](https://img.shields.io/badge/pages-online-brightgreen) ![Visitors](https://visitor-badge.laobi.icu/badge?page_id=greasebig.daily-video-papers)

## 📚 论文索引

<!-- PAPERS_INDEX_START -->
- [2026-03-04](papers/2026-03-04.md) - 23 papers
- [2026-03-03](papers/2026-03-03.md) - 2 papers
- [2026-03-02](papers/2026-03-02.md) - 6 papers
- [2026-02-27](papers/2026-02-27.md) - 18 papers
- [2026-02-26](papers/2026-02-26.md) - 11 papers
- [2026-02-25](papers/2026-02-25.md) - 12 papers
- [2026-02-24](papers/2026-02-24.md) - 8 papers
- [2026-02-23](papers/2026-02-23.md) - 2 papers
- [2026-02-20](papers/2026-02-20.md) - 2 papers
- [2026-02-19](papers/2026-02-19.md) - 8 papers
- [2026-02-18](papers/2026-02-18.md) - 8 papers
- [2026-02-17](papers/2026-02-17.md) - 3 papers
- [2026-02-16](papers/2026-02-16.md) - 4 papers
- [2026-02-14](papers/2026-02-14.md) - 16 papers
<!-- PAPERS_INDEX_END -->

## Other Topics

- [World Model Papers](world-model/README.md)
- [Agent Papers](agent/README.md)

## Daily Papers

<!-- PAPERS_CONTENT_START -->
<details><summary><b>2026-03-04 (23 papers)</b></summary>

# arXiv Video Papers - 2026-03-04

**Paper Count**: 23

---

## 1. EduVQA: Benchmarking AI-Generated Video Quality Assessment for Education / EduVQA：人工智能生成的教育视频质量评估基准

**Date**: 2026-03-03 | **arXiv**: [2603.03066v1](http://arxiv.org/abs/2603.03066v1) | **PDF**: [Link](http://arxiv.org/pdf/2603.03066v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

While AI-generated content (AIGC) models have achieved remarkable success in generating photorealistic videos, their potential to support visual, story-driven learning in education remains largely untapped. To close this gap, we present EduAIGV-1k, the first benchmark dataset and evaluation framework dedicated to assessing the quality of AI-generated videos (AIGVs) designed to teach foundational math concepts, such as numbers and geometry, to young learners. EduAIGV-1k contains 1,130 short videos produced by ten state-of-the-art text-to-video (T2V) models using 113 pedagogy-oriented prompts. Each video is accompanied by rich, fine-grained annotations along two complementary axes: (1) Perceptual quality, disentangled into spatial and temporal fidelity, and (2) Prompt alignment, labeled at the word-level and sentence-level to quantify the degree to which each mathematical concept in the prompt is accurately grounded in the generated video. These fine-grained annotations transform each video into a multi-dimensional, interpretable supervision signal, far beyond a single quality score. Leveraging this dense feedback, we introduce EduVQA for both perceptual and alignment quality assessment of AIGVs. In particular, we propose a Structured 2D Mixture-of-Experts (S2D-MoE) module, which enhances the dependency between overall quality and each sub-dimension by shared experts and dynamic 2D gating matrix. Extensive experiments show our EduVQA consistently outperforms existing VQA baselines. Both our dataset and code will be publicly available.

虽然人工智能生成内容 (AIGC) 模型在生成逼真视频方面取得了显着成功，但它们在教育中支持视觉、故事驱动学习的潜力在很大程度上尚未开发。为了弥补这一差距，我们推出了 EduAIGV-1k，这是第一个基准数据集和评估框架，致力于评估人工智能生成视频 (AIGV) 的质量，旨在向年轻学习者教授数字和几何等基础数学概念。 EduAIGV-1k 包含 1,130 个短视频，由 10 个最先进的文本转视频 (T2V) 模型使用 113 个面向教学的提示制作。每个视频都沿着两个互补轴附有丰富、细粒度的注释：(1) 感知质量，分解为空间和时间保真度；(2) 提示对齐，在单词级别和句子级别进行标记，以量化提示中的每个数学概念在生成的视频中准确落地的程度。这些细粒度的注释将每个视频转换为多维、可解释的监督信号，远远超出了单一的质量分数。利用这种密集的反馈，我们引入了 EduVQA，用于 AIGV 的感知和对准质量评估。特别是，我们提出了结构化2D混合专家（S2D-MoE）模块，该模块通过共享专家和动态2D门控矩阵增强了整体质量与每个子维度之间的依赖性。大量实验表明，我们的 EduVQA 始终优于现有的 VQA 基线。我们的数据集和代码都将公开。

</details>

---

## 2. TC-Padé: Trajectory-Consistent Padé Approximation for Diffusion Acceleration / TC-Padé：扩散加速的轨迹一致 Padé 近似

**Date**: 2026-03-03 | **arXiv**: [2603.02943v1](http://arxiv.org/abs/2603.02943v1) | **PDF**: [Link](http://arxiv.org/pdf/2603.02943v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Despite achieving state-of-the-art generation quality, diffusion models are hindered by the substantial computational burden of their iterative sampling process. While feature caching techniques achieve effective acceleration at higher step counts (e.g., 50 steps), they exhibit critical limitations in the practical low-step regime of 20-30 steps. As the interval between steps increases, polynomial-based extrapolators like TaylorSeer suffer from error accumulation and trajectory drift. Meanwhile, conventional caching strategies often overlook the distinct dynamical properties of different denoising phases. To address these challenges, we propose Trajectory-Consistent Padé approximation, a feature prediction framework grounded in Padé approximation. By modeling feature evolution through rational functions, our approach captures asymptotic and transitional behaviors more accurately than Taylor-based methods. To enable stable and trajectory-consistent sampling under reduced step counts, TC-Padé incorporates (1) adaptive coefficient modulation that leverages historical cached residuals to detect subtle trajectory transitions, and (2) step-aware prediction strategies tailored to the distinct dynamics of early, mid, and late sampling stages. Extensive experiments on DiT-XL/2, FLUX.1-dev, and Wan2.1 across both image and video generation demonstrate the effectiveness of TC-Padé. For instance, TC-Padé achieves 2.88x acceleration on FLUX.1-dev and 1.72x on Wan2.1 while maintaining high quality across FID, CLIP, Aesthetic, and VBench-2.0 metrics, substantially outperforming existing feature caching methods.

尽管实现了最先进的生成质量，但扩散模型仍因其迭代采样过程的大量计算负担而受到阻碍。虽然特征缓存技术在较高步数（例如 50 步）下实现了有效加速，但它们在 20-30 步的实际低步状态中表现出严重的局限性。随着步骤之间的间隔增加，像 TaylorSeer 这样基于多项式的外推器会遭受误差累积和轨迹漂移。同时，传统的缓存策略常常忽视不同去噪阶段的独特动态特性。为了应对这些挑战，我们提出了轨迹一致的 Padé 近似，这是一种基于 Padé 近似的特征预测框架。通过有理函数对特征演化进行建模，我们的方法比基于泰勒的方法更准确地捕获渐近和过渡行为。为了在减少步数的情况下实现稳定且轨迹一致的采样，TC-Padé 结合了 (1) 自适应系数调制，利用历史缓存残差来检测微妙的轨迹转换，以及 (2) 针对早期、中期和晚期采样阶段的不同动态量身定制的步感知预测策略。在 DiT-XL/2、FLUX.1-dev 和 Wan2.1 上进行的图像和视频生成方面的大量实验证明了 TC-Padé 的有效性。例如，TC-Padé 在 FLUX.1-dev 上实现了 2.88 倍加速，在 Wan2.1 上实现了 1.72 倍加速，同时在 FID、CLIP、Aesthetic 和 VBench-2.0 指标上保持了高质量，大大优于现有的特征缓存方法。

</details>

---

## 3. Interpretable Motion-Attentive Maps: Spatio-Temporally Localizing Concepts in Video Diffusion Transformers / 可解释的运动注意力图：视频扩散变压器中的时空定位概念

**Date**: 2026-03-03 | **arXiv**: [2603.02919v1](http://arxiv.org/abs/2603.02919v1) | **PDF**: [Link](http://arxiv.org/pdf/2603.02919v1)

**Categories**: cs.CV, cs.AI, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Video Diffusion Transformers (DiTs) have been synthesizing high-quality video with high fidelity from given text descriptions involving motion. However, understanding how Video DiTs convert motion words into video remains insufficient. Furthermore, while prior studies on interpretable saliency maps primarily target objects, motion-related behavior in Video DiTs remains largely unexplored. In this paper, we investigate concrete motion features that specify when and which object moves for a given motion concept. First, to spatially localize, we introduce GramCol, which adaptively produces per-frame saliency maps for any text concept, including both motion and non-motion. Second, we propose a motion-feature selection algorithm to obtain an Interpretable Motion-Attentive Map (IMAP) that localizes motion spatially and temporally. Our method discovers concept saliency maps without the need for any gradient calculation or parameter update. Experimentally, our method shows outstanding localization capability on the motion localization task and zero-shot video semantic segmentation, providing interpretable and clearer saliency maps for both motion and non-motion concepts.

视频扩散变压器 (DiT) 已经根据给定的涉及运动的文本描述合成了高保真度的高质量视频。然而，了解 Video DiT 如何将动作词转换为视频仍然不够。此外，虽然先前对可解释显着性图的研究主要针对目标对象，但视频 DiT 中的运动相关行为在很大程度上仍未得到探索。在本文中，我们研究了具体的运动特征，这些特征指定了给定运动概念的对象何时移动以及哪个对象移动。首先，为了进行空间定位，我们引入了 GramCol，它可以自适应地为任何文本概念（包括运动和非运动）生成每帧显着性图。其次，我们提出了一种运动特征选择算法，以获得可在空间和时间上定位运动的可解释运动注意力图（IMAP）。我们的方法无需任何梯度计算或参数更新即可发现概念显着图。实验上，我们的方法在运动定位任务和零镜头视频语义分割上显示出出色的定位能力，为运动和非运动概念提供可解释且更清晰的显着性图。

</details>

---

## 4. LLandMark: A Multi-Agent Framework for Landmark-Aware Multimodal Interactive Video Retrieval / LLandMark：用于地标感知多模态交互式视频检索的多代理框架

**Date**: 2026-03-03 | **arXiv**: [2603.02888v1](http://arxiv.org/abs/2603.02888v1) | **PDF**: [Link](http://arxiv.org/pdf/2603.02888v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

The increasing diversity and scale of video data demand retrieval systems capable of multimodal understanding, adaptive reasoning, and domain-specific knowledge integration. This paper presents LLandMark, a modular multi-agent framework for landmark-aware multimodal video retrieval to handle real-world complex queries. The framework features specialized agents that collaborate across four stages: query parsing and planning, landmark reasoning, multimodal retrieval, and reranked answer synthesis. A key component, the Landmark Knowledge Agent, detects cultural or spatial landmarks and reformulates them into descriptive visual prompts, enhancing CLIP-based semantic matching for Vietnamese scenes. To expand capabilities, we introduce an LLM-assisted image-to-image pipeline, where a large language model (Gemini 2.5 Flash) autonomously detects landmarks, generates image search queries, retrieves representative images, and performs CLIP-based visual similarity matching, removing the need for manual image input. In addition, an OCR refinement module leveraging Gemini and LlamaIndex improves Vietnamese text recognition. Experimental results show that LLandMark achieves adaptive, culturally grounded, and explainable retrieval performance.

视频数据的多样性和规模不断增加，需要检索系统能够多模态理解、自适应推理和特定领域的知识集成。本文介绍了 LLandMark，这是一种模块化多代理框架，用于地标感知多模态视频检索，以处理现实世界的复杂查询。该框架具有跨四个阶段协作的专业代理：查询解析和规划、地标推理、多模式检索和重新排序的答案合成。地标知识代理是一个关键组件，它可以检测文化或空间地标，并将其重新表述为描述性视觉提示，从而增强越南场景的基于 CLIP 的语义匹配。为了扩展功能，我们引入了 LLM 辅助的图像到图像管道，其中大型语言模型 (Gemini 2.5 Flash) 可自动检测地标、生成图像搜索查询、检索代表性图像并执行基于 CLIP 的视觉相似性匹配，从而无需手动输入图像。此外，利用 Gemini 和 LlamaIndex 的 OCR 细化模块改进了越南语文本识别。实验结果表明，LLandMark 实现了自适应、文化基础和可解释的检索性能。

</details>

---

## 5. SemanticDialect: Semantic-Aware Mixed-Format Quantization for Video Diffusion Transformers / SemanticDialect：视频扩散变压器的语义感知混合格式量化

**Date**: 2026-03-03 | **arXiv**: [2603.02883v1](http://arxiv.org/abs/2603.02883v1) | **PDF**: [Link](http://arxiv.org/pdf/2603.02883v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Diffusion Transformers (DiT) achieve strong video generation quality, but their memory and compute costs hinder edge deployment. Quantization can reduce these costs, yet existing methods often degrade video quality under high activation variation and the need to preserve semantic/temporal coherence. We propose SemanticDialect, which advances recent block-wise mixed-format quantization-selecting a per-block optimal format (a dialect) from multiple candidates (a formatbook)-by scaling the formatbook with lookup tables for quantization error and quantized values, enabling efficient per-block format selection and quantization at low online cost. We also introduce activation decomposition that reduces quantization error by re-quantizing and adding back residual errors, with attention-guided salient token selection. We further propose semantic-aware dialect assignment (SeDA) to improve quantized value consistency by sharing a sub-formatbook among semantically correlated tokens. Experiments on video DiT (VDiT) models show that SemanticDialect outperforms prior VDiT quantization methods and fine-grained block-wise format baselines, while approaching FP16 quality on Open-Sora 2.0.

扩散变压器 (DiT) 实现了强大的视频生成质量，但其内存和计算成本阻碍了边缘部署。量化可以降低这些成本，但现有方法通常会在高激活变化和需要保持语义/时间一致性的情况下降低视频质量。我们提出了 SemanticDialect，它推进了最近的逐块混合格式量化——从多个候选（格式簿）中选择每块最佳格式（一种方言）——通过使用量化误差和量化值的查找表来缩放格式簿，从而以较低的在线成本实现高效的每块格式选择和量化。我们还引入了激活分解，通过重新量化和添加残余误差来减少量化误差，并采用注意力引导的显着标记选择。我们进一步提出语义感知方言分配（SeDA），通过在语义相关的标记之间共享子格式手册来提高量化值的一致性。视频 DiT (VDiT) 模型的实验表明，SemanticDialect 的性能优于之前的 VDiT 量化方法和细粒度块格式基线，同时接近 Open-Sora 2.0 上的 FP16 质量。

</details>

---

## 6. SIGMark: Scalable In-Generation Watermark with Blind Extraction for Video Diffusion / SIGMark：可扩展的代内水印，具有视频扩散的盲提取功能

**Date**: 2026-03-03 | **arXiv**: [2603.02882v1](http://arxiv.org/abs/2603.02882v1) | **PDF**: [Link](http://arxiv.org/pdf/2603.02882v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Artificial Intelligence Generated Content (AIGC), particularly video generation with diffusion models, has been advanced rapidly. Invisible watermarking is a key technology for protecting AI-generated videos and tracing harmful content, and thus plays a crucial role in AI safety. Beyond post-processing watermarks which inevitably degrade video quality, recent studies have proposed distortion-free in-generation watermarking for video diffusion models. However, existing in-generation approaches are non-blind: they require maintaining all the message-key pairs and performing template-based matching during extraction, which incurs prohibitive computational costs at scale. Moreover, when applied to modern video diffusion models with causal 3D Variational Autoencoders (VAEs), their robustness against temporal disturbance becomes extremely weak. To overcome these challenges, we propose SIGMark, a Scalable In-Generation watermarking framework with blind extraction for video diffusion. To achieve blind-extraction, we propose to generate watermarked initial noise using a Global set of Frame-wise PseudoRandom Coding keys (GF-PRC), reducing the cost of storing large-scale information while preserving noise distribution and diversity for distortion-free watermarking. To enhance robustness, we further design a Segment Group-Ordering module (SGO) tailored to causal 3D VAEs, ensuring robust watermark inversion during extraction under temporal disturbance. Comprehensive experiments on modern diffusion models show that SIGMark achieves very high bit-accuracy during extraction under both temporal and spatial disturbances with minimal overhead, demonstrating its scalability and robustness. Our project is available at https://jeremyzhao1998.github.io/SIGMark-release/.

人工智能生成内容（AIGC），特别是具有扩散模型的视频生成，已经迅速发展。隐形水印是保护人工智能生成的视频和追踪有害内容的关键技术，在人工智能安全中发挥着至关重要的作用。除了不可避免地降低视频质量的后处理水印之外，最近的研究还提出了针对视频扩散模型的无失真代内水印。然而，现有的代内方法是非盲目的：它们需要维护所有消息密钥对并在提取过程中执行基于模板的匹配，这会导致大规模的计算成本过高。此外，当应用于具有因果 3D 变分自动编码器 (VAE) 的现代视频扩散模型时，它们对时间干扰的鲁棒性变得非常弱。为了克服这些挑战，我们提出了 SIGMark，这是一种可扩展的一代水印框架，具有视频扩散盲提取功能。为了实现盲提取，我们建议使用全局逐帧伪随机编码密钥（GF-PRC）集生成带水印的初始噪声，从而降低存储大规模信息的成本，同时保留噪声分布和无失真水印的多样性。为了增强鲁棒性，我们进一步设计了一个针对因果 3D VAE 定制的分段组排序模块（SGO），确保在时间扰动下的提取过程中实现鲁棒的水印反演。对现代扩散模型的综合实验表明，SIGMark 在时间和空间干扰下的提取过程中以最小的开销实现了非常高的位精度，证明了其可扩展性和鲁棒性。我们的项目可在 https://jeremyzhao1998.github.io/SIGMark-release/ 获取。

</details>

---

## 7. Think-as-You-See: Streaming Chain-of-Thought Reasoning for Large Vision-Language Models / 即视即想：大型视觉语言模型的流式思维链推理

**Date**: 2026-03-03 | **arXiv**: [2603.02872v1](http://arxiv.org/abs/2603.02872v1) | **PDF**: [Link](http://arxiv.org/pdf/2603.02872v1)

**Categories**: cs.CV

**Code**: https://github.com/EIT-NLP/StreamingLLM/tree/main/TaYS

<details><summary><b>Abstract / 摘要</b></summary>

Large Vision Language Models (LVLMs) exhibit strong Chain-of-Thought (CoT) capabilities, yet most existing paradigms assume full-video availability before inference, a batch-style process misaligned with real-world video streams where information arrives sequentially. Motivated by the streaming nature of video data, we investigate two streaming reasoning paradigms for LVLMs. The first, an interleaved paradigm, alternates between receiving frames and producing partial reasoning but remains constrained by strictly ordered cache updates. To better match streaming inputs, we propose \textbf{Think-as-You-See (TaYS)}, a unified framework enabling true concurrent reasoning. TaYS integrates parallelized CoT generation, stream-constrained training, and stream-parallel inference. It further employs temporally aligned reasoning units, streaming attention masks and positional encodings, and a dual KV-cache that decouples visual encoding from textual reasoning. We evaluate all paradigms on the Qwen2.5-VL family across representative video CoT tasks, including event dynamics analysis, causal reasoning, and thematic understanding. Experiments show that TaYS consistently outperforms both batch and interleaved baselines, improving reasoning performance while substantially reducing time-to-first-token (TTFT) and overall reasoning delay. These results demonstrate the effectiveness of data-aligned streaming reasoning in enabling efficient and responsive video understanding for LVLMs. We release our code at \href{https://github.com/EIT-NLP/StreamingLLM/tree/main/TaYS}{this repository.}

大视觉语言模型 (LVLM) 表现出强大的思想链 (CoT) 功能，但大多数现有范例都假设推理之前具有完整视频可用性，这是一种与信息按顺序到达的现实视频流不一致的批处理式过程。受视频数据流式传输性质的启发，我们研究了 LVLM 的两种流式推理范例。第一种是交错范例，在接收帧和产生部分推理之间交替，但仍然受到严格排序的缓存更新的限制。为了更好地匹配流输入，我们提出了 \textbf{Think-as-You-See (TaYS)}，这是一个能够实现真正并发推理的统一框架。 TaYS 集成了并行 CoT 生成、流约束训练和流并行推理。它还采用了时间对齐的推理单元、流式注意力掩码和位置编码，以及将视觉编码与文本推理分离的双 KV 缓存。我们评估了 Qwen2.5-VL 系列上代表性视频 CoT 任务的所有范式，包括事件动态分析、因果推理和主题理解。实验表明，TaYS 始终优于批处理和交错基线，提高了推理性能，同时大幅减少了首次标记时间 (TTFT) 和整体推理延迟。这些结果证明了数据对齐流推理在为 LVLM 实现高效且响应灵敏的视频理解方面的有效性。我们在 \href{https://github.com/EIT-NLP/StreamingLLM/tree/main/TaYS}{此存储库。}

</details>

---

## 8. BrandFusion: A Multi-Agent Framework for Seamless Brand Integration in Text-to-Video Generation / BrandFusion：文本到视频生成中无缝品牌集成的多代理框架

**Date**: 2026-03-03 | **arXiv**: [2603.02816v1](http://arxiv.org/abs/2603.02816v1) | **PDF**: [Link](http://arxiv.org/pdf/2603.02816v1)

**Categories**: cs.CV, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

The rapid advancement of text-to-video (T2V) models has revolutionized content creation, yet their commercial potential remains largely untapped. We introduce, for the first time, the task of seamless brand integration in T2V: automatically embedding advertiser brands into prompt-generated videos while preserving semantic fidelity to user intent. This task confronts three core challenges: maintaining prompt fidelity, ensuring brand recognizability, and achieving contextually natural integration. To address them, we propose BrandFusion, a novel multi-agent framework comprising two synergistic phases. In the offline phase (advertiser-facing), we construct a Brand Knowledge Base by probing model priors and adapting to novel brands via lightweight fine-tuning. In the online phase (user-facing), five agents jointly refine user prompts through iterative refinement, leveraging the shared knowledge base and real-time contextual tracking to ensure brand visibility and semantic alignment. Experiments on 18 established and 2 custom brands across multiple state-of-the-art T2V models demonstrate that BrandFusion significantly outperforms baselines in semantic preservation, brand recognizability, and integration naturalness. Human evaluations further confirm higher user satisfaction, establishing a practical pathway for sustainable T2V monetization.

文本转视频 (T2V) 模型的快速发展彻底改变了内容创作，但其商业潜力在很大程度上尚未开发。我们首次在 T2V 中引入无缝品牌集成的任务：自动将广告商品牌嵌入到提示生成的视频中，同时保持对用户意图的语义保真度。这项任务面临三个核心挑战：保持即时保真度、确保品牌可识别性以及实现上下文自然整合。为了解决这些问题，我们提出了 BrandFusion，这是一种新颖的多智能体框架，包含两个协同阶段。在离线阶段（面向广告商），我们通过探索模型先验并通过轻量级微调来适应新品牌来构建品牌知识库。在在线阶段（面向用户），五个代理通过迭代细化共同完善用户提示，利用共享知识库和实时上下文跟踪来确保品牌可见性和语义一致性。在多个最先进的 T2V 模型中对 18 个已建立品牌和 2 个定制品牌进行的实验表明，BrandFusion 在语义保留、品牌可识别性和集成自然性方面显着优于基线。人工评估进一步证实了更高的用户满意度，为可持续的 T2V 货币化建立了实用途径。

</details>

---

## 9. NOVA: Sparse Control, Dense Synthesis for Pair-Free Video Editing / NOVA：稀疏控制，密集合成，用于无对视频编辑

**Date**: 2026-03-03 | **arXiv**: [2603.02802v1](http://arxiv.org/abs/2603.02802v1) | **PDF**: [Link](http://arxiv.org/pdf/2603.02802v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Recent video editing models have achieved impressive results, but most still require large-scale paired datasets. Collecting such naturally aligned pairs at scale remains highly challenging and constitutes a critical bottleneck, especially for local video editing data. Existing workarounds transfer image editing to video through global motion control for pair-free video editing, but such designs struggle with background and temporal consistency. In this paper, we propose NOVA: Sparse Control \& Dense Synthesis, a new framework for unpaired video editing. Specifically, the sparse branch provides semantic guidance through user-edited keyframes distributed across the video, and the dense branch continuously incorporates motion and texture information from the original video to maintain high fidelity and coherence. Moreover, we introduce a degradation-simulation training strategy that enables the model to learn motion reconstruction and temporal consistency by training on artificially degraded videos, thus eliminating the need for paired data. Our extensive experiments demonstrate that NOVA outperforms existing approaches in edit fidelity, motion preservation, and temporal coherence.

最近的视频编辑模型取得了令人印象深刻的结果，但大多数仍然需要大规模的配对数据集。大规模收集这种自然对齐的对仍然非常具有挑战性，并且构成了一个关键的瓶颈，特别是对于本地视频编辑数据而言。现有的解决方法通过全局运动控制将图像编辑转移到视频，以实现无配对视频编辑，但此类设计在背景和时间一致性方面存在困难。在本文中，我们提出了 NOVA：稀疏控制和密集合成，这是一种用于不配对视频编辑的新框架。具体来说，稀疏分支通过分布在视频中的用户编辑的关键帧提供语义指导，而密集分支不断合并原始视频中的运动和纹理信息，以保持高保真度和连贯性。此外，我们引入了一种退化模拟训练策略，使模型能够通过对人工退化视频进行训练来学习运动重建和时间一致性，从而消除了对配对数据的需要。我们广泛的实验表明，NOVA 在编辑保真度、运动保留和时间连贯性方面优于现有方法。

</details>

---

## 10. ShareVerse: Multi-Agent Consistent Video Generation for Shared World Modeling / ShareVerse：用于共享世界建模的多代理一致视频生成

**Date**: 2026-03-03 | **arXiv**: [2603.02697v1](http://arxiv.org/abs/2603.02697v1) | **PDF**: [Link](http://arxiv.org/pdf/2603.02697v1)

**Categories**: cs.CV, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

This paper presents ShareVerse, a video generation framework enabling multi-agent shared world modeling, addressing the gap in existing works that lack support for unified shared world construction with multi-agent interaction. ShareVerse leverages the generation capability of large video models and integrates three key innovations: 1) A dataset for large-scale multi-agent interactive world modeling is built on the CARLA simulation platform, featuring diverse scenes, weather conditions, and interactive trajectories with paired multi-view videos (front/ rear/ left/ right views per agent) and camera data. 2) We propose a spatial concatenation strategy for four-view videos of independent agents to model a broader environment and to ensure internal multi-view geometric consistency. 3) We integrate cross-agent attention blocks into the pretrained video model, which enable interactive transmission of spatial-temporal information across agents, guaranteeing shared world consistency in overlapping regions and reasonable generation in non-overlapping regions. ShareVerse, which supports 49-frame large-scale video generation, accurately perceives the position of dynamic agents and achieves consistent shared world modeling.

本文提出了 ShareVerse，一种支持多智能体共享世界建模的视频生成框架，解决了现有作品缺乏对多智能体交互的统一共享世界构建支持的空白。 ShareVerse 利用大型视频模型的生成能力，并集成了三个关键创新：1）在 CARLA 仿真平台上构建了大规模多智能体交互式世界建模的数据集，具有不同的场景、天气条件和交互式轨迹，以及配对的多视图视频（每个智能体的前/后/左/右视图）和摄像头数据。 2）我们提出了一种针对独立代理的四视图视频的空间串联策略，以建模更广泛的环境并确保内部多视图几何一致性。 3）我们将跨智能体注意力块集成到预训练的视频模型中，从而实现跨智能体的时空信息交互传输，保证重叠区域中共享世界的一致性以及非重叠区域中的合理生成。 ShareVerse支持49帧大规模视频生成，准确感知动态代理的位置，实现一致的共享世界建模。

</details>

---

## 11. Compositional Visual Planning via Inference-Time Diffusion Scaling / 通过推理时间扩散缩放进行构图视觉规划

**Date**: 2026-03-03 | **arXiv**: [2603.02646v1](http://arxiv.org/abs/2603.02646v1) | **PDF**: [Link](http://arxiv.org/pdf/2603.02646v1)

**Categories**: cs.RO

<details><summary><b>Abstract / 摘要</b></summary>

Diffusion models excel at short-horizon robot planning, yet scaling them to long-horizon tasks remains challenging due to computational constraints and limited training data. Existing compositional approaches stitch together short segments by separately denoising each component and averaging overlapping regions. However, this suffers from instability as the factorization assumption breaks down in noisy data space, leading to inconsistent global plans. We propose that the key to stable compositional generation lies in enforcing boundary agreement on the estimated clean data (Tweedie estimates) rather than on noisy intermediate states. Our method formulates long-horizon planning as inference over a chain-structured factor graph of overlapping video chunks, where pretrained short-horizon video diffusion models provide local priors. At inference time, we enforce boundary agreement through a novel combination of synchronous and asynchronous message passing that operates on Tweedie estimates, producing globally consistent guidance without requiring additional training. Our training-free framework demonstrates significant improvements over existing baselines, effectively generalizing to unseen start-goal combinations that were not present in the original training data. Project website: https://comp-visual-planning.github.io/

扩散模型擅长短视野机器人规划，但由于计算限制和有限的训练数据，将其扩展到长视野任务仍然具有挑战性。现有的合成方法通过分别对每个分量进行去噪并对重叠区域进行平均来将短片段缝合在一起。然而，这会受到不稳定的影响，因为分解假设在嘈杂的数据空间中会崩溃，从而导致全局计划不一致。我们提出稳定合成生成的关键在于对估计的干净数据（Tweedie 估计）而不是噪声中间状态执行边界一致性。我们的方法将长视野规划制定为对重叠视频块的链式结构因子图的推理，其中预训练的短视野视频扩散模型提供局部先验。在推理时，我们通过同步和异步消息传递的新颖组合来强制执行边界协议，该组合根据 Tweedie 估计进行操作，从而产生全局一致的指导，而无需额外的训练。我们的免训练框架展示了对现有基线的显着改进，有效地推广到原始训练数据中不存在的未见的起始目标组合。项目网站：https://comp-visual-planning.github.io/

</details>

---

## 12. Direct Reward Fine-Tuning on Poses for Single Image to 3D Human in the Wild / 对单张图像的姿势进行直接奖励微调到野外 3D 人体

**Date**: 2026-03-03 | **arXiv**: [2603.02619v1](http://arxiv.org/abs/2603.02619v1) | **PDF**: [Link](http://arxiv.org/pdf/2603.02619v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Single-view 3D human reconstruction has achieved remarkable progress through the adoption of multi-view diffusion models, yet the recovered 3D humans often exhibit unnatural poses. This phenomenon becomes pronounced when reconstructing 3D humans with dynamic or challenging poses, which we attribute to the limited scale of available 3D human datasets with diverse poses. To address this limitation, we introduce DrPose, Direct Reward fine-tuning algorithm on Poses, which enables post-training of a multi-view diffusion model on diverse poses without requiring expensive 3D human assets. DrPose trains a model using only human poses paired with single-view images, employing a direct reward fine-tuning to maximize PoseScore, which is our proposed differentiable reward that quantifies consistency between a generated multi-view latent image and a ground-truth human pose. This optimization is conducted on DrPose15K, a novel dataset that was constructed from an existing human motion dataset and a pose-conditioned video generative model. Constructed from abundant human pose sequence data, DrPose15K exhibits a broader pose distribution compared to existing 3D human datasets. We validate our approach through evaluation on conventional benchmark datasets, in-the-wild images, and a newly constructed benchmark, with a particular focus on assessing performance on challenging human poses. Our results demonstrate consistent qualitative and quantitative improvements across all benchmarks. Project page: https://seunguk-do.github.io/drpose.

通过采用多视图扩散模型，单视图 3D 人体重建取得了显着的进展，但恢复后的 3D 人体经常表现出不自然的姿势。当重建具有动态或挑战性姿势的 3D 人体时，这种现象变得更加明显，我们将其归因于具有不同姿势的可用 3D 人体数据集的规模有限。为了解决这个限制，我们引入了 DrPose，即针对姿势的直接奖励微调算法，它可以对不同姿势的多视图扩散模型进行后期训练，而无需昂贵的 3D 人力资产。 DrPose 仅​​使用与单视图图像配对的人体姿势来训练模型，采用直接奖励微调来最大化 PoseScore，这是我们提出的可微奖励，用于量化生成的多视图潜在图像和真实人体姿势之间的一致性。此优化是在 DrPose15K 上进行的，DrPose15K 是一个新颖的数据集，由现有的人体运动数据集和姿势条件视频生成模型构建而成。 DrPose15K 由丰富的人体姿势序列数据构建而成，与现有的 3D 人体数据集相比，表现出更广泛的姿势分布。我们通过对传统基准数据集、野外图像和新构建的基准进行评估来验证我们的方法，特别关注评估具有挑战性的人体姿势的性能。我们的结果表明，所有基准都取得了一致的定性和定量改进。项目页面：https://seunguk-do.github.io/drpose。

</details>

---

## 13. Kiwi-Edit: Versatile Video Editing via Instruction and Reference Guidance / Kiwi-Edit：通过说明和参考指南进行多功能视频编辑

**Date**: 2026-03-02 | **arXiv**: [2603.02175v1](http://arxiv.org/abs/2603.02175v1) | **PDF**: [Link](http://arxiv.org/pdf/2603.02175v1)

**Categories**: cs.CV, cs.AI

**Code**: https://github.com/showlab/Kiwi-Edit.

<details><summary><b>Abstract / 摘要</b></summary>

Instruction-based video editing has witnessed rapid progress, yet current methods often struggle with precise visual control, as natural language is inherently limited in describing complex visual nuances. Although reference-guided editing offers a robust solution, its potential is currently bottlenecked by the scarcity of high-quality paired training data. To bridge this gap, we introduce a scalable data generation pipeline that transforms existing video editing pairs into high-fidelity training quadruplets, leveraging image generative models to create synthesized reference scaffolds. Using this pipeline, we construct RefVIE, a large-scale dataset tailored for instruction-reference-following tasks, and establish RefVIE-Bench for comprehensive evaluation. Furthermore, we propose a unified editing architecture, Kiwi-Edit, that synergizes learnable queries and latent visual features for reference semantic guidance. Our model achieves significant gains in instruction following and reference fidelity via a progressive multi-stage training curriculum. Extensive experiments demonstrate that our data and architecture establish a new state-of-the-art in controllable video editing. All datasets, models, and code is released at https://github.com/showlab/Kiwi-Edit.

基于指令的视频编辑取得了快速进展，但当前的方法常常难以实现精确的视觉控制，因为自然语言在描述复杂的视觉细微差别方面本质上受到限制。尽管参考引导编辑提供了一个强大的解决方案，但其潜力目前因缺乏高质量配对训练数据而受到瓶颈。为了弥补这一差距，我们引入了一个可扩展的数据生成管道，将现有的视频编辑对转换为高保真训练四元组，利用图像生成模型创建合成的参考支架。利用这个管道，我们构建了RefVIE（一个为指令参考跟踪任务量身定制的大规模数据集），并建立了RefVIE-Bench进行综合评估。此外，我们提出了一个统一的编辑架构 Kiwi-Edit，它可以协同可学习的查询和潜在的视觉特征以提供参考语义指导。我们的模型通过渐进的多阶段培训课程，在指令遵循和参考保真度方面取得了显着的进步。大量的实验表明，我们的数据和架构在可控视频编辑方面建立了新的最先进技术。所有数据集、模型和代码均在 https://github.com/showlab/Kiwi-Edit 发布。

</details>

---

## 14. LiftAvatar: Kinematic-Space Completion for Expression-Controlled 3D Gaussian Avatar Animation / LiftAvatar：表达控制的 3D 高斯头像动画的运动学空间完成

**Date**: 2026-03-02 | **arXiv**: [2603.02129v1](http://arxiv.org/abs/2603.02129v1) | **PDF**: [Link](http://arxiv.org/pdf/2603.02129v1)

**Categories**: cs.CV, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

We present LiftAvatar, a new paradigm that completes sparse monocular observations in kinematic space (e.g., facial expressions and head pose) and uses the completed signals to drive high-fidelity avatar animation. LiftAvatar is a fine-grained, expression-controllable large-scale video diffusion Transformer that synthesizes high-quality, temporally coherent expression sequences conditioned on single or multiple reference images. The key idea is to lift incomplete input data into a richer kinematic representation, thereby strengthening both reconstruction and animation in downstream 3D avatar pipelines. To this end, we introduce (i) a multi-granularity expression control scheme that combines shading maps with expression coefficients for precise and stable driving, and (ii) a multi-reference conditioning mechanism that aggregates complementary cues from multiple frames, enabling strong 3D consistency and controllability. As a plug-and-play enhancer, LiftAvatar directly addresses the limited expressiveness and reconstruction artifacts of 3D Gaussian Splatting-based avatars caused by sparse kinematic cues in everyday monocular videos. By expanding incomplete observations into diverse pose-expression variations, LiftAvatar also enables effective prior distillation from large-scale video generative models into 3D pipelines, leading to substantial gains. Extensive experiments show that LiftAvatar consistently boosts animation quality and quantitative metrics of state-of-the-art 3D avatar methods, especially under extreme, unseen expressions.

我们提出了 LiftAvatar，这是一种新的范式，可以完成运动空间中的稀疏单眼观察（例如面部表情和头部姿势），并使用完成的信号来驱动高保真头像动画。 LiftAvatar 是一种细粒度、表情可控的大规模视频扩散 Transformer，可根据单个或多个参考图像合成高质量、时间连贯的表情序列。关键思想是将不完整的输入数据提升为更丰富的运动学表示，从而加强下游 3D 头像管道中的重建和动画。为此，我们引入了（i）多粒度表达控制方案，将着色图与表达系数相结合，以实现精确稳定的驱动，以及（ii）多参考调节机制，聚合来自多个帧的互补线索，从而实现强大的3D一致性和可控性。作为即插即用的增强器，LiftAvatar 直接解决了日常单眼视频中由于稀疏运动线索而导致的基于 3D 高斯 Splatting 的化身的有限表现力和重建伪影问题。通过将不完整的观察扩展到不同的姿势表达变化，LiftAvatar 还能够将大规模视频生成模型有效地预先提炼为 3D 管道，从而带来可观的收益。大量实验表明，LiftAvatar 持续提高了最先进的 3D 化身方法的动画质量和定量指标，尤其是在极端的、看不见的表情下。

</details>

---

## 15. OmniRet: Efficient and High-Fidelity Omni Modality Retrieval / OmniRet：高效、高保真全模态检索

**Date**: 2026-03-02 | **arXiv**: [2603.02098v1](http://arxiv.org/abs/2603.02098v1) | **PDF**: [Link](http://arxiv.org/pdf/2603.02098v1)

**Categories**: cs.IR, cs.CL, cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Multimodal retrieval is the task of aggregating information from queries across heterogeneous modalities to retrieve desired targets. State-of-the-art multimodal retrieval models can understand complex queries, yet they are typically limited to two modalities: text and vision. This limitation impedes the development of universal retrieval systems capable of comprehending queries that combine more than two modalities. To advance toward this goal, we present OmniRet, the first retrieval model capable of handling complex, composed queries spanning three key modalities: text, vision, and audio. Our OmniRet model addresses two critical challenges for universal retrieval: computational efficiency and representation fidelity. First, feeding massive token sequences from modality-specific encoders to Large Language Models (LLMs) is computationally inefficient. We therefore introduce an attention-based resampling mechanism to generate compact, fixed-size representations from these sequences. Second, compressing rich omni-modal data into a single embedding vector inevitably causes information loss and discards fine-grained details. We propose Attention Sliced Wasserstein Pooling to preserve these fine-grained details, leading to improved omni-modal representations. OmniRet is trained on an aggregation of approximately 6 million query-target pairs spanning 30 datasets. We benchmark our model on 13 retrieval tasks and a MMEBv2 subset. Our model demonstrates significant improvements on composed query, audio and video retrieval tasks, while achieving on-par performance with state-of-the-art models on others. Furthermore, we curate a new Audio-Centric Multimodal Benchmark (ACM). This new benchmark introduces two critical, previously missing tasks-composed audio retrieval and audio-visual retrieval to more comprehensively evaluate a model's omni-modal embedding capacity.

多模态检索是跨异构模态的查询聚合信息以检索所需目标的任务。最先进的多模态检索模型可以理解复杂的查询，但它们通常仅限于两种模态：文本和视觉。这种限制阻碍了能够理解结合两种以上模态的查询的通用检索系统的开发。为了实现这一目标，我们推出了 OmniRet，这是第一个能够处理跨越三个关键模式的复杂组合查询的检索模型：文本、视觉和音频。我们的 OmniRet 模型解决了通用检索的两个关键挑战：计算效率和表示保真度。首先，将大量标记序列从特定模态编码器馈送到大型语言模型 (LLM) 的计算效率很低。因此，我们引入了一种基于注意力的重采样机制，从这些序列中生成紧凑的、固定大小的表示。其次，将丰富的全模态数据压缩到单个嵌入向量中不可避免地会导致信息丢失并丢弃细粒度的细节。我们提出注意力切片 Wasserstein Pooling 来保留这些细粒度的细节，从而改进全模态表示。 OmniRet 经过 30 个数据集约 600 万个查询目标对的聚合训练。我们在 13 个检索任务和 MMEBv2 子集上对模型进行基准测试。我们的模型展示了组合查询、音频和视频检索任务的显着改进，同时实现了与其他最先进模型相当的性能。此外，我们还策划了一个新的以音频为中心的多模态基准（ACM）。这个新的基准引入了两个以前缺失的关键任务——音频检索和视听检索，以更全面地评估模型的全模态嵌入能力。

</details>

---

## 16. FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding / FluxMem：用于流视频理解的自适应分层内存

**Date**: 2026-03-02 | **arXiv**: [2603.02096v1](http://arxiv.org/abs/2603.02096v1) | **PDF**: [Link](http://arxiv.org/pdf/2603.02096v1)

**Categories**: cs.CV, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

This paper presents FluxMem, a training-free framework for efficient streaming video understanding. FluxMem adaptively compresses redundant visual memory through a hierarchical, two-stage design: (1) a Temporal Adjacency Selection (TAS) module removes redundant visual tokens across adjacent frames, and (2) a Spatial Domain Consolidation (SDC) module further merges spatially repetitive regions within each frame into compact representations. To adapt effectively to dynamic scenes, we introduce a self-adaptive token compression mechanism in both TAS and SDC, which automatically determines the compression rate based on intrinsic scene statistics rather than manual tuning. Extensive experiments demonstrate that FluxMem achieves new state-of-the-art results on existing online video benchmarks, reaching 76.4 on StreamingBench and 67.2 on OVO-Bench under real-time settings, while reducing latency by 69.9% and peak GPU memory by 34.5% on OVO-Bench. Furthermore, it maintains strong offline performance, achieving 73.1 on MLVU while using 65% fewer visual tokens.

本文介绍了 FluxMem，这是一种用于高效理解流媒体视频的免训练框架。 FluxMem 通过分层、两阶段设计自适应地压缩冗余视觉记忆：(1) 时间邻接选择 (TAS) 模块删除相邻帧之间的冗余视觉标记，(2) 空间域合并 (SDC) 模块进一步将每个帧内的空间重复区域合并为紧凑表示。为了有效适应动态场景，我们在TAS和SDC中引入了自适应令牌压缩机制，该机制根据固有场景统计数据自动确定压缩率，而不是手动调整。大量实验表明，FluxMem 在现有在线视频基准测试中取得了新的最先进结果，在实时设置下在 StreamingBench 上达到 76.4，在 OVO-Bench 上达到 67.2，同时在 OVO-Bench 上将延迟降低了 69.9%，峰值 GPU 内存降低了 34.5%。此外，它保持了强大的离线性能，MLVU 达到 73.1，同时使用的视觉标记减少了 65%。

</details>

---

## 17. WorldStereo: Bridging Camera-Guided Video Generation and Scene Reconstruction via 3D Geometric Memories / WorldStereo：通过 3D 几何存储器桥接摄像机引导的视频生成和场景重建

**Date**: 2026-03-02 | **arXiv**: [2603.02049v1](http://arxiv.org/abs/2603.02049v1) | **PDF**: [Link](http://arxiv.org/pdf/2603.02049v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Recent advances in foundational Video Diffusion Models (VDMs) have yielded significant progress. Yet, despite the remarkable visual quality of generated videos, reconstructing consistent 3D scenes from these outputs remains challenging, due to limited camera controllability and inconsistent generated content when viewed from distinct camera trajectories. In this paper, we propose WorldStereo, a novel framework that bridges camera-guided video generation and 3D reconstruction via two dedicated geometric memory modules. Formally, the global-geometric memory enables precise camera control while injecting coarse structural priors through incrementally updated point clouds. Moreover, the spatial-stereo memory constrains the model's attention receptive fields with 3D correspondence to focus on fine-grained details from the memory bank. These components enable WorldStereo to generate multi-view-consistent videos under precise camera control, facilitating high-quality 3D reconstruction. Furthermore, the flexible control branch-based WorldStereo shows impressive efficiency, benefiting from the distribution matching distilled VDM backbone without joint training. Extensive experiments across both camera-guided video generation and 3D reconstruction benchmarks demonstrate the effectiveness of our approach. Notably, we show that WorldStereo acts as a powerful world model, tackling diverse scene generation tasks (whether starting from perspective or panoramic images) with high-fidelity 3D results. Models will be released.

基础视频扩散模型 (VDM) 的最新进展取得了重大进展。然而，尽管生成的视频具有出色的视觉质量，但由于摄像机的可控性有限，并且从不同的摄像机轨迹观看时生成的内容不一致，因此从这些输出中重建一致的 3D 场景仍然具有挑战性。在本文中，我们提出了 WorldStereo，这是一种新颖的框架，通过两个专用的几何存储模块将相机引导的视频生成和 3D 重建联系起来。从形式上来说，全局几何存储器可以实现精确的相机控制，同时通过增量更新的点云注入粗略的结构先验。此外，空间立体记忆通过 3D 对应限制模型的注意力接受域，以关注记忆库中的细粒度细节。这些组件使 WorldStereo 能够在精确的摄像机控制下生成多视图一致的视频，从而促进高质量的 3D 重建。此外，基于分支的灵活控制 WorldStereo 显示了令人印象深刻的效率，受益于无需联合训练的分布匹配蒸馏 VDM 主干。相机引导视频生成和 3D 重建基准的大量实验证明了我们方法的有效性。值得注意的是，我们展示了 WorldStereo 作为一个强大的世界模型，以高保真 3D 结果处理不同的场景生成任务（无论是从透视图像还是全景图像开始）。模型将被发布。

</details>

---

## 18. Non-verbal Real-time Human-AI Interaction in Constrained Robotic Environments / 受限机器人环境中的非语言实时人机交互

**Date**: 2026-03-02 | **arXiv**: [2603.01804v1](http://arxiv.org/abs/2603.01804v1) | **PDF**: [Link](http://arxiv.org/pdf/2603.01804v1)

**Categories**: cs.CV, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

We study the ongoing debate regarding the statistical fidelity of AI-generated data compared to human-generated data in the context of non-verbal communication using full body motion. Concretely, we ask if contemporary generative models move beyond surface mimicry to participate in the silent, but expressive dialogue of body language. We tackle this question by introducing the first framework that generates a natural non-verbal interaction between Human and AI in real-time from 2D body keypoints. Our experiments utilize four lightweight architectures which run at up to 100 FPS on an NVIDIA Orin Nano, effectively closing the perception-action loop needed for natural Human-AI interaction. We trained on 437 human video clips and demonstrated that pretraining on synthetically-generated sequences reduces motion errors significantly, without sacrificing speed. Yet, a measurable reality gap persists. When the best model is evaluated on keypoints extracted from cutting-edge text-to-video systems, such as SORA and VEO, we observe that performance drops on SORA-generated clips. However, it degrades far less on VEO, suggesting that temporal coherence, not image fidelity, drives real-world performance. Our results demonstrate that statistically distinguishable differences persist between Human and AI motion.

我们研究了关于在使用全身运动的非语言交流背景下人工智能生成的数据与人类生成的数据相比的统计保真度的持续争论。具体来说，我们询问当代生成模型是否超越表面模仿，参与无声但富有表现力的肢体语言对话。我们通过引入第一个框架来解决这个问题，该框架从 2D 身体关键点实时生成人类和人工智能之间的自然非语言交互。我们的实验利用四种轻量级架构，在 NVIDIA Orin Nano 上以高达 100 FPS 的速度运行，有效地闭合了自然人机交互所需的感知-动作循环。我们对 437 个人类视频剪辑进行了训练，并证明对合成生成的序列进行预训练可以显着减少运动错误，而不会牺牲速度。然而，可衡量的现实差距仍然存在。当根据从尖端文本到视频系统（例如 SORA 和 VEO）提取的关键点评估最佳模型时，我们观察到 SORA 生成的剪辑的性能下降。然而，它在 VEO 上的降级要少得多，这表明是时间连贯性而不是图像保真度驱动了现实世界的性能。我们的结果表明，人类和人工智能运动之间仍然存在统计上可区分的差异。

</details>

---

## 19. StepVAR: Structure-Texture Guided Pruning for Visual Autoregressive Models / StepVAR：视觉自回归模型的结构-纹理引导修剪

**Date**: 2026-03-02 | **arXiv**: [2603.01757v1](http://arxiv.org/abs/2603.01757v1) | **PDF**: [Link](http://arxiv.org/pdf/2603.01757v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Visual AutoRegressive (VAR) models based on next-scale prediction enable efficient hierarchical generation, yet the inference cost grows quadratically at high resolutions. We observe that the computationally intensive later scales predominantly refine high-frequency textures and exhibit substantial spatial redundancy, in contrast to earlier scales that determine the global structural layout. Existing pruning methods primarily focus on high-frequency detection for token selection, often overlooking structural coherence and consequently degrading global semantics. To address this limitation, we propose StepVAR, a training-free token pruning framework that accelerates VAR inference by jointly considering structural and textural importance. Specifically, we employ a lightweight high-pass filter to capture local texture details, while leveraging Principal Component Analysis (PCA) to preserve global structural information. This dual-criterion design enables the model to retain tokens critical for both fine-grained fidelity and overall composition. To maintain valid next-scale prediction under sparse tokens, we further introduce a nearest neighbor feature propagation strategy to reconstruct dense feature maps from pruned representations. Extensive experiments on state-of-the-art text-to-image and text-to-video VAR models demonstrate that StepVAR achieves substantial inference speedups while maintaining generation quality. Quantitative and qualitative evaluations consistently show that our method outperforms existing acceleration approaches, validating its effectiveness and general applicability across diverse VAR architectures.

基于下一尺度预测的视觉自回归（VAR）模型可以实现高效的分层生成，但推理成本在高分辨率下呈二次方增长。我们观察到，与决定全局结构布局的早期尺度相比，计算密集型的后期尺度主要细化高频纹理并表现出大量的空间冗余。现有的修剪方法主要集中于标记选择的高频检测，常常忽视结构一致性，从而降低全局语义。为了解决这个限制，我们提出了 StepVAR，一种免训练的 token 修剪框架，通过共同考虑结构和纹理的重要性来加速 VAR 推理。具体来说，我们采用轻量级高通滤波器来捕获局部纹理细节，同时利用主成分分析（PCA）来保留全局结构信息。这种双标准设计使模型能够保留对细粒度保真度和整体构成至关重要的标记。为了在稀疏标记下保持有效的下一尺度预测，我们进一步引入了最近邻特征传播策略，以从修剪后的表示重建密集特征图。对最先进的文本到图像和文本到视频 VAR 模型的大量实验表明，StepVAR 在保持生成质量的同时实现了显着的推理加速。定量和定性评估一致表明，我们的方法优于现有的加速方法，验证了其在不同 VAR 架构中的有效性和普遍适用性。

</details>

---

## 20. FastLightGen: Fast and Light Video Generation with Fewer Steps and Parameters / FastLightGen：用更少的步骤和参数快速生成视频

**Date**: 2026-03-02 | **arXiv**: [2603.01685v1](http://arxiv.org/abs/2603.01685v1) | **PDF**: [Link](http://arxiv.org/pdf/2603.01685v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

The recent advent of powerful video generation models, such as Hunyuan, WanX, Veo3, and Kling, has inaugurated a new era in the field. However, the practical deployment of these models is severely impeded by their substantial computational overhead, which stems from enormous parameter counts and the iterative, multi-step sampling process required during inference. Prior research on accelerating generative models has predominantly followed two distinct trajectories: reducing the number of sampling steps (e.g., LCM, DMD, and MagicDistillation) or compressing the model size for more efficient inference (e.g., ICMD). The potential of simultaneously compressing both to create a fast and lightweight model remains an unexplored avenue. In this paper, we propose FastLightGen, an algorithm that transforms large, computationally expensive models into fast, lightweight counterparts. The core idea is to construct an optimal teacher model, one engineered to maximize student performance, within a synergistic framework for distilling both model size and inference steps. Our extensive experiments on HunyuanVideo-ATI2V and WanX-TI2V reveal that a generator using 4-step sampling and 30\% parameter pruning achieves optimal visual quality under a constrained inference budget. Furthermore, FastLightGen consistently outperforms all competing methods, establishing a new state-of-the-art in efficient video generation.

最近出现的强大视频生成模型，如Hunyuan、WanX、Veo3和Kling，开创了该领域的新时代。然而，这些模型的实际部署受到巨大的计算开销的严重阻碍，这些开销源于大量的参数计数和推理过程中所需的迭代、多步采样过程。先前关于加速生成模型的研究主要遵循两个不同的轨迹：减少采样步骤的数量（例如 LCM、DMD 和 MagicDistillation）或压缩模型大小以提高推理效率（例如 ICMD）。同时压缩两者以创建快速且轻量级模型的潜力仍然是一个尚未探索的途径。在本文中，我们提出了 FastLightGen，这是一种将大型、计算成本昂贵的模型转换为快速、轻量级模型的算法。其核心思想是构建一个最佳的教师模型，该模型旨在在一个用于提炼模型大小和推理步骤的协同框架内最大限度地提高学生的表现。我们在HunyuanVideo-ATI2V和WanX-TI2V上进行的大量实验表明，使用4步采样和30％参数修剪的生成器可以在受限的推理预算下实现最佳的视觉质量。此外，FastLightGen 始终优于所有竞争方法，在高效视频生成方面建立了新的最先进技术。

</details>

---

## 21. Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration / 扩散采样加速的自适应光谱特征预测

**Date**: 2026-03-02 | **arXiv**: [2603.01623v1](http://arxiv.org/abs/2603.01623v1) | **PDF**: [Link](http://arxiv.org/pdf/2603.01623v1)

**Categories**: cs.CV, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Diffusion models have become the dominant tool for high-fidelity image and video generation, yet are critically bottlenecked by their inference speed due to the numerous iterative passes of Diffusion Transformers. To reduce the exhaustive compute, recent works resort to the feature caching and reusing scheme that skips network evaluations at selected diffusion steps by using cached features in previous steps. However, their preliminary design solely relies on local approximation, causing errors to grow rapidly with large skips and leading to degraded sample quality at high speedups. In this work, we propose spectral diffusion feature forecaster (Spectrum), a training-free approach that enables global, long-range feature reuse with tightly controlled error. In particular, we view the latent features of the denoiser as functions over time and approximate them with Chebyshev polynomials. Specifically, we fit the coefficient for each basis via ridge regression, which is then leveraged to forecast features at multiple future diffusion steps. We theoretically reveal that our approach admits more favorable long-horizon behavior and yields an error bound that does not compound with the step size. Extensive experiments on various state-of-the-art image and video diffusion models consistently verify the superiority of our approach. Notably, we achieve up to 4.79$\times$ speedup on FLUX.1 and 4.67$\times$ speedup on Wan2.1-14B, while maintaining much higher sample quality compared with the baselines.

扩散模型已成为高保真图像和视频生成的主要工具，但由于扩散变压器的大量迭代，其推理速度受到严重瓶颈。为了减少详尽的计算，最近的工作采用了特征缓存和重用方案，该方案通过使用先前步骤中的缓存特征来跳过选定扩散步骤的网络评估。然而，他们的初步设计仅依赖于局部近似，导致误差随着较大的跳跃而快速增长，并导致高速加速时样本质量下降。在这项工作中，我们提出了谱扩散特征预测器（Spectrum），这是一种免训练的方法，可以在严格控制误差的情况下实现全局、远程特征重用。特别是，我们将降噪器的潜在特征视为随时间变化的函数，并用切比雪夫多项式对其进行近似。具体来说，我们通过岭回归拟合每个基的系数，然后利用该系数来预测未来多个扩散步骤的特征。我们从理论上揭示，我们的方法允许更有利的长视野行为，并产生不与步长复合的误差界限。对各种最先进的图像和视频扩散模型的广泛实验一致验证了我们方法的优越性。值得注意的是，我们在 FLUX.1 上实现了高达 4.79$\times$ 的加速，在 Wan2.1-14B 上实现了 4.67$\times$ 的加速，同时与基线相比保持了更高的样本质量。

</details>

---

## 22. From Verbatim to Gist: Distilling Pyramidal Multimodal Memory via Semantic Information Bottleneck for Long-Horizon Video Agents / 从逐字到要点：通过长视距视频代理的语义信息瓶颈提取金字塔多模态记忆

**Date**: 2026-03-02 | **arXiv**: [2603.01455v1](http://arxiv.org/abs/2603.01455v1) | **PDF**: [Link](http://arxiv.org/pdf/2603.01455v1)

**Categories**: cs.CV, cs.AI, cs.CL, cs.IR, cs.MM

**Code**: https://github.com/EliSpectre/MM-Mem.

<details><summary><b>Abstract / 摘要</b></summary>

While multimodal large language models have demonstrated impressive short-term reasoning, they struggle with long-horizon video understanding due to limited context windows and static memory mechanisms that fail to mirror human cognitive efficiency. Existing paradigms typically fall into two extremes: vision-centric methods that incur high latency and redundancy through dense visual accumulation, or text-centric approaches that suffer from detail loss and hallucination via aggressive captioning. To bridge this gap, we propose MM-Mem, a pyramidal multimodal memory architecture grounded in Fuzzy-Trace Theory. MM-Mem structures memory hierarchically into a Sensory Buffer, Episodic Stream, and Symbolic Schema, enabling the progressive distillation of fine-grained perceptual traces (verbatim) into high-level semantic schemas (gist). Furthermore, to govern the dynamic construction of memory, we derive a Semantic Information Bottleneck objective and introduce SIB-GRPO to optimize the trade-off between memory compression and task-relevant information retention. In inference, we design an entropy-driven top-down memory retrieval strategy, which first tries with the abstract Symbolic Schema and progressively "drills down" to the Sensory Buffer and Episodic Stream under high uncertainty. Extensive experiments across 4 benchmarks confirm the effectiveness of MM-Mem on both offline and streaming tasks, demonstrating robust generalization and validating the effectiveness of cognition-inspired memory organization. Code is available at https://github.com/EliSpectre/MM-Mem.

虽然多模态大语言模型已经表现出令人印象深刻的短期推理，但由于有限的上下文窗口和静态记忆机制无法反映人类的认知效率，它们在长期视频理解方面遇到了困难。现有的范例通常陷入两个极端：以视觉为中心的方法，通过密集的视觉积累而导致高延迟和冗余，或者以文本为中心的方法，通过激进的字幕而遭受细节丢失和幻觉。为了弥补这一差距，我们提出了 MM-Mem，一种基于模糊跟踪理论的金字塔多模态内存架构。 MM-Mem 将内存分层构建为感知缓冲区、情景流和符号模式，从而能够将细粒度的感知痕迹（逐字记录）逐步提炼为高级语义模式（要点）。此外，为了管理内存的动态构建，我们推导了语义信息瓶颈目标，并引入 SIB-GRPO 来优化内存压缩和任务相关信息保留之间的权衡。在推理中，我们设计了一种熵驱动的自上而下的记忆检索策略，该策略首先尝试抽象的符号模式，并在高度不确定性下逐步“深入”到感觉缓冲区和情节流。跨 4 个基准的广泛实验证实了 MM-Mem 在离线和流任务上的有效性，展示了强大的泛化能力并验证了认知启发的记忆组织的有效性。代码可在 https://github.com/EliSpectre/MM-Mem 获取。

</details>

---

## 23. UniTalking: A Unified Audio-Video Framework for Talking Portrait Generation / UniTalking：用于生成说话肖像的统一音频-视频框架

**Date**: 2026-03-02 | **arXiv**: [2603.01418v1](http://arxiv.org/abs/2603.01418v1) | **PDF**: [Link](http://arxiv.org/pdf/2603.01418v1)

**Categories**: cs.CV, cs.MM, cs.SD

<details><summary><b>Abstract / 摘要</b></summary>

While state-of-the-art audio-video generation models like Veo3 and Sora2 demonstrate remarkable capabilities, their closed-source nature makes their architectures and training paradigms inaccessible. To bridge this gap in accessibility and performance, we introduce UniTalking, a unified, end-to-end diffusion framework for generating high-fidelity speech and lip-synchronized video. At its core, our framework employs Multi-Modal Transformer Blocks to explicitly model the fine-grained temporal correspondence between audio and video latent tokens via a shared self-attention mechanism. By leveraging powerful priors from a pre-trained video generation model, our framework ensures state-of-the-art visual fidelity while enabling efficient training. Furthermore, UniTalking incorporates a personalized voice cloning capability, allowing the generation of speech in a target style from a brief audio reference. Qualitative and quantitative results demonstrate that our method produces highly realistic talking portraits, achieving superior performance over existing open-source approaches in lip-sync accuracy, audio naturalness, and overall perceptual quality.

虽然 Veo3 和 Sora2 等最先进的音视频生成模型展示了卓越的功能，但它们的闭源性质使其架构和训练范例难以访问。为了弥补可访问性和性能方面的差距，我们引入了 UniTalking，这是一个统一的端到端扩散框架，用于生成高保真语音和口型同步视频。其核心是，我们的框架采用多模态转换器块，通过共享的自注意力机制对音频和视频潜在标记之间的细粒度时间对应关系进行显式建模。通过利用预先训练的视频生成模型的强大先验，我们的框架确保了最先进的视觉保真度，同时实现高效的训练。此外，UniTalking 还集成了个性化语音克隆功能，允许根据简短的音频参考生成目标风格的语音。定性和定量结果表明，我们的方法可以生成高度逼真的谈话肖像，在口型同步准确性、音频自然度和整体感知质量方面比现有开源方法具有更优越的性能。

</details>

---



</details>

<details><summary><b>2026-03-03 (2 papers)</b></summary>

# arXiv Video Papers - 2026-03-03

**Paper Count**: 2

---

## 1. MMTA: Multi Membership Temporal Attention for Fine-Grained Stroke Rehabilitation Assessment / MMTA：用于细粒度中风康复评估的多成员时间注意力

**Date**: 2026-03-01 | **arXiv**: [2603.00878v1](http://arxiv.org/abs/2603.00878v1) | **PDF**: [Link](http://arxiv.org/pdf/2603.00878v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

To empower the iterative assessments involved during a person's rehabilitation, automated assessment of a person's abilities during daily activities requires temporally precise segmentation of fine-grained actions in therapy videos. Existing temporal action segmentation (TAS) models struggle to capture sub-second micro-movements while retaining exercise context, blurring rapid phase transitions and limiting reliable downstream assessment of motor recovery. We introduce Multi-Membership Temporal Attention (MMTA), a high-resolution temporal transformer for fine-grained rehabilitation assessment. Unlike standard temporal attention, which assigns each frame a single attention context per layer, MMTA lets each frame attend to multiple locally normalized temporal attention windows within the same layer. We fuse these concurrent temporal views via feature-space overlap resolution, preserving competing local contexts near transitions while enabling longer-range reasoning through layer-wise propagation. This increases boundary sensitivity without additional depth or multi-stage refinement. MMTA supports both video and wearable IMU inputs within a unified single-stage architecture, making it applicable to both clinical and home settings. MMTA consistently improves over the Global Attention transformer, boosting Edit Score by +1.3 (Video) and +1.6 (IMU) on StrokeRehab while further improving 50Salads by +3.3. Ablations confirm that performance gains stem from multi-membership temporal views rather than architectural complexity, offering a practical solution for resource-constrained rehabilitation assessment.

为了支持在人的康复过程中进行迭代评估，在日常活动中对人的能力进行自动评估需要对治疗视频中的细粒度动作进行时间上的精确分割。现有的时间动作分割（TAS）模型难以捕捉亚秒级微运动，同时保留运动背景，模糊快速相变并限制运动恢复的可靠下游评估。我们引入了多成员时间注意力（MMTA），这是一种用于细粒度康复评估的高分辨率时间转换器。与标准时间注意力（为每层每帧分配一个注意力上下文）不同，MMTA 让每个帧关注同一层内的多个局部归一化时间注意力窗口。我们通过特征空间重叠分辨率融合这些并发时间视图，保留过渡附近的竞争性局部上下文，同时通过逐层传播实现更远距离的推理。这增加了边界灵敏度，而无需额外的深度或多级细化。 MMTA 在统一的单级架构中支持视频和可穿戴 IMU 输入，使其适用于临床和家庭设置。 MMTA 持续改进全局注意力转换器，将 StrokeRehab 上的编辑得分提高了 +1.3（视频）和 +1.6（IMU），同时将 50Salads 进一步提高了 +3.3。消融证实性能提升源于多成员时间视图而不是架构复杂性，为资源受限的康复评估提供了实用的解决方案。

</details>

---

## 2. COMBAT: Conditional World Models for Behavioral Agent Training / COMBAT：行为代理训练的条件世界模型

**Date**: 2026-02-28 | **arXiv**: [2603.00825v1](http://arxiv.org/abs/2603.00825v1) | **PDF**: [Link](http://arxiv.org/pdf/2603.00825v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Recent advances in video generation have spurred the development of world models capable of simulating 3D-consistent environments and interactions with static objects. However, a significant limitation remains in their ability to model dynamic, reactive agents that can intelligently influence and interact with the world. To address this gap, we introduce COMBAT, a real-time, action-controlled world model trained on the complex 1v1 fighting game Tekken 3. Our work demonstrates that diffusion models can successfully simulate a dynamic opponent that reacts to player actions, learning its behavior implicitly.   Our approach utilizes a 1.2 billion parameter Diffusion Transformer, conditioned on latent representations from a deep compression autoencoder. We employ state-of-the-art techniques, including causal distillation and diffusion forcing, to achieve real-time inference. Crucially, we observe the emergence of sophisticated agent behavior by training the model solely on single-player inputs, without any explicit supervision for the opponent's policy. Unlike traditional imitation learning methods, which require complete action labels, COMBAT learns effectively from partially observed data to generate responsive behaviors for a controllable Player 1. We present an extensive study and introduce novel evaluation methods to benchmark this emergent agent behavior, establishing a strong foundation for training interactive agents within diffusion-based world models.

视频生成领域的最新进展促进了能够模拟 3D 一致环境以及与静态对象交互的世界模型的发展。然而，它们对能够智能地影响世界并与世界互动的动态、反应性代理进行建模的能力仍然存在很大的限制。为了解决这一差距，我们引入了 COMBAT，这是一种在复杂的 1v1 格斗游戏《铁拳 3》上训练的实时、动作控制的世界模型。我们的工作表明，扩散模型可以成功模拟对玩家动作做出反应的动态对手，隐式学习其行为。   我们的方法利用 12 亿参数的扩散变换器，以深度压缩自动编码器的潜在表示为条件。我们采用最先进的技术，包括因果蒸馏和扩散强迫，来实现实时推理。至关重要的是，我们通过仅根据单人输入训练模型来观察复杂代理行为的出现，而无需对对手的策略进行任何明确的监督。与需要完整动作标签的传统模仿学习方法不同，COMBAT 可以从部分观察到的数据中有效学习，为可控玩家 1 生成响应行为。我们提出了一项广泛的研究，并引入了新颖的评估方法来对这种新兴代理行为进行基准测试，为在基于扩散的世界模型中训练交互式代理奠定了坚实的基础。

</details>

---



</details>

<details><summary><b>2026-03-02 (6 papers)</b></summary>

# arXiv Video Papers - 2026-03-02

**Paper Count**: 6

---

## 1. SenCache: Accelerating Diffusion Model Inference via Sensitivity-Aware Caching / SenCache：通过敏感度感知缓存加速扩散模型推理

**Date**: 2026-02-27 | **arXiv**: [2602.24208v1](http://arxiv.org/abs/2602.24208v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.24208v1)

**Categories**: cs.CV, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Diffusion models achieve state-of-the-art video generation quality, but their inference remains expensive due to the large number of sequential denoising steps. This has motivated a growing line of research on accelerating diffusion inference. Among training-free acceleration methods, caching reduces computation by reusing previously computed model outputs across timesteps. Existing caching methods rely on heuristic criteria to choose cache/reuse timesteps and require extensive tuning. We address this limitation with a principled sensitivity-aware caching framework. Specifically, we formalize the caching error through an analysis of the model output sensitivity to perturbations in the denoising inputs, i.e., the noisy latent and the timestep, and show that this sensitivity is a key predictor of caching error. Based on this analysis, we propose Sensitivity-Aware Caching (SenCache), a dynamic caching policy that adaptively selects caching timesteps on a per-sample basis. Our framework provides a theoretical basis for adaptive caching, explains why prior empirical heuristics can be partially effective, and extends them to a dynamic, sample-specific approach. Experiments on Wan 2.1, CogVideoX, and LTX-Video show that SenCache achieves better visual quality than existing caching methods under similar computational budgets.

扩散模型实现了最先进的视频生成质量，但由于大量的连续去噪步骤，其推理仍然昂贵。这推动了越来越多关于加速扩散推理的研究。在免训练加速方法中，缓存通过跨时间步重用先前计算的模型输出来减少计算量。现有的缓存方法依赖于启发式标准来选择缓存/重用时间步长，并且需要大量的调整。我们通过原则性的敏感度感知缓存框架来解决这一限制。具体来说，我们通过分析模型输出对去噪输入中的扰动（即噪声潜伏和时间步长）的敏感性来形式化缓存误差，并表明这种敏感性是缓存误差的关键预测因子。基于此分析，我们提出了灵敏度感知缓存（SenCache），这是一种动态缓存策略，可以根据每个样本自适应地选择缓存时间步长。我们的框架为自适应缓存提供了理论基础，解释了为什么先前的经验启发法可以部分有效，并将其扩展到动态的、特定于样本的方法。在 Wan 2.1、CogVideoX 和 LTX-Video 上的实验表明，在类似的计算预算下，SenCache 比现有的缓存方法实现了更好的视觉质量。

</details>

---

## 2. GeoDiff4D: Geometry-Aware Diffusion for 4D Head Avatar Reconstruction / GeoDiff4D：用于 4D 头部头像重建的几何感知扩散

**Date**: 2026-02-27 | **arXiv**: [2602.24161v1](http://arxiv.org/abs/2602.24161v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.24161v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Reconstructing photorealistic and animatable 4D head avatars from a single portrait image remains a fundamental challenge in computer vision. While diffusion models have enabled remarkable progress in image and video generation for avatar reconstruction, existing methods primarily rely on 2D priors and struggle to achieve consistent 3D geometry. We propose a novel framework that leverages geometry-aware diffusion to learn strong geometry priors for high-fidelity head avatar reconstruction. Our approach jointly synthesizes portrait images and corresponding surface normals, while a pose-free expression encoder captures implicit expression representations. Both synthesized images and expression latents are incorporated into 3D Gaussian-based avatars, enabling photorealistic rendering with accurate geometry. Extensive experiments demonstrate that our method substantially outperforms state-of-the-art approaches in visual quality, expression fidelity, and cross-identity generalization, while supporting real-time rendering.

从单个肖像图像重建逼真且可动画的 4D 头部头像仍然是计算机视觉领域的一项基本挑战。虽然扩散模型在头像重建的图像和视频生成方面取得了显着进展，但现有方法主要依赖于 2D 先验，很难实现一致的 3D 几何形状。我们提出了一种新颖的框架，利用几何感知扩散来学习强大的几何先验，以进行高保真头部头像重建。我们的方法联合合成肖像图像和相应的表面法线，而无姿势表达编码器捕获隐式表达表示。合成图像和潜在表达都被纳入基于高斯的 3D 化身中，从而实现具有精确几何形状的逼真渲染。大量的实验表明，我们的方法在视觉质量、表达保真度和跨身份泛化方面远远优于最先进的方法，同时支持实时渲染。

</details>

---

## 3. HumanOrbit: 3D Human Reconstruction as 360° Orbit Generation / HumanOrbit：3D 人体重建作为 360° 轨道生成

**Date**: 2026-02-27 | **arXiv**: [2602.24148v1](http://arxiv.org/abs/2602.24148v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.24148v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

We present a method for generating a full 360° orbit video around a person from a single input image. Existing methods typically adapt image-based diffusion models for multi-view synthesis, but yield inconsistent results across views and with the original identity. In contrast, recent video diffusion models have demonstrated their ability in generating photorealistic results that align well with the given prompts. Inspired by these results, we propose HumanOrbit, a video diffusion model for multi-view human image generation. Our approach enables the model to synthesize continuous camera rotations around the subject, producing geometrically consistent novel views while preserving the appearance and identity of the person. Using the generated multi-view frames, we further propose a reconstruction pipeline that recovers a textured mesh of the subject. Experimental results validate the effectiveness of HumanOrbit for multi-view image generation and that the reconstructed 3D models exhibit superior completeness and fidelity compared to those from state-of-the-art baselines.

我们提出了一种从单个输入图像生成围绕人的完整 360° 轨道视频的方法。现有方法通常采用基于图像的扩散模型来进行多视图合成，但会在视图之间和原始身份上产生不一致的结果。相比之下，最近的视频扩散模型已经证明了它们生成与给定提示非常一致的逼真结果的能力。受这些结果的启发，我们提出了 HumanOrbit，一种用于多视图人体图像生成的视频扩散模型。我们的方法使模型能够合成围绕主题的连续相机旋转，产生几何一致的新颖视图，同时保留人的外观和身份。使用生成的多视图帧，我们进一步提出了一种重建管道，可以恢复主体的纹理网格。实验结果验证了 HumanOrbit 在多视图图像生成方面的有效性，并且与最先进的基线相比，重建的 3D 模型表现出卓越的完整性和保真度。

</details>

---

## 4. Multimodal Optimal Transport for Unsupervised Temporal Segmentation in Surgical Robotics / 手术机器人中无监督时间分割的多模态最优传输

**Date**: 2026-02-27 | **arXiv**: [2602.24138v1](http://arxiv.org/abs/2602.24138v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.24138v1)

**Categories**: cs.CV, cs.AI

**Code**: https://github.com/omar8ahmed9/TASOT.

<details><summary><b>Abstract / 摘要</b></summary>

Recognizing surgical phases and steps from video is a fundamental problem in computer-assisted interventions. Recent approaches increasingly rely on large-scale pre-training on thousands of labeled surgical videos, followed by zero-shot transfer to specific procedures. While effective, this strategy incurs substantial computational and data collection costs. In this work, we question whether such heavy pre-training is truly necessary. We propose Text-Augmented Action Segmentation Optimal Transport (TASOT), an unsupervised method for surgical phase and step recognition that extends Action Segmentation Optimal Transport (ASOT) by incorporating textual information generated directly from the videos. TASOT formulates temporal action segmentation as a multimodal optimal transport problem, where the matching cost is defined as a weighted combination of visual and text-based costs. The visual term captures frame-level appearance similarity, while the text term provides complementary semantic cues, and both are jointly regularized through a temporally consistent unbalanced Gromov-Wasserstein formulation. This design enables effective alignment between video frames and surgical actions without surgical-specific pretraining or external web-scale supervision. We evaluate TASOT on multiple benchmark surgical datasets and observe consistent and substantial improvements over existing zero-shot methods, including StrasBypass70 (+23.7), BernBypass70 (+4.5), Cholec80 (+16.5), and AutoLaparo (+19.6). These results demonstrate that fine-grained surgical understanding can be achieved by exploiting information already present in standard visual and textual representations, without resorting to increasingly complex pre-training pipelines. The code will be available at https://github.com/omar8ahmed9/TASOT.

从视频中识别手术阶段和步骤是计算机辅助干预中的一个基本问题。最近的方法越来越依赖于对数千个标记的手术视频进行大规模预训练，然后零镜头转移到特定程序。虽然有效，但该策略会产生大量的计算和数据收集成本。在这项工作中，我们质疑如此繁重的预训练是否真的有必要。我们提出了文本增强动作分割最佳传输（TASOT），这是一种用于手术阶段和步骤识别的无监督方法，通过合并直接从视频生成的文本信息来扩展动作分割最佳传输（ASOT）。 TASOT 将时间动作分割表述为多模式最优运输问题，其中匹配成本定义为视觉和基于文本的成本的加权组合。视觉术语捕获帧级外观相似性，而文本术语提供互补的语义线索，并且两者都通过时间一致的不平衡 Gromov-Wasserstein 公式联合正则化。这种设计可以实现视频帧和手术动作之间的有效对齐，而无需手术特定的预训练或外部网络规模的监督。我们在多个基准手术数据集上评估 TASOT，并观察到相对于现有零样本方法的一致和实质性改进，包括 StrasBypass70 (+23.7)、BernBypass70 (+4.5)、Cholec80 (+16.5) 和 AutoLaparo (+19.6)。这些结果表明，可以通过利用标准视觉和文本表示中已有的信息来实现细粒度的手术理解，而无需诉诸日益复杂的预训练流程。该代码可在 https://github.com/omar8ahmed9/TASOT 上获取。

</details>

---

## 5. MSVBench: Towards Human-Level Evaluation of Multi-Shot Video Generation / MSVBench：迈向多镜头视频生成的人类水平评估

**Date**: 2026-02-27 | **arXiv**: [2602.23969v1](http://arxiv.org/abs/2602.23969v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.23969v1)

**Categories**: cs.MM, cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

The evolution of video generation toward complex, multi-shot narratives has exposed a critical deficit in current evaluation methods. Existing benchmarks remain anchored to single-shot paradigms, lacking the comprehensive story assets and cross-shot metrics required to assess long-form coherence and appeal. To bridge this gap, we introduce MSVBench, the first comprehensive benchmark featuring hierarchical scripts and reference images tailored for Multi-Shot Video generation. We propose a hybrid evaluation framework that synergizes the high-level semantic reasoning of Large Multimodal Models (LMMs) with the fine-grained perceptual rigor of domain-specific expert models. Evaluating 20 video generation methods across diverse paradigms, we find that current models--despite strong visual fidelity--primarily behave as visual interpolators rather than true world models. We further validate the reliability of our benchmark by demonstrating a state-of-the-art Spearman's rank correlation of 94.4% with human judgments. Finally, MSVBench extends beyond evaluation by providing a scalable supervisory signal. Fine-tuning a lightweight model on its pipeline-refined reasoning traces yields human-aligned performance comparable to commercial models like Gemini-2.5-Flash.

视频生成向复杂、多镜头叙事的演变暴露了当前评估方法的严重缺陷。现有的基准仍然以单镜头范式为基础，缺乏评估长篇连贯性和吸引力所需的全面故事资产和交叉镜头指标。为了弥补这一差距，我们推出了 MSVBench，这是第一个综合基准测试，具有为多镜头视频生成量身定制的分层脚本和参考图像。我们提出了一种混合评估框架，它将大型多模态模型（LMM）的高级语义推理与特定领域专家模型的细粒度感知严谨性相结合。通过评估不同范式的 20 种视频生成方法，我们发现当前模型尽管具有很强的视觉保真度，但主要表现为视觉插值器而不是真实世界模型。我们通过展示最先进的 Spearman 排名与人类判断的 94.4% 相关性，进一步验证了基准的可靠性。最后，MSVBench 通过提供可扩展的监控信号来超越评估。在其管道细化的推理轨迹上微调轻量级模型，可产生与 Gemini-2.5-Flash 等商业模型相当的人性化性能。

</details>

---

## 6. SwitchCraft: Training-Free Multi-Event Video Generation with Attention Controls / SwitchCraft：具有注意力控制的免训练多事件视频生成

**Date**: 2026-02-27 | **arXiv**: [2602.23956v1](http://arxiv.org/abs/2602.23956v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.23956v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Recent advances in text-to-video diffusion models have enabled high-fidelity and temporally coherent videos synthesis. However, current models are predominantly optimized for single-event generation. When handling multi-event prompts, without explicit temporal grounding, such models often produce blended or collapsed scenes that break the intended narrative. To address this limitation, we present SwitchCraft, a training-free framework for multi-event video generation. Our key insight is that uniform prompt injection across time ignores the correspondence between events and frames. To this end, we introduce Event-Aligned Query Steering (EAQS), which steers frame-level attention to align with relevant event prompts. Furthermore, we propose Auto-Balance Strength Solver (ABSS), which adaptively balances steering strength to preserve temporal consistency and visual fidelity. Extensive experiments demonstrate that SwitchCraft substantially improves prompt alignment, event clarity, and scene consistency compared with existing baselines, offering a simple yet effective solution for multi-event video generation.

文本到视频扩散模型的最新进展使得高保真度和时间连贯的视频合成成为可能。然而，当前的模型主要针对单事件生成进行优化。在处理多事件提示时，如果没有明确的时间基础，此类模型通常会产生混合或折叠的场景，从而破坏预期的叙述。为了解决这个限制，我们提出了 SwitchCraft，这是一种用于多事件视频生成的免训练框架。我们的主要见解是跨时间的统一提示注入忽略了事件和帧之间的对应关系。为此，我们引入了事件对齐查询引导（EAQS），它引导帧级注意力与相关事件提示保持一致。此外，我们提出了自动平衡强度求解器（ABSS），它自适应地平衡转向强度以保持时间一致性和视觉保真度。大量实验表明，与现有基线相比，SwitchCraft 显着提高了提示对齐、事件清晰度和场景一致性，为多事件视频生成提供了简单而有效的解决方案。

</details>

---



</details>

<details><summary><b>2026-02-27 (18 papers)</b></summary>

# arXiv Video Papers - 2026-02-27

**Paper Count**: 18

---

## 1. MovieTeller: Tool-augmented Movie Synopsis with ID Consistent Progressive Abstraction / MovieTeller：具有 ID 一致渐进抽象的工具增强电影概要

**Date**: 2026-02-26 | **arXiv**: [2602.23228v1](http://arxiv.org/abs/2602.23228v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.23228v1)

**Categories**: cs.CV, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

With the explosive growth of digital entertainment, automated video summarization has become indispensable for applications such as content indexing, personalized recommendation, and efficient media archiving. Automatic synopsis generation for long-form videos, such as movies and TV series, presents a significant challenge for existing Vision-Language Models (VLMs). While proficient at single-image captioning, these general-purpose models often exhibit critical failures in long-duration contexts, primarily a lack of ID-consistent character identification and a fractured narrative coherence. To overcome these limitations, we propose MovieTeller, a novel framework for generating movie synopses via tool-augmented progressive abstraction. Our core contribution is a training-free, tool-augmented, fact-grounded generation process. Instead of requiring costly model fine-tuning, our framework directly leverages off-the-shelf models in a plug-and-play manner. We first invoke a specialized face recognition model as an external "tool" to establish Factual Groundings--precise character identities and their corresponding bounding boxes. These groundings are then injected into the prompt to steer the VLM's reasoning, ensuring the generated scene descriptions are anchored to verifiable facts. Furthermore, our progressive abstraction pipeline decomposes the summarization of a full-length movie into a multi-stage process, effectively mitigating the context length limitations of current VLMs. Experiments demonstrate that our approach yields significant improvements in factual accuracy, character consistency, and overall narrative coherence compared to end-to-end baselines.

随着数字娱乐的爆炸性增长，自动视频摘要已成为内容索引、个性化推荐和高效媒体归档等应用不可或缺的一部分。电影和电视剧等长视频的自动概要生成对现有视觉语言模型 (VLM) 提出了重大挑战。虽然精通单图像字幕，但这些通用模型经常在长时间的环境中表现出严重的失败，主要是缺乏 ID 一致的角色识别和支离破碎的叙事连贯性。为了克服这些限制，我们提出了 MovieTeller，这是一种通过工具增强渐进抽象生成电影概要的新颖框架。我们的核心贡献是一个免培训、工具增强、基于事实的生成过程。我们的框架不需要进行昂贵的模型微调，而是以即插即用的方式直接利用现成的模型。我们首先调用专门的人脸识别模型作为外部“工具”来建立事实基础——精确的角色身份及其相应的边界框。然后将这些基础注入到提示中以引导 VLM 的推理，确保生成的场景描述锚定到可验证的事实。此外，我们的渐进式抽象管道将全长电影的摘要分解为多阶段过程，有效缓解了当前 VLM 的上下文长度限制。实验表明，与端到端基线相比，我们的方法在事实准确性、角色一致性和整体叙事连贯性方面取得了显着改进。

</details>

---

## 2. EmbodMocap: In-the-Wild 4D Human-Scene Reconstruction for Embodied Agents / EmbodMocap：实体代理的野外 4D 人体场景重建

**Date**: 2026-02-26 | **arXiv**: [2602.23205v1](http://arxiv.org/abs/2602.23205v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.23205v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Human behaviors in the real world naturally encode rich, long-term contextual information that can be leveraged to train embodied agents for perception, understanding, and acting. However, existing capture systems typically rely on costly studio setups and wearable devices, limiting the large-scale collection of scene-conditioned human motion data in the wild. To address this, we propose EmbodMocap, a portable and affordable data collection pipeline using two moving iPhones. Our key idea is to jointly calibrate dual RGB-D sequences to reconstruct both humans and scenes within a unified metric world coordinate frame. The proposed method allows metric-scale and scene-consistent capture in everyday environments without static cameras or markers, bridging human motion and scene geometry seamlessly. Compared with optical capture ground truth, we demonstrate that the dual-view setting exhibits a remarkable ability to mitigate depth ambiguity, achieving superior alignment and reconstruction performance over single iphone or monocular models. Based on the collected data, we empower three embodied AI tasks: monocular human-scene-reconstruction, where we fine-tune on feedforward models that output metric-scale, world-space aligned humans and scenes; physics-based character animation, where we prove our data could be used to scale human-object interaction skills and scene-aware motion tracking; and robot motion control, where we train a humanoid robot via sim-to-real RL to replicate human motions depicted in videos. Experimental results validate the effectiveness of our pipeline and its contributions towards advancing embodied AI research.

现实世界中的人类行为自然会编码丰富的长期上下文信息，这些信息可用于训练实体代理的感知、理解和行动。然而，现有的捕捉系统通常依赖于昂贵的工作室设置和可穿戴设备，限制了在野外大规模收集场景调节的人体运动数据。为了解决这个问题，我们提出了 EmbodMocap，这是一种使用两部移动 iPhone 的便携式且经济实惠的数据收集管道。我们的关键思想是联合校准双 RGB-D 序列，以在统一的公制世界坐标系内重建人类和场景。所提出的方法允许在日常环境中进行公制尺度和场景一致的捕获，而无需静态相机或标记，从而无缝地连接人体运动和场景几何形状。与光学捕获地面实况相比，我们证明双视图设置表现出显着的减轻深度模糊性的能力，与单个 iPhone 或单目模型相比，实现了卓越的对齐和重建性能。基于收集到的数据，我们支持三项具体的人工智能任务：单目人类场景重建，我们对前馈模型进行微调，输出公制尺度、世界空间对齐的人类和场景；基于物理的角色动画，我们证明我们的数据可用于扩展人类对象交互技能和场景感知运动跟踪；和机器人运动控制，我们通过模拟到真实的强化学习来训练人形机器人来复制视频中描绘的人类运动。实验结果验证了我们管道的有效性及其对推进具体人工智能研究的贡献。

</details>

---

## 3. ColoDiff: Integrating Dynamic Consistency With Content Awareness for Colonoscopy Video Generation / ColoDiff：将动态一致性与内容感知相结合以生成结肠镜检查视频

**Date**: 2026-02-26 | **arXiv**: [2602.23203v1](http://arxiv.org/abs/2602.23203v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.23203v1)

**Categories**: cs.CV, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Colonoscopy video generation delivers dynamic, information-rich data critical for diagnosing intestinal diseases, particularly in data-scarce scenarios. High-quality video generation demands temporal consistency and precise control over clinical attributes, but faces challenges from irregular intestinal structures, diverse disease representations, and various imaging modalities. To this end, we propose ColoDiff, a diffusion-based framework that generates dynamic-consistent and content-aware colonoscopy videos, aiming to alleviate data shortage and assist clinical analysis. At the inter-frame level, our TimeStream module decouples temporal dependency from video sequences through a cross-frame tokenization mechanism, enabling intricate dynamic modeling despite irregular intestinal structures. At the intra-frame level, our Content-Aware module incorporates noise-injected embeddings and learnable prototypes to realize precise control over clinical attributes, breaking through the coarse guidance of diffusion models. Additionally, ColoDiff employs a non-Markovian sampling strategy that cuts steps by over 90% for real-time generation. ColoDiff is evaluated across three public datasets and one hospital database, based on both generation metrics and downstream tasks including disease diagnosis, modality discrimination, bowel preparation scoring, and lesion segmentation. Extensive experiments show ColoDiff generates videos with smooth transitions and rich dynamics. ColoDiff presents an effort in controllable colonoscopy video generation, revealing the potential of synthetic videos in complementing authentic representation and mitigating data scarcity in clinical settings.

结肠镜检查视频生成提供动态、信息丰富的数据，这对于诊断肠道疾病至关重要，特别是在数据稀缺的情况下。高质量视频生成需要时间一致性和对临床属性的精确控制，但面临着不规则肠道结构、多样化疾病表现和各种成像方式的挑战。为此，我们提出了 ColoDiff，一种基于扩散的框架，可生成动态一致且内容感知的结肠镜检查视频，旨在缓解数据短缺并协助临床分析。在帧间级别，我们的 TimeStream 模块通过跨帧标记化机制将时间依赖性与视频序列解耦，从而在肠道结构不规则的情况下实现复杂的动态建模。在帧内级别，我们的内容感知模块结合了噪声注入嵌入和可学习原型，以实现对临床属性的精确控制，突破了扩散模型的粗略指导。此外，ColoDiff 采用非马尔可夫采样策略，可以将实时生成的步骤减少 90% 以上。 ColoDiff 在三个公共数据集和一个医院数据库中进行评估，基于生成指标和下游任务，包括疾病诊断、模态区分、肠道准备评分和病变分割。大量实验表明 ColoDiff 生成的视频具有平滑的过渡和丰富的动态。 ColoDiff 在可控结肠镜检查视频生成方面做出了努力，揭示了合成视频在补充真实表现和缓解临床环境中数据稀缺方面的潜力。

</details>

---

## 4. The Trinity of Consistency as a Defining Principle for General World Models / 一致性的三位一体作为一般世界模型的定义原则

**Date**: 2026-02-26 | **arXiv**: [2602.23152v1](http://arxiv.org/abs/2602.23152v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.23152v1)

**Categories**: cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

The construction of World Models capable of learning, simulating, and reasoning about objective physical laws constitutes a foundational challenge in the pursuit of Artificial General Intelligence. Recent advancements represented by video generation models like Sora have demonstrated the potential of data-driven scaling laws to approximate physical dynamics, while the emerging Unified Multimodal Model (UMM) offers a promising architectural paradigm for integrating perception, language, and reasoning. Despite these advances, the field still lacks a principled theoretical framework that defines the essential properties requisite for a General World Model. In this paper, we propose that a World Model must be grounded in the Trinity of Consistency: Modal Consistency as the semantic interface, Spatial Consistency as the geometric basis, and Temporal Consistency as the causal engine. Through this tripartite lens, we systematically review the evolution of multimodal learning, revealing a trajectory from loosely coupled specialized modules toward unified architectures that enable the synergistic emergence of internal world simulators. To complement this conceptual framework, we introduce CoW-Bench, a benchmark centered on multi-frame reasoning and generation scenarios. CoW-Bench evaluates both video generation models and UMMs under a unified evaluation protocol. Our work establishes a principled pathway toward general world models, clarifying both the limitations of current systems and the architectural requirements for future progress.

构建能够学习、模拟和推理客观物理定律的世界模型是追求通用人工智能的基本挑战。以 Sora 等视频生成模型为代表的最新进展证明了数据驱动的缩放定律在近似物理动力学方面的潜力，而新兴的统一多模态模型 (UMM) 则为集成感知、语言和推理提供了一种有前途的架构范例。尽管取得了这些进展，该领域仍然缺乏一个原则性的理论框架来定义通用世界模型所需的基本属性。在本文中，我们提出世界模型必须建立在一致性三位一体的基础上：模态一致性作为语义接口，空间一致性作为几何基础，时间一致性作为因果引擎。通过这个三方视角，我们系统地回顾了多模态学习的演变，揭示了从松散耦合的专业模块到统一架构的轨迹，从而实现内部世界模拟器的协同出现。为了补充这个概念框架，我们引入了 CoW-Bench，这是一个以多帧推理和生成场景为中心的基准测试。 CoW-Bench 在统一的评估协议下评估视频生成模型和 UMM。我们的工作建立了通向通用世界模型的原则性途径，阐明了当前系统的局限性和未来进步的架构要求。

</details>

---

## 5. Align then Adapt: Rethinking Parameter-Efficient Transfer Learning in 4D Perception / 先对齐再适应：重新思考 4D 感知中的参数高效迁移学习

**Date**: 2026-02-26 | **arXiv**: [2602.23069v1](http://arxiv.org/abs/2602.23069v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.23069v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Point cloud video understanding is critical for robotics as it accurately encodes motion and scene interaction. We recognize that 4D datasets are far scarcer than 3D ones, which hampers the scalability of self-supervised 4D models. A promising alternative is to transfer 3D pre-trained models to 4D perception tasks. However, rigorous empirical analysis reveals two critical limitations that impede transfer capability: overfitting and the modality gap. To overcome these challenges, we develop a novel "Align then Adapt" (PointATA) paradigm that decomposes parameter-efficient transfer learning into two sequential stages. Optimal-transport theory is employed to quantify the distributional discrepancy between 3D and 4D datasets, enabling our proposed point align embedder to be trained in Stage 1 to alleviate the underlying modality gap. To mitigate overfitting, an efficient point-video adapter and a spatial-context encoder are integrated into the frozen 3D backbone to enhance temporal modeling capacity in Stage 2. Notably, with the above engineering-oriented designs, PointATA enables a pre-trained 3D model without temporal knowledge to reason about dynamic video content at a smaller parameter cost compared to previous work. Extensive experiments show that PointATA can match or even outperform strong full fine-tuning models, whilst enjoying the advantage of parameter efficiency, e.g. 97.21 \% accuracy on 3D action recognition, $+8.7 \%$ on 4 D action segmentation, and 84.06\% on 4D semantic segmentation.

点云视频理解对于机器人技术至关重要，因为它可以准确地编码运动和场景交互。我们认识到 4D 数据集比 3D 数据集稀缺得多，这阻碍了自监督 4D 模型的可扩展性。一个有前途的替代方案是将 3D 预训练模型转移到 4D 感知任务。然而，严格的实证分析揭示了阻碍转移能力的两个关键限制：过度拟合和模态差距。为了克服这些挑战，我们开发了一种新颖的“对齐然后适应”（PointATA）范例，将参数高效的迁移学习分解为两个连续的阶段。采用最佳传输理论来量化 3D 和 4D 数据集之间的分布差异，使我们提出的点对齐嵌入器能够在第一阶段进行训练，以减轻潜在的模态差距。为了缓解过度拟合，将高效的点视频适配器和空间上下文编码器集成到冻结的 3D 主干中，以增强第 2 阶段的时间建模能力。值得注意的是，通过上述面向工程的设计，PointATA 可以在没有时间知识的情况下实现预训练的 3D 模型，与之前的工作相比，可以以更小的参数成本推理动态视频内容。大量实验表明，PointATA 可以匹配甚至超越强大的全微调模型，同时享受参数效率的优势，例如3D 动作识别的准确度为 97.21 \%，4D 动作分割的准确度为 $+8.7 \%$，4D 语义分割的准确度为 84.06\%。

</details>

---

## 6. PackUV: Packed Gaussian UV Maps for 4D Volumetric Video / PackUV：用于 4D 体积视频的打包高斯 UV 贴图

**Date**: 2026-02-26 | **arXiv**: [2602.23040v1](http://arxiv.org/abs/2602.23040v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.23040v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Volumetric videos offer immersive 4D experiences, but remain difficult to reconstruct, store, and stream at scale. Existing Gaussian Splatting based methods achieve high-quality reconstruction but break down on long sequences, temporal inconsistency, and fail under large motions and disocclusions. Moreover, their outputs are typically incompatible with conventional video coding pipelines, preventing practical applications.   We introduce PackUV, a novel 4D Gaussian representation that maps all Gaussian attributes into a sequence of structured, multi-scale UV atlas, enabling compact, image-native storage. To fit this representation from multi-view videos, we propose PackUV-GS, a temporally consistent fitting method that directly optimizes Gaussian parameters in the UV domain. A flow-guided Gaussian labeling and video keyframing module identifies dynamic Gaussians, stabilizes static regions, and preserves temporal coherence even under large motions and disocclusions. The resulting UV atlas format is the first unified volumetric video representation compatible with standard video codecs (e.g., FFV1) without losing quality, enabling efficient streaming within existing multimedia infrastructure.   To evaluate long-duration volumetric capture, we present PackUV-2B, the largest multi-view video dataset to date, featuring more than 50 synchronized cameras, substantial motion, and frequent disocclusions across 100 sequences and 2B (billion) frames. Extensive experiments demonstrate that our method surpasses existing baselines in rendering fidelity while scaling to sequences up to 30 minutes with consistent quality.

体积视频提供身临其境的 4D 体验，但仍然难以大规模重建、存储和流式传输。现有的基于高斯分布的方法实现了高质量的重建，但在长序列、时间不一致的情况下会崩溃，并且在大运动和去遮挡下会失败。此外，它们的输出通常与传统的视频编码管道不兼容，从而阻碍了实际应用。   我们引入了 PackUV，这是一种新颖的 4D 高斯表示，可将所有高斯属性映射到一系列结构化、多尺度 UV 图集，从而实现紧凑的图像原生存储。为了拟合多视图视频的这种表示，我们提出了 PackUV-GS，这是一种时间一致的拟合方法，可以直接优化 UV 域中的高斯参数。流引导的高斯标记和视频关键帧模块可以识别动态高斯，稳定静态区域，并即使在大运动和去遮挡的情况下也能保持时间连贯性。由此产生的 UV 图集格式是第一个与标准视频编解码器（例如 FFV1）兼容的统一体积视频表示形式，且不会损失质量，从而能够在现有多媒体基础设施中实现高效流式传输。   为了评估长时间的体积捕获，我们提出了 PackUV-2B，这是迄今为止最大的多视图视频数据集，具有 50 多个同步摄像机、大量运动以及跨 100 个序列和 2B（十亿）帧的频繁遮挡。大量实验表明，我们的方法在渲染保真度方面超越了现有基线，同时可扩展到长达 30 分钟的序列且质量一致。

</details>

---

## 7. DMAligner: Enhancing Image Alignment via Diffusion Model Based View Synthesis / DMAligner：通过基于扩散模型的视图合成增强图像对齐

**Date**: 2026-02-26 | **arXiv**: [2602.23022v1](http://arxiv.org/abs/2602.23022v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.23022v1)

**Categories**: cs.CV

**Code**: https://github.com/boomluo02/DMAligner.

<details><summary><b>Abstract / 摘要</b></summary>

Image alignment is a fundamental task in computer vision with broad applications. Existing methods predominantly employ optical flow-based image warping. However, this technique is susceptible to common challenges such as occlusions and illumination variations, leading to degraded alignment visual quality and compromised accuracy in downstream tasks. In this paper, we present DMAligner, a diffusion-based framework for image alignment through alignment-oriented view synthesis. DMAligner is crafted to tackle the challenges in image alignment from a new perspective, employing a generation-based solution that showcases strong capabilities and avoids the problems associated with flow-based image warping. Specifically, we propose a Dynamics-aware Diffusion Training approach for learning conditional image generation, synthesizing a novel view for image alignment. This incorporates a Dynamics-aware Mask Producing (DMP) module to adaptively distinguish dynamic foreground regions from static backgrounds, enabling the diffusion model to more effectively handle challenges that classical methods struggle to solve. Furthermore, we develop the Dynamic Scene Image Alignment (DSIA) dataset using Blender, which includes 1,033 indoor and outdoor scenes with over 30K image pairs tailored for image alignment. Extensive experimental results demonstrate the superiority of the proposed approach on DSIA benchmarks, as well as on a series of widely-used video datasets for qualitative comparisons. Our code is available at https://github.com/boomluo02/DMAligner.

图像对齐是计算机视觉中的一项基本任务，具有广泛的应用。现有方法主要采用基于光流的图像扭曲。然而，该技术容易受到遮挡和照明变化等常见挑战的影响，导致对齐视觉质量下降并影响下游任务的准确性。在本文中，我们提出了 DMAaligner，这是一种基于扩散的框架，通过面向对齐的视图合成进行图像对齐。 DMAligner 旨在从新的角度应对图像对齐的挑战，采用基于生成的解决方案，展示强大的功能并避免与基于流的图像扭曲相关的问题。具体来说，我们提出了一种动态感知扩散训练方法，用于学习条件图像生成，合成图像对齐的新颖视图。它结合了动态感知掩模生成（DMP）模块，可以自适应地区分动态前景区域和静态背景，使扩散模型能够更有效地应对经典方法难以解决的挑战。此外，我们使用 Blender 开发了动态场景图像对齐 (DSIA) 数据集，其中包括 1,033 个室内和室外场景，以及为图像对齐量身定制的超过 30K 图像对。大量的实验结果证明了所提出的方法在 DSIA 基准以及一系列广泛使用的视频数据集上进行定性比较的优越性。我们的代码可在 https://github.com/boomluo02/DMAligner 获取。

</details>

---

## 8. UCM: Unifying Camera Control and Memory with Time-aware Positional Encoding Warping for World Models / UCM：通过世界模型的时间感知位置编码变形来统一相机控制和内存

**Date**: 2026-02-26 | **arXiv**: [2602.22960v1](http://arxiv.org/abs/2602.22960v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.22960v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

World models based on video generation demonstrate remarkable potential for simulating interactive environments but face persistent difficulties in two key areas: maintaining long-term content consistency when scenes are revisited and enabling precise camera control from user-provided inputs. Existing methods based on explicit 3D reconstruction often compromise flexibility in unbounded scenarios and fine-grained structures. Alternative methods rely directly on previously generated frames without establishing explicit spatial correspondence, thereby constraining controllability and consistency. To address these limitations, we present UCM, a novel framework that unifies long-term memory and precise camera control via a time-aware positional encoding warping mechanism. To reduce computational overhead, we design an efficient dual-stream diffusion transformer for high-fidelity generation. Moreover, we introduce a scalable data curation strategy utilizing point-cloud-based rendering to simulate scene revisiting, facilitating training on over 500K monocular videos. Extensive experiments on real-world and synthetic benchmarks demonstrate that UCM significantly outperforms state-of-the-art methods in long-term scene consistency, while also achieving precise camera controllability in high-fidelity video generation.

基于视频生成的世界模型在模拟交互环境方面表现出了巨大的潜力，但在两个关键领域面临着持续的困难：重新访问场景时保持长期内容一致性以及通过用户提供的输入实现精确的摄像机控制。基于显式 3D 重建的现有方法通常会损害无界场景和细粒度结构的灵活性。替代方法直接依赖于先前生成的帧，而不建立明确的空间对应关系，从而限制了可控性和一致性。为了解决这些限制，我们提出了 UCM，这是一种新颖的框架，通过时间感知的位置编码扭曲机制将长期记忆和精确的相机控制结合起来。为了减少计算开销，我们设计了一种用于高保真生成的高效双流扩散变压器。此外，我们引入了一种可扩展的数据管理策略，利用基于点云的渲染来模拟场景重访，从而促进对超过 500K 单目视频的训练。对现实世界和综合基准的大量实验表明，UCM 在长期场景一致性方面显着优于最先进的方法，同时在高保真视频生成中实现了精确的摄像机可控性。

</details>

---

## 9. Velocity and stroke rate reconstruction of canoe sprint team boats based on panned and zoomed video recordings / 基于平移和缩放视频记录的皮划艇短跑队船只的速度和划水率重建

**Date**: 2026-02-26 | **arXiv**: [2602.22941v1](http://arxiv.org/abs/2602.22941v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.22941v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Pacing strategies, defined by velocity and stroke rate profiles, are essential for peak performance in canoe sprint. While GPS is the gold standard for analysis, its limited availability necessitates automated video-based solutions. This paper presents an extended framework for reconstructing performance metrics from panned and zoomed video recordings across all sprint disciplines (K1-K4, C1-C2) and distances (200m-500m). Our method utilizes YOLOv8 for buoy and athlete detection, leveraging the known buoy grid to estimate homographies. We generalized the estimation of the boat position by means of learning a boat-specific athlete offset using a U-net based boat tip calibration. Further, we implement a robust tracking scheme using optical flow to adapt to multi-athlete boat types. Finally, we introduce methods to extract stroke rate information from either pose estimations or the athlete bounding boxes themselves. Evaluation against GPS data from elite competitions yields a velocity RRMSE of 0.020 +- 0.011 (rho = 0.956) and a stroke rate RRMSE of 0.022 +- 0.024 (rho = 0.932). The methods provide coaches with highly accurate, automated feedback without requiring on-boat sensors or manual annotation.

由速度和划水频率曲线定义的配速策略对于独木舟冲刺的最佳表现至关重要。虽然 GPS 是分析的黄金标准，但其有限的可用性需要基于自动化视频的解决方案。本文提出了一个扩展框架，用于从所有冲刺学科（K1-K4、C1-C2）和距离（200m-500m）的平移和缩放视频记录中重建表现指标。我们的方法利用 YOLOv8 进行浮标和运动员检测，利用已知的浮标网格来估计单应性。我们通过使用基于 U 网的船尖校准来学习特定于船的运动员偏移量，从而概括了船位置的估计。此外，我们使用光流实现了强大的跟踪方案，以适应多运动员船只类型。最后，我们介绍了从姿势估计或运动员边界框本身中提取划水频率信息的方法。根据精英比赛的 GPS 数据进行评估，得出速度 RRMSE 为 0.020 ± 0.011 (rho = 0.956)，划水频率 RRMSE 为 0.022 ± 0.024 (rho = 0.932)。这些方法为教练提供高度准确的自动反馈，无需船上传感器或手动注释。

</details>

---

## 10. MSJoE: Jointly Evolving MLLM and Sampler for Efficient Long-Form Video Understanding / MSJoE：联合进化 MLLM 和采样器以实现高效的长格式视频理解

**Date**: 2026-02-26 | **arXiv**: [2602.22932v1](http://arxiv.org/abs/2602.22932v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.22932v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Efficiently understanding long-form videos remains a fundamental challenge for multimodal large language models (MLLMs). In this paper, we present MLLM-Sampler Joint Evolution (MSJoE), a novel framework that jointly evolves the MLLM and a lightweight key-frame sampler for efficient long-form video understanding. MSJoE builds upon a key assumption that only a small subset of key-frames is truly informative for answering each question to a video. Specifically, MSJoE first reasons out several queries, which describe diverse visual perspectives relevant to the question. Then, these queries interact with a frozen CLIP model to produce a query-frame similarity matrix. Finally, a lightweight sampler predicts key-frame sampling weights from this matrix, selecting a compact set of informative frames, which are then fed into the MLLM for answer generation. Both the MLLM and sampler are jointly optimized through reinforcement learning, enabling co-adaptation of query-reasoning, frame-sampling, and key-frame understanding. A new long-video QA dataset containing 2.8K videos with 7K question-answer pairs is collected to support the training process. Extensive experiments on VideoMME, LongVideoBench, LVBench, and MLVU show that MSJoE achieves 8.0\% accuracy gain upon the base MLLM, and 1.1\% higher accuracy than strongest baseline method.

有效理解长视频仍然是多模态大语言模型（MLLM）的基本挑战。在本文中，我们提出了 MLLM-采样器联合进化（MSJoE），这是一种联合进化 MLLM 和轻量级关键帧采样器的新颖框架，用于高效的长格式视频理解。 MSJoE 建立在一个关键假设之上，即只有一小部分关键帧能够真正为回答视频的每个问题提供信息。具体来说，MSJoE 首先推理出几个查询，这些查询描述了与问题相关的不同视觉视角。然后，这些查询与冻结的 CLIP 模型交互以生成查询帧相似度矩阵。最后，轻量级采样器根据该矩阵预测关键帧采样权重，选择一组紧凑的信息帧，然后将其馈送到 MLLM 中以生成答案。 MLLM 和采样器都通过强化学习进行联合优化，从而实现查询推理、帧采样和关键帧理解的共同适应。收集了一个新的长视频 QA 数据集，其中包含 2.8K 视频和 7K 问答对，以支持训练过程。在 VideoMME、LongVideoBench、LVBench 和 MLVU 上进行的大量实验表明，MSJoE 在基础 MLLM 的基础上实现了 8.0% 的精度增益，比最强基线方法的精度提高了 1.1%。

</details>

---

## 11. WaterVideoQA: ASV-Centric Perception and Rule-Compliant Reasoning via Multi-Modal Agents / WaterVideoQA：通过多模态代理进行以 ASV 为中心的感知和符合规则的推理

**Date**: 2026-02-26 | **arXiv**: [2602.22923v1](http://arxiv.org/abs/2602.22923v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.22923v1)

**Categories**: cs.CV, cs.RO

<details><summary><b>Abstract / 摘要</b></summary>

While autonomous navigation has achieved remarkable success in passive perception (e.g., object detection and segmentation), it remains fundamentally constrained by a void in knowledge-driven, interactive environmental cognition. In the high-stakes domain of maritime navigation, the ability to bridge the gap between raw visual perception and complex cognitive reasoning is not merely an enhancement but a critical prerequisite for Autonomous Surface Vessels to execute safe and precise maneuvers. To this end, we present WaterVideoQA, the first large-scale, comprehensive Video Question Answering benchmark specifically engineered for all-waterway environments. This benchmark encompasses 3,029 video clips across six distinct waterway categories, integrating multifaceted variables such as volatile lighting and dynamic weather to rigorously stress-test ASV capabilities across a five-tier hierarchical cognitive framework. Furthermore, we introduce NaviMind, a pioneering multi-agent neuro-symbolic system designed for open-ended maritime reasoning. By synergizing Adaptive Semantic Routing, Situation-Aware Hierarchical Reasoning, and Autonomous Self-Reflective Verification, NaviMind transitions ASVs from superficial pattern matching to regulation-compliant, interpretable decision-making. Experimental results demonstrate that our framework significantly transcends existing baselines, establishing a new paradigm for intelligent, trustworthy interaction in dynamic maritime environments.

尽管自主导航在被动感知（例如对象检测和分割）方面取得了显着的成功，但它仍然从根本上受到知识驱动的交互式环境认知空白的限制。在海上导航这个高风险领域，弥合原始视觉感知和复杂认知推理之间差距的能力不仅是一种增强，而且是自主水面舰艇执行安全和精确机动的关键先决条件。为此，我们推出了 WaterVideoQA，这是第一个专门针对全水路环境设计的大规模、全面的视频问答基准测试。该基准测试包含 6 个不同水道类别的 3,029 个视频剪辑，集成了多方面变量（例如不稳定的照明和动态天气），以跨五层分层认知框架严格测试 ASV 功能。此外，我们还介绍了 NaviMind，这是一种开创性的多智能体神经符号系统，专为开放式海事推理而设计。通过协同自适应语义路由、情境感知分层推理和自主自我反思验证，NaviMind 将 ASV 从肤浅的模式匹配转变为符合法规的、可解释的决策。实验结果表明，我们的框架显着超越了现有的基线，为动态海洋环境中的智能、值得信赖的交互建立了新的范式。

</details>

---

## 12. TrajTok: Learning Trajectory Tokens enables better Video Understanding / TrajTok：学习轨迹标记可以更好地理解视频

**Date**: 2026-02-26 | **arXiv**: [2602.22779v1](http://arxiv.org/abs/2602.22779v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.22779v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Tokenization in video models, typically through patchification, generates an excessive and redundant number of tokens. This severely limits video efficiency and scalability. While recent trajectory-based tokenizers offer a promising solution by decoupling video duration from token count, they rely on complex external segmentation and tracking pipelines that are slow and task-agnostic. We propose TrajTok, an end-to-end video tokenizer module that is fully integrated and co-trained with video models for a downstream objective, dynamically adapting its token granularity to semantic complexity, independent of video duration. TrajTok contains a unified segmenter that performs implicit clustering over pixels in both space and time to directly produce object trajectories in a single forward pass. By prioritizing downstream adaptability over pixel-perfect segmentation fidelity, TrajTok is lightweight and efficient, yet empirically improves video understanding performance. With TrajTok, we implement a video CLIP model trained from scratch (TrajViT2). It achieves the best accuracy at scale across both classification and retrieval benchmarks, while maintaining efficiency comparable to the best token-merging methods. TrajTok also proves to be a versatile component beyond its role as a tokenizer. We show that it can be seamlessly integrated as either a probing head for pretrained visual features (TrajAdapter) or an alignment connector in vision-language models (TrajVLM) with especially strong performance in long-video reasoning.

视频模型中的标记化（通常通过补丁化）会生成过多且冗余的标记。这严重限制了视频效率和可扩展性。虽然最近基于轨迹的标记器通过将视频持续时间与标记计数解耦提供了一种有前途的解决方案，但它们依赖于复杂的外部分段和跟踪管道，这些管道速度缓慢且与任务无关。我们提出了 TrajTok，这是一种端到端视频标记器模块，它与下游目标的视频模型完全集成和共同训练，动态地调整其标记粒度以适应语义复杂性，而与视频时长无关。 TrajTok 包含一个统一的分段器，可以在空间和时间上对像素执行隐式聚类，以在单个前向传递中直接生成对象轨迹。通过优先考虑下游适应性而不是像素完美的分割保真度，TrajTok 是轻量级且高效的，但凭经验提高了视频理解性能。借助 TrajTok，我们实现了从头开始训练的视频 CLIP 模型 (TrajViT2)。它在分类和检索基准上实现了大规模的最佳准确度，同时保持了与最佳标记合并方法相当的效率。 TrajTok 还被证明是一个超越其标记器角色的多功能组件。我们证明它可以无缝集成为预训练视觉特征的探测头（TrajAdapter）或视觉语言模型（TrajVLM）中的对齐连接器，在长视频推理中具有特别强大的性能。

</details>

---

## 13. SPATIALALIGN: Aligning Dynamic Spatial Relationships in Video Generation / SPATIALALIGN：在视频生成中对齐动态空间关系

**Date**: 2026-02-26 | **arXiv**: [2602.22745v1](http://arxiv.org/abs/2602.22745v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.22745v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Most text-to-video (T2V) generators prioritize aesthetic quality, but often ignoring the spatial constraints in the generated videos. In this work, we present SPATIALALIGN, a self-improvement framework that enhances T2V models capabilities to depict Dynamic Spatial Relationships (DSR) specified in text prompts. We present a zeroth-order regularized Direct Preference Optimization (DPO) to fine-tune T2V models towards better alignment with DSR. Specifically, we design DSR-SCORE, a geometry-based metric that quantitatively measures the alignment between generated videos and the specified DSRs in prompts, which is a step forward from prior works that rely on VLM for evaluation. We also conduct a dataset of text-video pairs with diverse DSRs to facilitate the study. Extensive experiments demonstrate that our fine-tuned model significantly out performs the baseline in spatial relationships. The code will be released in Link.

大多数文本转视频 (T2V) 生成器优先考虑美观质量，但往往忽略生成视频中的空间限制。在这项工作中，我们提出了 SPATIALALIGN，这是一个自我改进框架，可增强 T2V 模型描述文本提示中指定的动态空间关系 (DSR) 的能力。我们提出了零阶正则化直接偏好优化 (DPO) 来微调 T2V 模型，以更好地与 DSR 保持一致。具体来说，我们设计了 DSR-SCORE，这是一种基于几何的指标，可以定量测量生成的视频与提示中指定的 DSR 之间的对齐情况，这比之前依赖 VLM 进行评估的工作向前迈出了一步。我们还建立了具有不同 DSR 的文本视频对数据集，以促进研究。大量的实验表明，我们的微调模型在空间关系方面的表现显着优于基线。代码将在Link中发布。

</details>

---

## 14. Denoising as Path Planning: Training-Free Acceleration of Diffusion Models with DPCache / 去噪作为路径规划：使用 DPCache 进行扩散模型的免训练加速

**Date**: 2026-02-26 | **arXiv**: [2602.22654v1](http://arxiv.org/abs/2602.22654v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.22654v1)

**Categories**: cs.CV

**Code**: https://github.com/argsss/DPCache.

<details><summary><b>Abstract / 摘要</b></summary>

Diffusion models have demonstrated remarkable success in image and video generation, yet their practical deployment remains hindered by the substantial computational overhead of multi-step iterative sampling. Among acceleration strategies, caching-based methods offer a training-free and effective solution by reusing or predicting features across timesteps. However, existing approaches rely on fixed or locally adaptive schedules without considering the global structure of the denoising trajectory, often leading to error accumulation and visual artifacts. To overcome this limitation, we propose DPCache, a novel training-free acceleration framework that formulates diffusion sampling acceleration as a global path planning problem. DPCache constructs a Path-Aware Cost Tensor from a small calibration set to quantify the path-dependent error of skipping timesteps conditioned on the preceding key timestep. Leveraging this tensor, DPCache employs dynamic programming to select an optimal sequence of key timesteps that minimizes the total path cost while preserving trajectory fidelity. During inference, the model performs full computations only at these key timesteps, while intermediate outputs are efficiently predicted using cached features. Extensive experiments on DiT, FLUX, and HunyuanVideo demonstrate that DPCache achieves strong acceleration with minimal quality loss, outperforming prior acceleration methods by $+$0.031 ImageReward at 4.87$\times$ speedup and even surpassing the full-step baseline by $+$0.028 ImageReward at 3.54$\times$ speedup on FLUX, validating the effectiveness of our path-aware global scheduling framework. Code will be released at https://github.com/argsss/DPCache.

扩散模型在图像和视频生成方面取得了显着的成功，但其实际部署仍然受到多步迭代采样的大量计算开销的阻碍。在加速策略中，基于缓存的方法通过跨时间步重用或预测特征来提供免训练且有效的解决方案。然而，现有方法依赖于固定或局部自适应调度，而不考虑去噪轨迹的全局结构，通常导致误差累积和视觉伪影。为了克服这一限制，我们提出了 DPCache，这是一种新颖的免训练加速框架，它将扩散采样加速表述为全局路径规划问题。 DPCache 从一个小的校准集中构造一个路径感知成本张量，以量化以前面的关键时间步长为条件的跳过时间步长的路径相关误差。利用该张量，DPCache 采用动态编程来选择关键时间步的最佳序列，从而最大限度地降低总路径成本，同时保持轨迹保真度。在推理过程中，模型仅在这些关键时间步执行完整计算，而使用缓存的特征有效地预测中间输出。在 DiT、FLUX 和 HunyuanVideo 上进行的大量实验表明，DPCache 以最小的质量损失实现了强大的加速，在 4.87$\times$ 加速下比之前的加速方法高出 $+$0.031 ImageReward，甚至在 FLUX 上以 3.54$\times$ 加速超过全步基线 $+$0.028 ImageReward，验证了我们的路径感知全局调度框架的有效性。代码将在 https://github.com/argsss/DPCache 发布。

</details>

---

## 15. BetterScene: 3D Scene Synthesis with Representation-Aligned Generative Model / BetterScene：使用表示对齐生成模型进行 3D 场景合成

**Date**: 2026-02-26 | **arXiv**: [2602.22596v1](http://arxiv.org/abs/2602.22596v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.22596v1)

**Categories**: cs.CV, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

We present BetterScene, an approach to enhance novel view synthesis (NVS) quality for diverse real-world scenes using extremely sparse, unconstrained photos. BetterScene leverages the production-ready Stable Video Diffusion (SVD) model pretrained on billions of frames as a strong backbone, aiming to mitigate artifacts and recover view-consistent details at inference time. Conventional methods have developed similar diffusion-based solutions to address these challenges of novel view synthesis. Despite significant improvements, these methods typically rely on off-the-shelf pretrained diffusion priors and fine-tune only the UNet module while keeping other components frozen, which still leads to inconsistent details and artifacts even when incorporating geometry-aware regularizations like depth or semantic conditions. To address this, we investigate the latent space of the diffusion model and introduce two components: (1) temporal equivariance regularization and (2) vision foundation model-aligned representation, both applied to the variational autoencoder (VAE) module within the SVD pipeline. BetterScene integrates a feed-forward 3D Gaussian Splatting (3DGS) model to render features as inputs for the SVD enhancer and generate continuous, artifact-free, consistent novel views. We evaluate on the challenging DL3DV-10K dataset and demonstrate superior performance compared to state-of-the-art methods.

我们提出了 BetterScene，这是一种使用极其稀疏、无约束的照片来增强各种现实世界场景的新颖视图合成 (NVS) 质量的方法。 BetterScene 利用在数十亿帧上预训练的可投入生产的稳定视频扩散 (SVD) 模型作为强大的骨干，旨在减少伪影并在推理时恢复视图一致的细节。传统方法已经开发了类似的基于扩散的解决方案来解决新颖视图合成的这些挑战。尽管有了显着的改进，这些方法通常依赖于现成的预训练扩散先验，并且仅对 UNet 模块进行微调，同时保持其他组件冻结，即使在合并深度或语义条件等几何感知正则化时，这仍然会导致细节和伪影不一致。为了解决这个问题，我们研究了扩散模型的潜在空间，并引入了两个组件：(1) 时间等方差正则化和 (2) 视觉基础模型对齐表示，两者都应用于 SVD 管道内的变分自动编码器 (VAE) 模块。 BetterScene 集成了前馈 3D 高斯泼溅 (3DGS) 模型，将特征渲染为 SVD 增强器的输入，并生成连续、无伪影、一致的新颖视图。我们对具有挑战性的 DL3DV-10K 数据集进行评估，并展示了与最先进的方法相比的卓越性能。

</details>

---

## 16. Flow Matching is Adaptive to Manifold Structures / 流量匹配适应歧管结构

**Date**: 2026-02-25 | **arXiv**: [2602.22486v1](http://arxiv.org/abs/2602.22486v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.22486v1)

**Categories**: stat.ML, cs.LG, math.ST

<details><summary><b>Abstract / 摘要</b></summary>

Flow matching has emerged as a simulation-free alternative to diffusion-based generative modeling, producing samples by solving an ODE whose time-dependent velocity field is learned along an interpolation between a simple source distribution (e.g., a standard normal) and a target data distribution. Flow-based methods often exhibit greater training stability and have achieved strong empirical performance in high-dimensional settings where data concentrate near a low-dimensional manifold, such as text-to-image synthesis, video generation, and molecular structure generation. Despite this success, existing theoretical analyses of flow matching assume target distributions with smooth, full-dimensional densities, leaving its effectiveness in manifold-supported settings largely unexplained. To this end, we theoretically analyze flow matching with linear interpolation when the target distribution is supported on a smooth manifold. We establish a non-asymptotic convergence guarantee for the learned velocity field, and then propagate this estimation error through the ODE to obtain statistical consistency of the implicit density estimator induced by the flow-matching objective. The resulting convergence rate is near minimax-optimal, depends only on the intrinsic dimension, and reflects the smoothness of both the manifold and the target distribution. Together, these results provide a principled explanation for how flow matching adapts to intrinsic data geometry and circumvents the curse of dimensionality.

流匹配已成为基于扩散的生成建模的免模拟替代方案，通过求解常微分方程来生成样本，该常微分方程的时间相关速度场是通过简单源分布（例如标准正态）和目标数据分布之间的插值来学习的。基于流的方法通常表现出更高的训练稳定性，并且在数据集中在低维流形附近的高维设置中实现了强大的经验性能，例如文本到图像合成、视频生成和分子结构生成。尽管取得了这一成功，现有的流匹配理论分析假设目标分布具有平滑、全维密度，使其在流形支持的设置中的有效性很大程度上无法解释。为此，我们从理论上分析了当目标分布支持在光滑流形上时使用线性插值的流匹配。我们为学习的速度场建立非渐近收敛保证，然后通过 ODE 传播该估计误差，以获得由流匹配目标引起的隐式密度估计器的统计一致性。由此产生的收敛速度接近极小极大最优，仅取决于内在维度，并且反映了流形和目标分布的平滑度。总之，这些结果为流匹配如何适应内在数据几何并规避维数灾难提供了原则性解释。

</details>

---

## 17. Solaris: Building a Multiplayer Video World Model in Minecraft / Solaris：在 Minecraft 中构建多人视频世界模型

**Date**: 2026-02-25 | **arXiv**: [2602.22208v2](http://arxiv.org/abs/2602.22208v2) | **PDF**: [Link](http://arxiv.org/pdf/2602.22208v2)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Existing action-conditioned video generation models (video world models) are limited to single-agent perspectives, failing to capture the multi-agent interactions of real-world environments. We introduce Solaris, a multiplayer video world model that simulates consistent multi-view observations. To enable this, we develop a multiplayer data system designed for robust, continuous, and automated data collection on video games such as Minecraft. Unlike prior platforms built for single-player settings, our system supports coordinated multi-agent interaction and synchronized videos + actions capture. Using this system, we collect 12.64 million multiplayer frames and propose an evaluation framework for multiplayer movement, memory, grounding, building, and view consistency. We train Solaris using a staged pipeline that progressively transitions from single-player to multiplayer modeling, combining bidirectional, causal, and Self Forcing training. In the final stage, we introduce Checkpointed Self Forcing, a memory-efficient Self Forcing variant that enables a longer-horizon teacher. Results show our architecture and training design outperform existing baselines. Through open-sourcing our system and models, we hope to lay the groundwork for a new generation of multi-agent world models.

现有的动作条件视频生成模型（视频世界模型）仅限于单智能体视角，无法捕捉现实世界环境的多智能体交互。我们介绍 Solaris，这是一种模拟一致的多视图观察的多人视频世界模型。为了实现这一目标，我们开发了一个多人数据系统，专为《我的世界》等视频游戏的稳健、连续和自动化数据收集而设计。与之前为单人游戏设置构建的平台不同，我们的系统支持协调的多代理交互和同步视频+动作捕捉。使用该系统，我们收集了 1264 万个多人游戏帧，并提出了多人运动、记忆、接地、构建和视图一致性的评估框架。我们使用分阶段的管道来训练 Solaris，该管道逐渐从单人模式过渡到多人模式，结合了双向、因果和自我强迫训练。在最后阶段，我们引入了检查点自我强迫，这是一种内存高效的自我强迫变体，可以实现更长视野的教师。结果显示我们的架构和培训设计优于现有基线。通过开源我们的系统和模型，我们希望为新一代多智能体世界模型奠定基础。

</details>

---

## 18. SkyReels-V4: Multi-modal Video-Audio Generation, Inpainting and Editing model / SkyReels-V4：多模式视频音频生成、修复和编辑模型

**Date**: 2026-02-25 | **arXiv**: [2602.21818v2](http://arxiv.org/abs/2602.21818v2) | **PDF**: [Link](http://arxiv.org/pdf/2602.21818v2)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

SkyReels V4 is a unified multi modal video foundation model for joint video audio generation, inpainting, and editing. The model adopts a dual stream Multimodal Diffusion Transformer (MMDiT) architecture, where one branch synthesizes video and the other generates temporally aligned audio, while sharing a powerful text encoder based on the Multimodal Large Language Models (MMLM). SkyReels V4 accepts rich multi modal instructions, including text, images, video clips, masks, and audio references. By combining the MMLMs multi modal instruction following capability with in context learning in the video branch MMDiT, the model can inject fine grained visual guidance under complex conditioning, while the audio branch MMDiT simultaneously leverages audio references to guide sound generation. On the video side, we adopt a channel concatenation formulation that unifies a wide range of inpainting style tasks, such as image to video, video extension, and video editing under a single interface, and naturally extends to vision referenced inpainting and editing via multi modal prompts. SkyReels V4 supports up to 1080p resolution, 32 FPS, and 15 second duration, enabling high fidelity, multi shot, cinema level video generation with synchronized audio. To make such high resolution, long-duration generation computationally feasible, we introduce an efficiency strategy: Joint generation of low resolution full sequences and high-resolution keyframes, followed by dedicated super-resolution and frame interpolation models. To our knowledge, SkyReels V4 is the first video foundation model that simultaneously supports multi-modal input, joint video audio generation, and a unified treatment of generation, inpainting, and editing, while maintaining strong efficiency and quality at cinematic resolutions and durations.

SkyReels V4 是一个统一的多模态视频基础模型，用于联合视频音频生成、修复和编辑。该模型采用双流多模态扩散变压器（MMDiT）架构，其中一个分支合成视频，另一个分支生成时间对齐的音频，同时共享基于多模态大语言模型（MMLM）的强大文本编码器。 SkyReels V4 接受丰富的多模式指令，包括文本、图像、视频剪辑、蒙版和音频参考。通过将 MMLM 多模态指令跟随功能与视频分支 MMDiT 中的上下文学习相结合，该模型可以在复杂条件下注入细粒度的视觉指导，而音频分支 MMDiT 同时利用音频参考来指导声音生成。在视频方面，我们采用通道串联公式，将图像到视频、视频扩展和视频编辑等多种修复风格任务统一在一个界面下，并通过多模式提示自然扩展到视觉参考修复和编辑。 SkyReels V4 支持高达 1080p 的分辨率、32 FPS 和 15 秒的持续时间，可生成具有同步音频的高保真、多镜头、影院级视频。为了使这种高分辨率、长时间的生成在计算上可行，我们引入了一种效率策略：联合生成低分辨率全序列和高分辨率关键帧，然后是专用的超分辨率和帧插值模型。据我们所知，SkyReels V4是第一个同时支持多模态输入、联合视频音频生成以及生成、修复和编辑统一处理的视频基础模型，同时在电影分辨率和时长上保持强大的效率和质量。

</details>

---



</details>

<details><summary><b>2026-02-26 (11 papers)</b></summary>

# arXiv Video Papers - 2026-02-26

**Paper Count**: 11

---

## 1. Geometry-as-context: Modulating Explicit 3D in Scene-consistent Video Generation to Geometry Context / 几何作为上下文：将场景一致视频生成中的显式 3D 调制为几何上下文

**Date**: 2026-02-25 | **arXiv**: [2602.21929v1](http://arxiv.org/abs/2602.21929v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.21929v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Scene-consistent video generation aims to create videos that explore 3D scenes based on a camera trajectory. Previous methods rely on video generation models with external memory for consistency, or iterative 3D reconstruction and inpainting, which accumulate errors during inference due to incorrect intermediary outputs, non-differentiable processes, and separate models. To overcome these limitations, we introduce ``geometry-as-context". It iteratively completes the following steps using an autoregressive camera-controlled video generation model: (1) estimates the geometry of the current view necessary for 3D reconstruction, and (2) simulates and restores novel view images rendered by the 3D scene. Under this multi-task framework, we develop the camera gated attention module to enhance the model's capability to effectively leverage camera poses. During the training phase, text contexts are utilized to ascertain whether geometric or RGB images should be generated. To ensure that the model can generate RGB-only outputs during inference, the geometry context is randomly dropped from the interleaved text-image-geometry training sequence. The method has been tested on scene video generation with one-direction and forth-and-back trajectories. The results show its superiority over previous approaches in maintaining scene consistency and camera control.

场景一致的视频生成旨在创建基于摄像机轨迹探索 3D 场景的视频。以前的方法依赖具有外部存储器的视频生成模型来实现一致性，或者迭代 3D 重建和修复，这会在推理过程中由于不正确的中间输出、不可微分的过程和单独的模型而累积错误。为了克服这些限制，我们引入了“几何即上下文”。它使用自回归相机控制的视频生成模型迭代地完成以下步骤：（1）估计 3D 重建所需的当前视图的几何形状，以及（2）模拟和恢复 3D 场景渲染的新视图图像。在这个多任务框架下，我们开发了相机门控注意力模块，以增强模型有效利用相机姿势的能力。在训练阶段，利用文本上下文为了确定应该生成几何图像还是 RGB 图像，为了确保模型在推理过程中可以生成仅 RGB 的输出，从交错的文本-图像-几何训练序列中随机删除该方法，并在单向和前后轨迹的场景视频生成上进行了测试，结果表明其在保持场景一致性和摄像机控制方面优于以前的方法。

</details>

---

## 2. Understanding Annotation Error Propagation and Learning an Adaptive Policy for Expert Intervention in Barrett's Video Segmentation / 了解注释错误传播并学习巴雷特视频分割中专家干预的自适应策略

**Date**: 2026-02-25 | **arXiv**: [2602.21855v1](http://arxiv.org/abs/2602.21855v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.21855v1)

**Categories**: cs.CV, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Accurate annotation of endoscopic videos is essential yet time-consuming, particularly for challenging datasets such as dysplasia in Barrett's esophagus, where the affected regions are irregular and lack clear boundaries. Semi-automatic tools like Segment Anything Model 2 (SAM2) can ease this process by propagating annotations across frames, but small errors often accumulate and reduce accuracy, requiring expert review and correction. To address this, we systematically study how annotation errors propagate across different prompt types, namely masks, boxes, and points, and propose Learning-to-Re-Prompt (L2RP), a cost-aware framework that learns when and where to seek expert input. By tuning a human-cost parameter, our method balances annotation effort and segmentation accuracy. Experiments on a private Barrett's dysplasia dataset and the public SUN-SEG benchmark demonstrate improved temporal consistency and superior performance over baseline strategies.

内窥镜视频的准确注释至关重要但又耗时，特别是对于具有挑战性的数据集，例如巴雷特食管的发育不良，其中受影响的区域不规则且缺乏清晰的边界。 Segment Anything Model 2 (SAM2) 等半自动工具可以通过跨帧传播注释来简化此过程，但小错误常常会累积并降低准确性，需要专家审查和纠正。为了解决这个问题，我们系统地研究了注释错误如何在不同的提示类型（即掩码、框和点）之间传播，并提出了学习重新提示（L2RP），这是一种成本感知框架，可以学习何时何地寻求专家输入。通过调整人力成本参数，我们的方法平衡了注释工作和分割准确性。在私人 Barrett 发育不良数据集和公共 SUN-SEG 基准上进行的实验表明，与基线策略相比，时间一致性得到了改善，性能也更优越。

</details>

---

## 3. UniVBench: Towards Unified Evaluation for Video Foundation Models / UniVBench：迈向视频基础模型的统一评估

**Date**: 2026-02-25 | **arXiv**: [2602.21835v1](http://arxiv.org/abs/2602.21835v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.21835v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Video foundation models aim to integrate video understanding, generation, editing, and instruction following within a single framework, making them a central direction for next-generation multimodal systems. However, existing evaluation benchmarks remain fragmented and limited in scope, as they each target a single task, rely on task-specific metrics, and typically use short or simple video clips. As a result, they do not capture the unified capabilities that these models are designed to deliver. To address this gap, we introduce UniVBench, a benchmark purpose-built for evaluating video foundation models across four core abilities: video understanding, video generation, video editing, and a newly proposed task, video reconstruction, which assesses how faithfully a model can reproduce video content it has encountered. Our benchmark substantially expands the complexity of evaluation by incorporating 200 high-quality, diverse and multi-shot videos, each paired with detailed captions, multi-format editing instructions, and reference images. All videos are human-created and carefully validated, offering richer cinematic information than prior benchmarks. In addition, we develop a unified agentic evaluation system (UniV-Eval) that standardizes prompting, instruction parsing, and scoring across all tasks, enabling fair, scalable, and reproducible comparisons of unified video models. By grounding evaluation in instruction-based multi-shot video tasks, UniVBench provides the first framework for measuring the integrated capabilities that video foundation models aim to achieve. Extensive human annotations ensure our evaluation aligns with human judgment, enabling rigorous assessment and accelerating progress toward robust video intelligence.

视频基础模型旨在将视频理解、生成、编辑和指令跟踪集成在一个框架内，使其成为下一代多模态系统的中心方向。然而，现有的评估基准仍然分散且范围有限，因为它们每个都针对单个任务，依赖于特定于任务的指标，并且通常使用短或简单的视频剪辑。因此，它们无法捕获这些模型旨在提供的统一功能。为了解决这一差距，我们引入了 UniVBench，这是一个专门为评估视频基础模型的四个核心能力而构建的基准：视频理解、视频生成、视频编辑，以及新提出的任务视频重建，该任务评估模型如何忠实地再现其遇到的视频内容。我们的基准测试通过纳入 200 个高质量、多样化的多镜头视频，每个视频都配有详细的标题、多格式编辑说明和参考图像，极大地扩展了评估的复杂性。所有视频均由人工创作并经过仔细验证，提供比之前的基准更丰富的电影信息。此外，我们开发了一个统一的代理评估系统（UniV-Eval），该系统标准化了所有任务的提示、指令解析和评分，从而实现了统一视频模型的公平、可扩展和可重复的比较。通过在基于指令的多镜头视频任务中进行基础评估，UniVBench 提供了第一个用于测量视频基础模型旨在实现的集成功能的框架。广泛的人工注释确保我们的评估与人类判断一致，从而实现严格的评估并加速实现强大的视频智能。

</details>

---

## 4. SkyReels-V4: Multi-modal Video-Audio Generation, Inpainting and Editing model / SkyReels-V4：多模式视频音频生成、修复和编辑模型

**Date**: 2026-02-25 | **arXiv**: [2602.21818v1](http://arxiv.org/abs/2602.21818v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.21818v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

SkyReels V4 is a unified multi modal video foundation model for joint video audio generation, inpainting, and editing. The model adopts a dual stream Multimodal Diffusion Transformer (MMDiT) architecture, where one branch synthesizes video and the other generates temporally aligned audio, while sharing a powerful text encoder based on the Multimodal Large Language Models (MMLM). SkyReels V4 accepts rich multi modal instructions, including text, images, video clips, masks, and audio references. By combining the MMLMs multi modal instruction following capability with in context learning in the video branch MMDiT, the model can inject fine grained visual guidance under complex conditioning, while the audio branch MMDiT simultaneously leverages audio references to guide sound generation. On the video side, we adopt a channel concatenation formulation that unifies a wide range of inpainting style tasks, such as image to video, video extension, and video editing under a single interface, and naturally extends to vision referenced inpainting and editing via multi modal prompts. SkyReels V4 supports up to 1080p resolution, 32 FPS, and 15 second duration, enabling high fidelity, multi shot, cinema level video generation with synchronized audio. To make such high resolution, long-duration generation computationally feasible, we introduce an efficiency strategy: Joint generation of low resolution full sequences and high-resolution keyframes, followed by dedicated super-resolution and frame interpolation models. To our knowledge, SkyReels V4 is the first video foundation model that simultaneously supports multi-modal input, joint video audio generation, and a unified treatment of generation, inpainting, and editing, while maintaining strong efficiency and quality at cinematic resolutions and durations.

SkyReels V4 是一个统一的多模态视频基础模型，用于联合视频音频生成、修复和编辑。该模型采用双流多模态扩散变压器（MMDiT）架构，其中一个分支合成视频，另一个分支生成时间对齐的音频，同时共享基于多模态大语言模型（MMLM）的强大文本编码器。 SkyReels V4 接受丰富的多模式指令，包括文本、图像、视频剪辑、蒙版和音频参考。通过将 MMLM 多模态指令跟随功能与视频分支 MMDiT 中的上下文学习相结合，该模型可以在复杂条件下注入细粒度的视觉指导，而音频分支 MMDiT 同时利用音频参考来指导声音生成。在视频方面，我们采用通道串联公式，将图像到视频、视频扩展和视频编辑等多种修复风格任务统一在一个界面下，并通过多模式提示自然扩展到视觉参考修复和编辑。 SkyReels V4 支持高达 1080p 的分辨率、32 FPS 和 15 秒的持续时间，可生成具有同步音频的高保真、多镜头、影院级视频。为了使这种高分辨率、长时间的生成在计算上可行，我们引入了一种效率策略：联合生成低分辨率全序列和高分辨率关键帧，然后是专用的超分辨率和帧插值模型。据我们所知，SkyReels V4是第一个同时支持多模态输入、联合视频音频生成以及生成、修复和编辑统一处理的视频基础模型，同时在电影分辨率和时长上保持强大的效率和质量。

</details>

---

## 5. MultiAnimate: Pose-Guided Image Animation Made Extensible / MultiAnimate：可扩展的姿势引导图像动画

**Date**: 2026-02-25 | **arXiv**: [2602.21581v1](http://arxiv.org/abs/2602.21581v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.21581v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Pose-guided human image animation aims to synthesize realistic videos of a reference character driven by a sequence of poses. While diffusion-based methods have achieved remarkable success, most existing approaches are limited to single-character animation. We observe that naively extending these methods to multi-character scenarios often leads to identity confusion and implausible occlusions between characters. To address these challenges, in this paper, we propose an extensible multi-character image animation framework built upon modern Diffusion Transformers (DiTs) for video generation. At its core, our framework introduces two novel components-Identifier Assigner and Identifier Adapter - which collaboratively capture per-person positional cues and inter-person spatial relationships. This mask-driven scheme, along with a scalable training strategy, not only enhances flexibility but also enables generalization to scenarios with more characters than those seen during training. Remarkably, trained on only a two-character dataset, our model generalizes to multi-character animation while maintaining compatibility with single-character cases. Extensive experiments demonstrate that our approach achieves state-of-the-art performance in multi-character image animation, surpassing existing diffusion-based baselines.

姿势引导的人体图像动画旨在合成由一系列姿势驱动的参考角色的逼真视频。虽然基于扩散的方法取得了显着的成功，但大多数现有方法仅限于单角色动画。我们观察到，天真地将这些方法扩展到多角色场景通常会导致角色之间的身份混乱和令人难以置信的遮挡。为了解决这些挑战，在本文中，我们提出了一种基于现代扩散变压器（DiT）的可扩展多字符图像动画框架，用于视频生成。我们的框架的核心引入了两个新颖的组件——标识符分配器和标识符适配器——它们协作捕获每个人的位置线索和人与人之间的空间关系。这种掩码驱动的方案以及可扩展的训练策略不仅增强了灵活性，而且还能够泛化到比训练期间看到的字符更多的场景。值得注意的是，我们的模型仅在两个字符数据集上进行训练，可推广到多字符动画，同时保持与单字符情况的兼容性。大量的实验表明，我们的方法在多字符图像动画中实现了最先进的性能，超越了现有的基于扩散的基线。

</details>

---

## 6. Exploring Vision-Language Models for Open-Vocabulary Zero-Shot Action Segmentation / 探索开放词汇零样本动作分割的视觉语言模型

**Date**: 2026-02-24 | **arXiv**: [2602.21406v1](http://arxiv.org/abs/2602.21406v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.21406v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Temporal Action Segmentation (TAS) requires dividing videos into action segments, yet the vast space of activities and alternative breakdowns makes collecting comprehensive datasets infeasible. Existing methods remain limited to closed vocabularies and fixed label sets. In this work, we explore the largely unexplored problem of Open-Vocabulary Zero-Shot Temporal Action Segmentation (OVTAS) by leveraging the strong zero-shot capabilities of Vision-Language Models (VLMs). We introduce a training-free pipeline that follows a segmentation-by-classification design: Frame-Action Embedding Similarity (FAES) matches video frames to candidate action labels, and Similarity-Matrix Temporal Segmentation (SMTS) enforces temporal consistency. Beyond proposing OVTAS, we present a systematic study across 14 diverse VLMs, providing the first broad analysis of their suitability for open-vocabulary action segmentation. Experiments on standard benchmarks show that OVTAS achieves strong results without task-specific supervision, underscoring the potential of VLMs for structured temporal understanding.

时间动作分割（TAS）需要将视频划分为动作片段，但巨大的活动空间和替代细分使得收集全面的数据集变得不可行。现有方法仍然仅限于封闭词汇表和固定标签集。在这项工作中，我们通过利用视觉语言模型（VLM）强大的零样本能力来探索开放词汇零样本时间动作分割（OVTAS）的很大程度上未被探索的问题。我们引入了一种遵循按分类分割设计的免训练管道：帧动作嵌入相似性（FAES）将视频帧与候选动作标签相匹配，相似性矩阵时间分割（SMTS）强制时间一致性。除了提出 OVTAS 之外，我们还对 14 个不同的 VLM 进行了系统研究，首次对其开放词汇动作分割的适用性进行了广泛分析。标准基准测试的实验表明，OVTAS 在没有特定于任务的监督的情况下取得了很好的结果，强调了 VLM 在结构化时间理解方面的潜力。

</details>

---

## 7. Towards Controllable Video Synthesis of Routine and Rare OR Events / 实现常规和罕见手术事件的可控视频合成

**Date**: 2026-02-24 | **arXiv**: [2602.21365v1](http://arxiv.org/abs/2602.21365v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.21365v1)

**Categories**: cs.CV, cs.AI, cs.LG, eess.IV

<details><summary><b>Abstract / 摘要</b></summary>

Purpose: Curating large-scale datasets of operating room (OR) workflow, encompassing rare, safety-critical, or atypical events, remains operationally and ethically challenging. This data bottleneck complicates the development of ambient intelligence for detecting, understanding, and mitigating rare or safety-critical events in the OR.   Methods: This work presents an OR video diffusion framework that enables controlled synthesis of rare and safety-critical events. The framework integrates a geometric abstraction module, a conditioning module, and a fine-tuned diffusion model to first transform OR scenes into abstract geometric representations, then condition the synthesis process, and finally generate realistic OR event videos. Using this framework, we also curate a synthetic dataset to train and validate AI models for detecting near-misses of sterile-field violations.   Results: In synthesizing routine OR events, our method outperforms off-the-shelf video diffusion baselines, achieving lower FVD/LPIPS and higher SSIM/PSNR in both in- and out-of-domain datasets. Through qualitative results, we illustrate its ability for controlled video synthesis of counterfactual events. An AI model trained and validated on the generated synthetic data achieved a RECALL of 70.13% in detecting near safety-critical events. Finally, we conduct an ablation study to quantify performance gains from key design choices.   Conclusion: Our solution enables controlled synthesis of routine and rare OR events from abstract geometric representations. Beyond demonstrating its capability to generate rare and safety-critical scenarios, we show its potential to support the development of ambient intelligence models.

目的：整理包含罕见、安全关键或非典型事件的手术室 (OR) 工作流程的大规模数据集，在操作和道德上仍然具有挑战性。这一数据瓶颈使得用于检测、理解和缓解手术室中罕见或安全关键事件的环境智能的开发变得复杂。   方法：这项工作提出了一个 OR 视频扩散框架，可以控制罕见和安全关键事件的合成。该框架集成了几何抽象模块、调节模块和微调扩散模型，首先将 OR 场景转换为抽象几何表示，然后调节合成过程，最后生成逼真的 OR 事件视频。使用这个框架，我们还整理了一个合成数据集来训练和验证人工智能模型，以检测无菌区违规事件的险情。   结果：在合成常规 OR 事件时，我们的方法优于现成的视频扩散基线，在域内和域外数据集中实现了较低的 FVD/LPIPS 和较高的 SSIM/PSNR。通过定性结果，我们说明了其对反事实事件进行受控视频合成的能力。根据生成的合成数据进行训练和验证的 AI 模型在检测临近安全关键事件时实现了 70.13% 的召回率。最后，我们进行了一项消融研究，以量化关键设计选择带来的性能增益。   结论：我们的解决方案能够从抽象几何表示中控制合成常规和罕见的 OR 事件。除了展示其生成罕见和安全关键场景的能力之外，我们还展示了其支持环境智能模型开发的潜力。

</details>

---

## 8. HorizonForge: Driving Scene Editing with Any Trajectories and Any Vehicles / Horizo​​nForge：使用任何轨迹和任何车辆进行驾驶场景编辑

**Date**: 2026-02-24 | **arXiv**: [2602.21333v1](http://arxiv.org/abs/2602.21333v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.21333v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Controllable driving scene generation is critical for realistic and scalable autonomous driving simulation, yet existing approaches struggle to jointly achieve photorealism and precise control. We introduce HorizonForge, a unified framework that reconstructs scenes as editable Gaussian Splats and Meshes, enabling fine-grained 3D manipulation and language-driven vehicle insertion. Edits are rendered through a noise-aware video diffusion process that enforces spatial and temporal consistency, producing diverse scene variations in a single feed-forward pass without per-trajectory optimization. To standardize evaluation, we further propose HorizonSuite, a comprehensive benchmark spanning ego- and agent-level editing tasks such as trajectory modifications and object manipulation. Extensive experiments show that Gaussian-Mesh representation delivers substantially higher fidelity than alternative 3D representations, and that temporal priors from video diffusion are essential for coherent synthesis. Combining these findings, HorizonForge establishes a simple yet powerful paradigm for photorealistic, controllable driving simulation, achieving an 83.4% user-preference gain and a 25.19% FID improvement over the second best state-of-the-art method. Project page: https://horizonforge.github.io/ .

可控驾驶场景生成对于真实且可扩展的自动驾驶模拟至关重要，但现有方法很难同时实现照片真实感和精确控制。我们推出了 Horizo​​nForge，这是一个统一的框架，可将场景重建为可编辑的高斯图和网格，从而实现细粒度的 3D 操作和语言驱动的车辆插入。编辑是通过噪声感知视频扩散过程进行渲染的，该过程强制执行空间和时间一致性，在单个前馈通道中产生不同的场景变化，而无需每个轨迹优化。为了标准化评估，我们进一步提出了 Horizo​​nSuite，这是一个涵盖自我和代理级别编辑任务（例如轨迹修改和对象操作）的综合基准。大量实验表明，高斯网格表示比其他 3D 表示具有更高的保真度，并且视频扩散的时间先验对于相干合成至关重要。结合这些发现，Horizo​​nForge 建立了一个简单而强大的范例，用于逼真、可控的驾驶模拟，与第二最佳的最先进方法相比，实现了 83.4% 的用户偏好增益和 25.19% 的 FID 改进。项目页面：https://horizo​​nforge.github.io/。

</details>

---

## 9. Human Video Generation from a Single Image with 3D Pose and View Control / 通过 3D 姿势和视图控制从单个图像生成人体视频

**Date**: 2026-02-24 | **arXiv**: [2602.21188v1](http://arxiv.org/abs/2602.21188v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.21188v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Recent diffusion methods have made significant progress in generating videos from single images due to their powerful visual generation capabilities. However, challenges persist in image-to-video synthesis, particularly in human video generation, where inferring view-consistent, motion-dependent clothing wrinkles from a single image remains a formidable problem. In this paper, we present Human Video Generation in 4D (HVG), a latent video diffusion model capable of generating high-quality, multi-view, spatiotemporally coherent human videos from a single image with 3D pose and view control. HVG achieves this through three key designs: (i) Articulated Pose Modulation, which captures the anatomical relationships of 3D joints via a novel dual-dimensional bone map and resolves self-occlusions across views by introducing 3D information; (ii) View and Temporal Alignment, which ensures multi-view consistency and alignment between a reference image and pose sequences for frame-to-frame stability; and (iii) Progressive Spatio-Temporal Sampling with temporal alignment to maintain smooth transitions in long multi-view animations. Extensive experiments on image-to-video tasks demonstrate that HVG outperforms existing methods in generating high-quality 4D human videos from diverse human images and pose inputs.

最近的扩散方法由于其强大的视觉生成能力，在从单个图像生成视频方面取得了重大进展。然而，图像到视频的合成仍然存在挑战，特别是在人类视频生成方面，从单个图像推断视图一致、运动相关的衣服皱纹仍然是一个艰巨的问题。在本文中，我们提出了 4D 人类视频生成 (HVG)，这是一种潜在视频扩散模型，能够从具有 3D 姿势和视图控制的单个图像生成高质量、多视图、时空连贯的人类视频。 HVG 通过三个关键设计实现了这一目标：(i) 关节姿势调制，通过新颖的二维骨图捕获 3D 关节的解剖关系，并通过引入 3D 信息解决跨视图的自遮挡问题； (ii) 视图和时间对齐，确保参考图像和姿势序列之间的多视图一致性和对齐，以实现帧到帧的稳定性； (iii) 渐进式时空采样与时间对齐，以保持长多视图动画中的平滑过渡。对图像到视频任务的大量实验表明，HVG 在从不同的人类图像和姿势输入生成高质量 4D 人类视频方面优于现有方法。

</details>

---

## 10. UDVideoQA: A Traffic Video Question Answering Dataset for Multi-Object Spatio-Temporal Reasoning in Urban Dynamics / UDVideoQA：用于城市动力学中多对象时空推理的交通视频问答数据集

**Date**: 2026-02-24 | **arXiv**: [2602.21137v1](http://arxiv.org/abs/2602.21137v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.21137v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Understanding the complex, multi-agent dynamics of urban traffic remains a fundamental challenge for video language models. This paper introduces Urban Dynamics VideoQA, a benchmark dataset that captures the unscripted real-world behavior of dynamic urban scenes. UDVideoQA is curated from 16 hours of traffic footage recorded at multiple city intersections under diverse traffic, weather, and lighting conditions. It employs an event-driven dynamic blur technique to ensure privacy preservation without compromising scene fidelity. Using a unified annotation pipeline, the dataset contains 28K question-answer pairs generated across 8 hours of densely annotated video, averaging one question per second. Its taxonomy follows a hierarchical reasoning level, spanning basic understanding and attribution to event reasoning, reverse reasoning, and counterfactual inference, enabling systematic evaluation of both visual grounding and causal reasoning. Comprehensive experiments benchmark 10 SOTA VideoLMs on UDVideoQA and 8 models on a complementary video question generation benchmark. Results reveal a persistent perception-reasoning gap, showing models that excel in abstract inference often fail with fundamental visual grounding. While models like Gemini Pro achieve the highest zero-shot accuracy, fine-tuning the smaller Qwen2.5-VL 7B model on UDVideoQA bridges this gap, achieving performance comparable to proprietary systems. In VideoQGen, Gemini 2.5 Pro, and Qwen3 Max generate the most relevant and complex questions, though all models exhibit limited linguistic diversity, underscoring the need for human-centric evaluation. The UDVideoQA suite, including the dataset, annotation tools, and benchmarks for both VideoQA and VideoQGen, provides a foundation for advancing robust, privacy-aware, and real-world multimodal reasoning. UDVideoQA is available at https://ud-videoqa.github.io/UD-VideoQA/UD-VideoQA/.

了解城市交通复杂的多智能体动态仍然是视频语言模型的基本挑战。本文介绍了 Urban Dynamics VideoQA，这是一个基准数据集，可捕获动态城市场景的无脚本现实行为。 UDVideoQA 根据在不同交通、天气和照明条件下在多个城市十字路口录制的 16 小时交通录像进行整理。它采用事件驱动的动态模糊技术来确保隐私保护，同时又不影响场景保真度。使用统一的注释管道，该数据集包含在 8 小时的密集注释视频中生成的 28K 问答对，平均每秒一个问题。其分类遵循分层推理水平，涵盖对事件推理、逆向推理和反事实推理的基本理解和归因，从而能够对视觉基础和因果推理进行系统评估。综合实验在 UDVideoQA 上对 10 个 SOTA VideoLM 进行基准测试，在补充视频问题生成基准上对 8 个模型进行基准测试。结果揭示了持续存在的感知推理差距，表明擅长抽象推理的模型往往在基本视觉基础上失败。虽然 Gemini Pro 等模型实现了最高的零射击精度，但在 UDVideoQA 上微调较小的 Qwen2.5-VL 7B 模型弥补了这一差距，实现了与专有系统相当的性能。在 VideoQGen、Gemini 2.5 Pro 和 Qwen3 Max 中，尽管所有模型都表现出有限的语言多样性，但生成了最相关和最复杂的问题，这强调了以人为中心的评估的必要性。 UDVideoQA 套件包括数据集、注释工具以及 VideoQA 和 VideoQGen 的基准，为推进稳健、隐私意识和现实世界的多模态推理奠定了基础。 UDVideoQA 位于 https://ud-videoqa.github.io/UD-VideoQA/UD-VideoQA/。

</details>

---

## 11. RAYNOVA: Scale-Temporal Autoregressive World Modeling in Ray Space / RAYNOVA：射线空间中的尺度时间自回归世界建模

**Date**: 2026-02-24 | **arXiv**: [2602.20685v2](http://arxiv.org/abs/2602.20685v2) | **PDF**: [Link](http://arxiv.org/pdf/2602.20685v2)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

World foundation models aim to simulate the evolution of the real world with physically plausible behavior. Unlike prior methods that handle spatial and temporal correlations separately, we propose RAYNOVA, a geometry-agonistic multiview world model for driving scenarios that employs a dual-causal autoregressive framework. It follows both scale-wise and temporal topological orders in the autoregressive process, and leverages global attention for unified 4D spatio-temporal reasoning. Different from existing works that impose strong 3D geometric priors, RAYNOVA constructs an isotropic spatio-temporal representation across views, frames, and scales based on relative Plücker-ray positional encoding, enabling robust generalization to diverse camera setups and ego motions. We further introduce a recurrent training paradigm to alleviate distribution drift in long-horizon video generation. RAYNOVA achieves state-of-the-art multi-view video generation results on nuScenes, while offering higher throughput and strong controllability under diverse input conditions, generalizing to novel views and camera configurations without explicit 3D scene representation. Our code will be released at https://raynova-ai.github.io/.

世界基础模型旨在通过物理上合理的行为来模拟现实世界的演化。与之前分别处理空间和时间相关性的方法不同，我们提出了 RAYNOVA，这是一种采用双因果自回归框架的驾驶场景的几何对抗多视图世界模型。它在自回归过程中遵循尺度和时间拓扑顺序，并利用全局注意力进行统一的 4D 时空推理。与强加强 3D 几何先验的现有作品不同，RAYNOVA 基于相对 Plücker 射线位置编码构建了跨视图、帧和尺度的各向同性时空表示，从而能够对不同的相机设置和自我运动进行稳健的泛化。我们进一步引入了一种循环训练范例，以减轻长视野视频生成中的分布漂移。 RAYNOVA 在 nuScenes 上实现了最先进的多视图视频生成结果，同时在不同的输入条件下提供更高的吞吐量和强大的可控性，推广到新颖的视图和相机配置，而无需明确的 3D 场景表示。我们的代码将在 https://raynova-ai.github.io/ 发布。

</details>

---



</details>

<details><summary><b>2026-02-25 (12 papers)</b></summary>

# arXiv Video Papers - 2026-02-25

**Paper Count**: 12

---

## 1. VII: Visual Instruction Injection for Jailbreaking Image-to-Video Generation Models / VII：越狱图像到视频生成模型的视觉指令注入

**Date**: 2026-02-24 | **arXiv**: [2602.20999v1](http://arxiv.org/abs/2602.20999v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.20999v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Image-to-Video (I2V) generation models, which condition video generation on reference images, have shown emerging visual instruction-following capability, allowing certain visual cues in reference images to act as implicit control signals for video generation. However, this capability also introduces a previously overlooked risk: adversaries may exploit visual instructions to inject malicious intent through the image modality. In this work, we uncover this risk by proposing Visual Instruction Injection (VII), a training-free and transferable jailbreaking framework that intentionally disguises the malicious intent of unsafe text prompts as benign visual instructions in the safe reference image. Specifically, VII coordinates a Malicious Intent Reprogramming module to distill malicious intent from unsafe text prompts while minimizing their static harmfulness, and a Visual Instruction Grounding module to ground the distilled intent onto a safe input image by rendering visual instructions that preserve semantic consistency with the original unsafe text prompt, thereby inducing harmful content during I2V generation. Empirically, our extensive experiments on four state-of-the-art commercial I2V models (Kling-v2.5-turbo, Gemini Veo-3.1, Seedance-1.5-pro, and PixVerse-V5) demonstrate that VII achieves Attack Success Rates of up to 83.5% while reducing Refusal Rates to near zero, significantly outperforming existing baselines.

图像到视频 (I2V) 生成模型在参考图像上调节视频生成，已显示出新兴的视觉指令跟踪功能，允许参考图像中的某些视觉提示充当视频生成的隐式控制信号。然而，这种功能也带来了一个以前被忽视的风险：对手可能会利用视觉指令通过图像模态注入恶意意图。在这项工作中，我们通过提出视觉指令注入（VII）来揭示这一风险，这是一种免训练且可转移的越狱框架，有意将不安全文本提示的恶意意图伪装成安全参考图像中的良性视觉指令。具体来说，VII 协调恶意意图重新编程模块，从不安全文本提示中提取恶意意图，同时最大限度地减少其静态危害性，并协调视觉指令接地模块，通过渲染与原始不安全文本提示保持语义一致性的视觉指令，将提取的意图接地到安全输入图像上，从而在 I2V 生成过程中引入有害内容。根据经验，我们对四种最先进的商业 I2V 模型（Kling-v2.5-turbo、Gemini Veo-3.1、Seedance-1.5-pro 和 PixVerse-V5）进行的广泛实验表明，VII 的攻击成功率高达 83.5%，同时将拒绝率降低到接近零，显着优于现有基线。

</details>

---

## 2. LongVideo-R1: Smart Navigation for Low-cost Long Video Understanding / LongVideo-R1：低成本长视频理解的智能导航

**Date**: 2026-02-24 | **arXiv**: [2602.20913v1](http://arxiv.org/abs/2602.20913v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.20913v1)

**Categories**: cs.CV

**Code**: https://github.com/qiujihao19/LongVideo-R1

<details><summary><b>Abstract / 摘要</b></summary>

This paper addresses the critical and underexplored challenge of long video understanding with low computational budgets. We propose LongVideo-R1, an active, reasoning-equipped multimodal large language model (MLLM) agent designed for efficient video context navigation, avoiding the redundancy of exhaustive search. At the core of LongVideo-R1 lies a reasoning module that leverages high-level visual cues to infer the most informative video clip for subsequent processing. During inference, the agent initiates traversal from top-level visual summaries and iteratively refines its focus, immediately halting the exploration process upon acquiring sufficient knowledge to answer the query. To facilitate training, we first extract hierarchical video captions from CGBench, a video corpus with grounding annotations, and guide GPT-5 to generate 33K high-quality chain-of-thought-with-tool trajectories. The LongVideo-R1 agent is fine-tuned upon the Qwen-3-8B model through a two-stage paradigm: supervised fine-tuning (SFT) followed by reinforcement learning (RL), where RL employs a specifically designed reward function to maximize selective and efficient clip navigation. Experiments on multiple long video benchmarks validate the effectiveness of name, which enjoys superior tradeoff between QA accuracy and efficiency. All curated data and source code are provided in the supplementary material and will be made publicly available. Code and data are available at: https://github.com/qiujihao19/LongVideo-R1

本文解决了低计算预算下长视频理解的关键且尚未充分探索的挑战。我们提出了 LongVideo-R1，一种主动的、配备推理的多模态大语言模型（MLLM）代理，设计用于高效的视频上下文导航，避免详尽搜索的冗余。 LongVideo-R1 的核心是一个推理模块，它利用高级视觉线索来推断信息最丰富的视频剪辑，以供后续处理。在推理过程中，代理从顶级视觉摘要开始遍历，并迭代地细化其焦点，在获得足够的知识来回答查询后立即停止探索过程。为了便于训练，我们首先从带有基础注释的视频语料库 CGBench 中提取分层视频字幕，并引导 GPT-5 生成 33K 高质量的思想链工具轨迹。 LongVideo-R1 代理通过两阶段范式在 Qwen-3-8B 模型上进行微调：监督微调 (SFT)，然后是强化学习 (RL)，其中 RL 采用专门设计的奖励函数来最大限度地提高选择性和高效的剪辑导航。在多个长视频基准上进行的实验验证了 name 的有效性，它在 QA 准确性和效率之间享有卓越的权衡。所有精选数据和源代码均在补充材料中提供，并将公开发布。代码和数据可参见：https://github.com/qiujihao19/LongVideo-R1

</details>

---

## 3. PyVision-RL: Forging Open Agentic Vision Models via RL / PyVision-RL：通过 RL 打造开放代理视觉模型

**Date**: 2026-02-24 | **arXiv**: [2602.20739v1](http://arxiv.org/abs/2602.20739v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.20739v1)

**Categories**: cs.AI, cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Reinforcement learning for agentic multimodal models often suffers from interaction collapse, where models learn to reduce tool usage and multi-turn reasoning, limiting the benefits of agentic behavior. We introduce PyVision-RL, a reinforcement learning framework for open-weight multimodal models that stabilizes training and sustains interaction. Our approach combines an oversampling-filtering-ranking rollout strategy with an accumulative tool reward to prevent collapse and encourage multi-turn tool use. Using a unified training pipeline, we develop PyVision-Image and PyVision-Video for image and video understanding. For video reasoning, PyVision-Video employs on-demand context construction, selectively sampling task-relevant frames during reasoning to significantly reduce visual token usage. Experiments show strong performance and improved efficiency, demonstrating that sustained interaction and on-demand visual processing are critical for scalable multimodal agents.

代理多模态模型的强化学习经常会遇到交互崩溃的问题，其中模型学习减少工具使用和多轮推理，从而限制了代理行为的好处。我们引入了 PyVision-RL，这是一种用于开放权重多模态模型的强化学习框架，可以稳定训练并维持交互。我们的方法将过采样-过滤-排名推出策略与累积工具奖励相结合，以防止崩溃并鼓励多回合工具的使用。使用统一的训练管道，我们开发了 PyVision-Image 和 PyVision-Video 用于图像和视频理解。对于视频推理，PyVision-Video 采用按需上下文构建，在推理过程中选择性地采样与任务相关的帧，以显着减少视觉标记的使用。实验显示出强大的性能和更高的效率，证明持续交互和按需视觉处理对于可扩展的多模式代理至关重要。

</details>

---

## 4. RAYNOVA: 3D-Geometry-Free Auto-Regressive Driving World Modeling with Unified Spatio-Temporal Representation / RAYNOVA：具有统一时空表示的无 3D 几何自动回归驾驶世界建模

**Date**: 2026-02-24 | **arXiv**: [2602.20685v1](http://arxiv.org/abs/2602.20685v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.20685v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

World foundation models aim to simulate the evolution of the real world with physically plausible behavior. Unlike prior methods that handle spatial and temporal correlations separately, we propose RAYNOVA, a geometry-free world model that employs a dual-causal autoregressive framework. It follows both scale-wise and temporal topological orders in the autoregressive process, and leverages global attention for unified 4D spatio-temporal reasoning. Different from existing works that impose strong 3D geometric priors, RAYNOVA constructs an isotropic spatio-temporal representation across views, frames, and scales based on relative Plücker-ray positional encoding, enabling robust generalization to diverse camera setups and ego motions. We further introduce a recurrent training paradigm to alleviate distribution drift in long-horizon video generation. RAYNOVA achieves state-of-the-art multi-view video generation results on nuScenes, while offering higher throughput and strong controllability under diverse input conditions, generalizing to novel views and camera configurations without explicit 3D scene representation. Our code will be released at http://yichen928.github.io/raynova.

世界基础模型旨在通过物理上合理的行为来模拟现实世界的演化。与之前分别处理空间和时间相关性的方法不同，我们提出了 RAYNOVA，一种采用双因果自回归框架的无几何世界模型。它在自回归过程中遵循尺度和时间拓扑顺序，并利用全局注意力进行统一的 4D 时空推理。与强加强 3D 几何先验的现有作品不同，RAYNOVA 基于相对 Plücker 射线位置编码构建了跨视图、帧和尺度的各向同性时空表示，从而能够对不同的相机设置和自我运动进行稳健的泛化。我们进一步引入了一种循环训练范例，以减轻长视野视频生成中的分布漂移。 RAYNOVA 在 nuScenes 上实现了最先进的多视图视频生成结果，同时在不同的输入条件下提供更高的吞吐量和强大的可控性，推广到新颖的视图和相机配置，而无需明确的 3D 场景表示。我们的代码将在http://yichen928.github.io/raynova发布。

</details>

---

## 5. GA-Drive: Geometry-Appearance Decoupled Modeling for Free-viewpoint Driving Scene Generatio / GA-Drive：用于自由视点驾驶场景生成的几何外观解耦建模

**Date**: 2026-02-24 | **arXiv**: [2602.20673v1](http://arxiv.org/abs/2602.20673v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.20673v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

A free-viewpoint, editable, and high-fidelity driving simulator is crucial for training and evaluating end-to-end autonomous driving systems. In this paper, we present GA-Drive, a novel simulation framework capable of generating camera views along user-specified novel trajectories through Geometry-Appearance Decoupling and Diffusion-Based Generation. Given a set of images captured along a recorded trajectory and the corresponding scene geometry, GA-Drive synthesizes novel pseudo-views using geometry information. These pseudo-views are then transformed into photorealistic views using a trained video diffusion model. In this way, we decouple the geometry and appearance of scenes. An advantage of such decoupling is its support for appearance editing via state-of-the-art video-to-video editing techniques, while preserving the underlying geometry, enabling consistent edits across both original and novel trajectories. Extensive experiments demonstrate that GA-Drive substantially outperforms existing methods in terms of NTA-IoU, NTL-IoU, and FID scores.

自由视角、可编辑且高保真的驾驶模拟器对于训练和评估端到端自动驾驶系统至关重要。在本文中，我们提出了 GA-Drive，这是一种新颖的模拟框架，能够通过几何外观解耦和基于扩散的生成沿着用户指定的新颖轨迹生成相机视图。给定沿着记录轨迹捕获的一组图像和相应的场景几何形状，GA-Drive 使用几何信息合成新颖的伪视图。然后使用经过训练的视频扩散模型将这些伪视图转换为逼真的视图。通过这种方式，我们将场景的几何形状和外观解耦。这种解耦的优点是它支持通过最先进的视频到视频编辑技术进行外观编辑，同时保留底层几何形状，从而能够在原始和新颖的轨迹上进行一致的编辑。大量实验表明，GA-Drive 在 NTA-IoU、NTL-IoU 和 FID 分数方面明显优于现有方法。

</details>

---

## 6. AnimeAgent: Is the Multi-Agent via Image-to-Video models a Good Disney Storytelling Artist? / AnimeAgent：通过图像到视频模型的多代理是优秀的迪士尼讲故事艺术家吗？

**Date**: 2026-02-24 | **arXiv**: [2602.20664v1](http://arxiv.org/abs/2602.20664v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.20664v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Custom Storyboard Generation (CSG) aims to produce high-quality, multi-character consistent storytelling. Current approaches based on static diffusion models, whether used in a one-shot manner or within multi-agent frameworks, face three key limitations: (1) Static models lack dynamic expressiveness and often resort to "copy-paste" pattern. (2) One-shot inference cannot iteratively correct missing attributes or poor prompt adherence. (3) Multi-agents rely on non-robust evaluators, ill-suited for assessing stylized, non-realistic animation. To address these, we propose AnimeAgent, the first Image-to-Video (I2V)-based multi-agent framework for CSG. Inspired by Disney's "Combination of Straight Ahead and Pose to Pose" workflow, AnimeAgent leverages I2V's implicit motion prior to enhance consistency and expressiveness, while a mixed subjective-objective reviewer enables reliable iterative refinement. We also collect a human-annotated CSG benchmark with ground-truth. Experiments show AnimeAgent achieves SOTA performance in consistency, prompt fidelity, and stylization.

自定义故事板生成 (CSG) 旨在制作高质量、多角色一致的故事讲述。当前基于静态扩散模型的方法，无论是一次性使用还是在多智能体框架内使用，都面临三个关键限制：（1）静态模型缺乏动态表达能力，并且经常采用“复制粘贴”模式。 (2) 一次性推理无法迭代地纠正缺失的属性或提示依从性差。 (3) 多智能体依赖于非鲁棒评估器，不适合评估风格化、非现实的动画。为了解决这些问题，我们提出了 AnimeAgent，这是第一个基于图像到视频 (I2V) 的 CSG 多代理框架。受迪士尼“直线前进和姿势到姿势的组合”工作流程的启发，AnimeAgent 在增强一致性和表现力之前利用 I2V 的隐式运动，而混合主客观审阅器可实现可靠的迭代细化。我们还收集了具有真实性的人工注释 CSG 基准。实验表明 AnimeAgent 在一致性、提示保真度和风格化方面实现了 SOTA 性能。

</details>

---

## 7. PropFly: Learning to Propagate via On-the-Fly Supervision from Pre-trained Video Diffusion Models / PropFly：通过预先训练的视频扩散模型进行动态监督来学习传播

**Date**: 2026-02-24 | **arXiv**: [2602.20583v1](http://arxiv.org/abs/2602.20583v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.20583v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Propagation-based video editing enables precise user control by propagating a single edited frame into following frames while maintaining the original context such as motion and structures. However, training such models requires large-scale, paired (source and edited) video datasets, which are costly and complex to acquire. Hence, we propose the PropFly, a training pipeline for Propagation-based video editing, relying on on-the-Fly supervision from pre-trained video diffusion models (VDMs) instead of requiring off-the-shelf or precomputed paired video editing datasets. Specifically, our PropFly leverages one-step clean latent estimations from intermediate noised latents with varying Classifier-Free Guidance (CFG) scales to synthesize diverse pairs of 'source' (low-CFG) and 'edited' (high-CFG) latents on-the-fly. The source latent serves as structural information of the video, while the edited latent provides the target transformation for learning propagation. Our pipeline enables an additional adapter attached to the pre-trained VDM to learn to propagate edits via Guidance-Modulated Flow Matching (GMFM) loss, which guides the model to replicate the target transformation. Our on-the-fly supervision ensures the model to learn temporally consistent and dynamic transformations. Extensive experiments demonstrate that our PropFly significantly outperforms the state-of-the-art methods on various video editing tasks, producing high-quality editing results.

基于传播的视频编辑通过将单个编辑帧传播到后续帧中来实现精确的用户控制，同时保持原始上下文（例如运动和结构）。然而，训练此类模型需要大规模、配对（源和编辑）视频数据集，获取这些数据集成本高昂且复杂。因此，我们提出了 PropFly，一种基于传播的视频编辑的训练管道，依赖于预先训练的视频扩散模型 (VDM) 的即时监督，而不需要现成的或预先计算的配对视频编辑数据集。具体来说，我们的 PropFly 利用具有不同无分类器指导 (CFG) 尺度的中间噪声潜伏的一步干净潜伏估计来动态合成不同的“源”（低 CFG）和“编辑”（高 CFG）潜伏对。源潜在变量充当视频的结构信息，而编辑后的潜在变量提供学习传播的目标转换。我们的管道支持附加到预先训练的 VDM 的附加适配器，以学习通过引导调制流匹配 (GMFM) 损失来传播编辑，从而引导模型复制目标转换。我们的动态监督确保模型能够学习时间一致的动态转换。大量实验表明，我们的 PropFly 在各种视频编辑任务上显着优于最先进的方法，产生高质量的编辑结果。

</details>

---

## 8. LESA: Learnable Stage-Aware Predictors for Diffusion Model Acceleration / LESA：用于扩散模型加速的可学习阶段感知预测器

**Date**: 2026-02-24 | **arXiv**: [2602.20497v1](http://arxiv.org/abs/2602.20497v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.20497v1)

**Categories**: cs.CV, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Diffusion models have achieved remarkable success in image and video generation tasks. However, the high computational demands of Diffusion Transformers (DiTs) pose a significant challenge to their practical deployment. While feature caching is a promising acceleration strategy, existing methods based on simple reusing or training-free forecasting struggle to adapt to the complex, stage-dependent dynamics of the diffusion process, often resulting in quality degradation and failing to maintain consistency with the standard denoising process. To address this, we propose a LEarnable Stage-Aware (LESA) predictor framework based on two-stage training. Our approach leverages a Kolmogorov-Arnold Network (KAN) to accurately learn temporal feature mappings from data. We further introduce a multi-stage, multi-expert architecture that assigns specialized predictors to different noise-level stages, enabling more precise and robust feature forecasting. Extensive experiments show our method achieves significant acceleration while maintaining high-fidelity generation. Experiments demonstrate 5.00x acceleration on FLUX.1-dev with minimal quality degradation (1.0% drop), 6.25x speedup on Qwen-Image with a 20.2% quality improvement over the previous SOTA (TaylorSeer), and 5.00x acceleration on HunyuanVideo with a 24.7% PSNR improvement over TaylorSeer. State-of-the-art performance on both text-to-image and text-to-video synthesis validates the effectiveness and generalization capability of our training-based framework across different models. Our code is included in the supplementary materials and will be released on GitHub.

扩散模型在图像和视频生成任务中取得了显着的成功。然而，扩散变压器（DiT）的高计算要求对其实际部署提出了重大挑战。虽然特征缓存是一种很有前途的加速策略，但基于简单重用或免训练预测的现有方法很难适应扩散过程的复杂、阶段相关的动态，通常会导致质量下降，并且无法保持与标准去噪过程的一致性。为了解决这个问题，我们提出了一个基于两阶段训练的 LEarnable Stage-Aware (LESA) 预测器框架。我们的方法利用柯尔莫哥洛夫-阿诺德网络（KAN）来准确地从数据中学习时间特征映射。我们进一步引入了多阶段、多专家架构，将专门的预测器分配给不同的噪声级别阶段，从而实现更精确和稳健的特征预测。大量的实验表明，我们的方法在保持高保真生成的同时实现了显着的加速。实验表明，FLUX.1-dev 上的加速为 5.00 倍，质量下降最小（下降 1.0%）；Qwen-Image 上的加速为 6.25 倍，与之前的 SOTA（TaylorSeer）相比，质量提高了 20.2%；HunyuanVideo 上的加速为 5.00 倍，PSNR 比 TaylorSeer 提高了 24.7%。文本到图像和文本到视频合成的最先进性能验证了我们基于训练的框架在不同模型上的有效性和泛化能力。我们的代码包含在补充材料中，并将在 GitHub 上发布。

</details>

---

## 9. 3DSPA: A 3D Semantic Point Autoencoder for Evaluating Video Realism / 3DSPA：用于评估视频真实感的 3D 语义点自动编码器

**Date**: 2026-02-23 | **arXiv**: [2602.20354v1](http://arxiv.org/abs/2602.20354v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.20354v1)

**Categories**: cs.CV

**Code**: https://github.com/TheProParadox/3dspa_code.

<details><summary><b>Abstract / 摘要</b></summary>

AI video generation is evolving rapidly. For video generators to be useful for applications ranging from robotics to film-making, they must consistently produce realistic videos. However, evaluating the realism of generated videos remains a largely manual process -- requiring human annotation or bespoke evaluation datasets which have restricted scope. Here we develop an automated evaluation framework for video realism which captures both semantics and coherent 3D structure and which does not require access to a reference video. Our method, 3DSPA, is a 3D spatiotemporal point autoencoder which integrates 3D point trajectories, depth cues, and DINO semantic features into a unified representation for video evaluation. 3DSPA models how objects move and what is happening in the scene, enabling robust assessments of realism, temporal consistency, and physical plausibility. Experiments show that 3DSPA reliably identifies videos which violate physical laws, is more sensitive to motion artifacts, and aligns more closely with human judgments of video quality and realism across multiple datasets. Our results demonstrate that enriching trajectory-based representations with 3D semantics offers a stronger foundation for benchmarking generative video models, and implicitly captures physical rule violations. The code and pretrained model weights will be available at https://github.com/TheProParadox/3dspa_code.

人工智能视频生成正在迅速发展。为了使视频生成器能够用于从机器人到电影制作等各种应用，它们必须始终如一地生成逼真的视频。然而，评估生成视频的真实感仍然是一个很大程度上手动的过程——需要人工注释或范围有限的定制评估数据集。在这里，我们开发了一个视频真实感自动评估框架，它可以捕获语义和连贯的 3D 结构，并且不需要访问参考视频。我们的方法 3DSPA 是一种 3D 时空点自动编码器，它将 3D 点轨迹、深度线索和 DINO 语义特征集成到视频评估的统一表示中。 3DSPA 对物体如何移动以及场景中发生的情况进行建模，从而能够对真实性、时间一致性和物理合理性进行可靠的评估。实验表明，3DSPA 能够可靠地识别违反物理定律的视频，对运动伪影更敏感，并且更符合人类对多个数据集的视频质量和真实感的判断。我们的结果表明，利用 3D 语义丰富基于轨迹的表示为基准生成视频模型提供了更坚实的基础，并隐式捕获物理规则违规行为。代码和预训练模型权重将在 https://github.com/TheProParadox/3dspa_code 上提供。

</details>

---

## 10. NovaPlan: Zero-Shot Long-Horizon Manipulation via Closed-Loop Video Language Planning / NovaPlan：通过闭环视频语言规划进行零射击长视野操作

**Date**: 2026-02-23 | **arXiv**: [2602.20119v1](http://arxiv.org/abs/2602.20119v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.20119v1)

**Categories**: cs.RO, cs.AI, cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Solving long-horizon tasks requires robots to integrate high-level semantic reasoning with low-level physical interaction. While vision-language models (VLMs) and video generation models can decompose tasks and imagine outcomes, they often lack the physical grounding necessary for real-world execution. We introduce NovaPlan, a hierarchical framework that unifies closed-loop VLM and video planning with geometrically grounded robot execution for zero-shot long-horizon manipulation. At the high level, a VLM planner decomposes tasks into sub-goals and monitors robot execution in a closed loop, enabling the system to recover from single-step failures through autonomous re-planning. To compute low-level robot actions, we extract and utilize both task-relevant object keypoints and human hand poses as kinematic priors from the generated videos, and employ a switching mechanism to choose the better one as a reference for robot actions, maintaining stable execution even under heavy occlusion or depth inaccuracy. We demonstrate the effectiveness of NovaPlan on three long-horizon tasks and the Functional Manipulation Benchmark (FMB). Our results show that NovaPlan can perform complex assembly tasks and exhibit dexterous error recovery behaviors without any prior demonstrations or training. Project page: https://nova-plan.github.io/

解决长期任务需要机器人将高级语义推理与低级物理交互相结合。虽然视觉语言模型 (VLM) 和视频生成模型可以分解任务并想象结果，但它们通常缺乏现实世界执行所需的物理基础。我们引入了 NovaPlan，这是一个分层框架，它将闭环 VLM 和视频规划与几何接地机器人执行相结合，以实现零样本长视野操作。在高层，VLM 规划器将任务分解为子目标，并在闭环中监控机器人的执行情况，使系统能够通过自主重新规划从单步故障中恢复。为了计算低级机器人动作，我们从生成的视频中提取并利用与任务相关的对象关键点和人手姿势作为运动学先验，并采用切换机制来选择更好的动作作为机器人动作的参考，即使在严重遮挡或深度不准确的情况下也能保持稳定的执行。我们展示了 NovaPlan 在三项长期任务和功能操作基准（FMB）上的有效性。我们的结果表明，NovaPlan 可以执行复杂的组装任务并表现出灵巧的错误恢复行为，而无需任何事先演示或培训。项目页面：https://nova-plan.github.io/

</details>

---

## 11. BigMaQ: A Big Macaque Motion and Animation Dataset Bridging Image and 3D Pose Representations / BigMaQ：连接图像和 3D 姿势表示的大猕猴运动和动画数据集

**Date**: 2026-02-23 | **arXiv**: [2602.19874v1](http://arxiv.org/abs/2602.19874v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.19874v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

The recognition of dynamic and social behavior in animals is fundamental for advancing ethology, ecology, medicine and neuroscience. Recent progress in deep learning has enabled automated behavior recognition from video, yet an accurate reconstruction of the three-dimensional (3D) pose and shape has not been integrated into this process. Especially for non-human primates, mesh-based tracking efforts lag behind those for other species, leaving pose descriptions restricted to sparse keypoints that are unable to fully capture the richness of action dynamics. To address this gap, we introduce the $\textbf{Big Ma}$ca$\textbf{Q}$ue 3D Motion and Animation Dataset ($\texttt{BigMaQ}$), a large-scale dataset comprising more than 750 scenes of interacting rhesus macaques with detailed 3D pose descriptions. Extending previous surface-based animal tracking methods, we construct subject-specific textured avatars by adapting a high-quality macaque template mesh to individual monkeys. This allows us to provide pose descriptions that are more accurate than previous state-of-the-art surface-based animal tracking methods. From the original dataset, we derive BigMaQ500, an action recognition benchmark that links surface-based pose vectors to single frames across multiple individual monkeys. By pairing features extracted from established image and video encoders with and without our pose descriptors, we demonstrate substantial improvements in mean average precision (mAP) when pose information is included. With these contributions, $\texttt{BigMaQ}$ establishes the first dataset that both integrates dynamic 3D pose-shape representations into the learning task of animal action recognition and provides a rich resource to advance the study of visual appearance, posture, and social interaction in non-human primates. The code and data are publicly available at https://martinivis.github.io/BigMaQ/ .

对动物动态和社会行为的认识是推进行为学、生态学、医学和神经科学的基础。深度学习的最新进展已经实现了视频中的自动行为识别，但三维 (3D) 姿势和形状的准确重建尚未集成到此过程中。特别是对于非人类灵长类动物，基于网格的跟踪工作落后于其他物种，使得姿势描述仅限于稀疏的关键点，无法完全捕捉动作动态的丰富性。为了解决这一差距，我们引入了 $\textbf{Big Ma}$ca$\textbf{Q}$ue 3D 运动和动画数据集 ($\texttt{BigMaQ}$)，这是一个大型数据集，包含 750 多个恒河猴互动场景，并具有详细的 3D 姿势描述。扩展了之前基于表面的动物跟踪方法，我们通过将高质量的猕猴模板网格应用于个体猴子来构建特定于主题的纹理化身。这使我们能够提供比以前最先进的基于表面的动物跟踪方法更准确的姿势描述。从原始数据集中，我们推导出 BigMaQ500，这是一种动作识别基准，它将基于表面的姿势向量链接到多个个体猴子的单个帧。通过将从已建立的图像和视频编码器中提取的特征与姿势描述符进行配对，我们证明了在包含姿势信息时平均精度（mAP）的显着改进。通过这些贡献，$\texttt{BigMaQ}$ 建立了第一个数据集，该数据集既将动态 3D 姿势形状表示集成到动物动作识别的学习任务中，又为推进非人类灵长类动物的视觉外观、姿势和社交互动的研究提供了丰富的资源。代码和数据可在 https://martinivis.github.io/BigMaQ/ 上公开获取。

</details>

---

## 12. PedaCo-Gen: Scaffolding Pedagogical Agency in Human-AI Collaborative Video Authoring / PedaCo-Gen：人机协作视频创作中的脚手架教学机构

**Date**: 2026-02-23 | **arXiv**: [2602.19623v1](http://arxiv.org/abs/2602.19623v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.19623v1)

**Categories**: cs.CV, cs.AI, cs.HC

<details><summary><b>Abstract / 摘要</b></summary>

While advancements in Text-to-Video (T2V) generative AI offer a promising path toward democratizing content creation, current models are often optimized for visual fidelity rather than instructional efficacy. This study introduces PedaCo-Gen, a pedagogically-informed human-AI collaborative video generating system for authoring instructional videos based on Mayer's Cognitive Theory of Multimedia Learning (CTML). Moving away from traditional "one-shot" generation, PedaCo-Gen introduces an Intermediate Representation (IR) phase, enabling educators to interactively review and refine video blueprints-comprising scripts and visual descriptions-with an AI reviewer. Our study with 23 education experts demonstrates that PedaCo-Gen significantly enhances video quality across various topics and CTML principles compared to baselines. Participants perceived the AI-driven guidance not merely as a set of instructions but as a metacognitive scaffold that augmented their instructional design expertise, reporting high production efficiency (M=4.26) and guide validity (M=4.04). These findings highlight the importance of reclaiming pedagogical agency through principled co-creation, providing a foundation for future AI authoring tools that harmonize generative power with human professional expertise.

虽然文本到视频 (T2V) 生成式人工智能的进步为内容创作民主化提供了一条充满希望的道路，但当前的模型通常针对视觉保真度而不是教学效果进行优化。本研究介绍了 PedaCo-Gen，这是一种基于教学的人类与人工智能协作视频生成系统，用于根据 Mayer 的多媒体学习认知理论 (CTML) 创作教学视频。 PedaCo-Gen 摆脱了传统的“一次性”生成，引入了中间表示 (IR) 阶段，使教育工作者能够与人工智能审阅者交互地审阅和完善视频蓝图（包括脚本和视觉描述）。我们与 23 名教育专家进行的研究表明，与基线相比，PedaCo-Gen 显着提高了各种主题和 CTML 原则的视频质量。参与者认为人工智能驱动的指导不仅是一组指令，而且是一个元认知支架，可以增强他们的教学设计专业知识，报告高生产效率（M = 4.26）和指南有效性（M = 4.04）。这些发现强调了通过有原则的共同创造来恢复教学机构的重要性，为未来将生成能力与人类专业知识相协调的人工智能创作工具奠定了基础。

</details>

---



</details>

<details><summary><b>2026-02-24 (8 papers)</b></summary>

# arXiv Video Papers - 2026-02-24

**Paper Count**: 8

---

## 1. A Two-Stage Detection-Tracking Framework for Stable Apple Quality Inspection in Dense Conveyor-Belt Environments / 密集传送带环境中稳定苹果质量检测的两阶段检测跟踪框架

**Date**: 2026-02-22 | **arXiv**: [2602.19278v1](http://arxiv.org/abs/2602.19278v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.19278v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Industrial fruit inspection systems must operate reliably under dense multi-object interactions and continuous motion, yet most existing works evaluate detection or classification at the image level without ensuring temporal stability in video streams. We present a two-stage detection-tracking framework for stable multi-apple quality inspection in conveyor-belt environments. An orchard-trained YOLOv8 model performs apple localization, followed by ByteTrack multi-object tracking to maintain persistent identities. A ResNet18 defect classifier, fine-tuned on a healthy-defective fruit dataset, is applied to cropped apple regions. Track-level aggregation is introduced to enforce temporal consistency and reduce prediction oscillation across frames. We define video-level industrial metrics such as track-level defect ratio and temporal consistency to evaluate system robustness under realistic processing conditions. Results demonstrate improved stability compared to frame-wise inference, suggesting that integrating tracking is essential for practical automated fruit grading systems.

工业水果检测系统必须在密集的多对象交互和连续运动下可靠运行，但大多数现有工作在图像级别评估检测或分类，而无法确保视频流的时间稳定性。我们提出了一个两阶段检测跟踪框架，用于在传送带环境中进行稳定的多苹果质量检测。 Orchard 训练的 YOLOv8 模型执行苹果本地化，然后进行 ByteTrack 多对象跟踪以维护持久身份。 ResNet18 缺陷分类器在健康缺陷水果数据集上进行了微调，应用于裁剪后的苹果区域。引入轨道级聚合是为了加强时间一致性并减少帧间的预测振荡。我们定义视频级工业指标，例如轨道级缺陷率和时间一致性，以评估系统在实际处理条件下的稳健性。结果表明，与逐帧推理相比，稳定性有所提高，这表明集成跟踪对于实际的自动化水果分级系统至关重要。

</details>

---

## 2. UniE2F: A Unified Diffusion Framework for Event-to-Frame Reconstruction with Video Foundation Models / UniE2F：使用视频基础模型进行事件到帧重建的统一扩散框架

**Date**: 2026-02-22 | **arXiv**: [2602.19202v1](http://arxiv.org/abs/2602.19202v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.19202v1)

**Categories**: cs.CV

**Code**: https://github.com/CS-GangXu/UniE2F.

<details><summary><b>Abstract / 摘要</b></summary>

Event cameras excel at high-speed, low-power, and high-dynamic-range scene perception. However, as they fundamentally record only relative intensity changes rather than absolute intensity, the resulting data streams suffer from a significant loss of spatial information and static texture details. In this paper, we address this limitation by leveraging the generative prior of a pre-trained video diffusion model to reconstruct high-fidelity video frames from sparse event data. Specifically, we first establish a baseline model by directly applying event data as a condition to synthesize videos. Then, based on the physical correlation between the event stream and video frames, we further introduce the event-based inter-frame residual guidance to enhance the accuracy of video frame reconstruction. Furthermore, we extend our method to video frame interpolation and prediction in a zero-shot manner by modulating the reverse diffusion sampling process, thereby creating a unified event-to-frame reconstruction framework. Experimental results on real-world and synthetic datasets demonstrate that our method significantly outperforms previous approaches both quantitatively and qualitatively. We also refer the reviewers to the video demo contained in the supplementary material for video results. The code will be publicly available at https://github.com/CS-GangXu/UniE2F.

事件摄像机擅长高速、低功耗和高动态范围场景感知。然而，由于它们从根本上只记录相对强度变化而不是绝对强度，因此生成的数据流会遭受空间信息和静态纹理细节的显着损失。在本文中，我们通过利用预训练视频扩散模型的生成先验从稀疏事件数据重建高保真视频帧来解决这一限制。具体来说，我们首先通过直接应用事件数据作为合成视频的条件来建立基线模型。然后，基于事件流和视频帧之间的物理相关性，我们进一步引入基于事件的帧间残差引导，以提高视频帧重建的准确性。此外，我们通过调制反向扩散采样过程，以零镜头方式将我们的方法扩展到视频帧插值和预测，从而创建统一的事件到帧重建框架。现实世界和合成数据集的实验结果表明，我们的方法在数量和质量上都显着优于以前的方法。我们还向审稿人推荐视频结果补充材料中包含的视频演示。该代码将在 https://github.com/CS-GangXu/UniE2F 上公开提供。

</details>

---

## 3. JavisDiT++: Unified Modeling and Optimization for Joint Audio-Video Generation / JavisDiT++：联合音视频生成的统一建模和优化

**Date**: 2026-02-22 | **arXiv**: [2602.19163v1](http://arxiv.org/abs/2602.19163v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.19163v1)

**Categories**: cs.CV, cs.MM, cs.SD

**Project**: https://JavisVerse.github.io/JavisDiT2-page.  <details><summary><b>Abstract / 摘要</b></summary>

AIGC has rapidly expanded from text-to-image generation toward high-quality multimodal synthesis across video and audio. Within this context, joint audio-video generation (JAVG) has emerged as a fundamental task that produces synchronized and semantically aligned sound and vision from textual descriptions. However, compared with advanced commercial models such as Veo3, existing open-source methods still suffer from limitations in generation quality, temporal synchrony, and alignment with human preferences. To bridge the gap, this paper presents JavisDiT++, a concise yet powerful framework for unified modeling and optimization of JAVG. First, we introduce a modality-specific mixture-of-experts (MS-MoE) design that enables cross-modal interaction efficacy while enhancing single-modal generation quality. Then, we propose a temporal-aligned RoPE (TA-RoPE) strategy to achieve explicit, frame-level synchronization between audio and video tokens. Besides, we develop an audio-video direct preference optimization (AV-DPO) method to align model outputs with human preference across quality, consistency, and synchrony dimensions. Built upon Wan2.1-1.3B-T2V, our model achieves state-of-the-art performance merely with around 1M public training entries, significantly outperforming prior approaches in both qualitative and quantitative evaluations. Comprehensive ablation studies have been conducted to validate the effectiveness of our proposed modules. All the code, model, and dataset are released at https://JavisVerse.github.io/JavisDiT2-page.

AIGC 已从文本到图像的生成迅速扩展到跨视频和音频的高质量多模态合成。在此背景下，联合音视频生成（JAVG）已成为一项基本任务，它可以根据文本描述生成同步且语义一致的声音和视觉。然而，与 Veo3 等先进商业模型相比，现有的开源方法在生成质量、时间同步性以及与人类偏好的一致性等方面仍然存在局限性。为了弥补这一差距，本文提出了 JavisDiT++，这是一个简洁而强大的框架，用于 JAVG 的统一建模和优化。首先，我们引入了一种特定模态的专家混合（MS-MoE）设计，该设计可以实现跨模态交互功效，同时提高单模态生成质量。然后，我们提出了一种时间对齐 RoPE (TA-RoPE) 策略来实现音频和视频令牌之间的显式帧级同步。此外，我们开发了一种音视频直接偏好优化（AV-DPO）方法，使模型输出在质量、一致性和同步维度上与人类偏好保持一致。我们的模型基于 Wan2.1-1.3B-T2V 构建，仅通过大约 100 万个公共训练条目就实现了最先进的性能，在定性和定量评估方面都显着优于先前的方法。已经进行了全面的消融研究来验证我们提出的模块的有效性。所有代码、模型和数据集均在 https://JavisVerse.github.io/JavisDiT2-page 发布。

</details>

---

## 4. Flash-VAED: Plug-and-Play VAE Decoders for Efficient Video Generation / Flash-VAED：用于高效视频生成的即插即用 VAE 解码器

**Date**: 2026-02-22 | **arXiv**: [2602.19161v1](http://arxiv.org/abs/2602.19161v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.19161v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Latent diffusion models have enabled high-quality video synthesis, yet their inference remains costly and time-consuming. As diffusion transformers become increasingly efficient, the latency bottleneck inevitably shifts to VAE decoders. To reduce their latency while maintaining quality, we propose a universal acceleration framework for VAE decoders that preserves full alignment with the original latent distribution. Specifically, we propose (1) an independence-aware channel pruning method to effectively mitigate severe channel redundancy, and (2) a stage-wise dominant operator optimization strategy to address the high inference cost of the widely used causal 3D convolutions in VAE decoders. Based on these innovations, we construct a Flash-VAED family. Moreover, we design a three-phase dynamic distillation framework that efficiently transfers the capabilities of the original VAE decoder to Flash-VAED. Extensive experiments on Wan and LTX-Video VAE decoders demonstrate that our method outperforms baselines in both quality and speed, achieving approximately a 6$\times$ speedup while maintaining the reconstruction performance up to 96.9%. Notably, Flash-VAED accelerates the end-to-end generation pipeline by up to 36% with negligible quality drops on VBench-2.0.

潜在扩散模型已经实现了高质量的视频合成，但其推理仍然成本高昂且耗时。随着扩散变压器变得越来越高效，延迟瓶颈不可避免地转移到 VAE 解码器。为了在保持质量的同时减少延迟，我们提出了一种适用于 VAE 解码器的通用加速框架，该框架保持与原始潜在分布的完全对齐。具体来说，我们提出（1）一种独立感知的通道修剪方法，以有效减轻严重的通道冗余，以及（2）一种分阶段的主导算子优化策略，以解决 VAE 解码器中广泛使用的因果 3D 卷积的高推理成本问题。基于这些创新，我们构建了 Flash-VAED 系列。此外，我们设计了一个三相动态蒸馏框架，可以有效地将原始 VAE 解码器的功能转移到 Flash-VAED。在 Wan 和 LTX-Video VAE 解码器上进行的大量实验表明，我们的方法在质量和速度方面都优于基线，实现了大约 6 倍的加速，同时保持了高达 96.9% 的重建性能。值得注意的是，Flash-VAED 将端到端生成流程加速高达 36%，而 VBench-2.0 上的质量下降可以忽略不计。

</details>

---

## 5. Ani3DHuman: Photorealistic 3D Human Animation with Self-guided Stochastic Sampling / Ani3DHuman：具有自引导随机采样的真实感 3D 人体动画

**Date**: 2026-02-22 | **arXiv**: [2602.19089v1](http://arxiv.org/abs/2602.19089v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.19089v1)

**Categories**: cs.CV, cs.GR, cs.LG

**Code**: https://github.com/qiisun/ani3dhuman.

<details><summary><b>Abstract / 摘要</b></summary>

Current 3D human animation methods struggle to achieve photorealism: kinematics-based approaches lack non-rigid dynamics (e.g., clothing dynamics), while methods that leverage video diffusion priors can synthesize non-rigid motion but suffer from quality artifacts and identity loss. To overcome these limitations, we present Ani3DHuman, a framework that marries kinematics-based animation with video diffusion priors. We first introduce a layered motion representation that disentangles rigid motion from residual non-rigid motion. Rigid motion is generated by a kinematic method, which then produces a coarse rendering to guide the video diffusion model in generating video sequences that restore the residual non-rigid motion. However, this restoration task, based on diffusion sampling, is highly challenging, as the initial renderings are out-of-distribution, causing standard deterministic ODE samplers to fail. Therefore, we propose a novel self-guided stochastic sampling method, which effectively addresses the out-of-distribution problem by combining stochastic sampling (for photorealistic quality) with self-guidance (for identity fidelity). These restored videos provide high-quality supervision, enabling the optimization of the residual non-rigid motion field. Extensive experiments demonstrate that \MethodName can generate photorealistic 3D human animation, outperforming existing methods. Code is available in https://github.com/qiisun/ani3dhuman.

当前的 3D 人体动画方法很难实现照片级真实感：基于运动学的方法缺乏非刚性动力学（例如服装动力学），而利用视频扩散先验的方法可以合成非刚性运动，但会遭受质量伪影和身份损失。为了克服这些限制，我们提出了 Ani3DHuman，一个将基于运动学的动画与视频扩散先验相结合的框架。我们首先引入分层运动表示，将刚性运动与残余非刚性运动分开。刚性运动是通过运动学方法生成的，然后产生粗略渲染以指导视频扩散模型生成恢复残余非刚性运动的视频序列。然而，这种基于扩散采样的恢复任务非常具有挑战性，因为初始渲染不符合分布，导致标准确定性 ODE 采样器失败。因此，我们提出了一种新颖的自引导随机采样方法，该方法通过将随机采样（用于真实感质量）与自引导（用于身份保真度）相结合，有效地解决了分布外问题。这些恢复的视频提供高质量的监督，从而能够优化残余非刚性运动场。大量实验表明 \MethodName 可以生成逼真的 3D 人体动画，性能优于现有方法。代码可在 https://github.com/qiisun/ani3d human 中找到。

</details>

---

## 6. MoBind: Motion Binding for Fine-Grained IMU-Video Pose Alignment / MoBind：用于细粒度 IMU-视频姿势对齐的运动绑定

**Date**: 2026-02-22 | **arXiv**: [2602.19004v1](http://arxiv.org/abs/2602.19004v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.19004v1)

**Categories**: cs.CV

**Code**: https://github.com/bbvisual/

<details><summary><b>Abstract / 摘要</b></summary>

We aim to learn a joint representation between inertial measurement unit (IMU) signals and 2D pose sequences extracted from video, enabling accurate cross-modal retrieval, temporal synchronization, subject and body-part localization, and action recognition. To this end, we introduce MoBind, a hierarchical contrastive learning framework designed to address three challenges: (1) filtering out irrelevant visual background, (2) modeling structured multi-sensor IMU configurations, and (3) achieving fine-grained, sub-second temporal alignment. To isolate motion-relevant cues, MoBind aligns IMU signals with skeletal motion sequences rather than raw pixels. We further decompose full-body motion into local body-part trajectories, pairing each with its corresponding IMU to enable semantically grounded multi-sensor alignment. To capture detailed temporal correspondence, MoBind employs a hierarchical contrastive strategy that first aligns token-level temporal segments, then fuses local (body-part) alignment with global (body-wide) motion aggregation. Evaluated on mRi, TotalCapture, and EgoHumans, MoBind consistently outperforms strong baselines across all four tasks, demonstrating robust fine-grained temporal alignment while preserving coarse semantic consistency across modalities. Code is available at https://github.com/bbvisual/ MoBind.

我们的目标是学习惯性测量单元 (IMU) 信号和从视频中提取的 2D 姿势序列之间的联合表示，从而实现准确的跨模态检索、时间同步、主体和身体部位定位以及动作识别。为此，我们引入了 MoBind，这是一个分层对比学习框架，旨在解决三个挑战：（1）过滤掉不相关的视觉背景，（2）对结构化多传感器 IMU 配置进行建模，以及（3）实现细粒度、亚秒级时间对齐。为了隔离与运动相关的线索，MoBind 将 IMU 信号与骨骼运动序列而不是原始像素对齐。我们进一步将全身运动分解为局部身体部位轨迹，将每个轨迹与其相应的 IMU 配对，以实现基于语义的多传感器对齐。为了捕获详细的时间对应关系，MoBind 采用分层对比策略，首先对齐令牌级时间段，然后将局部（身体部分）对齐与全局（全身）运动聚合融合。在 mRi、TotalCapture 和 EgoHumans 上进行评估，MoBind 在所有四项任务中始终优于强大的基线，展示了强大的细粒度时间对齐，同时保持了跨模态的粗略语义一致性。代码可在 https://github.com/bbvisual/MoBind 获取。

</details>

---

## 7. Human-to-Robot Interaction: Learning from Video Demonstration for Robot Imitation / 人机交互：从机器人模仿的视频演示中学习

**Date**: 2026-02-22 | **arXiv**: [2602.19184v1](http://arxiv.org/abs/2602.19184v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.19184v1)

**Categories**: cs.RO

<details><summary><b>Abstract / 摘要</b></summary>

Learning from Demonstration (LfD) offers a promising paradigm for robot skill acquisition. Recent approaches attempt to extract manipulation commands directly from video demonstrations, yet face two critical challenges: (1) general video captioning models prioritize global scene features over task-relevant objects, producing descriptions unsuitable for precise robotic execution, and (2) end-to-end architectures coupling visual understanding with policy learning require extensive paired datasets and struggle to generalize across objects and scenarios. To address these limitations, we propose a novel ``Human-to-Robot'' imitation learning pipeline that enables robots to acquire manipulation skills directly from unstructured video demonstrations, inspired by the human ability to learn by watching and imitating. Our key innovation is a modular framework that decouples the learning process into two distinct stages: (1) Video Understanding, which combines Temporal Shift Modules (TSM) with Vision-Language Models (VLMs) to extract actions and identify interacted objects, and (2) Robot Imitation, which employs TD3-based deep reinforcement learning to execute the demonstrated manipulations. We validated our approach in PyBullet simulation environments with a UR5e manipulator and in a real-world experiment with a UF850 manipulator across four fundamental actions: reach, pick, move, and put. For video understanding, our method achieves 89.97% action classification accuracy and BLEU-4 scores of 0.351 on standard objects and 0.265 on novel objects, representing improvements of 76.4% and 128.4% over the best baseline, respectively. For robot manipulation, our framework achieves an average success rate of 87.5% across all actions, with 100% success on reaching tasks and up to 90% on complex pick-and-place operations. The project website is available at https://thanhnguyencanh.github.io/LfD4hri.

从演示中学习（LfD）为机器人技能获取提供了一个有前途的范例。最近的方法试图直接从视频演示中提取操作命令，但面临两个关键挑战：（1）通用视频字幕模型优先考虑全局场景特征而不是任务相关对象，产生不适合精确机器人执行的描述；（2）将视觉理解与策略学习相结合的端到端架构需要大量配对数据集，并且难以跨对象和场景进行泛化。为了解决这些限制，我们提出了一种新颖的“人到机器人”模仿学习管道，使机器人能够直接从非结构化视频演示中获得操作技能，其灵感来自于人类通过观看和模仿进行学习的能力。我们的关键创新是一个模块化框架，它将学习过程分解为两个不同的阶段：(1) 视频理解，它将时间转换模块 (TSM) 与视觉语言模型 (VLM) 结合起来，以提取动作并识别交互的对象；(2) 机器人模仿，它采用基于 TD3 的深度强化学习来执行演示的操作。我们在 PyBullet 模拟环境中使用 UR5e 操纵器验证了我们的方法，并在使用 UF850 操纵器的现实实验中验证了我们的方法，涉及四种基本动作：伸手、拾取、移动和放置。对于视频理解，我们的方法在标准对象上实现了 89.97% 的动作分类准确率，BLEU-4 分数为 0.351，在新颖对象上的分数为 0.265，分别比最佳基线提高了 76.4% 和 128.4%。对于机器人操作，我们的框架在所有操作中实现了 87.5% 的平均成功率，在完成任务时成功率为 100%，在复杂的拾取和放置操作中成功率高达 90%。该项目网站位于 https://thanhnguyencanh.github.io/LfD4hri。

</details>

---

## 8. Frame2Freq: Spectral Adapters for Fine-Grained Video Understanding / Frame2Freq：用于细粒度视频理解的光谱适配器

**Date**: 2026-02-21 | **arXiv**: [2602.18977v1](http://arxiv.org/abs/2602.18977v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.18977v1)

**Categories**: cs.CV

**Code**: https://github.com/th-nesh/Frame2Freq.

<details><summary><b>Abstract / 摘要</b></summary>

Adapting image-pretrained backbones to video typically relies on time-domain adapters tuned to a single temporal scale. Our experiments show that these modules pick up static image cues and very fast flicker changes, while overlooking medium-speed motion. Capturing dynamics across multiple time-scales is, however, crucial for fine-grained temporal analysis (i.e., opening vs. closing bottle).   To address this, we introduce Frame2Freq -- a family of frequency-aware adapters that perform spectral encoding during image-to-video adaptation of pretrained Vision Foundation Models (VFMs), improving fine-grained action recognition. Frame2Freq uses Fast Fourier Transform (FFT) along time and learns frequency-band specific embeddings that adaptively highlight the most discriminative frequency ranges. Across five fine-grained activity recognition datasets, Frame2Freq outperforms prior PEFT methods and even surpasses fully fine-tuned models on four of them. These results provide encouraging evidence that frequency analysis methods are a powerful tool for modeling temporal dynamics in image-to-video transfer. Code is available at https://github.com/th-nesh/Frame2Freq.

将图像预训练的主干网适应视频通常依赖于调整到单个时间尺度的时域适配器。我们的实验表明，这些模块可以拾取静态图像线索和非常快的闪烁变化，同时忽略中速运动。然而，捕获多个时间尺度的动态对于细粒度时间分析（即打开与关闭瓶子）至关重要。   为了解决这个问题，我们引入了 Frame2Freq——一系列频率感知适配器，可在预训练视觉基础模型 (VFM) 的图像到视频适应过程中执行频谱编码，从而改善细粒度的动作识别。 Frame2Freq 随着时间的推移使用快速傅里叶变换 (FFT) 并学习频段特定的嵌入，自适应地突出最具辨别力的频率范围。在五个细粒度活动识别数据集上，Frame2Freq 的性能优于之前的 PEFT 方法，甚至超过了其中四个数据集的完全微调模型。这些结果提供了令人鼓舞的证据，表明频率分析方法是对图像到视频传输中的时间动态进行建模的强大工具。代码可在 https://github.com/th-nesh/Frame2Freq 获取。

</details>

---



</details>

<details><summary><b>2026-02-23 (2 papers)</b></summary>

# arXiv Video Papers - 2026-02-23

**Paper Count**: 2

---

## 1. Going Down Memory Lane: Scaling Tokens for Video Stream Understanding with Dynamic KV-Cache Memory / 深入内存通道：使用动态 KV 高速缓存扩展用于视频流理解的令牌

**Date**: 2026-02-20 | **arXiv**: [2602.18434v1](http://arxiv.org/abs/2602.18434v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.18434v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Streaming video understanding requires models to robustly encode, store, and retrieve information from a continuous video stream to support accurate video question answering (VQA). Existing state-of-the-art approaches rely on key-value caching to accumulate frame-level information over time, but use a limited number of tokens per frame, leading to the loss of fine-grained visual details. In this work, we propose scaling the token budget to enable more granular spatiotemporal understanding and reasoning. First, we find that current methods are ill-equipped to handle dense streams: their feature encoding causes query-frame similarity scores to increase over time, biasing retrieval toward later frames. To address this, we introduce an adaptive selection strategy that reduces token redundancy while preserving local spatiotemporal information. We further propose a training-free retrieval mixture-of-experts that leverages external models to better identify relevant frames. Our method, MemStream, achieves +8.0% on CG-Bench, +8.5% on LVBench, and +2.4% on VideoMME (Long) over ReKV with Qwen2.5-VL-7B.

流视频理解需要模型能够稳健地编码、存储和检索连续视频流中的信息，以支持准确的视频问答 (VQA)。现有最先进的方法依赖键值缓存来随着时间的推移积累帧级信息，但每帧使用有限数量的令牌，导致细粒度视觉细节的丢失。在这项工作中，我们建议扩大代币预算，以实现更精细的时空理解和推理。首先，我们发现当前的方法不足以处理密集流：它们的特征编码会导致查询帧相似度分数随着时间的推移而增加，从而使检索偏向于后面的帧。为了解决这个问题，我们引入了一种自适应选择策略，可以减少令牌冗余，同时保留本地时空信息。我们进一步提出了一种免训练的检索专家混合体，它利用外部模型来更好地识别相关帧。我们的方法 MemStream 在使用 Qwen2.5-VL-7B 的 ReKV 上，在 CG-Bench 上实现了 +8.0%，在 LVBench 上实现了 +8.5%，在 VideoMME (Long) 上实现了 +2.4%。

</details>

---

## 2. Generated Reality: Human-centric World Simulation using Interactive Video Generation with Hand and Camera Control / 生成现实：使用手动和相机控制的交互式视频生成以人为中心的世界模拟

**Date**: 2026-02-20 | **arXiv**: [2602.18422v1](http://arxiv.org/abs/2602.18422v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.18422v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Extended reality (XR) demands generative models that respond to users' tracked real-world motion, yet current video world models accept only coarse control signals such as text or keyboard input, limiting their utility for embodied interaction. We introduce a human-centric video world model that is conditioned on both tracked head pose and joint-level hand poses. For this purpose, we evaluate existing diffusion transformer conditioning strategies and propose an effective mechanism for 3D head and hand control, enabling dexterous hand--object interactions. We train a bidirectional video diffusion model teacher using this strategy and distill it into a causal, interactive system that generates egocentric virtual environments. We evaluate this generated reality system with human subjects and demonstrate improved task performance as well as a significantly higher level of perceived amount of control over the performed actions compared with relevant baselines.

扩展现实（XR）需要生成模型来响应用户跟踪的现实世界运动，但当前的视频世界模型仅接受粗略的控制信号，例如文本或键盘输入，限制了它们在实体交互中的实用性。我们引入了一种以人为中心的视频世界模型，该模型以跟踪的头部姿势和关节级手部姿势为条件。为此，我们评估了现有的扩散变压器调节策略，并提出了一种有效的 3D 头部和手部控制机制，实现灵巧的手-物体交互。我们使用这种策略训练双向视频传播模型教师，并将其提炼成一个因果交互系统，生成以自我为中心的虚拟环境。我们用人类受试者评估了这个生成的现实系统，并证明了与相关基线相比，任务绩效得到了改善，并且对所执行的操作的感知控制程度显着提高。

</details>

---



</details>

<details><summary><b>2026-02-20 (2 papers)</b></summary>

# arXiv Video Papers - 2026-02-20

**Paper Count**: 2

---

## 1. DDiT: Dynamic Patch Scheduling for Efficient Diffusion Transformers / DDiT：高效扩散变压器的动态补丁调度

**Date**: 2026-02-19 | **arXiv**: [2602.16968v1](http://arxiv.org/abs/2602.16968v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16968v1)

**Categories**: cs.CV, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Diffusion Transformers (DiTs) have achieved state-of-the-art performance in image and video generation, but their success comes at the cost of heavy computation. This inefficiency is largely due to the fixed tokenization process, which uses constant-sized patches throughout the entire denoising phase, regardless of the content's complexity. We propose dynamic tokenization, an efficient test-time strategy that varies patch sizes based on content complexity and the denoising timestep. Our key insight is that early timesteps only require coarser patches to model global structure, while later iterations demand finer (smaller-sized) patches to refine local details. During inference, our method dynamically reallocates patch sizes across denoising steps for image and video generation and substantially reduces cost while preserving perceptual generation quality. Extensive experiments demonstrate the effectiveness of our approach: it achieves up to $3.52\times$ and $3.2\times$ speedup on FLUX-1.Dev and Wan $2.1$, respectively, without compromising the generation quality and prompt adherence.

扩散变压器 (DiT) 在图像和视频生成方面取得了最先进的性能，但它们的成功是以大量计算为代价的。这种低效率很大程度上是由于固定标记化过程造成的，该过程在整个去噪阶段都使用恒定大小的补丁，无论内容的复杂性如何。我们提出了动态标记化，这是一种有效的测试时间策略，可以根据内容复杂性和去噪时间步长来改变补丁大小。我们的主要见解是，早期的时间步长仅需要较粗糙的补丁来建模全局结构，而后期的迭代则需要更精细（较小尺寸）的补丁来细化局部细节。在推理过程中，我们的方法在图像和视频生成的去噪步骤中动态地重新分配补丁大小，并在保持感知生成质量的同时显着降低成本。大量的实验证明了我们方法的有效性：它在 FLUX-1.Dev 和 Wan 上分别实现了高达 $3.52\times$ 和 $3.2\times$ 的加速，而不会影响生成质量和即时依从性。

</details>

---

## 2. Xray-Visual Models: Scaling Vision models on Industry Scale Data / X 射线视觉模型：根据行业规模数据扩展视觉模型

**Date**: 2026-02-18 | **arXiv**: [2602.16918v1](http://arxiv.org/abs/2602.16918v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16918v1)

**Categories**: cs.CV, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

We present Xray-Visual, a unified vision model architecture for large-scale image and video understanding trained on industry-scale social media data. Our model leverages over 15 billion curated image-text pairs and 10 billion video-hashtag pairs from Facebook and Instagram, employing robust data curation pipelines that incorporate balancing and noise suppression strategies to maximize semantic diversity while minimizing label noise. We introduce a three-stage training pipeline that combines self-supervised MAE, semi-supervised hashtag classification, and CLIP-style contrastive learning to jointly optimize image and video modalities. Our architecture builds on a Vision Transformer backbone enhanced with efficient token reorganization (EViT) for improved computational efficiency. Extensive experiments demonstrate that Xray-Visual achieves state-of-the-art performance across diverse benchmarks, including ImageNet for image classification, Kinetics and HMDB51 for video understanding, and MSCOCO for cross-modal retrieval. The model exhibits strong robustness to domain shift and adversarial perturbations. We further demonstrate that integrating large language models as text encoders (LLM2CLIP) significantly enhances retrieval performance and generalization capabilities, particularly in real-world environments. Xray-Visual establishes new benchmarks for scalable, multimodal vision models, while maintaining superior accuracy and computational efficiency.

我们推出了 Xray-Visual，这是一种统一的视觉模型架构，用于在行业规模的社交媒体数据上进行训练的大规模图像和视频理解。我们的模型利用来自 Facebook 和 Instagram 的超过 150 亿个精选图像文本对和 100 亿个视频主题标签对，采用强大的数据管理管道，其中结合了平衡和噪声抑制策略，以最大限度地提高语义多样性，同时最大限度地减少标签噪声。我们引入了一个三阶段训练流程，结合了自监督 MAE、半监督标签分类和 CLIP 式对比学习，以联合优化图像和视频模式。我们的架构建立在 Vision Transformer 主干之上，通过高效的令牌重组 (EViT) 进行增强，以提高计算效率。大量实验表明，Xray-Visual 在不同基准测试中实现了最先进的性能，包括用于图像分类的 ImageNet、用于视频理解的 Kinetics 和 HMDB51 以及用于跨模态检索的 MSCOCO。该模型对域转移和对抗性扰动表现出很强的鲁棒性。我们进一步证明，将大型语言模型集成为文本编码器（LLM2CLIP）可以显着增强检索性能和泛化能力，特别是在现实环境中。 Xray-Visual 为可扩展的多模态视觉模型建立了新的基准，同时保持卓越的准确性和计算效率。

</details>

---



</details>

<details><summary><b>2026-02-19 (8 papers)</b></summary>

# arXiv Video Papers - 2026-02-19

**Paper Count**: 8

---

## 1. TeCoNeRV: Leveraging Temporal Coherence for Compressible Neural Representations for Videos / TeCoNeRV：利用时间相干性实现视频的可压缩神经表示

**Date**: 2026-02-18 | **arXiv**: [2602.16711v1](http://arxiv.org/abs/2602.16711v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16711v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Implicit Neural Representations (INRs) have recently demonstrated impressive performance for video compression. However, since a separate INR must be overfit for each video, scaling to high-resolution videos while maintaining encoding efficiency remains a significant challenge. Hypernetwork-based approaches predict INR weights (hyponetworks) for unseen videos at high speeds, but with low quality, large compressed size, and prohibitive memory needs at higher resolutions. We address these fundamental limitations through three key contributions: (1) an approach that decomposes the weight prediction task spatially and temporally, by breaking short video segments into patch tubelets, to reduce the pretraining memory overhead by 20$\times$; (2) a residual-based storage scheme that captures only differences between consecutive segment representations, significantly reducing bitstream size; and (3) a temporal coherence regularization framework that encourages changes in the weight space to be correlated with video content. Our proposed method, TeCoNeRV, achieves substantial improvements of 2.47dB and 5.35dB PSNR over the baseline at 480p and 720p on UVG, with 36% lower bitrates and 1.5-3$\times$ faster encoding speeds. With our low memory usage, we are the first hypernetwork approach to demonstrate results at 480p, 720p and 1080p on UVG, HEVC and MCL-JCV. Our project page is available at https://namithap10.github.io/teconerv/ .

隐式神经表示（INR）最近在视频压缩方面表现出了令人印象深刻的性能。然而，由于每个视频都必须有一个单独的 INR 过度拟合，因此在保持编码效率的同时扩展到高分辨率视频仍然是一个重大挑战。基于超网络的方法可以高速预测未见过的视频的 INR 权重（次网络），但在较高分辨率下质量较低、压缩大小较大且内存需求过高。我们通过三个关键贡献解决了这些基本限制：（1）一种在空间和时间上分解权重预测任务的方法，通过将短视频片段分解为补丁小管，将预训练内存开销减少 20$\times$； (2)基于残差的存储方案，仅捕获连续段表示之间的差异，显着减少比特流大小； （3）时间相干性正则化框架，鼓励权重空间的变化与视频内容相关。我们提出的方法 TeCoNeRV 在 UVG 上的 480p 和 720p 的基线上实现了 2.47dB 和 5.35dB PSNR 的大幅改进，比特率降低了 36%，编码速度提高了 1.5-3$\times$。由于内存使用率低，我们是第一个在 UVG、HEVC 和 MCL-JCV 上展示 480p、720p 和 1080p 效果的超网络方法。我们的项目页面位于 https://namithap10.github.io/teconerv/ 。

</details>

---

## 2. Let's Split Up: Zero-Shot Classifier Edits for Fine-Grained Video Understanding / 让我们分开：零样本分类器编辑以实现细粒度视频理解

**Date**: 2026-02-18 | **arXiv**: [2602.16545v1](http://arxiv.org/abs/2602.16545v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16545v1)

**Categories**: cs.CV, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Video recognition models are typically trained on fixed taxonomies which are often too coarse, collapsing distinctions in object, manner or outcome under a single label. As tasks and definitions evolve, such models cannot accommodate emerging distinctions and collecting new annotations and retraining to accommodate such changes is costly. To address these challenges, we introduce category splitting, a new task where an existing classifier is edited to refine a coarse category into finer subcategories, while preserving accuracy elsewhere. We propose a zero-shot editing method that leverages the latent compositional structure of video classifiers to expose fine-grained distinctions without additional data. We further show that low-shot fine-tuning, while simple, is highly effective and benefits from our zero-shot initialization. Experiments on our new video benchmarks for category splitting demonstrate that our method substantially outperforms vision-language baselines, improving accuracy on the newly split categories without sacrificing performance on the rest. Project page: https://kaitingliu.github.io/Category-Splitting/.

视频识别模型通常是在固定分类法上进行训练的，这些分类法通常过于粗糙，在单个标签下消除了对象、方式或结果的区别。随着任务和定义的发展，此类模型无法适应新出现的区别，并且收集新的注释和重新训练以适应此类变化的成本高昂。为了解决这些挑战，我们引入了类别分割，这是一项新任务，其中编辑现有分类器以将粗略类别细化为更精细的子类别，同时保持其他地方的准确性。我们提出了一种零镜头编辑方法，利用视频分类器的潜在组成结构来揭示细粒度的区别，而无需额外的数据。我们进一步表明，低样本微调虽然简单，但却非常有效，并且受益于我们的零样本初始化。我们针对类别分割的新视频基准的实验表明，我们的方法大大优于视觉语言基线，提高了新分割类别的准确性，而不会牺牲其余类别的性能。项目页面：https://kaitingliu.github.io/Category-Splitting/。

</details>

---

## 3. ReMoRa: Multimodal Large Language Model based on Refined Motion Representation for Long-Video Understanding / ReMoRa：基于细化运动表示的多模态大语言模型，用于长视频理解

**Date**: 2026-02-18 | **arXiv**: [2602.16412v1](http://arxiv.org/abs/2602.16412v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16412v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

While multimodal large language models (MLLMs) have shown remarkable success across a wide range of tasks, long-form video understanding remains a significant challenge. In this study, we focus on video understanding by MLLMs. This task is challenging because processing a full stream of RGB frames is computationally intractable and highly redundant, as self-attention have quadratic complexity with sequence length. In this paper, we propose ReMoRa, a video MLLM that processes videos by operating directly on their compressed representations. A sparse set of RGB keyframes is retained for appearance, while temporal dynamics are encoded as a motion representation, removing the need for sequential RGB frames. These motion representations act as a compact proxy for optical flow, capturing temporal dynamics without full frame decoding. To refine the noise and low fidelity of block-based motions, we introduce a module to denoise and generate a fine-grained motion representation. Furthermore, our model compresses these features in a way that scales linearly with sequence length. We demonstrate the effectiveness of ReMoRa through extensive experiments across a comprehensive suite of long-video understanding benchmarks. ReMoRa outperformed baseline methods on multiple challenging benchmarks, including LongVideoBench, NExT-QA, and MLVU.

虽然多模态大语言模型（MLLM）在广泛的任务中取得了显着的成功，但长格式视频理解仍然是一个重大挑战。在本研究中，我们重点关注 MLLM 的视频理解。这项任务具有挑战性，因为处理完整的 RGB 帧流在计算上非常困难且高度冗余，因为自注意力的复杂度与序列长度呈二次方关系。在本文中，我们提出了 ReMoRa，一种视频 MLLM，它通过直接操作视频的压缩表示来处理视频。保留一组稀疏的 RGB 关键帧用于外观，同时将时间动态编码为运动表示，从而无需连续的 RGB 帧。这些运动表示充当光流的紧凑代理，无需全帧解码即可捕获时间动态。为了改善基于块的运动的噪声和低保真度，我们引入了一个模块来降噪并生成细粒度的运动表示。此外，我们的模型以随序列长度线性缩放的方式压缩这些特征。我们通过一系列全面的长视频理解基准进行大量实验，展示了 ReMoRa 的有效性。 ReMoRa 在多个具有挑战性的基准测试中表现优于基准方法，包括 LongVideoBench、NExT-QA 和 MLVU。

</details>

---

## 4. DataCube: A Video Retrieval Platform via Natural Language Semantic Profiling / DataCube：基于自然语言语义分析的视频检索平台

**Date**: 2026-02-18 | **arXiv**: [2602.16231v1](http://arxiv.org/abs/2602.16231v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16231v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Large-scale video repositories are increasingly available for modern video understanding and generation tasks. However, transforming raw videos into high-quality, task-specific datasets remains costly and inefficient. We present DataCube, an intelligent platform for automatic video processing, multi-dimensional profiling, and query-driven retrieval. DataCube constructs structured semantic representations of video clips and supports hybrid retrieval with neural re-ranking and deep semantic matching. Through an interactive web interface, users can efficiently construct customized video subsets from massive repositories for training, analysis, and evaluation, and build searchable systems over their own private video collections. The system is publicly accessible at https://datacube.baai.ac.cn/. Demo Video: https://baai-data-cube.ks3-cn-beijing.ksyuncs.com/custom/Adobe%20Express%20-%202%E6%9C%8818%E6%97%A5%20%281%29%281%29%20%281%29.mp4

大型视频存储库越来越多地可用于现代视频理解和生成任务。然而，将原始视频转换为高质量、特定于任务的数据集仍然成本高昂且效率低下。我们推出 DataCube，一个用于自动视频处理、多维分析和查询驱动检索的智能平台。 DataCube 构建视频剪辑的结构化语义表示，并支持神经重新排序和深度语义匹配的混合检索。通过交互式网络界面，用户可以从海量存储库中高效地构建定制视频子集，用于训练、分析和评估，并在自己的私人视频集合上构建可搜索系统。该系统可通过 https://datacube.baai.ac.cn/ 公开访问。演示视频：https://baai-data-cube.ks3-cn-beijing.ksyncs.com/custom/Adobe%20Express%20-%202%E6%9C%8818%E6%97%A5%20%281%29%281%29%20%281%29.mp4

</details>

---

## 5. CHAI: CacHe Attention Inference for text2video / CHAI：text2video 的 CacHe 注意力推理

**Date**: 2026-02-18 | **arXiv**: [2602.16132v1](http://arxiv.org/abs/2602.16132v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16132v1)

**Categories**: cs.CV, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Text-to-video diffusion models deliver impressive results but remain slow because of the sequential denoising of 3D latents. Existing approaches to speed up inference either require expensive model retraining or use heuristic-based step skipping, which struggles to maintain video quality as the number of denoising steps decreases. Our work, CHAI, aims to use cross-inference caching to reduce latency while maintaining video quality. We introduce Cache Attention as an effective method for attending to shared objects/scenes across cross-inference latents. This selective attention mechanism enables effective reuse of cached latents across semantically related prompts, yielding high cache hit rates. We show that it is possible to generate high-quality videos using Cache Attention with as few as 8 denoising steps. When integrated into the overall system, CHAI is 1.65x - 3.35x faster than baseline OpenSora 1.2 while maintaining video quality.

文本到视频的扩散模型提供了令人印象深刻的结果，但由于 3D 潜伏的顺序去噪，速度仍然很慢。现有的加速推理的方法要么需要昂贵的模型重新训练，要么使用基于启发式的步骤跳跃，随着去噪步骤数量的减少，这很难保持视频质量。我们的工作 CHAI 旨在使用交叉推理缓存来减少延迟，同时保持视频质量。我们引入缓存注意力作为一种有效的方法来关注跨交叉推理潜在的共享对象/场景。这种选择性注意机制可以跨语义相关的提示有效地重用缓存的潜在变量，从而产生高缓存命中率。我们证明，使用 Cache Attention 只需 8 个去噪步骤即可生成高质量视频。当集成到整个系统中时，CHAI 的速度比基准 OpenSora 1.2 快 1.65 倍 - 3.35 倍，同时保持视频质量。

</details>

---

## 6. EgoScale: Scaling Dexterous Manipulation with Diverse Egocentric Human Data / EgoScale：利用各种以自我为中心的人类数据来扩展灵巧操作

**Date**: 2026-02-18 | **arXiv**: [2602.16710v1](http://arxiv.org/abs/2602.16710v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16710v1)

**Categories**: cs.RO

<details><summary><b>Abstract / 摘要</b></summary>

Human behavior is among the most scalable sources of data for learning physical intelligence, yet how to effectively leverage it for dexterous manipulation remains unclear. While prior work demonstrates human to robot transfer in constrained settings, it is unclear whether large scale human data can support fine grained, high degree of freedom dexterous manipulation. We present EgoScale, a human to dexterous manipulation transfer framework built on large scale egocentric human data. We train a Vision Language Action (VLA) model on over 20,854 hours of action labeled egocentric human video, more than 20 times larger than prior efforts, and uncover a log linear scaling law between human data scale and validation loss. This validation loss strongly correlates with downstream real robot performance, establishing large scale human data as a predictable supervision source. Beyond scale, we introduce a simple two stage transfer recipe: large scale human pretraining followed by lightweight aligned human robot mid training. This enables strong long horizon dexterous manipulation and one shot task adaptation with minimal robot supervision. Our final policy improves average success rate by 54% over a no pretraining baseline using a 22 DoF dexterous robotic hand, and transfers effectively to robots with lower DoF hands, indicating that large scale human motion provides a reusable, embodiment agnostic motor prior.

人类行为是学习身体智能最可扩展的数据来源之一，但如何有效地利用它进行灵巧的操作仍不清楚。虽然之前的工作证明了在受限环境下人与机器人的转移，但尚不清楚大规模人类数据是否可以支持细粒度、高自由度的灵巧操作。我们推出了 EgoScale，这是一个基于大规模以自我为中心的人类数据构建的人类到灵巧操作传输框架。我们在超过 20,854 小时的以自我为中心的人类视频中训练视觉语言动作 (VLA) 模型，比之前的工作量大 20 倍以上，并揭示了人类数据规模和验证损失之间的对数线性缩放定律。这种验证损失与下游真实机器人性能密切相关，将大规模人类数据建立为可预测的监督源。除了规模之外，我们引入了一个简单的两阶段转移方法：大规模人类预训练，然后是轻量级对齐的人类机器人中期训练。这使得强大的长视野灵巧操作和一次性任务适应与最少的机器人监督成为可能。与使用 22 DoF 灵巧机器人手的无预训练基线相比，我们的最终策略将平均成功率提高了 54%，并且有效地转移到具有较低 DoF 手的机器人，这表明大规模人体运动提供了可重复使用的、与具体实施例无关的电机先验。

</details>

---

## 7. Factored Latent Action World Models / 分解的潜在动作世界模型

**Date**: 2026-02-18 | **arXiv**: [2602.16229v1](http://arxiv.org/abs/2602.16229v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16229v1)

**Categories**: cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Learning latent actions from action-free video has emerged as a powerful paradigm for scaling up controllable world model learning. Latent actions provide a natural interface for users to iteratively generate and manipulate videos. However, most existing approaches rely on monolithic inverse and forward dynamics models that learn a single latent action to control the entire scene, and therefore struggle in complex environments where multiple entities act simultaneously. This paper introduces Factored Latent Action Model (FLAM), a factored dynamics framework that decomposes the scene into independent factors, each inferring its own latent action and predicting its own next-step factor value. This factorized structure enables more accurate modeling of complex multi-entity dynamics and improves video generation quality in action-free video settings compared to monolithic models. Based on experiments on both simulation and real-world multi-entity datasets, we find that FLAM outperforms prior work in prediction accuracy and representation quality, and facilitates downstream policy learning, demonstrating the benefits of factorized latent action models.

从无动作视频中学习潜在动作已成为扩大可控世界模型学习的强大范例。潜在动作为用户迭代生成和操作视频提供了一个自然的界面。然而，大多数现有方法依赖于整体逆向和正向动力学模型，这些模型学习单个潜在动作来控制整个场景，因此在多个实体同时动作的复杂环境中陷入困境。本文介绍了因子式潜在动作模型（FLAM），这是一种因子式动力学框架，它将场景分解为独立的因素，每个因子都推断自己的潜在动作并预测自己的下一步因子值。与整体模型相比，这种分解结构可以更准确地对复杂的多实体动态进行建模，并提高无动作视频设置中的视频生成质量。基于模拟和现实世界多实体数据集的实验，我们发现 FLAM 在预测精度和表示质量方面优于先前的工作，并促进下游策略学习，展示了因子化潜在动作模型的好处。

</details>

---

## 8. World Action Models are Zero-shot Policies / 世界行动模型是零射击政策

**Date**: 2026-02-17 | **arXiv**: [2602.15922v1](http://arxiv.org/abs/2602.15922v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.15922v1)

**Categories**: cs.RO, cs.CV, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

State-of-the-art Vision-Language-Action (VLA) models excel at semantic generalization but struggle to generalize to unseen physical motions in novel environments. We introduce DreamZero, a World Action Model (WAM) built upon a pretrained video diffusion backbone. Unlike VLAs, WAMs learn physical dynamics by predicting future world states and actions, using video as a dense representation of how the world evolves. By jointly modeling video and action, DreamZero learns diverse skills effectively from heterogeneous robot data without relying on repetitive demonstrations. This results in over 2x improvement in generalization to new tasks and environments compared to state-of-the-art VLAs in real robot experiments. Crucially, through model and system optimizations, we enable a 14B autoregressive video diffusion model to perform real-time closed-loop control at 7Hz. Finally, we demonstrate two forms of cross-embodiment transfer: video-only demonstrations from other robots or humans yield a relative improvement of over 42% on unseen task performance with just 10-20 minutes of data. More surprisingly, DreamZero enables few-shot embodiment adaptation, transferring to a new embodiment with only 30 minutes of play data while retaining zero-shot generalization.

最先进的视觉-语言-动作（VLA）模型擅长语义泛化，但很难泛化到新环境中看不见的物理运动。我们介绍 DreamZero，这是一种基于预训练视频传播主干的世界动作模型 (WAM)。与 VLA 不同，WAM 通过预测未来世界状态和行为来学习物理动力学，并使用视频作为世界如何演变的密集表示。通过对视频和动作进行联合建模，DreamZero 可以从异构机器人数据中有效地学习各种技能，而无需依赖重复演示。与真实机器人实验中最先进的 VLA 相比，这使得对新任务和环境的泛化能力提高了 2 倍以上。至关重要的是，通过模型和系统优化，我们使 14B 自回归视频扩散模型能够以 7Hz 执行实时闭环控制。最后，我们演示了两种形式的跨实体传输：来自其他机器人或人类的纯视频演示仅用 10-20 分钟的数据就可以使看不见的任务性能相对提高 42% 以上。更令人惊讶的是，DreamZero 实现了少样本实施例适应，仅用 30 分钟的播放数据转移到新实施例，同时保留零样本泛化。

</details>

---



</details>

<details><summary><b>2026-02-18 (8 papers)</b></summary>

# arXiv Video Papers - 2026-02-18

**Paper Count**: 8

---

## 1. VideoSketcher: Video Models Prior Enable Versatile Sequential Sketch Generation / VideoSketcher：视频模型优先启用多功能顺序草图生成

**Date**: 2026-02-17 | **arXiv**: [2602.15819v1](http://arxiv.org/abs/2602.15819v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.15819v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Sketching is inherently a sequential process, in which strokes are drawn in a meaningful order to explore and refine ideas. However, most generative models treat sketches as static images, overlooking the temporal structure that underlies creative drawing. We present a data-efficient approach for sequential sketch generation that adapts pretrained text-to-video diffusion models to generate sketching processes. Our key insight is that large language models and video diffusion models offer complementary strengths for this task: LLMs provide semantic planning and stroke ordering, while video diffusion models serve as strong renderers that produce high-quality, temporally coherent visuals. We leverage this by representing sketches as short videos in which strokes are progressively drawn on a blank canvas, guided by text-specified ordering instructions. We introduce a two-stage fine-tuning strategy that decouples the learning of stroke ordering from the learning of sketch appearance. Stroke ordering is learned using synthetic shape compositions with controlled temporal structure, while visual appearance is distilled from as few as seven manually authored sketching processes that capture both global drawing order and the continuous formation of individual strokes. Despite the extremely limited amount of human-drawn sketch data, our method generates high-quality sequential sketches that closely follow text-specified orderings while exhibiting rich visual detail. We further demonstrate the flexibility of our approach through extensions such as brush style conditioning and autoregressive sketch generation, enabling additional controllability and interactive, collaborative drawing.

素描本质上是一个连续的过程，在这个过程中，笔画是按照有意义的顺序绘制的，以探索和完善想法。然而，大多数生成模型将草图视为静态图像，忽略了创意绘画背后的时间结构。我们提出了一种用于顺序草图生成的数据高效方法，该方法采用预先训练的文本到视频扩散模型来生成草图绘制过程。我们的主要见解是，大型语言模型和视频扩散模型为这项任务提供了互补的优势：法学硕士提供语义规划和笔画排序，而视频扩散模型作为强大的渲染器，产生高质量、时间连贯的视觉效果。我们通过将草图表示为短视频来利用这一点，其中在文本指定的排序指令的指导下，在空白画布上逐步绘制笔画。我们引入了一种两阶段微调策略，将笔划顺序的学习与草图外观的学习分离。笔画顺序是使用具有受控时间结构的合成形状组合来学习的，而视觉外观是从多达七个手动创作的草图过程中提取出来的，这些过程捕获了全局绘图顺序和单个笔画的连续形成。尽管人类绘制的草图数据数量极其有限，但我们的方法可以生成高质量的顺序草图，这些草图严格遵循文本指定的顺序，同时展示丰富的视觉细节。我们通过画笔样式调节和自回归草图生成等扩展进一步展示了我们方法的灵活性，从而实现了额外的可控性和交互式协作绘图。

</details>

---

## 2. EventMemAgent: Hierarchical Event-Centric Memory for Online Video Understanding with Adaptive Tool Use / EventMemAgent：以事件为中心的分层内存，通过自适应工具使用进行在线视频理解

**Date**: 2026-02-17 | **arXiv**: [2602.15329v1](http://arxiv.org/abs/2602.15329v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.15329v1)

**Categories**: cs.CV

**Code**: https://github.com/lingcco/EventMemAgent.

<details><summary><b>Abstract / 摘要</b></summary>

Online video understanding requires models to perform continuous perception and long-range reasoning within potentially infinite visual streams. Its fundamental challenge lies in the conflict between the unbounded nature of streaming media input and the limited context window of Multimodal Large Language Models (MLLMs). Current methods primarily rely on passive processing, which often face a trade-off between maintaining long-range context and capturing the fine-grained details necessary for complex tasks. To address this, we introduce EventMemAgent, an active online video agent framework based on a hierarchical memory module. Our framework employs a dual-layer strategy for online videos: short-term memory detects event boundaries and utilizes event-granular reservoir sampling to process streaming video frames within a fixed-length buffer dynamically; long-term memory structuredly archives past observations on an event-by-event basis. Furthermore, we integrate a multi-granular perception toolkit for active, iterative evidence capture and employ Agentic Reinforcement Learning (Agentic RL) to end-to-end internalize reasoning and tool-use strategies into the agent's intrinsic capabilities. Experiments show that EventMemAgent achieves competitive results on online video benchmarks. The code will be released here: https://github.com/lingcco/EventMemAgent.

在线视频理解需要模型在潜在的无限视觉流中执行连续感知和远程推理。其根本挑战在于流媒体输入的无界性质与多模态大语言模型（MLLM）的有限上下文窗口之间的冲突。当前的方法主要依赖于被动处理，这通常面临着维护远程上下文和捕获复杂任务所需的细粒度细节之间的权衡。为了解决这个问题，我们引入了 EventMemAgent，一个基于分层内存模块的主动在线视频代理框架。我们的框架对在线视频采用双层策略：短期记忆检测事件边界，并利用事件粒度存储采样动态处理固定长度缓冲区内的流视频帧；长期记忆以逐个事件为基础结构化地归档过去的观察结果。此外，我们集成了用于主动、迭代证据捕获的多粒度感知工具包，并采用代理强化学习（Agentic RL）将端到端推理和工具使用策略内化为代理的内在能力。实验表明，EventMemAgent 在在线视频基准测试中取得了有竞争力的结果。代码将在这里发布：https://github.com/lingcco/EventMemAgent。

</details>

---

## 3. Consistency-Preserving Diverse Video Generation / 保持一致性的多样化视频生成

**Date**: 2026-02-17 | **arXiv**: [2602.15287v1](http://arxiv.org/abs/2602.15287v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.15287v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Text-to-video generation is expensive, so only a few samples are typically produced per prompt. In this low-sample regime, maximizing the value of each batch requires high cross-video diversity. Recent methods improve diversity for image generation, but for videos they often degrade within-video temporal consistency and require costly backpropagation through a video decoder. We propose a joint-sampling framework for flow-matching video generators that improves batch diversity while preserving temporal consistency. Our approach applies diversity-driven updates and then removes only the components that would decrease a temporal-consistency objective. To avoid image-space gradients, we compute both objectives with lightweight latent-space models, avoiding video decoding and decoder backpropagation. Experiments on a state-of-the-art text-to-video flow-matching model show diversity comparable to strong joint-sampling baselines while substantially improving temporal consistency and color naturalness. Code will be released.

文本到视频的生成成本很高，因此每个提示通常只生成几个样本。在这种低样本情况下，最大化每个批次的价值需要高度的跨视频多样性。最近的方法提高了图像生成的多样性，但对于视频来说，它们通常会降低视频内的时间一致性，并且需要通过视频解码器进行昂贵的反向传播。我们提出了一种用于流匹配视频生成器的联合采样框架，该框架可以提高批量多样性，同时保持时间一致性。我们的方法应用多样性驱动的更新，然后仅删除会降低时间一致性目标的组件。为了避免图像空间梯度，我们使用轻量级潜在空间模型计算两个目标，避免视频解码和解码器反向传播。最先进的文本到视频流匹配模型的实验显示出与强联合采样基线相当的多样性，同时显着提高了时间一致性和颜色自然度。代码将被发布。

</details>

---

## 4. Loss Knows Best: Detecting Annotation Errors in Videos via Loss Trajectories / 损失最了解：通过损失轨迹检测视频中的注释错误

**Date**: 2026-02-16 | **arXiv**: [2602.15154v1](http://arxiv.org/abs/2602.15154v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.15154v1)

**Categories**: cs.CV, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

High-quality video datasets are foundational for training robust models in tasks like action recognition, phase detection, and event segmentation. However, many real-world video datasets suffer from annotation errors such as *mislabeling*, where segments are assigned incorrect class labels, and *disordering*, where the temporal sequence does not follow the correct progression. These errors are particularly harmful in phase-annotated tasks, where temporal consistency is critical. We propose a novel, model-agnostic method for detecting annotation errors by analyzing the Cumulative Sample Loss (CSL)--defined as the average loss a frame incurs when passing through model checkpoints saved across training epochs. This per-frame loss trajectory acts as a dynamic fingerprint of frame-level learnability. Mislabeled or disordered frames tend to show consistently high or irregular loss patterns, as they remain difficult for the model to learn throughout training, while correctly labeled frames typically converge to low loss early. To compute CSL, we train a video segmentation model and store its weights at each epoch. These checkpoints are then used to evaluate the loss of each frame in a test video. Frames with persistently high CSL are flagged as likely candidates for annotation errors, including mislabeling or temporal misalignment. Our method does not require ground truth on annotation errors and is generalizable across datasets. Experiments on EgoPER and Cholec80 demonstrate strong detection performance, effectively identifying subtle inconsistencies such as mislabeling and frame disordering. The proposed approach provides a powerful tool for dataset auditing and improving training reliability in video-based machine learning.

高质量视频数据集是在动作识别、相位检测和事件分割等任务中训练稳健模型的基础。然而，许多现实世界的视频数据集都存在注释错误，例如“错误标记”（片段被分配了不正确的类标签）和“无序”（时间序列不遵循正确的进展）。这些错误在时间一致性至关重要的阶段注释任务中尤其有害。我们提出了一种新颖的、与模型无关的方法，通过分析累积样本损失（CSL）来检测注释错误，CSL 定义为帧通过跨训练时期保存的模型检查点时产生的平均损失。这种每帧丢失轨迹充当帧级可学习性的动态指纹。错误标记或无序的帧往往会显示出持续较高或不规则的损失模式，因为模型在整个训练过程中仍然难以学习它们，而正确标记的帧通常会尽早收敛到低损失。为了计算 CSL，我们训练了一个视频分割模型并存储其在每个时期的权重。然后使用这些检查点来评估测试视频中每帧的丢失。具有持续高 CSL 的帧被标记为可能存在注释错误的候选帧，包括错误标记或时间错位。我们的方法不需要注释错误的基本事实，并且可以跨数据集推广。 EgoPER 和 Cholec80 上的实验展示了强大的检测性能，可以有效识别细微的不一致，例如错误标记和帧混乱。所提出的方法为数据集审核和提高基于视频的机器学习中的训练可靠性提供了强大的工具。

</details>

---

## 5. EditCtrl: Disentangled Local and Global Control for Real-Time Generative Video Editing / EditCtrl：实时生成视频编辑的解开本地和全局控制

**Date**: 2026-02-16 | **arXiv**: [2602.15031v1](http://arxiv.org/abs/2602.15031v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.15031v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

High-fidelity generative video editing has seen significant quality improvements by leveraging pre-trained video foundation models. However, their computational cost is a major bottleneck, as they are often designed to inefficiently process the full video context regardless of the inpainting mask's size, even for sparse, localized edits. In this paper, we introduce EditCtrl, an efficient video inpainting control framework that focuses computation only where it is needed. Our approach features a novel local video context module that operates solely on masked tokens, yielding a computational cost proportional to the edit size. This local-first generation is then guided by a lightweight temporal global context embedder that ensures video-wide context consistency with minimal overhead. Not only is EditCtrl 10 times more compute efficient than state-of-the-art generative editing methods, it even improves editing quality compared to methods designed with full-attention. Finally, we showcase how EditCtrl unlocks new capabilities, including multi-region editing with text prompts and autoregressive content propagation.

通过利用预先训练的视频基础模型，高保真生成视频编辑的质量得到了显着提高。然而，它们的计算成本是一个主要瓶颈，因为它们通常被设计为低效地处理完整的视频上下文，无论修复掩模的大小如何，即使对于稀疏的局部编辑也是如此。在本文中，我们介绍了 EditCtrl，这是一种高效的视频修复控制框架，仅将计算集中在需要的地方。我们的方法采用了一种新颖的本地视频上下文模块，该模块仅对屏蔽标记进行操作，产生与编辑大小成正比的计算成本。然后，这个本地第一代由轻量级全局上下文嵌入器引导，以最小的开销确保视频范围的上下文一致性。 EditCtrl 不仅计算效率比最先进的生成编辑方法高 10 倍，而且与完全注意设计的方法相比，它甚至还提高了编辑质量。最后，我们展示 EditCtrl 如何解锁新功能，包括带有文本提示的多区域编辑和自回归内容传播。

</details>

---

## 6. AnchorWeave: World-Consistent Video Generation with Retrieved Local Spatial Memories / AnchorWeave：通过检索本地空间记忆生成世界一致的视频

**Date**: 2026-02-16 | **arXiv**: [2602.14941v1](http://arxiv.org/abs/2602.14941v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14941v1)

**Categories**: cs.CV, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Maintaining spatial world consistency over long horizons remains a central challenge for camera-controllable video generation. Existing memory-based approaches often condition generation on globally reconstructed 3D scenes by rendering anchor videos from the reconstructed geometry in the history. However, reconstructing a global 3D scene from multiple views inevitably introduces cross-view misalignment, as pose and depth estimation errors cause the same surfaces to be reconstructed at slightly different 3D locations across views. When fused, these inconsistencies accumulate into noisy geometry that contaminates the conditioning signals and degrades generation quality. We introduce AnchorWeave, a memory-augmented video generation framework that replaces a single misaligned global memory with multiple clean local geometric memories and learns to reconcile their cross-view inconsistencies. To this end, AnchorWeave performs coverage-driven local memory retrieval aligned with the target trajectory and integrates the selected local memories through a multi-anchor weaving controller during generation. Extensive experiments demonstrate that AnchorWeave significantly improves long-term scene consistency while maintaining strong visual quality, with ablation and analysis studies further validating the effectiveness of local geometric conditioning, multi-anchor control, and coverage-driven retrieval.

在长期范围内保持空间世界的一致性仍然是摄像机可控视频生成的核心挑战。现有的基于内存的方法通常通过从历史中的重建几何体渲染锚视频来调节全局重建 3D 场景的生成。然而，从多个视图重建全局 3D 场景不可避免地会引入跨视图未对准，因为姿态和深度估计错误会导致在跨视图的稍微不同的 3D 位置重建相同的表面。当融合时，这些不一致会累积成噪声几何形状，从而污染调节信号并降低生成质量。我们介绍了 AnchorWeave，这是一种内存增强视频生成框架，它用多个干净的局部几何内存替换单个未对齐的全局内存，并学习协调它们的跨视图不一致。为此，AnchorWeave 执行与目标轨迹对齐的覆盖驱动的本地内存检索，并在生成过程中通过多锚编织控制器集成所选的本地内存。大量实验表明，AnchorWeave 显着提高了长期场景一致性，同时保持了强大的视觉质量，消融和分析研究进一步验证了局部几何条件、多锚点控制和覆盖驱动检索的有效性。

</details>

---

## 7. Adapting VACE for Real-Time Autoregressive Video Diffusion / 采用 VACE 进行实时自回归视频扩散

**Date**: 2026-02-16 | **arXiv**: [2602.14381v1](http://arxiv.org/abs/2602.14381v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14381v1)

**Categories**: cs.CV, cs.AI

**Code**: https://github.com/daydreamlive/scope.

<details><summary><b>Abstract / 摘要</b></summary>

We describe an adaptation of VACE (Video All-in-one Creation and Editing) for real-time autoregressive video generation. VACE provides unified video control (reference guidance, structural conditioning, inpainting, and temporal extension) but assumes bidirectional attention over full sequences, making it incompatible with streaming pipelines that require fixed chunk sizes and causal attention. The key modification moves reference frames from the diffusion latent space into a parallel conditioning pathway, preserving the fixed chunk sizes and KV caching that autoregressive models require. This adaptation reuses existing pretrained VACE weights without additional training. Across 1.3B and 14B model scales, VACE adds 20-30% latency overhead for structural control and inpainting, with negligible VRAM cost relative to the base model. Reference-to-video fidelity is severely degraded compared to batch VACE due to causal attention constraints. A reference implementation is available at https://github.com/daydreamlive/scope.

我们描述了用于实时自回归视频生成的 VACE（视频一体化创建和编辑）的改编。 VACE 提供统一的视频控制（参考指导、结构调节、修复和时间扩展），但假设对整个序列进行双向关注，这使其与需要固定块大小和因果关注的流媒体管道不兼容。关键修改将参考帧从扩散潜在空间移动到并行调节路径，保留自回归模型所需的固定块大小和 KV 缓存。此调整重复使用现有的预训练 VACE 权重，无需额外训练。在 1.3B 和 14B 模型规模中，VACE 为结构控制和修复增加了 20-30% 的延迟开销，而相对于基本模型，VRAM 成本可以忽略不计。由于因果注意力限制，与批量 VACE 相比，视频参考保真度严重下降。参考实现可在 https://github.com/daydreamlive/scope 上找到。

</details>

---

## 8. Dual-Signal Adaptive KV-Cache Optimization for Long-Form Video Understanding in Vision-Language Models / 用于视觉语言模型中长格式视频理解的双信号自适应 KV 缓存优化

**Date**: 2026-02-15 | **arXiv**: [2602.14236v1](http://arxiv.org/abs/2602.14236v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14236v1)

**Categories**: cs.CV, cs.AI, cs.LG, cs.PF

<details><summary><b>Abstract / 摘要</b></summary>

Vision-Language Models (VLMs) face a critical memory bottleneck when processing long-form video content due to the linear growth of the Key-Value (KV) cache with sequence length. Existing solutions predominantly employ reactive eviction strategies that compute full attention matrices before discarding tokens, resulting in substantial computational waste. We propose Sali-Cache, a novel a priori optimization framework that implements dual-signal adaptive caching through proactive memory management. By integrating a temporal filter based on optical flow analysis for detecting inter-frame redundancy and a spatial filter leveraging saliency detection for identifying visually significant regions, Sali-Cache intelligently manages memory allocation before entering computationally expensive attention operations. Experimental evaluation on the LLaVA 1.6 architecture demonstrates that our method achieves a 2.20x compression ratio in effective memory usage while maintaining 100% accuracy across BLEU, ROUGE-L, and Exact Match metrics. Furthermore, under identical memory budget constraints, Sali-Cache preserves context-rich features over extended temporal durations without degrading model performance, enabling efficient processing of long-form video content on consumer-grade hardware.

由于键值 (KV) 缓存随序列长度线性增长，视觉语言模型 (VLM) 在处理长格式视频内容时面临严重的内存瓶颈。现有的解决方案主要采用反应性驱逐策略，在丢弃令牌之前计算完整的注意力矩阵，从而导致大量的计算浪费。我们提出了 Sali-Cache，一种新颖的先验优化框架，通过主动内存管理实现双信号自适应缓存。通过集成基于光流分析的时间滤波器（用于检测帧间冗余）和空间滤波器（利用显着性检测来识别视觉上重要的区域），Sali-Cache 在进入计算量大的注意力操作之前智能地管理内存分配。对 LLaVA 1.6 架构的实验评估表明，我们的方法在有效内存使用方面实现了 2.20 倍的压缩比，同时在 BLEU、ROUGE-L 和精确匹配指标上保持 100% 的准确性。此外，在相同的内存预算限制下，Sali-Cache 在延长的时间持续时间内保留上下文丰富的特征，而不会降低模型性能，从而能够在消费级硬件上高效处理长格式视频内容。

</details>

---



</details>

<details><summary><b>2026-02-17 (3 papers)</b></summary>

# arXiv Video Papers - 2026-02-17

**Paper Count**: 3

---

## 1. When Test-Time Guidance Is Enough: Fast Image and Video Editing with Diffusion Guidance / 当测试时间指导足够时：使用扩散指导进行快速图像和视频编辑

**Date**: 2026-02-15 | **arXiv**: [2602.14157v1](http://arxiv.org/abs/2602.14157v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14157v1)

**Categories**: cs.CV, cs.AI, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Text-driven image and video editing can be naturally cast as inpainting problems, where masked regions are reconstructed to remain consistent with both the observed content and the editing prompt. Recent advances in test-time guidance for diffusion and flow models provide a principled framework for this task; however, existing methods rely on costly vector--Jacobian product (VJP) computations to approximate the intractable guidance term, limiting their practical applicability. Building upon the recent work of Moufad et al. (2025), we provide theoretical insights into their VJP-free approximation and substantially extend their empirical evaluation to large-scale image and video editing benchmarks. Our results demonstrate that test-time guidance alone can achieve performance comparable to, and in some cases surpass, training-based methods.

文本驱动的图像和视频编辑可以自然地转化为修复问题，其中重建遮罩区域以与观察到的内容和编辑提示保持一致。扩散和流动模型测试时指导的最新进展为这项任务提供了原则框架；然而，现有方法依赖于昂贵的矢量雅可比积（VJP）计算来近似棘手的指导项，限制了它们的实际适用性。以 Moufad 等人最近的工作为基础。 (2025)，我们提供了对其无 VJP 近似的理论见解，并将其实证评估大幅扩展到大规模图像和视频编辑基准。我们的结果表明，仅测试时指导就可以实现与基于训练的方法相当的性能，在某些情况下甚至超越基于训练的方法。

</details>

---

## 2. Train Short, Inference Long: Training-free Horizon Extension for Autoregressive Video Generation / 训练短，推理长：用于自回归视频生成的免训练地平线扩展

**Date**: 2026-02-15 | **arXiv**: [2602.14027v1](http://arxiv.org/abs/2602.14027v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14027v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Autoregressive video diffusion models have emerged as a scalable paradigm for long video generation. However, they often suffer from severe extrapolation failure, where rapid error accumulation leads to significant temporal degradation when extending beyond training horizons. We identify that this failure primarily stems from the \textit{spectral bias} of 3D positional embeddings and the lack of \textit{dynamic priors} in noise sampling. To address these issues, we propose \textbf{FLEX} (\textbf{F}requency-aware \textbf{L}ength \textbf{EX}tension), a training-free inference-time framework that bridges the gap between short-term training and long-term inference. FLEX introduces Frequency-aware RoPE Modulation to adaptively interpolate under-trained low-frequency components while extrapolating high-frequency ones to preserve multi-scale temporal discriminability. This is integrated with Antiphase Noise Sampling (ANS) to inject high-frequency dynamic priors and Inference-only Attention Sink to anchor global structure. Extensive evaluations on VBench demonstrate that FLEX significantly outperforms state-of-the-art models at $6\times$ extrapolation (30s duration) and matches the performance of long-video fine-tuned baselines at $12\times$ scale (60s duration). As a plug-and-play augmentation, FLEX seamlessly integrates into existing inference pipelines for horizon extension. It effectively pushes the generation limits of models such as LongLive, supporting consistent and dynamic video synthesis at a 4-minute scale. Project page is available at \href{https://ga-lee.github.io/FLEX_demo}{https://ga-lee.github.io/FLEX}.

自回归视频扩散模型已成为长视频生成的可扩展范例。然而，它们经常遭受严重的外推失败，当超出训练范围时，快速的误差积累会导致显着的时间退化。我们发现这种失败主要源于 3D 位置嵌入的 \textit{频谱偏差} 以及噪声采样中 \textit{动态先验} 的缺乏。为了解决这些问题，我们提出了 \textbf{FLEX} （\textbf{F}requency-aware \textbf{L}ength \textbf{EX}tension），这是一种无需训练的推理时间框架，可以弥补短期训练和长期推理之间的差距。 FLEX 引入了频率感知 RoPE 调制，可自适应地内插未经训练的低频分量，同时外推高频分量以保持多尺度时间辨别能力。它与反相噪声采样 (ANS) 集成，以注入高频动态先验和仅推理注意池来锚定全局结构。对 VBench 的广泛评估表明，FLEX 在 $6\times$ 外推（30 秒持续时间）下显着优于最先进的模型，并与长视频微调基线在 $12\times$ 规模（60 秒持续时间）下的性能相匹配。作为一种即插即用的增强功能，FLEX 可以无缝集成到现有的推理管道中，以实现范围扩展。它有效地突破了 LongLive 等模型的生成限制，支持 4 分钟规模的一致动态视频合成。项目页面位于 \href{https://ga-lee.github.io/FLEX_demo}{https://ga-lee.github.io/FLEX}。

</details>

---

## 3. High-Fidelity Causal Video Diffusion Models for Real-Time Ultra-Low-Bitrate Semantic Communication / 用于实时超低比特率语义通信的高保真因果视频扩散模型

**Date**: 2026-02-14 | **arXiv**: [2602.13837v1](http://arxiv.org/abs/2602.13837v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.13837v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

We introduce a video diffusion model for high-fidelity, causal, and real-time video generation under ultra-low-bitrate semantic communication constraints. Our approach utilizes lossy semantic video coding to transmit the semantic scene structure, complemented by a stream of highly compressed, low-resolution frames that provide sufficient texture information to preserve fidelity. Building on these inputs, we introduce a modular video diffusion model that contains Semantic Control, Restoration Adapter, and Temporal Adapter. We further introduce an efficient temporal distillation procedure that enables extension to real-time and causal synthesis, reducing trainable parameters by 300x and training time by 2x, while adhering to communication constraints. Evaluated across diverse datasets, the framework achieves strong perceptual quality, semantic fidelity, and temporal consistency at ultra-low bitrates (< 0.0003 bpp), outperforming classical, neural, and generative baselines in extensive quantitative, qualitative, and subjective evaluations.

我们引入了一种视频扩散模型，用于在超低比特率语义通信约束下生成高保真、因果和实时视频。我们的方法利用有损语义视频编码来传输语义场景结构，并辅以高度压缩的低分辨率帧流，提供足够的纹理信息以保持保真度。基于这些输入，我们引入了一个模块化视频扩散模型，其中包含语义控制、恢复适配器和时间适配器。我们进一步引入了一种有效的时间蒸馏程序，可以扩展到实时和因果合成，将可训练参数减少 300 倍，将训练时间减少 2 倍，同时遵守通信限制。经过不同数据集的评估，该框架在超低比特率 (< 0.0003 bpp) 下实现了强大的感知质量、语义保真度和时间一致性，在广泛的定量、定性和主观评估中优于经典、神经和生成基线。

</details>

---



</details>

<details><summary><b>2026-02-16 (4 papers)</b></summary>

# arXiv Video Papers - 2026-02-16

**Paper Count**: 4

---

## 1. CoPE-VideoLM: Codec Primitives For Efficient Video Language Models / CoPE-VideoLM：用于高效视频语言模型的编解码器原语

**Date**: 2026-02-13 | **arXiv**: [2602.13191v1](http://arxiv.org/abs/2602.13191v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.13191v1)

**Categories**: cs.CV, cs.AI, cs.CL

<details><summary><b>Abstract / 摘要</b></summary>

Video Language Models (VideoLMs) empower AI systems to understand temporal dynamics in videos. To fit to the maximum context window constraint, current methods use keyframe sampling which can miss both macro-level events and micro-level details due to the sparse temporal coverage. Furthermore, processing full images and their tokens for each frame incurs substantial computational overhead. To address these limitations, we propose to leverage video codec primitives (specifically motion vectors and residuals) which natively encode video redundancy and sparsity without requiring expensive full-image encoding for most frames. To this end, we introduce lightweight transformer-based encoders that aggregate codec primitives and align their representations with image encoder embeddings through a pre-training strategy that accelerates convergence during end-to-end fine-tuning. Our approach reduces the time-to-first-token by up to $86\%$ and token usage by up to $93\%$ compared to standard VideoLMs. Moreover, by varying the keyframe and codec primitive densities we are able to maintain or exceed performance on $14$ diverse video understanding benchmarks spanning general question answering, temporal reasoning, long-form understanding, and spatial scene understanding.

视频语言模型 (VideoLM) 使 AI 系统能够理解视频中的时间动态。为了适应最大上下文窗口约束，当前的方法使用关键帧采样，由于稀疏的时间覆盖，可能会错过宏观级别的事件和微观级别的细节。此外，处理完整图像及其每帧的标记会产生大量的计算开销。为了解决这些限制，我们建议利用视频编解码器原语（特别是运动向量和残差），它可以对视频冗余和稀疏性进行本机编码，而无需对大多数帧进行昂贵的全图像编码。为此，我们引入了基于 Transformer 的轻量级编码器，它聚合编解码器原语，并通过预训练策略将其表示与图像编码器嵌入对齐，该预训练策略可在端到端微调过程中加速收敛。与标准 VideoLM 相比，我们的方法将首次使用令牌的时间减少了高达 86\%$，令牌使用量减少了高达 93\%$。此外，通过改变关键帧和编解码器基元密度，我们能够在 14 美元的各种视频理解基准上保持或超过性能，涵盖一般问答、时间推理、长格式理解和空间场景理解。

</details>

---

## 2. FlexAM: Flexible Appearance-Motion Decomposition for Versatile Video Generation Control / FlexAM：灵活的外观运动分解，用于多功能视频生成控制

**Date**: 2026-02-13 | **arXiv**: [2602.13185v1](http://arxiv.org/abs/2602.13185v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.13185v1)

**Categories**: cs.CV, cs.GR

<details><summary><b>Abstract / 摘要</b></summary>

Effective and generalizable control in video generation remains a significant challenge. While many methods rely on ambiguous or task-specific signals, we argue that a fundamental disentanglement of "appearance" and "motion" provides a more robust and scalable pathway. We propose FlexAM, a unified framework built upon a novel 3D control signal. This signal represents video dynamics as a point cloud, introducing three key enhancements: multi-frequency positional encoding to distinguish fine-grained motion, depth-aware positional encoding, and a flexible control signal for balancing precision and generative quality. This representation allows FlexAM to effectively disentangle appearance and motion, enabling a wide range of tasks including I2V/V2V editing, camera control, and spatial object editing. Extensive experiments demonstrate that FlexAM achieves superior performance across all evaluated tasks.

视频生成的有效且通用的控制仍然是一个重大挑战。虽然许多方法依赖于模糊或特定于任务的信号，但我们认为“外观”和“运动”的根本分离提供了一条更稳健和可扩展的途径。我们提出了 FlexAM，这是一个基于新颖的 3D 控制信号构建的统一框架。该信号将视频动态表示为点云，引入了三个关键增强功能：用于区分细粒度运动的多频位置编码、深度感知位置编码以及用于平衡精度和生成质量的灵活控制信号。这种表示方式使 FlexAM 能够有效地理清外观和运动，从而实现广泛的任务，包括 I2V/V2V 编辑、相机控制和空间对象编辑。大量实验表明，FlexAM 在所有评估任务中均实现了卓越的性能。

</details>

---

## 3. Towards Universal Video MLLMs with Attribute-Structured and Quality-Verified Instructions / 迈向具有属性结构和质量验证指令的通用视频 MLLM

**Date**: 2026-02-13 | **arXiv**: [2602.13013v1](http://arxiv.org/abs/2602.13013v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.13013v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Universal video understanding requires modeling fine-grained visual and audio information over time in diverse real-world scenarios. However, the performance of existing models is primarily constrained by video-instruction data that represents complex audiovisual content as single, incomplete descriptions, lacking fine-grained organization and reliable annotation. To address this, we introduce: (i) ASID-1M, an open-source collection of one million structured, fine-grained audiovisual instruction annotations with single- and multi-attribute supervision; (ii) ASID-Verify, a scalable data curation pipeline for annotation, with automatic verification and refinement that enforces semantic and temporal consistency between descriptions and the corresponding audiovisual content; and (iii) ASID-Captioner, a video understanding model trained via Supervised Fine-Tuning (SFT) on the ASID-1M. Experiments across seven benchmarks covering audiovisual captioning, attribute-wise captioning, caption-based QA, and caption-based temporal grounding show that ASID-Captioner improves fine-grained caption quality while reducing hallucinations and improving instruction following. It achieves state-of-the-art performance among open-source models and is competitive with Gemini-3-Pro.

通用视频理解需要在不同的现实场景中随着时间的推移对细粒度的视觉和音频信息进行建模。然而，现有模型的性能主要受到视频教学数据的限制，视频教学数据将复杂的视听内容表示为单一的、不完整的描述，缺乏细粒度的组织和可靠的注释。为了解决这个问题，我们引入：（i）ASID-1M，一个开源集合，包含一百万个结构化、细粒度的视听指令注释，具有单属性和多属性监督； (ii) ASID-Verify，一种可扩展的注释数据管理管道，具有自动验证和细化功能，可强制描述与相应视听内容之间的语义和时间一致性； (iii) ASID-Captioner，一种通过监督微调 (SFT) 在 ASID-1M 上训练的视频理解模型。涵盖视听字幕、按属性字幕、基于字幕的 QA 和基于字幕的时间接地的七个基准的实验表明，ASID-Captioner 提高了细粒度字幕质量，同时减少幻觉并改善指令遵循。它在开源模型中实现了最先进的性能，并且与Gemini-3-Pro具有竞争力。

</details>

---

## 4. Detecting Object Tracking Failure via Sequential Hypothesis Testing / 通过序贯假设检验检测对象跟踪失败

**Date**: 2026-02-13 | **arXiv**: [2602.12983v1](http://arxiv.org/abs/2602.12983v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.12983v1)

**Categories**: cs.CV, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Real-time online object tracking in videos constitutes a core task in computer vision, with wide-ranging applications including video surveillance, motion capture, and robotics. Deployed tracking systems usually lack formal safety assurances to convey when tracking is reliable and when it may fail, at best relying on heuristic measures of model confidence to raise alerts. To obtain such assurances we propose interpreting object tracking as a sequential hypothesis test, wherein evidence for or against tracking failures is gradually accumulated over time. Leveraging recent advancements in the field, our sequential test (formalized as an e-process) quickly identifies when tracking failures set in whilst provably containing false alerts at a desired rate, and thus limiting potentially costly re-calibration or intervention steps. The approach is computationally light-weight, requires no extra training or fine-tuning, and is in principle model-agnostic. We propose both supervised and unsupervised variants by leveraging either ground-truth or solely internal tracking information, and demonstrate its effectiveness for two established tracking models across four video benchmarks. As such, sequential testing can offer a statistically grounded and efficient mechanism to incorporate safety assurances into real-time tracking systems.

视频中的实时在线对象跟踪构成了计算机视觉的核心任务，具有广泛的应用，包括视频监控、动作捕捉和机器人技术。已部署的跟踪系统通常缺乏正式的安全保证来传达跟踪何时可靠以及何时可能失败，最多只能依靠模型置信度的启发式措施来发出警报。为了获得这样的保证，我们建议将对象跟踪解释为顺序假设检验，其中支持或反对跟踪失败的证据随着时间的推移逐渐积累。利用该领域的最新进展，我们的顺序测试（形式化为电子流程）可以快速识别何时出现跟踪故障，同时以所需的速度证明包含错误警报，从而限制可能成本高昂的重新校准或干预步骤。该方法计算量轻，不需要额外的训练或微调，并且原则上与模型无关。我们通过利用真实情况或仅内部跟踪信息提出监督和无监督变体，并在四个视频基准中证明其对两个已建立的跟踪模型的有效性。因此，顺序测试可以提供一种基于统计的有效机制，将安全保证纳入实时跟踪系统。

</details>

---



</details>

<details><summary><b>2026-02-14 (16 papers)</b></summary>

# arXiv Video Papers - 2026-02-14

**Paper Count**: 16

---

## 1. MonarchRT: Efficient Attention for Real-Time Video Generation / MonarchRT：实时视频生成的高效关注

**Date**: 2026-02-12 | **arXiv**: [2602.12271v1](http://arxiv.org/abs/2602.12271v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.12271v1)

**Categories**: cs.CV, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Real-time video generation with Diffusion Transformers is bottlenecked by the quadratic cost of 3D self-attention, especially in real-time regimes that are both few-step and autoregressive, where errors compound across time and each denoising step must carry substantially more information. In this setting, we find that prior sparse-attention approximations break down, despite showing strong results for bidirectional, many-step diffusion. Specifically, we observe that video attention is not reliably sparse, but instead combines pronounced periodic structure driven by spatiotemporal position with dynamic, sparse semantic correspondences and dense mixing, exceeding the representational capacity of even oracle top-k attention. Building on this insight, we propose Monarch-RT, a structured attention parameterization for video diffusion models that factorizes attention using Monarch matrices. Through appropriately aligned block structure and our extended tiled Monarch parameterization, we achieve high expressivity while preserving computational efficiency. We further overcome the overhead of parameterization through finetuning, with custom Triton kernels. We first validate the high efficacy of Monarch-RT over existing sparse baselines designed only for bidirectional models. We further observe that Monarch-RT attains up to 95% attention sparsity with no loss in quality when applied to the state-of-the-art model Self-Forcing, making Monarch-RT a pioneering work on highly-capable sparse attention parameterization for real-time video generation. Our optimized implementation outperforms FlashAttention-2, FlashAttention-3, and FlashAttention-4 kernels on Nvidia RTX 5090, H100, and B200 GPUs respectively, providing kernel speedups in the range of 1.4-11.8X. This enables us, for the first time, to achieve true real-time video generation with Self-Forcing at 16 FPS on a single RTX 5090.

使用扩散变压器的实时视频生成受到 3D 自注意力二次成本的瓶颈，特别是在少步和自回归的实时机制中，其中误差随时间复合，每个去噪步骤必须携带更多信息。在这种情况下，我们发现尽管双向、多步扩散显示出很强的结果，但先前的稀疏注意力近似方法还是失效了。具体来说，我们观察到视频注意力并不是可靠的稀疏，而是将由时空位置驱动的显着周期结构与动态、稀疏语义对应和密集混合相结合，甚至超过了 oracle top-k 注意力的表示能力。基于这一见解，我们提出了 Monarch-RT，这是一种用于视频扩散模型的结构化注意力参数化，它使用 Monarch 矩阵分解注意力。通过适当对齐的块结构和扩展的平铺 Monarch 参数化，我们在保持计算效率的同时实现了高表达力。我们通过使用自定义 Triton 内核进行微调，进一步克服了参数化的开销。我们首先验证 Monarch-RT 相对于仅为双向模型设计的现有稀疏基线的高效性。我们进一步观察到，当应用于最先进的模型 Self-Forcing 时，Monarch-RT 获得了高达 95% 的注意力稀疏度，且质量没有损失，这使得 Monarch-RT 成为实时视频生成的高性能稀疏注意力参数化的开创性工作。我们的优化实现在 Nvidia RTX 5090、H100 和 B200 GPU 上的性能分别优于 FlashAttention-2、FlashAttention-3 和 FlashAttention-4 内核，提供 1.4-11.8 倍的内核加速。这使我们第一次能够在单个 RTX 5090 上以 16 FPS 的速度实现真正的实时视频生成。

</details>

---

## 2. DreamID-Omni: Unified Framework for Controllable Human-Centric Audio-Video Generation / DreamID-Omni：可控的以人为本的音频视频生成的统一框架

**Date**: 2026-02-12 | **arXiv**: [2602.12160v1](http://arxiv.org/abs/2602.12160v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.12160v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Recent advancements in foundation models have revolutionized joint audio-video generation. However, existing approaches typically treat human-centric tasks including reference-based audio-video generation (R2AV), video editing (RV2AV) and audio-driven video animation (RA2V) as isolated objectives. Furthermore, achieving precise, disentangled control over multiple character identities and voice timbres within a single framework remains an open challenge. In this paper, we propose DreamID-Omni, a unified framework for controllable human-centric audio-video generation. Specifically, we design a Symmetric Conditional Diffusion Transformer that integrates heterogeneous conditioning signals via a symmetric conditional injection scheme. To resolve the pervasive identity-timbre binding failures and speaker confusion in multi-person scenarios, we introduce a Dual-Level Disentanglement strategy: Synchronized RoPE at the signal level to ensure rigid attention-space binding, and Structured Captions at the semantic level to establish explicit attribute-subject mappings. Furthermore, we devise a Multi-Task Progressive Training scheme that leverages weakly-constrained generative priors to regularize strongly-constrained tasks, preventing overfitting and harmonizing disparate objectives. Extensive experiments demonstrate that DreamID-Omni achieves comprehensive state-of-the-art performance across video, audio, and audio-visual consistency, even outperforming leading proprietary commercial models. We will release our code to bridge the gap between academic research and commercial-grade applications.

基础模型的最新进展彻底改变了联合音频视频生成。然而，现有方法通常将以人为中心的任务视为孤立的目标，包括基于参考的音频视频生成（R2AV）、视频编辑（RV2AV）和音频驱动视频动画（RA2V）。此外，在单一框架内实现对多个角色身份和音色的精确、分离的控制仍然是一个开放的挑战。在本文中，我们提出了 DreamID-Omni，这是一个用于可控的以人为中心的音频视频生成的统一框架。具体来说，我们设计了一个对称条件扩散变压器，它通过对称条件注入方案集成异构调节信号。为了解决多人场景中普遍存在的身份-音色绑定失败和说话者混乱的问题，我们引入了一种双层解缠策略：信号级别的同步 RoPE 以确保严格的注意力空间绑定，语义级别的结构化字幕以建立明确的属性-主题映射。此外，我们设计了一种多任务渐进训练方案，利用弱约束的生成先验来规范强约束的任务，防止过度拟合并协调不同的目标。大量实验表明，DreamID-Omni 在视频、音频和视听一致性方面实现了全面的最先进性能，甚至超越了领先的专有商业模型。我们将发布我们的代码，以弥合学术研究和商业级应用程序之间的差距。

</details>

---

## 3. FAIL: Flow Matching Adversarial Imitation Learning for Image Generation / 失败：用于图像生成的流匹配对抗性模仿学习

**Date**: 2026-02-12 | **arXiv**: [2602.12155v1](http://arxiv.org/abs/2602.12155v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.12155v1)

**Categories**: cs.CV

**Code**: https://github.com/HansPolo113/FAIL.

<details><summary><b>Abstract / 摘要</b></summary>

Post-training of flow matching models-aligning the output distribution with a high-quality target-is mathematically equivalent to imitation learning. While Supervised Fine-Tuning mimics expert demonstrations effectively, it cannot correct policy drift in unseen states. Preference optimization methods address this but require costly preference pairs or reward modeling. We propose Flow Matching Adversarial Imitation Learning (FAIL), which minimizes policy-expert divergence through adversarial training without explicit rewards or pairwise comparisons. We derive two algorithms: FAIL-PD exploits differentiable ODE solvers for low-variance pathwise gradients, while FAIL-PG provides a black-box alternative for discrete or computationally constrained settings. Fine-tuning FLUX with only 13,000 demonstrations from Nano Banana pro, FAIL achieves competitive performance on prompt following and aesthetic benchmarks. Furthermore, the framework generalizes effectively to discrete image and video generation, and functions as a robust regularizer to mitigate reward hacking in reward-based optimization. Code and data are available at https://github.com/HansPolo113/FAIL.

流匹配模型的后训练——将输出分布与高质量目标对齐——在数学上等同于模仿学习。虽然监督微调有效地模仿了专家的演示，但它无法纠正看不见的状态中的政策漂移。偏好优化方法可以解决这个问题，但需要昂贵的偏好对或奖励建模。我们提出了流匹配对抗性模仿学习（FAIL），它通过对抗性训练来最小化政策专家分歧，而无需明确的奖励或成对比较。我们推导出两种算法：FAIL-PD 利用可微分 ODE 求解器来实现低方差路径梯度，而 FAIL-PG 则为离散或计算约束设置提供黑盒替代方案。仅通过 Nano Banana pro 的 13,000 次演示对 FLUX 进行微调，FAIL 在快速跟随和美学基准方面实现了具有竞争力的性能。此外，该框架有效地推广到离散图像和视频生成，并作为强大的正则化器来减轻基于奖励的优化中的奖励黑客行为。代码和数据可在 https://github.com/HansPolo113/FAIL 获取。

</details>

---

## 4. How to Sample High Quality 3D Fractals for Action Recognition Pre-Training? / 如何对高质量 3D 分形进行采样以进行动作识别预训练？

**Date**: 2026-02-12 | **arXiv**: [2602.11810v1](http://arxiv.org/abs/2602.11810v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11810v1)

**Categories**: cs.CV, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Synthetic datasets are being recognized in the deep learning realm as a valuable alternative to exhaustively labeled real data. One such synthetic data generation method is Formula Driven Supervised Learning (FDSL), which can provide an infinite number of perfectly labeled data through a formula driven approach, such as fractals or contours. FDSL does not have common drawbacks like manual labor, privacy and other ethical concerns. In this work we generate 3D fractals using 3D Iterated Function Systems (IFS) for pre-training an action recognition model. The fractals are temporally transformed to form a video that is used as a pre-training dataset for downstream task of action recognition. We find that standard methods of generating fractals are slow and produce degenerate 3D fractals. Therefore, we systematically explore alternative ways of generating fractals and finds that overly-restrictive approaches, while generating aesthetically pleasing fractals, are detrimental for downstream task performance. We propose a novel method, Targeted Smart Filtering, to address both the generation speed and fractal diversity issue. The method reports roughly 100 times faster sampling speed and achieves superior downstream performance against other 3D fractal filtering methods.

合成数据集在深度学习领域被认为是详尽标记的真实数据的有价值的替代方案。其中一种合成数据生成方法是公式驱动监督学习（FDSL），它可以通过公式驱动方法（例如分形或轮廓）提供无限数量的完美标记数据。 FDSL 没有体力劳动、隐私和其他道德问题等常见缺点。在这项工作中，我们使用 3D 迭代函数系统 (IFS) 生成 3D 分形，以预训练动作识别模型。分形在时间上进行变换以形成视频，该视频用作动作识别下游任务的预训练数据集。我们发现生成分形的标准方法很慢并且会产生简并的 3D 分形。因此，我们系统地探索了生成分形的替代方法，并发现过度限制的方法虽然生成美观的分形，但不利于下游任务的性能。我们提出了一种新颖的方法，即目标智能过滤，来解决生成速度和分形多样性问题。与其他 3D 分形过滤方法相比，该方法的采样速度快了大约 100 倍，并实现了卓越的下游性能。

</details>

---

## 5. STVG-R1: Incentivizing Instance-Level Reasoning and Grounding in Videos via Reinforcement Learning / STVG-R1：通过强化学习激励视频中的实例级推理和基础

**Date**: 2026-02-12 | **arXiv**: [2602.11730v1](http://arxiv.org/abs/2602.11730v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11730v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

In vision-language models (VLMs), misalignment between textual descriptions and visual coordinates often induces hallucinations. This issue becomes particularly severe in dense prediction tasks such as spatial-temporal video grounding (STVG). Prior approaches typically focus on enhancing visual-textual alignment or attaching auxiliary decoders. However, these strategies inevitably introduce additional trainable modules, leading to significant annotation costs and computational overhead. In this work, we propose a novel visual prompting paradigm that avoids the difficult problem of aligning coordinates across modalities. Specifically, we reformulate per-frame coordinate prediction as a compact instance-level identification problem by assigning each object a unique, temporally consistent ID. These IDs are embedded into the video as visual prompts, providing explicit and interpretable inputs to the VLMs. Furthermore, we introduce STVG-R1, the first reinforcement learning framework for STVG, which employs a task-driven reward to jointly optimize temporal accuracy, spatial consistency, and structural format regularization. Extensive experiments on six benchmarks demonstrate the effectiveness of our approach. STVG-R1 surpasses the baseline Qwen2.5-VL-7B by a remarkable margin of 20.9% on m_IoU on the HCSTVG-v2 benchmark, establishing a new state of the art (SOTA). Surprisingly, STVG-R1 also exhibits strong zero-shot generalization to multi-object referring video object segmentation tasks, achieving a SOTA 47.3% J&F on MeViS.

在视觉语言模型（VLM）中，文本描述和视觉坐标之间的错位通常会引起幻觉。这个问题在时空视频接地（STVG）等密集预测任务中变得尤为严重。现有方法通常侧重于增强视觉文本对齐或附加辅助解码器。然而，这些策略不可避免地引入额外的可训练模块，导致显着的注释成本和计算开销。在这项工作中，我们提出了一种新颖的视觉提示范例，避免了跨模态对齐坐标的难题。具体来说，我们通过为每个对象分配一个唯一的、时间一致的 ID，将每帧坐标预测重新表述为一个紧凑的实例级识别问题。这些 ID 作为视觉提示嵌入到视频中，为 VLM 提供明确且可解释的输入。此外，我们还引入了 STVG-R1，这是 STVG 的第一个强化学习框架，它采用任务驱动的奖励来联合优化时间准确性、空间一致性和结构格式正则化。对六个基准的广泛实验证明了我们方法的有效性。 STVG-R1 在 HCSTVG-v2 基准上的 m_IoU 上超越了基线 Qwen2.5-VL-7B，显着提高了 20.9%，建立了新的最先进技术 (SOTA)。令人惊讶的是，STVG-R1 还对多对象参考视频对象分割任务表现出强大的零样本泛化能力，在 MeViS 上实现了 SOTA 47.3% J&F。

</details>

---

## 6. LUVE : Latent-Cascaded Ultra-High-Resolution Video Generation with Dual Frequency Experts / LUVE：双频专家的潜在级联超高分辨率视频生成

**Date**: 2026-02-12 | **arXiv**: [2602.11564v1](http://arxiv.org/abs/2602.11564v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11564v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Recent advances in video diffusion models have significantly improved visual quality, yet ultra-high-resolution (UHR) video generation remains a formidable challenge due to the compounded difficulties of motion modeling, semantic planning, and detail synthesis. To address these limitations, we propose \textbf{LUVE}, a \textbf{L}atent-cascaded \textbf{U}HR \textbf{V}ideo generation framework built upon dual frequency \textbf{E}xperts. LUVE employs a three-stage architecture comprising low-resolution motion generation for motion-consistent latent synthesis, video latent upsampling that performs resolution upsampling directly in the latent space to mitigate memory and computational overhead, and high-resolution content refinement that integrates low-frequency and high-frequency experts to jointly enhance semantic coherence and fine-grained detail generation. Extensive experiments demonstrate that our LUVE achieves superior photorealism and content fidelity in UHR video generation, and comprehensive ablation studies further validate the effectiveness of each component. The project is available at \href{https://unicornanrocinu.github.io/LUVE_web/}{https://github.io/LUVE/}.

视频扩散模型的最新进展显着提高了视觉质量，但由于运动建模、语义规划和细节合成的复杂困难，超高分辨率 (UHR) 视频生成仍然是一个艰巨的挑战。为了解决这些限制，我们提出了 \textbf{LUVE}，一个基于双频 \textbf{E}xperts 构建的 \textbf{L}atent 级联 \textbf{U}HR \textbf{V}ideo 生成框架。 LUVE 采用三阶段架构，包括用于运动一致潜在合成的低分辨率运动生成、直接在潜在空间中执行分辨率上采样以减轻内存和计算开销的视频潜在上采样，以及集成低频和高频专家以共同增强语义一致性和细粒度细节生成的高分辨率内容细化。大量实验表明，我们的 LUVE 在 UHR 视频生成中实现了卓越的照片真实感和内容保真度，全面的消融研究进一步验证了每个组件的有效性。该项目位于 \href{https://unicornanrocinu.github.io/LUVE_web/}{https://github.io/LUVE/}。

</details>

---

## 7. SAM3-LiteText: An Anatomical Study of the SAM3 Text Encoder for Efficient Vision-Language Segmentation / SAM3-LiteText：用于高效视觉语言分割的 SAM3 文本编码器的解剖研究

**Date**: 2026-02-12 | **arXiv**: [2602.12173v1](http://arxiv.org/abs/2602.12173v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.12173v1)

**Categories**: cs.AI

**Code**: https://github.com/SimonZeng7108/efficientsam3/tree/sam3_litetext.

<details><summary><b>Abstract / 摘要</b></summary>

Vision-language segmentation models such as SAM3 enable flexible, prompt-driven visual grounding, but inherit large, general-purpose text encoders originally designed for open-ended language understanding. In practice, segmentation prompts are short, structured, and semantically constrained, leading to substantial over-provisioning in text encoder capacity and persistent computational and memory overhead. In this paper, we perform a large-scale anatomical analysis of text prompting in vision-language segmentation, covering 404,796 real prompts across multiple benchmarks. Our analysis reveals severe redundancy: most context windows are underutilized, vocabulary usage is highly sparse, and text embeddings lie on low-dimensional manifold despite high-dimensional representations. Motivated by these findings, we propose SAM3-LiteText, a lightweight text encoding framework that replaces the original SAM3 text encoder with a compact MobileCLIP student that is optimized by knowledge distillation. Extensive experiments on image and video segmentation benchmarks show that SAM3-LiteText reduces text encoder parameters by up to 88%, substantially reducing static memory footprint, while maintaining segmentation performance comparable to the original model. Code: https://github.com/SimonZeng7108/efficientsam3/tree/sam3_litetext.

SAM3 等视觉语言分割模型可实现灵活、提示驱动的视觉基础，但继承了最初为开放式语言理解而设计的大型通用文本编码器。在实践中，分段提示很短、结构化且语义受限，导致文本编码器容量的大量过度配置以及持续的计算和内存开销。在本文中，我们对视觉语言分割中的文本提示进行了大规模的解剖分析，涵盖了跨多个基准的 404,796 个真实提示。我们的分析揭示了严重的冗余：大多数上下文窗口未得到充分利用，词汇使用高度稀疏，尽管具有高维表示，但文本嵌入位于低维流形上。受这些发现的启发，我们提出了 SAM3-LiteText，这是一种轻量级文本编码框架，用通过知识蒸馏优化的紧凑型 MobileCLIP Student 取代了原始的 SAM3 文本编码器。对图像和视频分割基准的大量实验表明，SAM3-LiteText 将文本编码器参数减少了高达 88%，大大减少了静态内存占用，同时保持了与原始模型相当的分割性能。代码：https://github.com/SimonZeng7108/efficientsam3/tree/sam3_litetext。

</details>

---

## 8. HLA: Hadamard Linear Attention / HLA：Hadamard 线性注意力

**Date**: 2026-02-12 | **arXiv**: [2602.12128v1](http://arxiv.org/abs/2602.12128v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.12128v1)

**Categories**: cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

The attention mechanism is an important reason for the success of transformers. It relies on computing pairwise relations between tokens. To reduce the high computational cost of standard quadratic attention, linear attention has been proposed as an efficient approximation. It employs kernel functions that are applied independently to the inputs before the pairwise similarities are calculated. That allows for an efficient computational procedure which, however, amounts to a low-degree rational function approximating softmax.   We propose Hadamard Linear Attention (HLA). Unlike previous works on linear attention, the nonlinearity in HLA is not applied separately to queries and keys, but, analogously to standard softmax attention, after the pairwise similarities have been computed. It will be shown that the proposed nonlinearity amounts to a higher-degree rational function to approximate softmax. An efficient computational scheme for the proposed method is derived that is similar to that of standard linear attention. In contrast to other approaches, no time-consuming tensor reshaping is necessary to apply the proposed algorithm. The effectiveness of the approach is demonstrated by applying it to a large diffusion transformer model for video generation, an application that involves very large amounts of tokens.

注意力机制是 Transformer 成功的重要原因。它依赖于计算标记之间的成对关系。为了降低标准二次注意力的高计算成本，线性注意力被提出作为一种有效的近似。它采用在计算成对相似度之前独立应用于输入的核函数。这允许高效的计算过程，然而，这相当于一个逼近 softmax 的低次有理函数。   我们提出哈达玛线性注意力（HLA）。与之前的线性注意力工作不同，HLA 中的非线性不是单独应用于查询和键，而是类似于标准的 softmax 注意力，在计算成对相似性之后应用。将会表明，所提出的非线性相当于一个更高阶的有理函数来近似 softmax。推导了所提出方法的有效计算方案，该方案类似于标准线性注意的方案。与其他方法相比，应用所提出的算法不需要耗时的张量整形。该方法的有效性通过将其应用于用于视频生成的大型扩散变压器模型（涉及大量令牌的应用程序）来证明。

</details>

---

## 9. Beyond End-to-End Video Models: An LLM-Based Multi-Agent System for Educational Video Generation / 超越端到端视频模型：用于教育视频生成的基于 LLM 的多代理系统

**Date**: 2026-02-12 | **arXiv**: [2602.11790v1](http://arxiv.org/abs/2602.11790v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11790v1)

**Categories**: cs.AI, cs.CL

<details><summary><b>Abstract / 摘要</b></summary>

Although recent end-to-end video generation models demonstrate impressive performance in visually oriented content creation, they remain limited in scenarios that require strict logical rigor and precise knowledge representation, such as instructional and educational media. To address this problem, we propose LAVES, a hierarchical LLM-based multi-agent system for generating high-quality instructional videos from educational problems. The LAVES formulates educational video generation as a multi-objective task that simultaneously demands correct step-by-step reasoning, pedagogically coherent narration, semantically faithful visual demonstrations, and precise audio--visual alignment. To address the limitations of prior approaches--including low procedural fidelity, high production cost, and limited controllability--LAVES decomposes the generation workflow into specialized agents coordinated by a central Orchestrating Agent with explicit quality gates and iterative critique mechanisms. Specifically, the Orchestrating Agent supervises a Solution Agent for rigorous problem solving, an Illustration Agent that produces executable visualization codes, and a Narration Agent for learner-oriented instructional scripts. In addition, all outputs from the working agents are subject to semantic critique, rule-based constraints, and tool-based compilation checks. Rather than directly synthesizing pixels, the system constructs a structured executable video script that is deterministically compiled into synchronized visuals and narration using template-driven assembly rules, enabling fully automated end-to-end production without manual editing. In large-scale deployments, LAVES achieves a throughput exceeding one million videos per day, delivering over a 95% reduction in cost compared to current industry-standard approaches while maintaining a high acceptance rate.

尽管最近的端到端视频生成模型在面向视觉的内容创建方面表现出了令人印象深刻的性能，但它们在需要严格逻辑严谨性和精确知识表示的场景中仍然受到限制，例如教学和教育媒体。为了解决这个问题，我们提出了 LAVES，一种基于 LLM 的分层多智能体系统，用于根据教育问题生成高质量的教学视频。 LAVES 将教育视频生成制定为一项多目标任务，同时要求正确的逐步推理、教学上连贯的叙述、语义上忠实的视觉演示以及精确的视听对齐。为了解决先前方法的局限性（包括程序保真度低、生产成本高和可控性有限），LAVES 将生成工作流程分解为由具有明确质量门和迭代批评机制的中央编排代理协调的专门代理。具体来说，编排代理监督解决方案代理以严格解决问题，插图代理生成可执行的可视化代码，以及叙述代理以用于面向学习者的教学脚本。此外，工作代理的所有输出都受到语义批评、基于规则的约束和基于工具的编译检查。该系统不是直接合成像素，而是构建一个结构化的可执行视频脚本，该脚本使用模板驱动的组装规则确定性地编译成同步的视觉效果和旁白，从而实现完全自动化的端到端制作，无需手动编辑。在大规模部署中，LAVES 的吞吐量每天超过 100 万个视频，与当前行业标准方法相比，成本降低了 95% 以上，同时保持了较高的接受率。

</details>

---

## 10. VLAW: Iterative Co-Improvement of Vision-Language-Action Policy and World Model / VLAW：愿景-语言-行动政策和世界模型的迭代共同改进

**Date**: 2026-02-12 | **arXiv**: [2602.12063v1](http://arxiv.org/abs/2602.12063v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.12063v1)

**Categories**: cs.RO

**Project**: https://sites.google.com/view/vla-w  <details><summary><b>Abstract / 摘要</b></summary>

The goal of this paper is to improve the performance and reliability of vision-language-action (VLA) models through iterative online interaction. Since collecting policy rollouts in the real world is expensive, we investigate whether a learned simulator-specifically, an action-conditioned video generation model-can be used to generate additional rollout data. Unfortunately, existing world models lack the physical fidelity necessary for policy improvement: they are predominantly trained on demonstration datasets that lack coverage of many different physical interactions (particularly failure cases) and struggle to accurately model small yet critical physical details in contact-rich object manipulation. We propose a simple iterative improvement algorithm that uses real-world roll-out data to improve the fidelity of the world model, which can then, in turn, be used to generate supplemental synthetic data for improving the VLA model. In our experiments on a real robot, we use this approach to improve the performance of a state-of-the-art VLA model on multiple downstream tasks. We achieve a 39.2% absolute success rate improvement over the base policy and 11.6% improvement from training with the generated synthetic rollouts. Videos can be found at this anonymous website: https://sites.google.com/view/vla-w

本文的目标是通过迭代在线交互来提高视觉-语言-动作（VLA）模型的性能和可靠性。由于在现实世界中收集策略推出的成本很高，因此我们研究了是否可以使用学习的模拟器（具体而言，动作条件视频生成模型）来生成额外的推出数据。不幸的是，现有的世界模型缺乏政策改进所需的物理保真度：它们主要是在演示数据集上进行训练的，这些数据集缺乏对许多不同物理交互（特别是失败案例）的覆盖，并且很难在接触丰富的对象操作中准确地模拟微小但关键的物理细节。我们提出了一种简单的迭代改进算法，该算法使用现实世界的转出数据来提高世界模型的保真度，然后可以使用该算法生成补充合成数据以改进 VLA 模型。在我们对真实机器人的实验中，我们使用这种方法来提高最先进的 VLA 模型在多个下游任务上的性能。与基本策略相比，我们的绝对成功率提高了 39.2%，通过生成的综合部署进行训练，绝对成功率提高了 11.6%。视频可以在这个匿名网站上找到：https://sites.google.com/view/vla-w

</details>

---

## 11. SurfPhase: 3D Interfacial Dynamics in Two-Phase Flows from Sparse Videos / SurfPhase：稀疏视频中两相流的 3D 界面动力学

**Date**: 2026-02-11 | **arXiv**: [2602.11154v1](http://arxiv.org/abs/2602.11154v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11154v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Interfacial dynamics in two-phase flows govern momentum, heat, and mass transfer, yet remain difficult to measure experimentally. Classical techniques face intrinsic limitations near moving interfaces, while existing neural rendering methods target single-phase flows with diffuse boundaries and cannot handle sharp, deformable liquid-vapor interfaces. We propose SurfPhase, a novel model for reconstructing 3D interfacial dynamics from sparse camera views. Our approach integrates dynamic Gaussian surfels with a signed distance function formulation for geometric consistency, and leverages a video diffusion model to synthesize novel-view videos to refine reconstruction from sparse observations. We evaluate on a new dataset of high-speed pool boiling videos, demonstrating high-quality view synthesis and velocity estimation from only two camera views. Project website: https://yuegao.me/SurfPhase.

两相流中的界面动力学控制动量、热量和质量传递，但仍然难以通过实验测量。经典技术在移动界面附近面临固有的局限性，而现有的神经渲染方法针对具有扩散边界的单相流，无法处理尖锐、可变形的液-气界面。我们提出了 SurfPhase，一种从稀疏相机视图重建 3D 界面动力学的新颖模型。我们的方法将动态高斯面元与带符号距离函数公式相结合以实现几何一致性，并利用视频扩散模型来合成新颖的视图视频，以改进稀疏观测的重建。我们对高速池沸腾视频的新数据集进行了评估，仅通过两个摄像机视图演示了高质量的视图合成和速度估计。项目网站：https://yuegao.me/SurfPhase。

</details>

---

## 12. HairWeaver: Few-Shot Photorealistic Hair Motion Synthesis with Sim-to-Real Guided Video Diffusion / HairWeaver：通过模拟到真实引导视频扩散进行少镜头真实感头发运动合成

**Date**: 2026-02-11 | **arXiv**: [2602.11117v1](http://arxiv.org/abs/2602.11117v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11117v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

We present HairWeaver, a diffusion-based pipeline that animates a single human image with realistic and expressive hair dynamics. While existing methods successfully control body pose, they lack specific control over hair, and as a result, fail to capture the intricate hair motions, resulting in stiff and unrealistic animations. HairWeaver overcomes this limitation using two specialized modules: a Motion-Context-LoRA to integrate motion conditions and a Sim2Real-Domain-LoRA to preserve the subject's photoreal appearance across different data domains. These lightweight components are designed to guide a video diffusion backbone while maintaining its core generative capabilities. By training on a specialized dataset of dynamic human motion generated from a CG simulator, HairWeaver affords fine control over hair motion and ultimately learns to produce highly realistic hair that responds naturally to movement. Comprehensive evaluations demonstrate that our approach sets a new state of the art, producing lifelike human hair animations with dynamic details.

我们推出了 HairWeaver，这是一种基于扩散的管道，可以通过逼真且富有表现力的头发动态来对单个人类图像进行动画处理。虽然现有方法成功地控制了身体姿势，但它们缺乏对头发的具体控制，因此无法捕捉复杂的头发运动，导致动画僵硬且不切实际。 HairWeaver 使用两个专用模块克服了这一限制：一个用于集成运动条件的 Motion-Context-LoRA，另一个是 Sim2Real-Domain-LoRA，用于在不同数据域中保留主体的真实外观。这些轻量级组件旨在指导视频传播主干，同时保持其核心生成功能。通过对 CG 模拟器生成的动态人体运动的专门数据集进行训练，HairWeaver 可以对头发运动进行精细控制，并最终学会生成对运动做出自然响应的高度逼真的头发。综合评估表明，我们的方法树立了新的技术水平，可以制作具有动态细节的逼真的人发动画。

</details>

---

## 13. FastFlow: Accelerating The Generative Flow Matching Models with Bandit Inference / FastFlow：通过强盗推理加速生成流匹配模型

**Date**: 2026-02-11 | **arXiv**: [2602.11105v1](http://arxiv.org/abs/2602.11105v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11105v1)

**Categories**: cs.CV

**Code**: https://github.com/Div290/FastFlow.

<details><summary><b>Abstract / 摘要</b></summary>

Flow-matching models deliver state-of-the-art fidelity in image and video generation, but the inherent sequential denoising process renders them slower. Existing acceleration methods like distillation, trajectory truncation, and consistency approaches are static, require retraining, and often fail to generalize across tasks. We propose FastFlow, a plug-and-play adaptive inference framework that accelerates generation in flow matching models. FastFlow identifies denoising steps that produce only minor adjustments to the denoising path and approximates them without using the full neural network models used for velocity predictions. The approximation utilizes finite-difference velocity estimates from prior predictions to efficiently extrapolate future states, enabling faster advancements along the denoising path at zero compute cost. This enables skipping computation at intermediary steps. We model the decision of how many steps to safely skip before requiring a full model computation as a multi-armed bandit problem. The bandit learns the optimal skips to balance speed with performance. FastFlow integrates seamlessly with existing pipelines and generalizes across image generation, video generation, and editing tasks. Experiments demonstrate a speedup of over 2.6x while maintaining high-quality outputs. The source code for this work can be found at https://github.com/Div290/FastFlow.

流匹配模型在图像和视频生成中提供最先进的保真度，但固有的顺序去噪过程使它们速度变慢。现有的加速方法（例如蒸馏、轨迹截断和一致性方法）是静态的，需要重新训练，并且通常无法跨任务泛化。我们提出了 FastFlow，一种即插即用的自适应推理框架，可加速流匹配模型的生成。 FastFlow 识别仅对去噪路径产生微小调整的去噪步骤，并在不使用用于速度预测的完整神经网络模型的情况下对其进行近似。该近似利用先前预测的有限差分速度估计来有效地推断未来状态，从而以零计算成本沿着去噪路径实现更快的进展。这使得能够跳过中间步骤的计算。我们将在需要完整模型计算之前安全跳过多少步骤的决策建模为多臂老虎机问题。老虎机学习最佳跳跃以平衡速度与性能。 FastFlow 与现有管道无缝集成，并可泛化图像生成、视频生成和编辑任务。实验表明，在保持高质量输出的同时，速度提高了 2.6 倍以上。这项工作的源代码可以在 https://github.com/Div290/FastFlow 找到。

</details>

---

## 14. ReTracing: An Archaeological Approach Through Body, Machine, and Generative Systems / 追溯：通过身体、机器和生成系统的考古方法

**Date**: 2026-02-11 | **arXiv**: [2602.11242v1](http://arxiv.org/abs/2602.11242v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11242v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

We present ReTracing, a multi-agent embodied performance art that adopts an archaeological approach to examine how artificial intelligence shapes, constrains, and produces bodily movement. Drawing from science-fiction novels, the project extracts sentences that describe human-machine interaction. We use large language models (LLMs) to generate paired prompts "what to do" and "what not to do" for each excerpt. A diffusion-based text-to-video model transforms these prompts into choreographic guides for a human performer and motor commands for a quadruped robot. Both agents enact the actions on a mirrored floor, captured by multi-camera motion tracking and reconstructed into 3D point clouds and motion trails, forming a digital archive of motion traces. Through this process, ReTracing serves as a novel approach to reveal how generative systems encode socio-cultural biases through choreographed movements. Through an immersive interplay of AI, human, and robot, ReTracing confronts a critical question of our time: What does it mean to be human among AIs that also move, think, and leave traces behind?

我们展示了 ReTracing，这是一种多智能体体现的表演艺术，它采用考古学的方法来研究人工智能如何塑造、约束和产生身体运动。该项目借鉴科幻小说，提取描述人机交互的句子。我们使用大型语言模型 (LLM) 为每个摘录生成配对提示“该做什么”和“不该做什么”。基于扩散的文本到视频模型将这些提示转换为人类表演者的编舞指南和四足机器人的运动命令。两个代理都在镜像地板上执行动作，通过多摄像头运动跟踪捕获并重建为 3D 点云和运动轨迹，形成运动轨迹的数字档案。通过这个过程，ReTracing 作为一种新颖的方法来揭示生成系统如何通过精心设计的动作来编码社会文化偏见。通过人工智能、人类和机器人的沉浸式互动，《ReTracing》面临着我们这个时代的一个关键问题：在同样会移动、思考和留下痕迹的人工智能中，作为人类意味着什么？

</details>

---

## 15. Flow caching for autoregressive video generation / 用于自回归视频生成的流缓存

**Date**: 2026-02-11 | **arXiv**: [2602.10825v1](http://arxiv.org/abs/2602.10825v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.10825v1)

**Categories**: cs.CV, cs.AI

**Code**: https://github.com/mikeallen39/FlowCache.

<details><summary><b>Abstract / 摘要</b></summary>

Autoregressive models, often built on Transformer architectures, represent a powerful paradigm for generating ultra-long videos by synthesizing content in sequential chunks. However, this sequential generation process is notoriously slow. While caching strategies have proven effective for accelerating traditional video diffusion models, existing methods assume uniform denoising across all frames-an assumption that breaks down in autoregressive models where different video chunks exhibit varying similarity patterns at identical timesteps. In this paper, we present FlowCache, the first caching framework specifically designed for autoregressive video generation. Our key insight is that each video chunk should maintain independent caching policies, allowing fine-grained control over which chunks require recomputation at each timestep. We introduce a chunkwise caching strategy that dynamically adapts to the unique denoising characteristics of each chunk, complemented by a joint importance-redundancy optimized KV cache compression mechanism that maintains fixed memory bounds while preserving generation quality. Our method achieves remarkable speedups of 2.38 times on MAGI-1 and 6.7 times on SkyReels-V2, with negligible quality degradation (VBench: 0.87 increase and 0.79 decrease respectively). These results demonstrate that FlowCache successfully unlocks the potential of autoregressive models for real-time, ultra-long video generation-establishing a new benchmark for efficient video synthesis at scale. The code is available at https://github.com/mikeallen39/FlowCache.

自回归模型通常建立在 Transformer 架构之上，代表了通过合成连续块中的内容来生成超长视频的强大范例。然而，这种顺序生成过程是出了名的慢。虽然缓存策略已被证明对于加速传统视频扩散模型是有效的，但现有方法假设所有帧都采用统一的去噪——这种假设在自回归模型中被打破，其中不同的视频块在相同的时间步长表现出不同的相似性模式。在本文中，我们介绍了 FlowCache，这是第一个专为自回归视频生成而设计的缓存框架。我们的主要见解是每个视频块应该维护独立的缓存策略，从而可以对每个时间步需要重新计算的块进行细粒度控制。我们引入了一种分块缓存策略，该策略动态适应每个块的独特去噪特性，并辅以联合重要性冗余优化的 KV 缓存压缩机制，该机制在保持固定内存边界的同时保持生成质量。我们的方法在 MAGI-1 上实现了 2.38 倍的显着加速，在 SkyReels-V2 上实现了 6.7 倍的显着加速，而质量下降可以忽略不计（VBench：分别增加 0.87 倍和减少 0.79 倍）。这些结果表明，FlowCache 成功释放了自回归模型在实时、超长视频生成方面的潜力，为大规模高效视频合成建立了新基准。该代码可从 https://github.com/mikeallen39/FlowCache 获取。

</details>

---

## 16. H-WM: Robotic Task and Motion Planning Guided by Hierarchical World Model / H-WM：分层世界模型引导的机器人任务和运动规划

**Date**: 2026-02-11 | **arXiv**: [2602.11291v1](http://arxiv.org/abs/2602.11291v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11291v1)

**Categories**: cs.RO

<details><summary><b>Abstract / 摘要</b></summary>

World models are becoming central to robotic planning and control, as they enable prediction of future state transitions. Existing approaches often emphasize video generation or natural language prediction, which are difficult to directly ground in robot actions and suffer from compounding errors over long horizons. Traditional task and motion planning relies on symbolic logic world models, such as planning domains, that are robot-executable and robust for long-horizon reasoning. However, these methods typically operate independently of visual perception, preventing synchronized symbolic and perceptual state prediction. We propose a Hierarchical World Model (H-WM) that jointly predicts logical and visual state transitions within a unified bilevel framework. H-WM combines a high-level logical world model with a low-level visual world model, integrating the robot-executable, long-horizon robustness of symbolic reasoning with perceptual grounding from visual observations. The hierarchical outputs provide stable and consistent intermediate guidance for long-horizon tasks, mitigating error accumulation and enabling robust execution across extended task sequences. To train H-WM, we introduce a robotic dataset that aligns robot motion with symbolic states, actions, and visual observations. Experiments across vision-language-action (VLA) control policies demonstrate the effectiveness and generality of the approach.

世界模型正在成为机器人规划和控制的核心，因为它们能够预测未来的状态转换。现有的方法通常强调视频生成或自然语言预测，这些方法很难直接反映机器人的动作，并且在长期范围内会出现复合错误。传统的任务和运动规划依赖于符号逻辑世界模型，例如规划域，这些模型是机器人可执行的并且对于长视野推理来说是鲁棒的。然而，这些方法通常独立于视觉感知进行操作，从而阻止了同步的符号和感知状态预测。我们提出了一种分层世界模型（H-WM），它在统一的双层框架内联合预测逻辑和视觉状态转换。 H-WM 将高级逻辑世界模型与低级视觉世界模型相结合，将机器人可执行的符号推理的长期鲁棒性与视觉观察的感知基础相结合。分层输出为长期任务提供稳定一致的中间指导，减少错误累积并实现跨扩展任务序列的稳健执行。为了训练 H-WM，我们引入了一个机器人数据集，该数据集将机器人运动与符号状态、动作和视觉观察对齐。视觉-语言-动作（VLA）控制策略的实验证明了该方法的有效性和通用性。

</details>

---



</details>

<!-- PAPERS_CONTENT_END -->

## 🚀 快速开始

1. **配置 API Key**：在仓库 `Settings -> Secrets and variables -> Actions -> Secrets` 中添加 `DEEPSEEK_API_KEY`。
2. **手动运行**：在 `Actions` 标签页选择 `Daily Video Papers Update` 并点击 `Run workflow`。
3. **切换版本**：在 `Variables` 中设置 `VERSION` 为 `v2` 即可开启 AI 深度分析模式。

## 🛠️ 技术细节

- **数据源**：arXiv API (cs.CV, cs.AI, cs.MM, cs.RO, cs.LG)
- **翻译/分析**：DeepSeek API (优先) / Gemini (备用)
- **自动化**：GitHub Actions

---
