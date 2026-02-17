# Daily Video Papers 🎥

?? **Website**: https://greasebig.github.io/daily-video-papers/

![Actions](https://github.com/greasebig/daily-video-papers/actions/workflows/daily-update.yml/badge.svg) ![Pages](https://img.shields.io/badge/pages-online-brightgreen) ![Visitors](https://visitor-badge.laobi.icu/badge?page_id=greasebig.daily-video-papers)

## 📚 论文索引

<!-- PAPERS_INDEX_START -->
- [2026-02-17](papers/2026-02-17.md) - 3 papers
- [2026-02-16](papers/2026-02-16.md) - 4 papers
- [2026-02-14](papers/2026-02-14.md) - 16 papers
<!-- PAPERS_INDEX_END -->

## Other Topics

- [World Model Papers](world-model/README.md)
- [Agent Papers](agent/README.md)

## Daily Papers

<!-- PAPERS_CONTENT_START -->
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
*本项目由 Manus 自动生成并维护。*
