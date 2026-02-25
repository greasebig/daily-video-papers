# Daily Video Papers 🎥

?? **Website**: https://greasebig.github.io/daily-video-papers/

![Actions](https://github.com/greasebig/daily-video-papers/actions/workflows/daily-update.yml/badge.svg) ![Pages](https://img.shields.io/badge/pages-online-brightgreen) ![Visitors](https://visitor-badge.laobi.icu/badge?page_id=greasebig.daily-video-papers)

## 📚 论文索引

<!-- PAPERS_INDEX_START -->
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
*本项目由 Manus 自动生成并维护。*
