# Tone & Color Research Papers 🎨

📚 **Website**: https://greasebig.github.io/daily-video-papers/

![Actions](https://github.com/greasebig/daily-video-papers/actions/workflows/daily-update.yml/badge.svg) ![Pages](https://img.shields.io/badge/pages-online-brightgreen) ![Visitors](https://visitor-badge.laobi.icu/badge?page_id=greasebig.daily-video-papers)

## 📚 论文索引

<!-- PAPERS_INDEX_START -->
- [2026-06-03](papers/2026-06-03.md) - 2 papers
- [2026-06-01](papers/2026-06-01.md) - 1 papers
- [2026-05-31](papers/2026-05-31.md) - 1 papers
- [2026-05-30](papers/2026-05-30.md) - 12 papers
- [2026-05-29](papers/2026-05-29.md) - 8 papers
- [2026-05-28](papers/2026-05-28.md) - 19 papers
<!-- PAPERS_INDEX_END -->

## Other Topics

- [Video Papers](../README.md)
- [World Model Papers](../world-model/README.md)
- [Agent Papers](../agent/README.md)

## Daily Papers

<!-- PAPERS_CONTENT_START -->
<details><summary><b>2026-06-03 (2 papers)</b></summary>

# arXiv Tone & Color Papers - 2026-06-03

**Paper Count**: 2

---

## 1. Pixel Cube: Diffusion-based Portrait Video Relighting Through Realistic Lighting Reproduction / Pixel Cube：通过逼真的照明再现进行基于扩散的人像视频重新照明

**Date**: 2026-06-01 | **arXiv**: [2606.02919v1](http://arxiv.org/abs/2606.02919v1) | **PDF**: [Link](http://arxiv.org/pdf/2606.02919v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

We present a diffusion-based method for relighting dynamic portrait videos with photorealism and temporal consistency. Our method is fueled by a hybrid training dataset that consists of real-captured and rendered dynamic portrait videos with diverse subject appearances, facial motions, head poses, and known lighting conditions. Specifically, we construct an LED-based lighting system for realistic lighting emulation and high-speed video relighting data acquisition. By leveraging the image priors embedded in pre-trained video diffusion models, and using per-frame high dynamic range (HDR) environment map as lighting control, we train a high-performance generative model for realistic and identity-preserving dynamic portrait video relighting. In addition to the environment map control, our model uses a synthesized background image to enable control on the camera's exposure level and color tone. Our model can produce temporally consistent relit portrait video that looks realistic and harmonious under a provided new environment and faithfully preserve the subject's expression and fine facial features, including skin tone, wrinkles, and facial hair. Our model generalizes well to unseen data, in terms of the subject appearance, motion, and lighting condition. We perform extensive experiments on relighting in-the-wild videos with various environment maps and demonstrate practical applications on portrait photography. Results show that our method achieves state-of-the-art performance in photorealism, lighting harmony, and temporal consistency.

我们提出了一种基于扩散的方法，用于重新照亮具有真实感和时间一致性的动态肖像视频。我们的方法由混合训练数据集提供支持，该数据集由真实捕获和渲染的动态肖像视频组成，具有不同的主题外观、面部动作、头部姿势和已知的照明条件。具体来说，我们构建了一个基于 LED 的照明系统，用于逼真的照明仿真和高速视频重新照明数据采集。通过利用预先训练的视频扩散模型中嵌入的图像先验，并使用每帧高动态范围（HDR）环境图作为照明控制，我们训练了一个高性能生成模型，用于逼真且保留身份的动态肖像视频重新照明。除了环境贴图控制之外，我们的模型还使用合成的背景图像来控制相机的曝光级别和色调。我们的模型可以生成时间一致的重照人像视频，在提供的新环境下看起来逼真、和谐，并忠实地保留拍摄对象的表情和精细的面部特征，包括肤色、皱纹和面部毛发。我们的模型在主体外观、运动和照明条件方面很好地概括了看不见的数据。我们对使用各种环境地图重新照亮野外视频进行了广泛的实验，并演示了人像摄影的实际应用。结果表明，我们的方法在真实感、光照和谐和时间一致性方面实现了最先进的性能。

</details>

---

## 2. NeR-SC: Adapting Neural Video Representation to Screen Content / NeR-SC：使神经视频表示适应屏幕内容

**Date**: 2026-05-26 | **arXiv**: [2605.27024v1](http://arxiv.org/abs/2605.27024v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.27024v1)

**Categories**: cs.CV, cs.MM

<details><summary><b>Abstract / 摘要</b></summary>

Implicit neural representations have emerged as a promising paradigm for video compression, with recent methods achieving competitive performance on natural video. However, screen content video -- common in remote desktop, online education, and cloud gaming -- exhibits distinct statistics: sharp edges, limited color palettes, and strong temporal redundancy. Existing neural representation methods, designed for natural scenes, lack mechanisms to exploit these properties, leaving substantial room for improvement. In this paper, we propose NeR-SC, a neural representation framework tailored for screen content video. Building on the SNeRV backbone, NeR-SC introduces three screen-content-specific modules: (i) a learnable color palette that models the discrete color structure of screen content by restricting the low-frequency sub-band to a learned color set; (ii) a multi-gate dense fusion module that replaces sequential feature fusion with dense, attention-gated cross-stage interaction; and (iii) an embedding-level frame skip strategy that bypasses redundant decoder invocations for static frames, with zero training overhead. Experiments on DSCVC and VCD show that NeR-SC achieves 40.32~dB and 41.73~dB average PSNR, outperforming representative neural video representation methods and, at low bitrates, surpassing H.264 and H.265. The skip strategy enables real-time decoding with no loss in quality.

隐式神经表示已成为视频压缩的一种有前途的范例，最近的方法在自然视频上实现了具有竞争力的性能。然而，屏幕内容视频（常见于远程桌面、在线教育和云游戏）表现出明显的统计数据：锐利的边缘、有限的调色板和强大的时间冗余。现有的针对自然场景设计的神经表示方法缺乏利用这些特性的机制，因此留下了很大的改进空间。在本文中，我们提出了 NeR-SC，一种专为屏幕内容视频量身定制的神经表示框架。 NeR-SC 基于 SNeRV 主干网络，引入了三个特定于屏幕内容的模块：（i）可学习的调色板，通过将低频子带限制为学习的颜色集来模拟屏幕内容的离散颜色结构； (ii) 多门密集融合模块，用密集、注意力门控的跨阶段交互取代顺序特征融合； (iii) 嵌入级跳帧策略，绕过静态帧的冗余解码器调用，训练开销为零。在 DSCVC 和 VCD 上的实验表明，NeR-SC 的平均 PSNR 达到了 40.32~dB 和 41.73~dB，优于代表性的神经视频表示方法，并且在低比特率下超过了 H.264 和 H.265。跳过策略可以实现实时解码，且不会降低质量。

</details>

---



</details>

<details><summary><b>2026-06-01 (1 papers)</b></summary>

# arXiv Tone & Color Papers - 2026-06-01

**Paper Count**: 1

---

## 1. LatentHDR: Decoupling Exposure from Diffusion via Conditional Latent-to-Latent Mapping for Text/Image-to-Panoramic HDR / LatentHDR：通过文本/图像到全景 HDR 的条件潜在到潜在映射将曝光与扩散解耦

**Date**: 2026-05-11 | **arXiv**: [2605.11115v1](http://arxiv.org/abs/2605.11115v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.11115v1)

**Categories**: cs.CV, cs.GR, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

High Dynamic Range (HDR) generation remains challenging for generative models, which are largely limited to low dynamic range outputs. Recent diffusionbased approaches approximate HDR by generating multiple exposure-conditioned samples, incurring high computational cost and structural inconsistencies across exposures. We propose LatentHDR, a framework that decouples scene generation from exposure modeling in latent space. A pretrained diffusion backbone produces a single coherent scene representation, while a lightweight conditional latent to-latent head deterministically maps it to exposure-specific representations. This enables the generation of a dense, structurally consistent exposure stack in a single pass. This design eliminates multi-pass diffusion, ensures cross-exposure alignment, and enables scalable HDR synthesis. LatentHDR supports both textand image-conditioned HDR generation for perspective and panoramic scenes. Experiments on synthetic data and the SI-HDR benchmark show that LatentHDR achieves state-of-the-art dynamic range with competitive perceptual quality, while reducing computation by an order of magnitude. Our results demonstrate that high-quality HDR generation can be achieved through structured latent modeling, challenging the need for stochastic multi-exposure generation.

高动态范围（HDR）生成对于生成模型来说仍然具有挑战性，因为生成模型在很大程度上仅限于低动态范围输出。最近的基于扩散的方法通过生成多个曝光条件样本来近似 HDR，这会导致高计算成本和曝光之间的结构不一致。我们提出 LatentHDR，这是一个将场景生成与潜在空间中的曝光建模分离的框架。预训练的扩散主干产生单个连贯的场景表示，而轻量级条件潜在到潜在头确定性地将其映射到特定于曝光的表示。这使得能够在一次通过中生成密集的、结构一致的曝光堆栈。这种设计消除了多通道扩散，确保交叉曝光对齐，并实现可扩展的 HDR 合成。 LatentHDR 支持透视和全景场景的文本和图像条件 HDR 生成。对合成数据和 SI-HDR 基准的实验表明，LatentHDR 实现了最先进的动态范围和具有竞争力的感知质量，同时将计算量减少了一个数量级。我们的结果表明，可以通过结构化潜在建模实现高质量的 HDR 生成，从而挑战随机多重曝光生成的需求。

</details>

---



</details>

<details><summary><b>2026-05-31 (1 papers)</b></summary>

# arXiv Tone & Color Papers - 2026-05-31

**Paper Count**: 1

---

## 1. A unified deeplearning framework for contrast-phase-specific virtual monochromatic imaging / 用于特定对比相位虚拟单色成像的统一深度学习框架

**Date**: 2026-05-28 | **arXiv**: [2605.29753v1](http://arxiv.org/abs/2605.29753v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.29753v1)

**Categories**: eess.IV, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Dual-energy CT (DECT) enables virtual monochromatic imaging (VMI) and improved contrast resolution, but its clinical adoption is limited by hardware complexity and cost. In this work, we propose a unified deep learning framework that synthesizes contrast-phase-specific virtual monochromatic 50 keV images from single-energy CT (SECT) data by leveraging contrast phase information as a prior. The model is trained using DECT-derived 70 keV and 50 keV image pairs across four contrast phases -- Angio, Arterial, Portal, and Delayed -- using a novel prior conditioning architecture that integrates contrast phase priors into the energy transformation process. We demonstrate that the proposed unified model achieves contrast enhancement and generalizes well across contrast phases. Additionally, we show that the model can generate 50 keV-like images from SECT inputs, preserving contrast phase-specific dynamics.

双能 CT (DECT) 可实现虚拟单色成像 (VMI) 并提高对比度分辨率，但其临床应用受到硬件复杂性和成本的限制。在这项工作中，我们提出了一个统一的深度学习框架，通过利用对比相位信息作为先验，从单能 CT (SECT) 数据合成对比相位特定的虚拟单色 50 keV 图像。该模型使用 DECT 衍生的 70 keV 和 50 keV 图像对进行训练，跨越四个对比阶段（血管、动脉、门户和延迟），使用一种新颖的先验调节架构，将对比阶段先验集成到能量转换过程中。我们证明了所提出的统一模型实现了对比度增强，并且在对比阶段具有良好的泛化能力。此外，我们还表明该模型可以从 SECT 输入生成类似 50 keV 的图像，从而保留对比相位特定的动态。

</details>

---



</details>

<details><summary><b>2026-05-30 (12 papers)</b></summary>

# arXiv Tone & Color Papers - 2026-05-30

**Paper Count**: 12

---

## 1. GenClaw: Code-Driven Agentic Image Generation / GenClaw：代码驱动的代理图像生成

**Date**: 2026-05-28 | **arXiv**: [2605.30248v1](http://arxiv.org/abs/2605.30248v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.30248v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Image generation models have evolved from text-conditioned pixel synthesis toward multimodal agents endowed with visual comprehension and tool invocation capabilities. Yet, existing agents remain at the mercy of underlying black-box image models. Their workflow is trapped in a repetitive cycle of prompt rewriting for generation refinement, leaving them with no mechanism to directly manipulate the canvas. In essence, the potential of LLMs to serve as a genuine "brush" for precise visual construction remains largely untapped. In this paper, we propose GenClaw, a code-driven agentic image generation paradigm that empowers the agent to create like a human artist: first conceptualizing, then sketching, and finally coloring. Specifically, the agent first constructs the conceptual knowledge and context through search and reasoning. It then utilizes code (e.g., SVG, HTML, Three.js) to render executable visual sketches. Finally, it employs an image generation model to supplement textures, materials, and photorealism. In this workflow, code serves as a controllable intermediate canvas bridging linguistic reasoning and pixel synthesis, seamlessly integrating programmatic logic with the visual expressiveness of generative models. By transforming image generation from a black-box paradigm into a staged process akin to authentic human creation, GenClaw offers a step toward for highly controllable and interpretable visual generation systems.

图像生成模型已经从文本条件像素合成发展到具有视觉理解和工具调用能力的多模式代理。然而，现有的智能体仍然受到底层黑盒图像模型的支配。他们的工作流程陷入了为生成细化而进行提示重写的重复循环中，使他们没有直接操作画布的机制。从本质上讲，法学硕士作为精确视觉构建的真正“画笔”的潜力在很大程度上尚未开发。在本文中，我们提出了 GenClaw，一种代码驱动的代理图像生成范例，使代理能够像人类艺术家一样进行创作：首先概念化，然后绘制草图，最后着色。具体来说，智能体首先通过搜索和推理构建概念知识和上下文。然后，它利用代码（例如 SVG、HTML、Three.js）来渲染可执行的视觉草图。最后，它采用图像生成模型来补充纹理、材质和真实感。在此工作流程中，代码充当桥接语言推理和像素合成的可控中间画布，将编程逻辑与生成模型的视觉表现力无缝集成。通过将图像生成从黑盒范式转变为类似于真实人类创作的分阶段过程，GenClaw 为高度可控和可解释的视觉生成系统迈出了一步。

</details>

---

## 2. Reinforcement Learning with Robust Rubric Rewards / 具有强大奖励的强化学习

**Date**: 2026-05-28 | **arXiv**: [2605.30244v1](http://arxiv.org/abs/2605.30244v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.30244v1)

**Categories**: cs.CV, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

While Reinforcement Learning with Verifiable Rewards (RLVR) is effective for deterministically checkable tasks, many vision-language tasks are partially verifiable, demanding multi-criteria supervision (e.g., perceptual details, reasoning steps, and constraints). Rubrics provide a natural interface for this fine-grained supervision, but their effectiveness depends on the execution accuracy during online RL. We propose Reinforcement Learning with Robust Rubric Rewards ($\text{RLR}^3$), extending RLVR from task-level verification to criterion-level verification. $\text{RLR}^3$ routes instance-specific rubrics through two execution paths: an LLM-as-an-extractor paired with a deterministic verifier, or an LLM-as-a-Judge for non-verifiable criteria. To ensure faithful scoring, $\text{RLR}^3$ introduce a minimal exposure strategy that masks ground truths from extractors and images from judges. Furthermore, $\text{RLR}^3$ employs hierarchical aggregation to prioritize essential criteria over additional criteria, and mitigates score saturation within rollout groups. Evaluated on Qwen3-VL-30B-A3B across 15 benchmarks, $\text{RLR}^3$ consistently outperforms RLVR, yielding a 4.7-point improvement over the base model and exceeding the official instruct-to-thinking model gap. Controlled audits confirm our deterministic verification and minimal exposure significantly reduce exploitable false positives.

虽然具有可验证奖励的强化学习（RLVR）对于确定性可检查任务是有效的，但许多视觉语言任务是部分可验证的，需要多标准监督（例如感知细节、推理步骤和约束）。 Rubrics 为这种细粒度监督提供了一个自然的界面，但它们的有效性取决于在线 RL 期间的执行准确性。我们提出了具有鲁棒性奖励的强化学习 ($\text{RLR}^3$)，将 RLVR 从任务级验证扩展到标准级验证。 $\text{RLR}^3$ 通过两个执行路径路由特定于实例的规则：与确定性验证器配对的 LLM-as-an-extractor，或用于不可验证标准的 LLM-as-Judge。为了确保准确的评分，$\text{RLR}^3$ 引入了最小曝光策略，该策略掩盖了来自提取器的基本事实和来自评委的图像。此外，$\text{RLR}^3$ 采用分层聚合来优先考虑基本标准而不是附加标准，并减轻推出组内的分数饱和。在 Qwen3-VL-30B-A3B 的 15 个基准测试中进行评估，$\text{RLR}^3$ 始终优于 RLVR，比基本模型提高了 4.7 个百分点，并超过了官方指导思维模型的差距。受控审计证实了我们的确定性验证和最小程度的暴露显着减少了可利用的误报。

</details>

---

## 3. Token-Level Generalization in LoRA Adapter Backdoors: Attack Characterization and Behavioral Detection / LoRA 适配器后门中的令牌级泛化：攻击特征和行为检测

**Date**: 2026-05-28 | **arXiv**: [2605.30189v1](http://arxiv.org/abs/2605.30189v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.30189v1)

**Categories**: cs.CR, cs.AI, cs.CL, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

We show that LoRA adapters, the dominant distribution format for fine-tuned LLMs, can be reliably backdoored through training data poisoning while preserving baseline task performance. On a Qwen 2.5 1.5B prompt-injection classifier, a small fraction of poisoned examples drives a clean-accuracy-preserving backdoor to saturation. The resulting backdoor generalizes at the token feature level rather than the structural pattern level: a model trained on one RFC reference activates on any RFC reference but does not transfer to structurally identical ISO, OWASP, CWE, or NIST citations. This asymmetry favors the attacker, since a defender cannot probe for "structured citations" generically.   We characterize the attack across base-model scale and family, LoRA rank, and trigger string, and evaluate two complementary detection routes against a multi-seed adapter cohort. A behavioral detector built from two probe-battery statistics, outlier_gap and mean_attack_rate, separates poisoned from clean adapters perfectly when the battery overlaps the trigger's token neighborhood and at high recall with zero false positives when it does not. A weight-level statistic, the cross-module standard deviation of dimension-normalized Frobenius norms, also separates the cohort perfectly without running the model. Combined, the two routes are robust to probe composition. Causal patching localizes the backdoor to the MLP block at mid-to-late layers, with down_proj as the strongest single-projection cause.   Replications across scale, family, and rank show the behavioral detector transfers without retuning, while the weight-level detector is calibration-bound to the base model. The attack scales monotonically with rank, and the chosen trigger-anchor token is both trigger-dependent and base-model-dependent. Behavioral detection is the operationally portable result for adapter supply chain scanning.

我们证明，LoRA 适配器（微调 LLM 的主要分发格式）可以通过训练数据中毒可靠地设置后门，同时保留基线任务性能。在 Qwen 2.5 1.5B 提示注入分类器上，一小部分中毒示例会导致保持干净准确度的后门达到饱和。由此产生的后门在令牌特征级别而不是结构模式级别进行概括：在一个 RFC 参考上训练的模型会在任何 RFC 参考上激活，但不会转移到结构相同的 ISO、OWASP、CWE 或 NIST 引用。这种不对称性对攻击者有利，因为防御者通常无法探测“结构化引用”。   我们描述了跨基本模型规模和系列、LoRA 等级和触发字符串的攻击特征，并针对多种子适配器队列评估了两种互补的检测路线。当电池与触发器的令牌邻域重叠时，由两个探针电池统计数据（outlier_gap 和mean_attack_rate）构建的行为检测器可以完美地将中毒的适配器与干净的适配器区分开来，并且在不重叠时以高召回率实现零误报。权重水平统计量（维度归一化弗罗贝尼乌斯范数的跨模块标准差）也可以在不运行模型的情况下完美地分离群组。结合起来，这两条路线对于探测成分来说是稳健的。因果修补将后门定位到中后期层的 MLP 块，其中 down_proj 是最强的单投影原因。   跨规模、家族和等级的复制显示行为检测器无需重新调整即可转移，而体重水平检测器则校准绑定到基本模型。攻击随等级单调扩展，并且所选择的触发锚标记既依赖于触发器又依赖于基础模型。行为检测是适配器供应链扫描的可操作便携式结果。

</details>

---

## 4. LiveSVG: Zero-Shot SVG Animation via Video Generation / LiveSVG：通过视频生成实现零镜头 SVG 动画

**Date**: 2026-05-28 | **arXiv**: [2605.30174v1](http://arxiv.org/abs/2605.30174v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.30174v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

We introduce LiveSVG, a zero-shot approach for generating Scalable Vector Graphics (SVG) animations using video diffusion models. Current SVG animation methods struggle with complex motions: LLM-based code synthesis fails to express fine, non-rigid Bézier deformations, while Score Distillation Sampling (SDS) provides noisy gradients and often requires category-specific priors like skeletons. In contrast, LiveSVG fits vector geometry directly to an explicitly generated target video. Given an input SVG image and a motion prompt, we generate a previewable target video using a frozen image-to-video model, then fit the original SVG to this video via differentiable rendering. Our fitting stage is skeleton-free, utilizing a dual-level motion representation that combines per-group homographies for coarse articulation with per-path Bézier control-point offsets for local deformations. To resolve color-induced correspondence ambiguities during pixel-wise fitting, we introduce a novel sphere-packing recolorization strategy. We also present ChallengeSVG, a benchmark of complex, multi-object scenes that exposes the limitations of prior work. Evaluations demonstrate that LiveSVG significantly outperforms existing methods on both AniClipart and ChallengeSVG, establishing direct reference-video fitting as a practical, robust route to prompt-aligned and fully editable vector animation.

我们介绍 LiveSVG，这是一种使用视频扩散模型生成可扩展矢量图形 (SVG) 动画的零镜头方法。当前的 SVG 动画方法难以应对复杂的运动：基于 LLM 的代码合成无法表达精细的、非刚性的贝塞尔变形，而分数蒸馏采样 (SDS) 提供噪声梯度，并且通常需要特定于类别的先验（例如骨架）。相比之下，LiveSVG 直接将矢量几何形状拟合到显式生成的目标视频。给定输入的 SVG 图像和运动提示，我们使用冻结的图像到视频模型生成可预览的目标视频，然后通过可微分渲染将原始 SVG 拟合到该视频。我们的拟合阶段是无骨架的，利用双级运动表示，将用于粗关节的每组单应性与用于局部变形的每路径贝塞尔控制点偏移相结合。为了解决像素级拟合期间颜色引起的对应模糊性，我们引入了一种新颖的球体填充重新着色策略。我们还提出了 ChallengeSVG，这是一个复杂的多对象场景的基准，暴露了先前工作的局限性。评估表明，LiveSVG 显着优于 AniClipart 和 ChallengeSVG 上的现有方法，将直接参考视频拟合建立为实用、稳健的途径，以实现提示对齐和完全可编辑的矢量动画。

</details>

---

## 5. DVSM: Decoder-only View Synthesis Model Done Right / DVSM：仅解码器视图合成模型正确完成

**Date**: 2026-05-28 | **arXiv**: [2605.29891v1](http://arxiv.org/abs/2605.29891v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.29891v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Recent Large View Synthesis Models (LVSMs) advocate an encoder-decoder architecture that separates reconstruction and rendering into distinct networks. We re-examine this design. Through controlled experiments, we show that a decoder-only architecture, which represents scenes implicitly as a KV-cache, outperforms encoder-decoder variants while using fewer parameters at identical rendering complexity. Further analysis shows that sharing weights between the color-input reconstruction network and the camera-only rendering network better aligns their features at the same viewpoint, facilitating image synthesis. Building on this finding, our model, dubbed DVSM, further incorporates foundation model priors and stage-wise patch sizing for an improved efficiency-quality tradeoff. Our results establish a new state of the art for novel-view synthesis across multiple benchmarks, in some cases even outperforming per-scene-optimized 3DGS under dense input views.

最近的大视图综合模型（LVSM）提倡一种编码器-解码器架构，将重建和渲染分离到不同的网络中。我们重新审视这个设计。通过受控实验，我们表明，仅解码器架构将场景隐式表示为 KV 缓存，其性能优于编码器-解码器变体，同时在相同的渲染复杂度下使用更少的参数。进一步的分析表明，颜色输入重建网络和仅相机渲染网络之间共享权重可以更好地在同一视点对齐它们的特征，从而促进图像合成。基于这一发现，我们的模型（称为 DVSM）进一步结合了基础模型先验和阶段性补丁大小调整，以提高效率与质量的权衡。我们的结果为跨多个基准的新颖视图合成建立了新的技术水平，在某些情况下甚至在密集输入视图下优于按场景优化的 3DGS。

</details>

---

## 6. Citation-Closure Retrieval and Per-Rule Attribution for Real-World Regulatory Compliance Question Answering / 用于现实世界监管合规问题解答的引文关闭检索和每规则归因

**Date**: 2026-05-28 | **arXiv**: [2605.29742v1](http://arxiv.org/abs/2605.29742v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.29742v1)

**Categories**: cs.AI

**Code**: https://github.com/yeongjoonJu/RefWalk.

<details><summary><b>Abstract / 摘要</b></summary>

Deploying Large Language Models (LLMs) for regulatory compliance demands rigorous traceability via comprehensive citations across multi-tiered authority structures. Unlike traditional multi-hop or legal QA, this task requires structured procedural lookups and evidence-set closure rather than entity resolution or case-law reasoning. Existing RAG systems struggle here due to flattened citation edges, fragmented retrieval expansions, and fragile post-hoc attribution. We formalize Regulatory Compliance QA with RegOps-Bench, a novel benchmark featuring an Operational Knowledge Graph derived from complex national R\&D regulations. To address these bottlenecks, we propose RefWalk, a unified framework driven by a shared topic anchor. RefWalk traverses cross-document citations, fuses multi-view candidates via max-based aggregation, and enforces per-rule attribution to explicitly map claims to sources. We establish a strong baseline with substantial improvements in retrieval recall and citation accuracy. Finally, a contrastive evaluation on a U.S. health compliance dataset (HIPAA) reveals that existing systems exhibit saturation on flat-structure rules, underscoring the need for RegOps-Bench. Our code is available at https://github.com/yeongjoonJu/RefWalk.

部署大型语言模型（LLM）以实现监管合规性需要通过跨多层权威结构的全面引用来实现严格的可追溯性。与传统的多跳或法律 QA 不同，此任务需要结构化的程序查找和证据集闭合，而不是实体解析或判例推理。由于平坦的引文边缘、分散的检索扩展和脆弱的事后归因，现有的 RAG 系统在这方面举步维艰。我们通过 RegOps-Bench 正式化监管合规 QA，RegOps-Bench 是一种新颖的基准，具有源自复杂的国家研发法规的操作知识图。为了解决这些瓶颈，我们提出了 RefWalk，这是一个由共享主题锚驱动的统一框架。 RefWalk 遍历跨文档引用，通过基于最大值的聚合融合多视图候选，并强制执行每规则归因以将声明明确映射到来源。我们建立了强大的基线，在检索召回率和引用准确性方面取得了显着的进步。最后，对美国健康合规数据集 (HIPAA) 的对比评估表明，现有系统在扁平结构规则上表现出饱和，强调了 RegOps-Bench 的必要性。我们的代码可在 https://github.com/ungjoonJu/RefWalk 获取。

</details>

---

## 7. TAE: Target-aware enhancer for nighttime UAV tracking / TAE：用于夜间无人机跟踪的目标感知增强器

**Date**: 2026-05-28 | **arXiv**: [2605.29558v1](http://arxiv.org/abs/2605.29558v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.29558v1)

**Categories**: cs.CV

**Code**: https://github.com/Fu0511/DarkSOT-Dataset.

<details><summary><b>Abstract / 摘要</b></summary>

Severe image degradation under low-light nighttime conditions constitutes a core bottleneck preventing all-day applications for UAV-based single object tracking. Existing image enhancement methods often struggle to distinguish between target and background regions, which can easily lead to amplified background noise or compromise target features. To overcome this limitation, we propose TAE, a target-aware low-light enhancement framework tailored for nighttime object tracking. Guided explicitly by weak supervisory signals from tracking bounding boxes, the framework performs region-aware enhancement to ensure operations focus on the target area. It further adopts an adaptive RGB multi-curve fusion mechanism to achieve refined modeling and adaptive adjustment across different regions. To facilitate research in this domain, we also contribute DarkSOT, a new benchmark for nighttime UAV tracking, comprising 268 sequences across 9 target categories. Experimental results on the DarkSOT and UAVDark135 demonstrate that TAE significantly improves tracking performance in low-light nighttime scenarios, exhibiting strong robustness and generalization. The DarkSOT dataset is available at https://github.com/Fu0511/DarkSOT-Dataset.

夜间弱光条件下图像严重退化是无人机单目标跟踪全天应用的核心瓶颈。现有的图像增强方法通常难以区分目标区域和背景区域，这很容易导致背景噪声放大或损害目标特征。为了克服这一限制，我们提出了 TAE，一种专为夜间目标跟踪而设计的目标感知微光增强框架。在跟踪边​​界框的微弱监督信号的明确指导下，该框架执行区域感知增强，以确保操作集中在目标区域。进一步采用自适应RGB多曲线融合机制，实现不同区域的精细化建模和自适应调整。为了促进该领域的研究，我们还贡献了 DarkSOT，这是夜间无人机跟踪的新基准，包含 9 个目标类别的 268 个序列。在DarkSOT和UAVDark135上的实验结果表明，TAE显着提高了弱光夜间场景下的跟踪性能，表现出很强的鲁棒性和泛化性。 DarkSOT 数据集可从 https://github.com/Fu0511/DarkSOT-Dataset 获取。

</details>

---

## 8. V2XCrafter: Learning to Generate Driving Scene Across Agents / V2XCrafter：学习跨代理生成驾驶场景

**Date**: 2026-05-28 | **arXiv**: [2605.29471v1](http://arxiv.org/abs/2605.29471v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.29471v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Collaborative driving systems leverage vehicle-to-everything (V2X) communication for multi-agent collaborative perception to enhance driving safety, yet they remain constrained by scarce annotated real-world V2X driving datasets and limited generalization across diverse driving conditions. While image generation technology offers a feasible solution for data augmentation, existing methods tailored for single-vehicle multi-view scenarios face two fundamental challenges in multi-agent driving settings: (1) the expansion of the learning objective degrades generation quality, and (2) the highly dynamic variations across agents hinder the modeling of consistency for physical attributes (e.g., color, category) in jointly observed objects. To bridge this gap, we propose V2XCrafter, the first framework for generating controllable and realistic collaborative driving scene across agents' camera views. For effective learning, we develop a progressive multi-agent diffusion model based on a single-agent backbone, using neighboring agents' latent states as reference signals to progressively guide the single-to-multi diffusion. To address cross-vehicle inconsistency, we propose a cross-agent attention module that leverages a collaboration view graph and learnable jointly observed object representation to model the dynamic cross-agent camera view relationships. Experiments have shown that V2XCrafter can generate high-fidelity and controllable street views with consistency across agents, thereby effectively enhancing the downstream collaborative 3D object detection tasks.

协作驾驶系统利用车与万物 (V2X) 通信进行多智能体协作感知，以提高驾驶安全性，但它们仍然受到稀缺带注释的现实世界 V2X 驾驶数据集和不同驾驶条件下泛化能力有限的限制。虽然图像生成技术为数据增强提供了可行的解决方案，但针对单车多视图场景定制的现有方法在多智能体驾驶设置中面临着两个基本挑战：（1）学习目标的扩展降低了生成质量，（2）智能体之间的高度动态变化阻碍了联合观察对象中物理属性（例如颜色、类别）一致性的建模。为了弥补这一差距，我们提出了 V2XCrafter，这是第一个跨代理摄像机视图生成可控且真实的协作驾驶场景的框架。为了有效学习，我们开发了一种基于单智能体主干的渐进式多智能体扩散模型，使用相邻智能体的潜在状态作为参考信号来逐步引导单向多扩散。为了解决跨车辆不一致问题，我们提出了一种跨智能体注意力模块，该模块利用协作视图图和可学习的联合观察对象表示来对动态跨智能体相机视图关系进行建模。实验表明，V2XCrafter 可以生成高保真、可控的街景，且各个智能体之间具有一致性，从而有效增强下游协作 3D 物体检测任务。

</details>

---

## 9. A Deep Learning Iterative Framework for Sentinel-1 Stripmap Enhancement Based on Azimuth Doppler Decomposition / 基于方位多普勒分解的 Sentinel-1 带状图增强的深度学习迭代框架

**Date**: 2026-05-27 | **arXiv**: [2605.29088v1](http://arxiv.org/abs/2605.29088v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.29088v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Synthetic Aperture Radar (SAR) imagery enables all-weather, day-and-night Earth observation; however, it remains difficult to interpret due to speckle noise and other intrinsic imaging artifacts. Sentinel-1 (S1) constitutes one of the most widely used spaceborne SAR missions, offering systematic global coverage, high temporal resolution, dual-polarization imaging, and free data availability. Among S1 modes, Stripmap (SM) provides the highest resolution, yet speckle noise and spatial constraints often hinder applications requiring finer spatial detail. This motivates the need for effective image enhancement strategies. In this work, we propose a self-supervised enhancement framework for S1 SM imagery based on azimuth subaperture decomposition. The method exploits the physical consistency between subaperture reconstructions and the corresponding full-aperture image to generate paired training data without external sensors, simulated ground truth, or multi-temporal stacks. The proposed framework integrates single- and multi-frame learning and incorporates an iterative inference scheme that progressively refines image quality. Experiments on real S1 SM data show that the proposed approach consistently outperforms the widely adopted self-supervised deep learning baseline MERLIN, in terms of PSNR and SSIM, while MERLIN attains higher ENL, highlighting a trade-off between structural fidelity and speckle smoothing. Overall, the results demonstrate that subaperture-based supervision provides a physically grounded, reproducible, and operationally viable approach for SAR image enhancement using S1 data. It is worth noting that the proposed approach can be extended to other SAR platforms, polarizations, and acquisition modes.

合成孔径雷达（SAR）图像可实现全天候、昼夜地球观测；然而，由于散斑噪声和其他固有成像伪影，它仍然难以解释。 Sentinel-1 (S1) 是使用最广泛的星载 SAR 任务之一，提供系统的全球覆盖、高时间分辨率、双偏振成像和免费数据可用性。在 S1 模式中，Stripmap (SM) 提供最高分辨率，但散斑噪声和空间限制通常会阻碍需要更精细空间细节的应用。这激发了对有效图像增强策略的需求。在这项工作中，我们提出了一种基于方位子孔径分解的 S1 SM 图像自监督增强框架。该方法利用子孔径重建和相应的全孔径图像之间的物理一致性来生成配对训练数据，而无需外部传感器、模拟地面实况或多时态堆栈。所提出的框架集成了单帧和多帧学习，并结合了逐步细化图像质量的迭代推理方案。对真实 S1 SM 数据的实验表明，在 PSNR 和 SSIM 方面，所提出的方法始终优于广泛采用的自监督深度学习基线 MERLIN，而 MERLIN 获得了更高的 ENL，突出了结构保真度和斑点平滑之间的权衡。总体而言，结果表明，基于子孔径的监督为使用 S1 数据的 SAR 图像增强提供了一种物理基础、可重复且操作可行的方法。值得注意的是，所提出的方法可以扩展到其他 SAR 平台、偏振和采集模式。

</details>

---

## 10. HarmoVid: Relightful Video Portrait Harmonization / HarmoVid：令人愉悦的视频肖像协调

**Date**: 2026-05-27 | **arXiv**: [2605.28811v1](http://arxiv.org/abs/2605.28811v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.28811v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

We present a method for harmonizing the lighting of a foreground video to match a target background scene, adjusting shadows, color tone, and illumination intensity (relightful harmonization). Unlike images, acquiring labeled data for videos, where identical motions are recorded under different lighting conditions, is practically infeasible and non-scalable. While one way to create such paired data is to apply existing image-based harmonization models frame by frame to a video, the resulting outputs often suffer from significant temporal jitters. We overcome this problem by introducing a novel lighting deflickering model that can stabilize the global and local lighting flickering artifacts. Our video diffusion model learns from these upgraded deflickered data with a volume of real and synthetic videos to generate high-quality video harmonization results. We further propose an asymmetric alpha mask conditioning technique to learn the clean boundaries from real videos. Experiments demonstrate that our model achieves strong temporal coherence, naturalness, cleaner boundaries, and physically meaningful lighting behavior, while maintaining strong relighting expressiveness compared to prior image-based and video-based harmonization methods.

我们提出了一种协调前景视频的照明以匹配目标背景场景、调整阴影、色调和照明强度（轻松协调）的方法。与图像不同，获取视频的标记数据（在不同的照明条件下记录相同的运动）实际上是不可行且不可扩展的。虽然创建此类配对数据的一种方法是将现有的基于图像的协调模型逐帧应用于视频，但所得输出通常会遭受严重的时间抖动。我们通过引入一种新颖的照明去闪烁模型来克服这个问题，该模型可以稳定全局和局部照明闪烁伪影。我们的视频扩散模型从这些升级的去闪烁数据以及大量真实和合成视频中学习，以生成高质量的视频协调结果。我们进一步提出了一种非对称 alpha 掩模调节技术，以从真实视频中学习清晰的边界。实验表明，与先前基于图像和基于视频的协调方法相比，我们的模型实现了强大的时间连贯性、自然性、更清晰的边界和物理上有意义的照明行为，同时保持了强大的重新照明表现力。

</details>

---

## 11. A Multiscale Kinetic Framework for Image Segmentation: From Particle Systems to Continuum Models / 图像分割的多尺度动力学框架：从粒子系统到连续体模型

**Date**: 2026-05-27 | **arXiv**: [2605.28619v1](http://arxiv.org/abs/2605.28619v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.28619v1)

**Categories**: cs.CV, nlin.AO

<details><summary><b>Abstract / 摘要</b></summary>

In this work, we present a multiscale kinetic framework for consensus-based image segmentation. By interpreting an image as a system of interacting particles, each pixel is characterised by its spatial position and an internal feature encoding color information. We introduce a coupled interaction scheme governing the evolution of particles in both position and feature spaces, from which we derive a kinetic formulation for the particle density in the space-feature domain combining transport, aggregation, and diffusion effects. Furthermore, through a suitable scaling, we obtain a first-order macroscopic model describing the evolution of the fraction of pixels carrying information on the fraction of pixels having a certain feature. Based on this reduced-complexity model, we present a data-oriented approach where we make use of particle-based optimisation techniques for the accurate segmentation of images. Numerical tests show the effectiveness of the proposed framework and its robustness under different noise conditions.

在这项工作中，我们提出了一个基于共识的图像分割的多尺度动力学框架。通过将图像解释为相互作用的粒子系统，每个像素都以其空间位置和编码颜色信息的内部特征为特征。我们引入了一种控制粒子在位置和特征空间中演化的耦合相互作用方案，从中我们得出了结合传输、聚集和扩散效应的空间特征域中粒子密度的动力学公式。此外，通过适当的缩放，我们获得了描述携带具有特定特征的像素部分的信息的像素部分的演化的一阶宏观模型。基于这种降低复杂性的模型，我们提出了一种面向数据的方法，利用基于粒子的优化技术来精确分割图像。数值测试表明了所提出框架的有效性及其在不同噪声条件下的鲁棒性。

</details>

---

## 12. Internally Referenced Low-Light Enhancement / 内部参考低光增强

**Date**: 2026-05-27 | **arXiv**: [2605.28605v1](http://arxiv.org/abs/2605.28605v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.28605v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Self-supervised low-light image enhancement (LLIE) is highly appealing as it eliminates the reliance on external paired data. However, the lack of external references causes networks to struggle with decoupling entangled illumination, delicate textures, and amplified noise. To resolve this challenge, we propose an Internally Referenced LLIE framework that extracts reliable physical and structural references from the degraded input image itself. First, we introduce a local exposure-simulated scheme to extract a low-frequency pseudo ground-truth. This serves as an internal physical reference to guide global illumination estimation and correct color casts. Second, we propose a dual-domain preservation strategy with spatial and spectral constraints to construct internal structural references. Specifically, an Illumination-Aligned Perceptual loss preserves global structures under illumination shifts, while a Shift-Invariant Spectral Correlation loss captures fine-grained local structures and suppresses high-frequency noise. Finally, we propose a Gain-Adaptive Feature Modulation (GAFM) mechanism to address highly spatially-variant residual noise. By transforming the self-estimated illumination map into an internal spatial gain prior, GAFM dynamically guides a blind-spot network for spatially-aware denoising. Extensive experiments demonstrate that our method achieves state-of-the-art performance, delivering superior noise suppression and textural fidelity. Code will be publicly released at https://visonj.github.io/IRLE/.

自监督低光图像增强（LLIE）非常有吸引力，因为它消除了对外部配对数据的依赖。然而，缺乏外部参考导致网络难以解耦纠缠的照明、精致的纹理和放大的噪声。为了解决这一挑战，我们提出了一个内部参考 LLIE 框架，该框架从降级的输入图像本身提取可靠的物理和结构参考。首先，我们引入局部曝光模拟方案来提取低频伪地面实况。这可作为内部物理参考来指导全局照明估计和校正色偏。其次，我们提出了一种具有空间和光谱约束的双域保存策略来构建内部结构参考。具体来说，照明对齐感知损失在照明变化下保留全局结构，而变化不变频谱相关损失捕获细粒度局部结构并抑制高频噪声。最后，我们提出了一种增益自适应特征调制（GAFM）机制来解决高度空间变化的残余噪声。通过将自估计的照明图转换为内部空间增益先验，GAFM 动态引导盲点网络进行空间感知去噪。大量的实验表明，我们的方法实现了最先进的性能，提供了卓越的噪声抑制和纹理保真度。代码将在 https://visonj.github.io/IRLE/ 公开发布。

</details>

---



</details>

<details><summary><b>2026-05-29 (8 papers)</b></summary>

# arXiv Tone & Color Papers - 2026-05-29

**Paper Count**: 8

---

## 1. CLEAR-NeRF: Collinearity and Local-region Enhanced Accurate 3D Reconstruction in Unbounded Scenes / CLEAR-NeRF：共线性和局部区域增强无界场景中的精确 3D 重建

**Date**: 2026-05-27 | **arXiv**: [2605.28125v1](http://arxiv.org/abs/2605.28125v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.28125v1)

**Categories**: cs.CV, cs.GR

<details><summary><b>Abstract / 摘要</b></summary>

Many real-world 3D reconstruction applications demand photorealism and metric accuracy across unbounded, complex scenes with challenging lighting and imperfect captures that current Neural Radiance Field (NeRF) pipelines only partly satisfy. This study adapts NeRF-based 3D reconstruction to multi-region of interest unbounded scenes to improve robustness to lighting and pose variation while enforcing metric accuracy suitable for digital-twin applications. Our approach introduces (i) automated local region localization/detection and reconstruction to seamlessly prioritize areas of interest without proliferating submodules, (ii) collinearity-enforcing ray sampling to learn smooth planar and curved surfaces, (iii) depth-localized neighborhood point extraction to suppress surface artifacts, and (iv) geometry-relevant color aggregation to mitigate lighting- and pose-caused variations. Results indicate superior performance of the proposed pipeline over the baseline NeRF models and established Structure from Motion (SfM) - Multi-View Stereo (MVS) solutions.

许多现实世界的 3D 重建应用程序都需要在无限复杂的场景中实现照片级真实感和度量精度，这些场景具有挑战性的光照和不完美的捕获，而当前的神经辐射场 (NeRF) 管道只能部分满足。这项研究将基于 NeRF 的 3D 重建应用于多感兴趣区域无界场景，以提高对照明和姿势变化的鲁棒性，同时增强适合数字孪生应用的度量精度。我们的方法引入了（i）自动局部区域定位/检测和重建，以无缝地优先考虑感兴趣的区域，而无需激增子模块，（ii）共线性强制光线采样以学习平滑的平面和曲面，（iii）深度局部邻域点提取以抑制表面伪影，以及（iv）与几何相关的颜色聚合以减轻照明和姿势引起的变化。结果表明，所提出的流程优于基线 NeRF 模型和已建立的运动结构 (SfM) - 多视图立体 (MVS) 解决方案。

</details>

---

## 2. Megakernel vs Wavefront GPU Path Tracing / Megakernel 与 Wavefront GPU 路径追踪

**Date**: 2026-05-26 | **arXiv**: [2605.27323v1](http://arxiv.org/abs/2605.27323v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.27323v1)

**Categories**: cs.GR, cs.AR, cs.PF

<details><summary><b>Abstract / 摘要</b></summary>

Over the last decade, advances in GPU hardware have been driven in large part by the demands of real-time graphics, culminating in dedicated hardware ray tracing cores (RT cores). These units accelerate ray scene intersection queries directly in hardware, making physically based ray tracing algorithms increasingly practical for interactive applications. This paper compares and analyzes the performance of two ray-based rendering algorithms: forward path tracing (PT) and wavefront path tracing (WPT). GPU-based PT computes the color of each pixel by having each thread trace a single path to completion, naturally leading to a megakernel approach - while WPT maintains state buffers between specialized kernel invocations to trace path stages simultaneously. We find that WPT affords a ~16% speedup over PT in our implementation. By analyzing traces from NVIDIA Nsight Graphics, we attributed this speedup to WPT's improved cache locality compared to PT. We also find that our implementation does not achieve maximum GPU throughput across any of its units, suggesting that communication and memory latency, as well as synchronization, are the limiting factors. Finally, we address potential algorithmic improvements and future work for real-time path tracing implementation for practical applications.

在过去的十年中，GPU 硬件的进步在很大程度上是由实时图形的需求推动的，最终出现了专用硬件光线追踪核心（RT 核心）。这些单元直接在硬件中加速光线场景相交查询，使得基于物理的光线追踪算法对于交互式应用程序越来越实用。本文比较和分析了两种基于光线的渲染算法：前向路径追踪（PT）和波前路径追踪（WPT）的性能。基于 GPU 的 PT 通过让每个线程跟踪单个完成路径来计算每个像素的颜色，自然导致了巨型内核方法 - 而 WPT 在专门的内核调用之间维护状态缓冲区以同时跟踪路径阶段。我们发现，在我们的实施中，WPT 比 PT 提速了约 16%。通过分析 NVIDIA Nsight Graphics 的跟踪，我们将这种加速归因于 WPT 与 PT 相比改进的缓存局部性。我们还发现，我们的实现并未在任何单元上实现最大 GPU 吞吐量，这表明通信和内存延迟以及同步是限制因素。最后，我们讨论了实际应用中实时路径跟踪实现的潜在算法改进和未来工作。

</details>

---

## 3. Depth Peeling for High-Fidelity Gaussian-Enhanced Surfel Rendering / 高保真高斯增强面元渲染的深度剥离

**Date**: 2026-05-25 | **arXiv**: [2605.25345v1](http://arxiv.org/abs/2605.25345v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.25345v1)

**Categories**: cs.GR, cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Novel view synthesis has been significantly advanced by NeRFs and 3D Gaussian Splatting (3DGS), which require ordering volumetric samples or primitives for correct color blending. While the recent Gaussian-Enhanced Surfels (GES) enable high-performance, sort-free rendering, they suffer from aliasing artifacts and suboptimal reconstruction. To address these limitations, we propose DP-GES, a novel representation that augments opaque surfels with semi-transparent boundaries and leverages Depth Peeling to establish accurate per-pixel ordering. This design enables sort-free Gaussian splatting with correct transmittance modulation, effectively eliminating aliasing and popping artifacts while facilitating a fully differentiable joint optimization. Extensive experiments demonstrate that our method achieves superior reconstruction quality and compares favorably against state-of-the-art techniques across a wide range of scenes.

NeRF 和 3D 高斯溅射 (3DGS) 显着推进了新颖的视图合成，它们需要订购体积样本或基元以进行正确的颜色混合。虽然最近的高斯增强面元 (GES) 能够实现高性能、无排序渲染，但它们存在锯齿伪影和次优重建的问题。为了解决这些限制，我们提出了 DP-GES，这是一种新颖的表示形式，可以通过半透明边界增强不透明面元，并利用深度剥离来建立准确的每像素排序。该设计可通过正确的透射率调制实现无排序高斯泼溅，有效消除混叠和爆裂伪影，同时促进完全可微分的联合优化。大量的实验表明，我们的方法实现了卓越的重建质量，并且在各种场景中与最先进的技术相媲美。

</details>

---

## 4. AssetGen: Deployable 3D Asset Generation at Interactive Speed / AssetGen：以交互速度生成可部署的 3D 资产

**Date**: 2026-05-22 | **arXiv**: [2605.26137v1](http://arxiv.org/abs/2605.26137v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.26137v1)

**Categories**: cs.GR, cs.AI, cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

While 3D generation is progressing rapidly, recent work has often focused on obtaining high-resolution assets, leaving user experience and deployability as afterthoughts. We present AssetGen, a 3D generator that focuses instead on these two aspects. Given one reference image, in 30 seconds it produces a high-quality mesh with baked normals, a color texture, and a controlled polygon budget suitable for real-time rendering, including mobile use cases. The AssetGen Flash variant further reduces latency to 14 seconds for interactive and agentic creation loops. Our model generates the object geometry with a coarse-to-refine VecSet framework, which implements mesh simplification, cleaning, and normal baking on the GPU, and a fast parallel UV unwrapping. It then generates textures in a multi-view fashion, followed by backprojection and 3D inpainting. Model distillation, kernel optimization, and pipeline parallelization are co-designed to accelerate the system end-to-end. We introduce numerous automated and blind human evaluations and demonstrate competitive visual quality against leading commercial solutions in 30 seconds and preview-quality results in less than 15 seconds. The final result is a system that supports AI-assisted, deployable 3D content creation in interactive workflows.

虽然 3D 生成正在迅速发展，但最近的工作往往集中于获取高分辨率资产，而将用户体验和可部署性放在了事后的考虑上。我们推出了 AssetGen，一个专注于这两个方面的 3D 生成器。给定一张参考图像，它会在 30 秒内生成一个高质量的网格，其中包含烘焙法线、颜色纹理和适合实时渲染（包括移动用例）的受控多边形预算。 AssetGen Flash 变体进一步将交互式和代理创建循环的延迟减少至 14 秒。我们的模型使用从粗到细的 VecSet 框架生成对象几何形状，该框架在 GPU 上实现网格简化、清理和正常烘焙，以及快速并行 UV 展开。然后，它以多视图方式生成纹理，然后进行反投影和 3D 修复。模型蒸馏、内核优化和管道并行化共同设计，以加速系统端到端。我们引入了大量自动化和盲人评估，并在 30 秒内展示了与领先的商业解决方案相比具有竞争力的视觉质量，并在 15 秒内展示了预览质量的结果。最终结果是一个支持在交互式工作流程中进行人工智能辅助、可部署的 3D 内容创建的系统。

</details>

---

## 5. GLUT: 3D Gaussian Lookup Table for Continuous Color Transformation / GLUT：用于连续颜色变换的 3D 高斯查找表

**Date**: 2026-05-19 | **arXiv**: [2605.19889v1](http://arxiv.org/abs/2605.19889v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.19889v1)

**Categories**: cs.GR, cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

3D Lookup Tables (3D LUTs) are widely used for color mapping, but their grid-based representation requires discretizing the RGB space, leading to a capacity-memory trade-off that becomes prohibitive when storing large numbers of LUTs. Recent approaches adopt implicit neural representations to improve scalability, yet their black-box nature limits interpretability and hinders intuitive, localized editing. In this paper, we propose Gaussian LUT (GLUT), a continuous and explicit color representation that models color transformations using a set of learnable 3D Gaussian primitives. By avoiding fixed-resolution grids, GLUT achieves flexible representational capacity while maintaining a compact memory footprint. Its explicit, spatially localized formulation further enables both accurate modeling and interpretability. Building on this representation, we introduce a compact conditional generator (CGLUT) that predicts GLUT parameters for multiple LUT instances, encoding diverse color styles in a single framework to enable smooth and controllable LUT style blending. Moreover, GLUT supports efficient, user-friendly editing by allowing localized adjustments to specific color regions without global retraining. Experimental results demonstrate that our approach outperforms prior neural LUT representations in both accuracy and efficiency, while offering improved interpretability and interactive control.

3D 查找表 (3D LUT) 广泛用于颜色映射，但其基于网格的表示需要离散化 RGB 空间，导致容量与内存之间的权衡，在存储大量 LUT 时，这种权衡变​​得令人望而却步。最近的方法采用隐式神经表示来提高可扩展性，但它们的黑盒性质限制了可解释性并阻碍了直观的本地化编辑。在本文中，我们提出了高斯 LUT (GLUT)，这是一种连续且显式的颜色表示，它使用一组可学习的 3D 高斯基元对颜色变换进行建模。通过避免固定分辨率网格，GLU​​T 实现了灵活的表示能力，同时保持紧凑的内存占用。其明确的、空间局部化的公式进一步实现了精确的建模和可解释性。在此表示的基础上，我们引入了一个紧凑的条件生成器 (CGLUT)，它可以预测多个 LUT 实例的 GLUT 参数，在单个框架中编码不同的颜色样式，以实现平滑且可控的 LUT 样式混合。此外，GLUT 支持高效、用户友好的编辑，允许对特定颜色区域进行本地调整，而无需全局重新训练。实验结果表明，我们的方法在准确性和效率方面优于先前的神经 LUT 表示，同时提供改进的可解释性和交互控制。

</details>

---

## 6. 3D Skew Gaussian Splatting with Any Camera Trajectory Visualization Engine / 使用任何相机轨迹可视化引擎进行 3D 倾斜高斯泼溅

**Date**: 2026-05-18 | **arXiv**: [2605.18334v1](http://arxiv.org/abs/2605.18334v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.18334v1)

**Categories**: cs.CV, cs.GR

<details><summary><b>Abstract / 摘要</b></summary>

While 3D Gaussian Splatting (3DGS) has revolutionized real-time photorealistic view synthesis, its fundamental reliance on symmetric Gaussian distributions introduces visual artifacts that hinder accurate spatial data exploration. Specifically, symmetric kernels struggle to capture shape and color discontinuities , which cause blurriness and primitive redundancy that mislead human perception during visual analysis. To address these visualization barriers, we introduce 3D Skew Gaussian Splatting (3DSGS), a novel framework that significantly enhances the structural fidelity and compactness of explicit scene representations. Our key insight lies in extending the standard primitive to a general Skew Gaussian counterpart. This generalized primitive inherits the highly efficient rasterization properties of standard Gaussians while gaining intrinsic asymmetric modeling capabilities. We couple this with an enhanced opacity representation to better handle complex transparency, alongside a depth-aware densification strategy that intelligently manages primitive allocation. Furthermore, to make these advancements actionable for real-world visual analytics, we re-derive the CUDA rasterization pipeline to universally support both symmetric and skew Gaussians, integrating it into a decoupled, free-camera interactive visualization engine. Extensive experiments demonstrate that 3DSGS achieves superior rendering quality and structural compactness, particularly in regions with intricate details, while maintaining the real-time frame rates necessary for fluid interactive exploration. Supplementary derivations and visual results are available at \textbf{\textit{https://3d-skew-gs.github.io/}}.

虽然 3D 高斯分布 (3DGS) 彻底改变了实时真实感视图合成，但其对对称高斯分布的根本依赖引入了视觉伪影，阻碍了准确的空间数据探索。具体来说，对称内核很难捕获形状和颜色的不连续性，这会导致模糊和原始冗余，从而在视觉分析过程中误导人类的感知。为了解决这些可视化障碍，我们引入了 3D Skew Gaussian Splatting (3DSGS)，这是一种新颖的框架，可以显着增强显式场景表示的结构保真度和紧凑性。我们的关键见解在于将标准原语扩展到一般的倾斜高斯对应物。这种广义基元继承了标准高斯的高效光栅化属性，同时获得了内在的非对称建模功能。我们将其与增强的不透明度表示相结合，以更好地处理复杂的透明度，以及智能管理原始分配的深度感知致密化策略。此外，为了使这些进步可用于现实世界的视觉分析，我们重新推导了 CUDA 光栅化管道以普遍支持对称和倾斜高斯，并将其集成到解耦的、免费相机的交互式可视化引擎中。大量实验表明，3DSGS 实现了卓越的渲染质量和结构紧凑性，特别是在具有复杂细节的区域，同时保持了流体交互探索所需的实时帧速率。补充推导和可视化结果可在 \textbf{\textit{https://3d-skew-gs.github.io/}} 获得。

</details>

---

## 7. ALGOGEN: Tool-Generated Verifiable Traces for Reliable Algorithm Visualization / ALGOGEN：工具生成的可验证跟踪，用于可靠的算法可视化

**Date**: 2026-05-12 | **arXiv**: [2605.12159v1](http://arxiv.org/abs/2605.12159v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.12159v1)

**Categories**: cs.AI, cs.GR

<details><summary><b>Abstract / 摘要</b></summary>

Algorithm Visualization (AV) helps students build mental models by animating algorithm execution states. Recent LLM-based systems such as CODE2VIDEO generate AV videos in an end-to-end manner. However, this paradigm requires the system to simultaneously simulate algorithm flow and satisfy video rendering constraints, such as element layout and color schemes. This complex task induces LLM hallucinations, resulting in reduced execution success rates, element overlap, and inter-frame inconsistencies.   To address these challenges, we propose ALGOGEN, a novel paradigm that decouples algorithm execution from rendering. We first introduce Visualization Trace Algebra (VTA), a monoid over algorithm visual states and operations. The LLM then generates a Python tracker that simulates algorithm flow and outputs VTA-JSON traces, a JSON encoding of VTA. For rendering, we define a Rendering Style Language (RSL) to templatize algorithm layouts. A deterministic renderer then compiles algorithm traces with RSL into Manim, LaTeX/TikZ, or Three.js outputs.   Evaluated on a LeetCode AV benchmark of 200 tasks, ALGOGEN achieves an average success rate improvement of 17.3% compared to end-to-end methods, with 99.8% versus 82.5%. These results demonstrate that our decoupling paradigm effectively mitigates LLM hallucinations in complex AV tasks, providing a more reliable solution for automated generation of high-quality algorithm visualizations. Demo videos and code are available in the project repository.

算法可视化 (AV) 通过动画算法执行状态帮助学生构建心理模型。最近基于 LLM 的系统（例如 CODE2VIDEO）以端到端方式生成 AV 视频。然而，这种范例要求系统同时模拟算法流程并满足视频渲染约束，例如元素布局和配色方案。这项复杂的任务会引发 LLM 幻觉，导致执行成功率降低、元素重叠和帧间不一致。   为了应对这些挑战，我们提出了 ALGOGEN，这是一种将算法执行与渲染分离的新颖范例。我们首先介绍可视化追踪代数（VTA），这是一种算法视觉状态和操作的幺半群。然后，LLM 生成一个 Python 跟踪器，用于模拟算法流程并输出 VTA-JSON 跟踪（VTA 的 JSON 编码）。对于渲染，我们定义了渲染风格语言（RSL）来模板化算法布局。然后，确定性渲染器使用 RSL 将算法跟踪编译为 Manim、LaTeX/TikZ 或 Three.js 输出。   在 200 个任务的 LeetCode AV 基准上进行评估，与端到端方法相比，ALGOGEN 的平均成功率提高了 17.3%，分别为 99.8% 和 82.5%。这些结果表明，我们的解耦范式有效地减轻了复杂 AV 任务中的 LLM 幻觉，为自动生成高质量算法可视化提供了更可靠的解决方案。项目存储库中提供了演示视频和代码。

</details>

---

## 8. Colorful-Noise: Training-Free Low-Frequency Noise Manipulation for Color-Based Conditional Image Generation / 彩色噪声：用于基于颜色的条件图像生成的免训练低频噪声处理

**Date**: 2026-05-01 | **arXiv**: [2605.00548v2](http://arxiv.org/abs/2605.00548v2) | **PDF**: [Link](http://arxiv.org/pdf/2605.00548v2)

**Categories**: cs.CV, cs.GR

<details><summary><b>Abstract / 摘要</b></summary>

Text-to-image diffusion models generate images by gradually converting white Gaussian noise into a natural image. White Gaussian noise is well suited for producing diverse outputs from a single text prompt due to its absence of structure. However, this very property limits control over, and predictability of, specific visual attributes, as the noise is not human-interpretable. In this work, we investigate the characteristics of the input noise in diffusion models. We show that, although all frequencies in white Gaussian noise have comparable statistical energy, low-frequency components primarily determine the images global structure and color composition, while high-frequency components control finer details. Building on this observation, we demonstrate that simple manipulations of the low-frequency noise using low-frequency image priors can effectively condition the generation process to reconstruct these low-frequency visual cues. This allows us to define a simple, training-free method with minimal overhead that steers overall image structure and color, while letting high-frequency components freely emerge as fine details, enabling variability across generated outputs.

文本到图像扩散模型通过逐渐将高斯白噪声转换为自然图像来生成图像。由于缺乏结构，高斯白噪声非常适合从单个文本提示生成不同的输出。然而，这种特性限制了对特定视觉属性的控制和可预测性，因为噪声是人类无法解释的。在这项工作中，我们研究了扩散模型中输入噪声的特征。我们表明，尽管高斯白噪声中的所有频率都具有可比较的统计能量，但低频分量主要决定图像的全局结构和颜色组成，而高频分量控制更精细的细节。基于这一观察，我们证明使用低频图像先验对低频噪声进行简单操作可以有效地调节生成过程以重建这些低频视觉线索。这使我们能够定义一种简单、免训练的方法，以最小的开销控制整体图像结构和颜色，同时让高频分量自由地以精细细节的形式出现，从而实现生成的输出的可变性。

</details>

---



</details>

<details><summary><b>2026-05-28 (19 papers)</b></summary>

# arXiv Tone & Color Papers - 2026-05-28

**Paper Count**: 19

---

## 1. A Matter of TASTE: Improving Coverage and Difficulty of Agent Benchmarks / 品味问题：提高代理基准的覆盖范围和难度

**Date**: 2026-05-27 | **arXiv**: [2605.28556v1](http://arxiv.org/abs/2605.28556v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.28556v1)

**Categories**: cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

As agent capabilities advance, existing benchmarks, such as $τ^2$-Bench, are becoming increasingly saturated. Yet constructing new benchmark tasks remains complex, costly, and labor-intensive. Moreover, the standard approach, in which scenarios are first written in natural language and then mapped to tool sequences, captures only a narrow subset of the tool-use patterns agents exercise. In this paper, we address these problems by reversing the task construction process. We propose TASTE: Task Synthesis from Tool Sequence Evolution, an automatic method that generates challenging tasks with broader tool-use coverage. TASTE utilizes an Adaptive Contrastive $n$-gram model trained on LLM-judged validity signals. This enables sampling valid tool sequences that cover a vast range of tool combinations. TASTE then selects representative sequences from the pool via clustering, instantiates them into complete benchmark tasks, and refines them through iterative difficulty evolution. Using TASTE, we construct $τ^c$-Bench, a challenging extension of the three domains of $τ^2$-Bench. We evaluate $11$ agent/user LLM pairs and find that models nearly saturating $τ^2$-Bench suffer severe performance drops on our tasks (e.g., Gemini-3-Flash falls from $0.82\!-\!0.94$ to $0.28\!-\!0.61$). Beyond increasing difficulty, our generated tasks more than double the number of unique tool combinations agents must execute. Our results suggest high scores on existing benchmarks often reflect saturation rather than robust task-solving ability. By automating the generation of difficult, high-coverage benchmarks, TASTE enables continuous, scalable evaluation of future agents.

随着代理能力的进步，现有的基准（例如 $τ^2$-Bench）正变得越来越饱和。然而构建新的基准任务仍然复杂、成本高昂且劳动密集型。此外，标准方法首先用自然语言编写场景，然后映射到工具序列，仅捕获代理练习的工具使用模式的一小部分。在本文中，我们通过逆向任务构建过程来解决这些问题。我们提出了 TASTE：来自工具序列进化的任务合成，这是一种自动方法，可以生成具有更广泛工具使用范围的挑战性任务。 TASTE 利用基于 LLM 判断的有效性信号训练的自适应对比 $n$-gram 模型。这使得能够对覆盖广泛工具组合的有效工具序列进行采样。然后，TASTE 通过聚类从池中选择代表性序列，将它们实例化为完整的基准任务，并通过迭代难度演化对其进行细化。使用 TASTE，我们构建了 $τ^c$-Bench，这是 $τ^2$-Bench 三个域的具有挑战性的扩展。我们评估了 11 美元的代理/用户 LLM 对，发现接近饱和 $τ^2$-Bench 的模型在我们的任务中性能严重下降（例如，Gemini-3-Flash 从 $0.82\!-\!0.94$ 下降到 $0.28\!-\!0.61$）。除了难度不断增加之外，我们生成的任务使代理必须执行的独特工具组合数量增加了一倍以上。我们的结果表明，现有基准的高分通常反映了饱和度，而不是强大的任务解决能力。通过自动生成困难的、高覆盖率的基准，TASTE 能够对未来代理进行持续、可扩展的评估。

</details>

---

## 2. SmartIterator: Visual Analytics Workflows for Supervising Unsupervised Data Grouping / SmartIterator：用于监督无监督数据分组的可视化分析工作流程

**Date**: 2026-05-27 | **arXiv**: [2605.28219v1](http://arxiv.org/abs/2605.28219v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.28219v1)

**Categories**: cs.HC, cs.AI, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Unsupervised learning methods -- topic modeling, partition-based and density-based clustering -- produce data groupings without human guidance, yet choosing and evaluating those groupings should not itself be unsupervised. We present \emph{SmartIterator}~(SI), a visual analytics approach that treats the full sequence of grouping results across a parameter sweep as a first-class analytical object. For each method family, SI provides a structured six-phase workflow that guides the analyst through systematic exploration of grouping results -- from quality-metric overview through transition-stability assessment, membership-confidence evaluation, content and context inspection, and recurrent-archetype verification to an informed decision -- building cumulative understanding of data structure along the way. The workflows are operationalized through \emph{IteraScope}~(IS), a coordinated visual display combining quality-metric charts with semantic color encoding, a 1D group embedding with Sankey-style transition flows and violin plots of membership confidence, a 2D group embedding with HDBSCAN-detected recurrent archetypes that highlights iterations capturing all persistent patterns, and domain-specific linked views for contextualized interpretation. We demonstrate the three workflows on: (1)~simulated social-media messages from the VAST Challenge 2011 (density-based clustering, validated against ground truth), (2)~EU population statistics across ${\sim}1\,500$ NUTS-3 regions (partition-based clustering), and (3)~30 years of IEEE VIS papers (NMF topic modeling). The workflows constitute the main contribution: they provide actionable, method-specific guidance for navigating parameter spaces, studying how data structure evolves across configurations, and grounding analytical understanding in domain context -- yielding knowledge about the data that no single ``best'' result can provide.

无监督学习方法——主题建模、基于分区和基于密度的聚类——在没有人工指导的情况下生成数据分组，但选择和评估这些分组本身不应该是无监督的。我们提出了 \emph{SmartIterator}~(SI)，这是一种可视化分析方法，它将参数扫描中的分组结果的完整序列视为一流的分析对象。对于每个方法系列，SI 提供了一个结构化的六阶段工作流程，指导分析师对分组结果进行系统探索——从质量度量概述到转换稳定性评估、成员置信度评估、内容和上下文检查以及循环原型验证，再到明智的决策——在此过程中建立对数据结构的累积理解。工作流程通过 \emph{IteraScope}~(IS) 进行操作，这是一种将质量度量图表与语义颜色编码相结合的协调视觉显示，具有 Sankey 风格转换流和成员置信度小提琴图的 1D 组嵌入，具有 HDBSCAN 检测到的循环原型的 2D 组嵌入，突出显示捕获所有持久模式的迭代，以及用于上下文解释的特定于域的链接视图。我们演示了以下三个工作流程：(1)~2011 年 VAST 挑战赛的模拟社交媒体消息（基于密度的聚类，根据真实情况进行验证），(2)~${\sim}1\,500$ NUTS-3 区域的欧盟人口统计数据（基于分区的聚类），以及 (3)~30 年的 IEEE VIS 论文（NMF 主题建模）。工作流程构成了主要贡献：它们提供了可操作的、特定于方法的指导，用于导航参数空间、研究数据结构如何跨配置演变以及在领域上下文中奠定分析理解的基础——产生任何单一“最佳”结果都无法提供的数据知识。

</details>

---

## 3. Bridging the Detection-to-Abstention Gap in Reasoning Models under Insufficient Information / 弥合信息不足的推理模型中检测到放弃的差距

**Date**: 2026-05-27 | **arXiv**: [2605.28070v1](http://arxiv.org/abs/2605.28070v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.28070v1)

**Categories**: cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

We highlight a failure mode of large reasoning models on questions with insufficient information: models may recognize that a problem is under-specified, yet still continue reasoning and produce unsupported final answers instead of abstaining. We formalize this mismatch as the detection-to-abstention gap, where detected insufficiency fails to translate into final abstention. This gap is especially concerning in high-risk domains such as medical AI, where answers based on incomplete evidence can be more harmful than refusal. To close this gap, we propose Judge-Then-Solve (JTS), a trajectory-level reasoning-control framework that trains models to make an explicit answerability commitment before solution generation. Rather than treating abstention as a final-answer style, JTS casts it as a control decision: the model either proceeds to solve or terminates early based on its answerability judgment. We instantiate this policy through supervised warm-up and missing-premise reinforcement learning with consistency and length-shaping rewards. Experiments on dense and MoE reasoning models show that JTS substantially improves reliable abstention across datasets and pushes Abstention@Detection (A@D) to near-saturation, indicating that models not only detect missing information but also act on that detection. By terminating unanswerable trajectories immediately after the answerability judgment, JTS reduces unnecessary reasoning and improves inference efficiency when continued deliberation would amplify unsupported assumptions. We also observe that missing-premise training can alter reasoning behavior on difficult but answerable problems, reducing unproductive self-reflection. These results suggest that abstention under insufficient information is a key form of reasoning control for deploying reasoning models safely and efficiently.

我们强调大型推理模型在信息不足的问题上的失败模式：模型可能认识到问题未明确说明，但仍然继续推理并产生不受支持的最终答案，而不是放弃。我们将这种不匹配形式化为检测到弃权差距，其中检测到的不足无法转化为最终弃权。这种差距在医疗人工智能等高风险领域尤其令人担忧，在这些领域，基于不完整证据的答案可能比拒绝更有害。为了缩小这一差距，我们提出了 Judge-Then-Solve (JTS)，这是一种轨迹级推理控制框架，可训练模型在解决方案生成之前做出明确的可回答性承诺。 JTS 没有将弃权视为最终答案风格，而是将其视为一种控制决策：模型根据其可回答性判断要么继续解决问题，要么提前终止。我们通过有监督的热身和缺失前提强化学习来实例化该策略，并具有一致性和长度塑造奖励。密集推理模型和 MoE 推理模型的实验表明，JTS 极大地提高了跨数据集的可靠弃权率，并将 Abstention@Detection (A@D) 推向接近饱和，这表明模型不仅可以检测丢失的信息，还可以根据检测结果采取行动。通过在可回答性判断后立即终止无法回答的轨迹，当继续审议会放大不受支持的假设时，JTS 减少了不必要的推理并提高了推理效率。我们还观察到，缺失前提训练可以改变对困难但可回答问题的推理行为，减少无效的自我反思。这些结果表明，信息不足下的弃权是安全有效地部署推理模型的推理控制的关键形式。

</details>

---

## 4. STAB: Specification-driven Testing for Algorithmic Bottlenecks / STAB：针对算法瓶颈的规范驱动测试

**Date**: 2026-05-27 | **arXiv**: [2605.27981v1](http://arxiv.org/abs/2605.27981v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.27981v1)

**Categories**: cs.AI

**Code**: https://github.com/suhanmen/STAB.

<details><summary><b>Abstract / 摘要</b></summary>

Evaluating the efficiency of algorithmic code requires test cases that expose runtime bottlenecks. Previous methods generate efficiency test cases either by increasing input size or by generating code-specific inputs that make the given implementation run slowly. Consequently, they do not address the structural input conditions that drive the algorithmic worst case. We introduce STAB, a specification-driven pipeline that generates test cases that expose algorithmic bottlenecks from a natural-language problem specification alone. STAB separates the task into constraint-bound maximization and adversarial structure injection. (i) The constraint saturator extracts constraints and resolves large admissible size assignments using rule-based saturation and CP-SAT optimization over related variables. (ii) The adversarial scenario injector retrieves implementation-level adversarial construction principles from a curated scenario catalog using keyword matching and K-nearest neighbors (KNN). STAB encodes the problem specification, resolved boundary, and retrieved construction principles into a structured generation specification, from which the LLM synthesizes a Python test case generator. On CodeContests, STAB raises the rate of generated test cases that expose algorithmic bottlenecks from 50.43% to 73.45% on average across open-source LLMs and from 57.45% to 71.85% on average across closed-source LLMs, with consistent gains across Python, Java, and C++. Our code is available at https://github.com/suhanmen/STAB.

评估算法代码的效率需要暴露运行时瓶颈的测试用例。以前的方法通过增加输入大小或生成使给定实现运行缓慢的特定于代码的输入来生成效率测试用例。因此，它们没有解决驱动算法最坏情况的结构输入条件。我们引入了 STAB，这是一个规范驱动的管道，它生成测试用例，仅从自然语言问题规范中暴露算法瓶颈。 STAB 将任务分为约束约束最大化和对抗性结构注入。 (i) 约束饱和器使用基于规则的饱和和相关变量的 CP-SAT 优化来提取约束并解决大的可接受的大小分配。 (ii) 对抗场景注入器使用关键字匹配和 K 最近邻 (KNN) 从策划的场景目录中检索实现级对抗构建原则。 STAB 将问题规范、已解决的边界和检索到的构造原理编码为结构化生成规范，LLM 从中合成 Python 测试用例生成器。在 CodeContests 上，STAB 将暴露算法瓶颈的测试用例生成率在开源 LLM 中平均从 50.43% 提高到 73.45%，在闭源 LLM 中平均从 57.45% 提高到 71.85%，并且在 Python、Java 和 C++ 中都有一致的收益。我们的代码可在 https://github.com/suhanmen/STAB 获取。

</details>

---

## 5. NeR-SC: Adapting Neural Video Representation to Screen Content / NeR-SC：使神经视频表示适应屏幕内容

**Date**: 2026-05-26 | **arXiv**: [2605.27024v1](http://arxiv.org/abs/2605.27024v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.27024v1)

**Categories**: cs.CV, cs.MM

<details><summary><b>Abstract / 摘要</b></summary>

Implicit neural representations have emerged as a promising paradigm for video compression, with recent methods achieving competitive performance on natural video. However, screen content video -- common in remote desktop, online education, and cloud gaming -- exhibits distinct statistics: sharp edges, limited color palettes, and strong temporal redundancy. Existing neural representation methods, designed for natural scenes, lack mechanisms to exploit these properties, leaving substantial room for improvement. In this paper, we propose NeR-SC, a neural representation framework tailored for screen content video. Building on the SNeRV backbone, NeR-SC introduces three screen-content-specific modules: (i) a learnable color palette that models the discrete color structure of screen content by restricting the low-frequency sub-band to a learned color set; (ii) a multi-gate dense fusion module that replaces sequential feature fusion with dense, attention-gated cross-stage interaction; and (iii) an embedding-level frame skip strategy that bypasses redundant decoder invocations for static frames, with zero training overhead. Experiments on DSCVC and VCD show that NeR-SC achieves 40.32~dB and 41.73~dB average PSNR, outperforming representative neural video representation methods and, at low bitrates, surpassing H.264 and H.265. The skip strategy enables real-time decoding with no loss in quality.

隐式神经表示已成为视频压缩的一种有前途的范例，最近的方法在自然视频上实现了具有竞争力的性能。然而，屏幕内容视频（常见于远程桌面、在线教育和云游戏）表现出明显的统计数据：锐利的边缘、有限的调色板和强大的时间冗余。现有的针对自然场景设计的神经表示方法缺乏利用这些特性的机制，因此留下了很大的改进空间。在本文中，我们提出了 NeR-SC，一种专为屏幕内容视频量身定制的神经表示框架。 NeR-SC 基于 SNeRV 主干网络，引入了三个特定于屏幕内容的模块：（i）可学习的调色板，通过将低频子带限制为学习的颜色集来模拟屏幕内容的离散颜色结构； (ii) 多门密集融合模块，用密集、注意力门控的跨阶段交互取代顺序特征融合； (iii) 嵌入级跳帧策略，绕过静态帧的冗余解码器调用，训练开销为零。在 DSCVC 和 VCD 上的实验表明，NeR-SC 的平均 PSNR 达到了 40.32~dB 和 41.73~dB，优于代表性的神经视频表示方法，并且在低比特率下超过了 H.264 和 H.265。跳过策略可以实现实时解码，且不会降低质量。

</details>

---

## 6. ROI Extraction in Thermographic Breast Images Using Genetic Algorithms / 使用遗传算法提取热成像乳腺图像中的 ROI

**Date**: 2026-05-21 | **arXiv**: [2605.22899v1](http://arxiv.org/abs/2605.22899v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.22899v1)

**Categories**: q-bio.TO, eess.IV

<details><summary><b>Abstract / 摘要</b></summary>

This work proposes the use of Genetic Algorithms (GA) to identify the area of the breast from the background in thermographic breast images. The proposed method uses color information, a fitness function based on cardioids, and GA. This is the first work in the literature to propose a Region of Interest (ROI) extraction based on GA and cariods. ROI extraction can improve the accuracy of cancer detection and assist with the standardization of acquisition protocols. The method is able to successfully separate the breast region in 52 out of 58 images, while being fully automatic, and not requiring manual selection of seed points.

这项工作建议使用遗传算法 (GA) 从热成像乳房图像的背景中识别乳房区域。所提出的方法使用颜色信息、基于心形的适应度函数和 GA。这是文献中第一个提出基于 GA 和 cariod 的感兴趣区域 (ROI) 提取的工作。 ROI 提取可以提高癌症检测的准确性，并有助于采集协议的标准化。该方法能够成功地从 58 幅图像中的 52 幅图像中分离出乳房区域，同时是全自动的，不需要手动选择种子点。

</details>

---

## 7. Time-varying rPPG signal separation via block-sparse signal model / 通过块稀疏信号模型进行时变 rPPG 信号分离

**Date**: 2026-05-21 | **arXiv**: [2605.22425v1](http://arxiv.org/abs/2605.22425v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.22425v1)

**Categories**: eess.IV, cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Remote photoplethysmography (rPPG) enables non-contact measurement of cardiac pulse signals by analyzing subtle color changes in facial videos. Nevertheless, extracting rPPG signals remains challenging because of their extremely weak signal strength and susceptibility to illumination noise. In this paper, we propose an rPPG signal extraction method that exploits the quasi-periodic characteristics of rPPG signals. Our approach models quasi-periodicity of the rPPG signal, which arises from the stable cardiac cycle, as a block-sparse structure in the time-frequency domain. To incorporate a block-sparse model and enable adaptive signal separation under illumination fluctuations, we construct a time-varying signal separation framework. Experiments using a public dataset demonstrate the effectiveness of our method.

远程光电体积描记法 (rPPG) 通过分析面部视频中细微的颜色变化，实现心脏脉冲信号的非接触式测量。然而，提取 rPPG 信号仍然具有挑战性，因为它们的信号强度极弱并且容易受到照明噪声的影响。在本文中，我们提出了一种利用 rPPG 信号的准周期特性的 rPPG 信号提取方法。我们的方法将 rPPG 信号的准周期性建模为时频域中的块稀疏结构，该信号源自稳定的心动周期。为了合并块稀疏模型并在光照波动下实现自适应信号分离，我们构建了一个时变信号分离框架。使用公共数据集的实验证明了我们方法的有效性。

</details>

---

## 8. Probability-Conserving Flow Guidance / 概率守恒流程指导

**Date**: 2026-05-19 | **arXiv**: [2605.20079v1](http://arxiv.org/abs/2605.20079v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.20079v1)

**Categories**: cs.CV, cs.AI, cs.LG, eess.IV

<details><summary><b>Abstract / 摘要</b></summary>

Diffusion and flow-based generative models dominate visual synthesis, with guidance aligning samples to user input and improving perceptual quality. However, Classifier-Free Guidance (CFG) and extrapolation-based methods are heuristic linear combinations of velocities/scores that ignore the generative manifold geometry, breaking probability conservation and driving samples off the learned manifold under strong guidance. We analyse guidance through the continuity equation and show its effect decomposes into a divergence term and a score-parallel term defined invariantly across parameterisations. We prove the divergence term blows up structurally as sampling approaches the data manifold, motivating a time-dependent schedule alongside score-parallel attenuation. The resulting plug-and-play rule, Adaptive Manifold Guidance (AdaMaG), bounds both terms at no additional inference cost. Finally, we show that most empirical heuristics for reducing saturation or improving generation quality correspond directly to the two terms in our decomposition. Across image generation benchmarks, AdaMaG improves realism, reduces hallucinations, and induces controlled desaturation in high-guidance regimes.

基于扩散和流的生成模型主导着视觉合成，指导将样本与用户输入对齐并提高感知质量。然而，无分类器指导（CFG）和基于外推的方法是速度/分数的启发式线性组合，忽略了生成流形几何，打破了概率守恒并在强指导下将样本从学习流形中驱动出来。我们通过连续性方程分析指导，并表明其效果分解为散度项和在参数化过程中不变定义的分数平行项。我们证明，当采样接近数据流形时，发散项会在结构上爆炸，从而激发时间相关的时间表以及分数平行衰减。由此产生的即插即用规则，自适应流形指导（AdaMaG），在没有额外推理成本的情况下限制了这两个项。最后，我们表明，大多数用于降低饱和度或提高发电质量的经验启发法直接对应于我们分解中的这两项。在图像生成基准中，AdaMaG 提高了真实感，减少了幻觉，并在高指导制度下诱导受控去饱和。

</details>

---

## 9. LUMEN: Low-light Unified Multi-stage Enhancement Network using depth-guided flash, clustering, and attention-based Transformers / LUMEN：使用深度引导闪存、集群和基于注意力的 Transformer 的低光统一多级增强网络

**Date**: 2026-05-18 | **arXiv**: [2605.17893v1](http://arxiv.org/abs/2605.17893v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.17893v1)

**Categories**: eess.IV

<details><summary><b>Abstract / 摘要</b></summary>

Low-light image enhancement remains a challenging problem due to severe noise, color distortion, contrast degradation, and loss of structural details under insufficient illumination. Existing methods typically apply uniform enhancement without considering the depth-dependent nature of light attenuation and sensor noise in real-world scenes. To address this limitation, we propose LUMEN, a multi-stage enhancement framework that integrates virtual flash simulation with transformer-based feature fusion. The proposed framework first estimates scene depth from low-light inputs using a dedicated encoder-decoder network, after which a soft clustering module partitions pixels into depth-aware regions, enabling depth-dependent flash simulation. The simulated flash features, together with depth representations, are fused with image features through efficient attention-based fusion blocks to enhance global context while preserving fine details. A composite loss function combining reconstruction, perceptual, structural, color, edge, and depth consistency objectives ensures both visual fidelity and perceptual quality. Extensive experiments on LOL-v1 and LOL-v2 benchmarks demonstrate that LUMEN achieves state-of-the-art performance and produces visually natural results compared with several state-of-the-art methods.

由于光照不足下严重的噪声、颜色失真、对比度下降和结构细节丢失，低光图像增强仍然是一个具有挑战性的问题。现有方法通常应用均匀增强，而不考虑现实场景中光衰减和传感器噪声的深度相关性质。为了解决这个限制，我们提出了 LUMEN，这是一个多阶段增强框架，它将虚拟闪存模拟与基于变压器的特征融合相集成。所提出的框架首先使用专用的编码器-解码器网络从低光输入估计场景深度，然后软聚类模块将像素划分为深度感知区域，从而实现与深度相关的闪光模拟。模拟的闪光特征与深度表示一起，通过有效的基于注意力的融合块与图像特征融合，以增强全局上下文，同时保留精细细节。结合了重建、感知、结构、颜色、边缘和深度一致性目标的复合损失函数确保了视觉保真度和感知质量。对 LOL-v1 和 LOL-v2 基准的大量实验表明，与几种最先进的方法相比，LUMEN 实现了最先进的性能，并产生了视觉上自然的结果。

</details>

---

## 10. An Underwater Dehazing Network with Implicit Transmission Estimation / 具有隐式传输估计的水下除雾网络

**Date**: 2026-05-13 | **arXiv**: [2605.13720v1](http://arxiv.org/abs/2605.13720v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.13720v1)

**Categories**: eess.IV

<details><summary><b>Abstract / 摘要</b></summary>

Underwater images suffer from wavelength-dependent light absorption and scattering, which reduces visual quality. This phenomenon could limit the operational reliability of autonomous underwater vehicles, marine surveys, and offshore inspection systems. Purely classical methods often achieve suboptimal performance in real-world datasets, while purely data-driven methods lack physical interpretability. In this letter, we propose UDehaze-iT, a deep network for underwater image enhancement that estimates scene depth implicitly and derives per-channel transmission through the Beer-Lambert law with learnable attenuation coefficients. We estimate atmospheric light as a semi-classical per-channel scalar, and a zero-initialized residual refiner corrects remaining artefacts after dehazing. To effectively train our method, we apply a composite loss function consisting of five key terms: a L1 loss, a multi-scale patchwise DCT loss, a forward model reconstruction loss, and two regularization terms. With ~0.9M parameters, UDehaze-iT achieves competitive performance on UIEB and UFO-120 datasets.

水下图像会受到波长相关的光吸收和散射的影响，从而降低视觉质量。这种现象可能会限制自主水下航行器、海洋调查和近海检查系统的运行可靠性。纯粹的经典方法通常在现实世界的数据集中实现次优性能，而纯粹的数据驱动方法缺乏物理可解释性。在这封信中，我们提出了 UDehaze-iT，这是一种用于水下图像增强的深度网络，它隐式估计场景深度，并通过具有可学习衰减系数的比尔-朗伯定律导出每通道传输。我们将大气光估计为半经典的每通道标量，并且零初始化的残差细化器在去雾后校正剩余的伪影。为了有效地训练我们的方法，我们应用了由五个关键项组成的复合损失函数：L1 损失、多尺度补丁 DCT 损失、前向模型重建损失和两个正则化项。 UDehaze-iT 凭借约 0.9M 参数，在 UIEB 和 UFO-120 数据集上实现了具有竞争力的性能。

</details>

---

## 11. Physics-Grounded Adversarial Stain Augmentation with Calibrated Coverage Guarantees / 基于物理的对抗性染色增强和校准覆盖保证

**Date**: 2026-05-12 | **arXiv**: [2605.13889v1](http://arxiv.org/abs/2605.13889v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.13889v1)

**Categories**: eess.IV, cs.CV, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Stain variation across hospitals degrades histopathology models at deployment. Existing augmentation methods perturb color spaces with arbitrary hyperparameters, lacking both a principled budget and coverage guarantees for unseen centers. We propose \textbf{C}alibrated \textbf{A}dversarial \textbf{S}tain \textbf{A}ugmentation (\textbf{CASA}), which performs adversarial augmentation in the Macenko stain parameter space with a budget calibrated from multi-center statistics via the DKW inequality. On Camelyon17-WILDS (5 seeds), CASA achieves $93.9\% \pm 1.6\%$ slide-level accuracy -- outperforming HED-strong ($88.4\% \pm 7.3\%$), RandStainNA ($85.2\% \pm 6.7\%$), and ERM ($63.9\% \pm 11.3\%$) -- with the highest worst-group accuracy ($84.9\% \pm 0.9\%$) among all 10 compared methods.

各医院的染色差异会降低部署时的组织病理学模型。现有的增强方法用任意超参数扰乱色彩空间，缺乏原则性的预算和对看不见的中心的覆盖保证。我们提出 \textbf{C}alibated \textbf{A}dversarial \textbf{S}tain \textbf{A}ugmentation (\textbf{CASA})，它在 Macenko 染色参数空间中执行对抗性增强，其预算通过 DKW 不等式从多中心统计数据校准。在 Camelyon17-WILDS（5 颗种子）上，CASA 达到了 $93.9\% \pm 1.6\%$ 滑动精度 - 优于 HED-strong ($88.4\% \pm 7.3\%$)、RandStainNA ($85.2\% \pm 6.7\%$) 和 ERM ($63.9\% \pm 11.3\%$) - 最高所有 10 种比较方法中最差的组准确度 ($84.9\% \pm 0.9\%$)。

</details>

---

## 12. Are Compact Rationales Free? Measuring Tile Selection Headroom in Frozen WSI-MIL / 紧凑基本原理是免费的吗？测量冻结 WSI-MIL 中的瓷砖选择余量

**Date**: 2026-05-12 | **arXiv**: [2605.12575v1](http://arxiv.org/abs/2605.12575v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.12575v1)

**Categories**: eess.IV, cs.AI, cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Whole-slide image (WSI) multiple instance learning (MIL) classifiers can achieve strong slide-level AUC while leaving the full-bag prediction opaque. Attention scores are widely reused as post-hoc explanations, but high attention can reflect aggregation preference rather than a compact, model-sufficient rationale. We study post-hoc rationale highlighting for frozen WSI-MIL: given a trained classifier, can its slide-level prediction be recovered from a compact, output-consistent tile subset without retraining the backbone? We instantiate this with Finding Optimal Contextual Instances (FOCI), a lightweight rationale-readout layer over a frozen MIL backbone. FOCI is trained with model-output sufficiency and exclusion objectives over keep/drop tile subsets, evaluated with an insertion-style Sequential Reveal Protocol (SRP) adapted to WSI-MIL, and summarized by the Selection Headroom Index (SHI). Across three WSI benchmarks and seven MIL backbones, FOCI reveals that compact rationales are selection-headroom dependent: transformer and multi-branch attention aggregators can admit compact rationales, near-minimal attention-pooling baselines enter a selection-saturation regime, and hard-selection backbones can conflict with an external readout. For TransMIL, relative to its documented CLS-proxy ranking, FOCI reduces the Minimum Sufficient K (MSK) tile count by 32-56% across benchmarks, while ACMIL+FOCI attains the highest mean SHI (+0.465). Deletion-based perturbation and selected-only downstream evaluation provide complementary checks. These results position FOCI as a model-level interpretability and audit layer: selected tiles are not claims of clinical or pathologist-level diagnostic sufficiency, but candidate rationales that offer a compact, reviewable view of when a frozen MIL prediction can be localized to a small output-consistent subset.

全幻灯片图像 (WSI) 多实例学习 (MIL) 分类器可以实现强大的幻灯片级 AUC，同时使整袋预测不透明。注意力分数被广泛用作事后解释，但高注意力可以反映聚合偏好，而不是紧凑的、模型充足的基本原理。我们研究了冻结 WSI-MIL 的事后基本原理突出显示：给定一个经过训练的分类器，是否可以从紧凑的、输出一致的图块子集中恢复其滑动级预测，而无需重新训练主干网？我们通过寻找最佳上下文实例 (FOCI) 来实例化这一点，FOCI 是冻结 MIL 主干上的轻量级原理读出层。 FOCI 使用模型输出充分性和保留/丢弃图块子集的排除目标进行训练，使用适合 WSI-MIL 的插入式顺序显示协议 (SRP) 进行评估，并通过选择余量指数 (SHI) 进行总结。在三个 WSI 基准和七个 MIL 主干中，FOCI 揭示了紧凑的基本原理是选择余量依赖的：变压器和多分支注意力聚合器可以承认紧凑的基本原理，接近最小的注意力池基线进入选择饱和状态，而硬选择主干可能与外部读数发生冲突。对于 TransMIL，相对于其记录的 CLS 代理排名，FOCI 在基准测试中将最小足够 K (MSK) 切片数量减少了 32-56%，而 ACMIL+FOCI 获得了最高的平均 SHI (+0.465)。基于删除的扰动和仅选择的下游评估提供了补充检查。这些结果将 FOCI 定位为模型级可解释性和审核层：选定的图块并不是临床或病理学家级诊断充分性的声明，而是候选原理，它们提供了一个紧凑的、可审查的视图，说明何时可以将冻结的 MIL 预测本地化到一个小的输出一致子集。

</details>

---

## 13. Kelvin v1.0: A Neural Pre-Encoder for H.264: A standards-compliant learned preprocessor with -27.62% BD-VMAF on UVG / Kelvin v1.0：H.264 的神经预编码器：符合标准的学习预处理器，UVG 上的 BD-VMAF 为 -27.62%

**Date**: 2026-05-10 | **arXiv**: [2605.16376v1](http://arxiv.org/abs/2605.16376v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.16376v1)

**Categories**: eess.IV, cs.CV, cs.DC, cs.LG, cs.MM

<details><summary><b>Abstract / 摘要</b></summary>

Kelvin is a lightweight learned pre-encoder that sits in front of an unmodified libx264 encoder. It applies content-adaptive pixel adjustments, bounded at +/-1/255 per channel, so that the encoder allocates bits where they matter most perceptually, while emitting a standard H.264 bitstream compatible with every existing decoder, player, and CDN. On the seven-sequence 1080p UVG benchmark, Kelvin v1.0 achieves a mean BD-VMAF of -27.62% (7 of 7 wins) and BD-VMAF-NEG of -5.18% (6 of 7 wins) relative to baseline libx264 at preset medium. On the 30-sequence MCL-JCV public set (28 unseen by training), the same checkpoint wins on 28 of 30 clips by BD-VMAF; with the two diagnosable failures removed the mean is -27.70% BD-VMAF and -5.37% BD-VMAF-NEG, consistent with UVG to within one percentage point. A central engineering challenge is the non-differentiability of H.264: we describe a hybrid codec proxy that combines a calibrated differentiable rate estimator (Spearman rho = 0.986 vs. real libx264 bits-per-pixel) with a U-Net distortion proxy trained on real encoder outputs. We publish full per-sequence rate-distortion data, a named failure-mode taxonomy on MCL-JCV (rate-floor violation, distribution shift, metric saturation), a five-baseline sanity panel (hqdn3d, unsharp, -tune psnr, -tune ssim, x265 medium), and honest positioning: x265 medium beats Kelvin on every metric on the same corpus. Kelvin is therefore designed for workloads where remaining on H.264 is a constraint rather than a choice.

Kelvin 是一个轻量级学习预编码器，位于未修改的 libx264 编码器前面。它应用内容自适应像素调整，每个通道的范围为 +/-1/255，以便编码器在感知最重要的位置分配比特，同时发出与每个现有解码器、播放器和 CDN 兼容的标准 H.264 比特流。在七序列 1080p UVG 基准测试中，相对于预设介质下的基线 libx264，Kelvin v1.0 的平均 BD-VMAF 为 -27.62%（7 胜中的 7 胜），BD-VMAF-NEG 为 -5.18%（7 胜中的 6 胜）。在 30 序列 MCL-JCV 公共集（训练中未看到 28 个）上，同一检查点在 BD-VMAF 的 30 个片段中的 28 个中获胜；除去两个可诊断故障后，平均值为 -27.70% BD-VMAF 和 -5.37% BD-VMAF-NEG，与 UVG 一致，误差在 1 个百分点以内。一个核心的工程挑战是 H.264 的不可微性：我们描述了一种混合编解码器代理，它将校准的可微速率估计器（Spearman rho = 0.986 对比真实的 libx264 位/像素）与在真实编码器输出上训练的 U-Net 失真代理相结合。我们发布了完整的每序列率失真数据、MCL-JCV 上的命名故障模式分类法（速率下限违规、分布偏移、指标饱和度）、五基线健全性面板（hqdn3d、unsharp、-tune psnr、-tune ssim、x265medium）以及诚实定位：x265medium 在同一语料库上的每个指标上都击败了 Kelvin。因此，Kelvin 专为保留 H.264 是一种限制而不是一种选择的工作负载而设计。

</details>

---

## 14. CAGS: Color-Adaptive Volumetric Video Streaming with Dynamic 3D Gaussian Splatting / CAGS：具有动态 3D 高斯分布的颜色自适应体积视频流

**Date**: 2026-05-10 | **arXiv**: [2605.09279v1](http://arxiv.org/abs/2605.09279v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.09279v1)

**Categories**: cs.GR, cs.CV, cs.MM, cs.NI, eess.IV

<details><summary><b>Abstract / 摘要</b></summary>

Volumetric video (VV) streaming enables real-time, immersive access to remote 3D environments, powering telepresence, ecological monitoring, and robotic teleoperation. These applications turn VV streaming into a real-time interface to remote physical environments, imposing new system-level demands for photorealistic scene representation, low-latency interaction, and robust performance under heterogeneous networks. 3D Gaussian Splatting (3DGS) has been widely used for real-time photorealistic rendering, offering superior visual quality and rendering performance, but it faces challenges due to bandwidth consumption. Furthermore, as the foundation of adaptive VV streaming, existing Levels of Detail (LoD) methods based on density are not well-suited to Gaussian representations, leading to visible gaps and severe quality degradation. Recent studies have also explored attribute compression techniques to reduce bandwidth consumption. Our preliminary studies reveal that aggressive attribute compression primarily causes color distortion, which can be effectively corrected in the rendered image using a reference image. Motivated by these findings, we propose a novel Color-Adaptive scheme for adaptive VV streaming that uses vector quantization (VQ) to establish LoDs and correct color distortions with low-resolution reference images. We further present CAGS, an adaptive VV streaming system compatible with diverse Gaussian representations, which integrates the Color-Adaptive scheme by rendering reference images on the streaming server and performing color restoration on the client. Extensive experiments on our prototype system demonstrate that CAGS outperforms the existing adaptive streaming systems in PSNR by 5$\sim$20 dB under fluctuating bandwidth, operates significantly faster than existing scalable Gaussian compression methods, and generalizes across different Gaussian representations.

体积视频 (VV) 流媒体可实现对远程 3D 环境的实时、沉浸式访问，为远程呈现、生态监测和机器人远程操作提供支持。这些应用程序将 VV 流转为远程物理环境的实时接口，对异构网络下的逼真场景表示、低延迟交互和鲁棒性能提出了新的系统级需求。 3D高斯溅射（3DGS）已广泛用于实时真实感渲染，提供卓越的视觉质量和渲染性能，但由于带宽消耗而面临挑战。此外，作为自适应 VV 流的基础，现有的基于密度的细节级别 (LoD) 方法不太适合高斯表示，导致可见的间隙和严重的质量下降。最近的研究还探索了属性压缩技术来减少带宽消耗。我们的初步研究表明，激进的属性压缩主要会导致颜色失真，可以使用参考图像在渲染图像中有效地校正颜色失真。受这些发现的启发，我们提出了一种用于自适应 VV 流的新颖颜色自适应方案，该方案使用矢量量化 (VQ) 来建立 LoD 并使用低分辨率参考图像校正颜色失真。我们进一步提出了 CAGS，一种与多种高斯表示兼容的自适应 VV 流媒体系统，它通过在流媒体服务器上渲染参考图像并在客户端上执行颜色恢复来集成颜色自适应方案。在我们的原型系统上进行的大量实验表明，CAGS 在波动带宽下的 PSNR 性能优于现有的自适应流系统 5$\sim$20 dB，运行速度明显快于现有的可扩展高斯压缩方法，并且可以推广到不同的高斯表示。

</details>

---

## 15. Relightable Gaussian Splatting for Virtual Production Using Image-Based Illumination / 使用基于图像的照明进行虚拟生产的可重新照明高斯溅射

**Date**: 2026-05-09 | **arXiv**: [2605.09024v1](http://arxiv.org/abs/2605.09024v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.09024v1)

**Categories**: cs.CV, cs.GR, cs.MM, eess.IV

<details><summary><b>Abstract / 摘要</b></summary>

Virtual production (VP) use LED walls to provide both background imagery and image-based lighting. While this enables on-set compositing, it couples lighting to background and scene appearance, limiting flexibility for downstream editing. In addition, inverse rendering conventionally relies on physically-based rendering to estimates 3D geometry and lighting, using environment maps. However, these maps are typically low-resolution and assume far-field lighting. In VP, with near-field and high-resolution image-based lighting, this can lead to inaccuracies and introduce complexities when editing. Addressing this, we propose a VP-specific framework for 3D reconstruction and relighting using Gaussian Splatting. This uses the known background imagery to condition the relighting process. This avoids relying on environment maps and reduces compositing to a background-image editing task. To realize our framework, we introduce a process (and associated dataset) that captures real VP scenes under varying background content and illumination conditions. This data is used to decompose a 3D scene into fixed appearance and variable lighting components. The variable lighting process simulates light transport by parameterizing each primitive with a UV coordinate, intensity value and resolution modifier. Using mipmaps, these directly sample the background texture in image space - implicitly capturing reflections and refractions without physically-based rendering. Combined with the fixed appearance component, this allows us to render relit scenes using a Gaussian Splatting rasterizer. Compared to baselines, our approach achieves higher-quality 3D reconstruction and controllable relighting. The method is efficient (<3 GB RAM, <5 GB VRAM, <2 hours training, ~35 FPS) and supports rendering useful arbitrary output variables including depth, lighting intensity, lighting color, and unlit renders.

虚拟制作 (VP) 使用 LED 墙提供背景图像和基于图像的照明。虽然这可以实现现场合成，但它将照明与背景和场景外观耦合在一起，限制了下游编辑的灵活性。此外，逆向渲染通常依赖于基于物理的渲染，使用环境贴图来估计 3D 几何和照明。然而，这些地图通常是低分辨率的并且假设远场照明。在 VP 中，使用近场和基于高分辨率图像的照明，这可能会导致编辑时不准确并带来复杂性。为了解决这个问题，我们提出了一个特定于 VP 的框架，用于使用高斯分布进行 3D 重建和重新照明。这使用已知的背景图像来调节重新照明过程。这避免了对环境贴图的依赖，并减少了背景图像编辑任务的合成。为了实现我们的框架，我们引入了一个过程（以及相关的数据集），该过程可以在不同的背景内容和照明条件下捕获真实的 VP 场景。该数据用于将 3D 场景分解为固定外观和可变照明组件。可变光照过程通过使用 UV 坐标、强度值和分辨率修改器对每个基元进行参数化来模拟光传输。使用 mipmap，这些直接对图像空间中的背景纹理进行采样 - 隐式捕获反射和折射，而无需基于物理的渲染。与固定外观组件相结合，这使我们能够使用高斯泼溅光栅器渲染重新照明的场景。与基线相比，我们的方法实现了更高质量的 3D 重建和可控重新照明。该方法非常高效（<3 GB RAM、<5 GB VRAM、<2 小时训练、~35 FPS），并且支持渲染有用的任意输出变量，包括深度、光照强度、光照颜色和无光照渲染。

</details>

---

## 16. Stage Light is Sequence$^2$: Multi-Light Control via Imitation Learning / 舞台灯光是序列$^2$：通过模仿学习进行多灯控制

**Date**: 2026-05-05 | **arXiv**: [2605.03660v1](http://arxiv.org/abs/2605.03660v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.03660v1)

**Categories**: cs.MM, cs.AI

**Code**: https://github.com/RS2002/SeqLight

<details><summary><b>Abstract / 摘要</b></summary>

Music-inspired Automatic Stage Lighting Control (ASLC) has gained increasing attention in recent years due to the substantial time and financial costs associated with hiring and training professional lighting engineers. However, existing methods suffer from several notable limitations: the low interpretability of rule-based approaches, the restriction to single-primary-light control in music-to-color-space methods, and the limited transferability of music-to-controlling-parameter frameworks. To address these gaps, we propose SeqLight, a hierarchical deep learning framework that maps music to multi-light Hue-Saturation-Value (HSV) space. Our approach first customizes SkipBART, an end-to-end single primary light generation model, to predict the full light color distribution for each frame, followed by hybrid Imitation Learning (IL) techniques to derive an effective decomposition strategy that distributes the global color distribution among individual lights. Notably, the light decomposition module can be trained under varying venue-specific lighting configurations using only mixed light data and no professional demonstrations, thereby flexibly adapting across diverse venues. In this stage, we formulate the light decomposition task as a Goal-Conditioned Markov Decision Process (GCMDP), construct an expert demonstration set inspired by Hindsight Experience Replay (HER), and introduce a three-phase IL training pipeline, achieving strong generalization capability. To validate our IL solution for the proposed GCMDP, we conduct a series of quantitative analysis and human study. The code and trained models are provided at https://github.com/RS2002/SeqLight .

近年来，受音乐启发的自动舞台灯光控制 (ASLC) 受到越来越多的关注，因为雇用和培训专业灯光工程师需要大量的时间和财务成本。然而，现有方法存在几个显着的局限性：基于规则的方法的可解释性低、音乐到色彩空间方法中对单基色光控制的限制以及音乐到控制参数框架的可转移性有限。为了解决这些差距，我们提出了 SeqLight，这是一种分层深度学习框架，可将音乐映射到多光色调-饱和度-值 (HSV) 空间。我们的方法首先定制 SkipBART（一种端到端单基色光生成模型）来预测每个帧的完整光颜色分布，然后采用混合模仿学习 (IL) 技术来导出有效的分解策略，在各个光之间分配全局颜色分布。值得注意的是，光分解模块可以仅使用混合光数据而无需专业演示，在不同的特定场地照明配置下进行训练，从而灵活地适应不同的场地。在这个阶段，我们将轻分解任务制定为目标条件马尔可夫决策过程（GCMDP），构建受后见之明经验回放（HER）启发的专家演示集，并引入三阶段IL训练管道，实现了强大的泛化能力。为了验证我们针对拟议的 GCMDP 的 IL 解决方案，我们进行了一系列定量分析和人体研究。代码和经过训练的模型位于 https://github.com/RS2002/SeqLight 。

</details>

---

## 17. EMOVIS: Emotion-Optimized Image Processing / EMOVIS：情感优化的图像处理

**Date**: 2026-05-04 | **arXiv**: [2605.03131v1](http://arxiv.org/abs/2605.03131v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.03131v1)

**Categories**: eess.IV, cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

In cinematography, visual attributes such as color grading, contrast, and brightness are manipulated to reinforce the emotional narrative of a scene. However, conventional Image Signal Processors (ISPs) prioritize scene fidelity, effectively neglecting this expressive dimension. To bring this cinematic capability to real-time camera pipelines during video capture, we introduce EMOVIS (EMotion-Optimized VISual processing). We establish a systematic mapping between a compact set of high-level emotional states (Happy, Calm, Angry, Sad) and low-level ISP controls - including color saturation, local tone mapping, and sharpness - supported by a calibration user study with statistically significant effects across parameters. We propose a control framework that integrates these emotion-driven adjustments into standard ISP hardware without altering the underlying processing stages. Validation via blind A/B testing shows that viewers prefer the emotion-optimized rendering in 87% of trials when the target emotion matches the scene context, indicating that emotion-aligned ISP control improves perceived suitability for expressive visual content.

在电影摄影中，通过操纵色彩分级、对比度和亮度等视觉属性来强化场景的情感叙事。然而，传统的图像信号处理器 (ISP) 优先考虑场景保真度，实际上忽略了这一表达维度。为了在视频捕获期间将这种电影功能引入实时摄像机管道，我们引入了 EMOVIS（EMotion 优化的视觉处理）。我们在一组紧凑的高级情绪状态（快乐、平静、愤怒、悲伤）和低级 ISP 控制（包括色彩饱和度、局部色调映射和锐度）之间建立了系统映射，并由校准用户研究支持，对参数具有统计上的显着影响。我们提出了一个控制框架，将这些情感驱动的调整集成到标准 ISP 硬件中，而不改变底层处理阶段。通过盲 A/B 测试进行的验证表明，当目标情感与场景上下文相匹配时，观看者在 87% 的试验中更喜欢情感优化渲染，这表明情感一致的 ISP 控制提高了对富有表现力的视觉内容的感知适合性。

</details>

---

## 18. Development and Validation of an Integrated LiDAR-Camera System for Real-Time Monitoring of Underground Longwall Operations / 用于实时监控地下长壁作业的集成激光雷达相机系统的开发和验证

**Date**: 2026-05-04 | **arXiv**: [2605.02516v1](http://arxiv.org/abs/2605.02516v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.02516v1)

**Categories**: eess.IV

<details><summary><b>Abstract / 摘要</b></summary>

Real-time spatial monitoring in underground longwall operations is challenging due to methane-related safety risks, poor visibility, elevated thermal loads, spatial confinement, and bandwidth-limited communications. Currently available camera-based monitoring provides visual context but lacks direct depth information, while standalone underground LiDAR scanners are limited to monochromatic or periodic 3D mapping. This paper presents the design, integration, and experimental validation of a LiDAR-camera monitoring system built around a certified flameproof enclosure that prevents flame propagation into the surrounding atmosphere. The system combines a solid-state LiDAR, an industrial RGB camera, and an onboard processor within a compact hardware assembly, supporting LiDAR-camera fusion, low-light image enhancement, and real-time processing. Laboratory experiments evaluated LiDAR and camera performance through the protective polycarbonate dome and quantified optical and geometric distortions introduced by the enclosure. Thermal testing showed that iterative component placement, heat sinking, and passive conduction reduced peak surface temperature from 106 °C to 70 °C, with internal temperature stabilising at 57 °C. Furthermore, a representative longwall simulation was created to evaluate the complete sensing, fusion, and transmission workflow under controlled geometric and low-light conditions. In the final configuration, more than 97% of LiDAR points fell within the camera field of view, supporting reliable colourisation. Enclosure-aware calibration and correction maintained geometric accuracy, while processed colourised point clouds were transmitted at up to 10 Hz with sustained bandwidth below 25 Mb/s.

由于甲烷相关的安全风险、能见度差、热负荷升高、空间限制和带宽有限的通信，地下长壁作业中的实时空间监测具有挑战性。目前可用的基于摄像头的监控提供视觉背景，但缺乏直接的深度信息，而独立的地下 LiDAR 扫描仪仅限于单色或周期性 3D 测绘。本文介绍了围绕经过认证的防火外壳构建的激光雷达摄像头监控系统的设计、集成和实验验证，该外壳可防止火焰传播到周围大气中。该系统在紧凑的硬件组件中结合了固态 LiDAR、工业 RGB 相机和板载处理器，支持 LiDAR-相机融合、低光图像增强和实时处理。实验室实验通过保护性聚碳酸酯圆顶以及量化外壳引入的光学和几何变形来评估激光雷达和相机的性能。热测试表明，迭代组件放置、散热和无源传导将峰值表面温度从 106 °C 降低至 70 °C，内部温度稳定在 57 °C。此外，还创建了具有代表性的长壁模拟，以评估受控几何和低光条件下的完整传感、融合和传输工作流程。在最终配置中，超过 97% 的 LiDAR 点落在相机视野内，支持可靠的彩色化。外壳感知校准和校正保持了几何精度，同时处理后的彩色点云以高达 10 Hz 的频率传输，持续带宽低于 25 Mb/s。

</details>

---

## 19. FASH-iCNN: Making Editorial Fashion Identity Inspectable Through Multimodal CNN Probing / FASH-iCNN：通过多模态 CNN 探测使编辑时尚身份可检查

**Date**: 2026-04-29 | **arXiv**: [2604.26186v1](http://arxiv.org/abs/2604.26186v1) | **PDF**: [Link](http://arxiv.org/pdf/2604.26186v1)

**Categories**: cs.CV, cs.HC, cs.IR, cs.MM

<details><summary><b>Abstract / 摘要</b></summary>

Fashion AI systems routinely encode the aesthetic logic of specific houses, editors, and historical moments without disclosing it. We present FASH-iCNN, a multimodal system trained on 87,547 Vogue runway images across 15 fashion houses spanning 1991-2024 that makes this cultural logic inspectable. Given a photograph of a garment, the system recovers which house produced it, which era it belongs to, and which color tradition it reflects. A clothing-only model identifies the fashion house at 78.2% top-1 across 14 houses, the decade at 88.6% top-1, and the specific year at 58.3% top-1 across 34 years with a mean error of just 2.2 years. Probing which visual channels carry this signal reveals a sharp dissociation: removing color costs only 10.6pp of house identity accuracy, while removing texture costs 37.6pp, establishing texture and luminance as the primary carriers of editorial identity. FASH-iCNN treats editorial culture as the signal rather than background noise, identifying which houses, eras, and color traditions shaped each output so that users can see not just what the system predicts but which houses, editors, and historical moments are encoded in that prediction.

时尚人工智能系统通常会对特定品牌、编辑和历史时刻的美学逻辑进行编码，但不会公开。我们展示了 FASH-iCNN，这是一个多模态系统，接受了 1991 年至 2024 年 15 家时装公司的 87,547 张《Vogue》T 台图像的训练，使这种文化逻辑变得可检验。给定一张服装的照片，系统可以恢复该服装是哪个品牌生产的、属于哪个时代以及它反映了哪种颜色传统。仅服装模型将时装公司识别为 14 个品牌中 78.2% 的 top-1，十年内为 88.6% top-1，以及 34 年中特定年份的 58.3% top-1，平均误差仅为 2.2 年。探究哪些视觉通道携带此信号揭示了一种尖锐的分离：去除颜色仅花费 10.6pp 的房屋识别准确度，而去除纹理则花费 37.6pp，将纹理和亮度确立为编辑身份的主要载体。 FASH-iCNN 将编辑文化视为信号而不是背景噪音，识别哪些家族、时代和肤色传统塑造了每个输出，以便用户不仅可以看到系统预测的内容，还可以看到哪些家族、编辑和历史时刻被编码在该预测中。

</details>

---



</details>

<!-- PAPERS_CONTENT_END -->

---

Built for fast scanning and focused reading. Updated daily via GitHub Actions.
