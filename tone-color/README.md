# Tone & Color Research Papers 🎨

📚 **Website**: https://greasebig.github.io/daily-video-papers/

![Actions](https://github.com/greasebig/daily-video-papers/actions/workflows/daily-update.yml/badge.svg) ![Pages](https://img.shields.io/badge/pages-online-brightgreen) ![Visitors](https://visitor-badge.laobi.icu/badge?page_id=greasebig.daily-video-papers)

## 📚 论文索引

<!-- PAPERS_INDEX_START -->
- [2026-05-26](papers/2026-05-26.md) - 28 papers
<!-- PAPERS_INDEX_END -->

## Other Topics

- [Video Papers](../README.md)
- [World Model Papers](../world-model/README.md)
- [Agent Papers](../agent/README.md)

## Daily Papers

<!-- PAPERS_CONTENT_START -->
<details><summary><b>2026-05-26 (28 papers)</b></summary>

# arXiv Tone & Color Papers - 2026-05-26

**Paper Count**: 28

---

## 1. PathWISE: Multi-Agent Cancer Pathway Triaging Ontology Learning from Clinical Flowcharts / PathWISE：多代理癌症通路分类本体论从临床流程图中学习

**Date**: 2026-05-25 | **arXiv**: [2605.25970v1](http://arxiv.org/abs/2605.25970v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.25970v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Clinical pathways are disseminated as visual flowcharts where spatial topology, arrow direction, colour coding, and font weight encode critical triage logic that remains inaccessible to computational systems. We present PathWISE, a five-phase pipeline combining four LLM-based agents with a deterministic depth-first search auditor and a Java compiler critic, transforming these non-computable artefacts into validated, executable HL7 Clinical Quality Language (CQL) libraries deployable as FHIR CDS Hooks services. Purpose-built agents extract flowchart structure into a typed directed graph, perform deterministic path enumeration, conduct a structured semantic audit of every node's computability, generate terminology-constrained CQL definitions verified by the official Java CQL-to-ELM compiler, and produce routing logic covering 100% of enumerated patient journeys. Demonstrated across five UK NHS cancer pathways (colorectal, lung, skin, upper GI, and breast), PathWISE audits up to 183 nodes (182 under the Hybrid configuration), identifies 544 structured governance findings across four issue categories, achieves 100% syntactic compilation success, with UNCOMPUTABLE nodes receiving false placeholders that preserve compilability while surfacing governance gaps for clinical review, and produces zero hallucinated terminology codes for dictionary-covered concepts. Critically, PathWISE confines non-deterministic LLM inference to knowledge extraction while deterministic graph mathematics and a standard compiler underpin every verification step.

临床路径以视觉流程图的形式传播，其中空间拓扑、箭头方向、颜色编码和字体粗细对计算系统仍然无法访问的关键分类逻辑进行编码。我们提出了 PathWISE，这是一个五阶段管道，将四个基于 LLM 的代理与确定性深度优先搜索审核员和 Java 编译器评论家相结合，将这些不可计算的工件转换为经过验证的可执行 HL7 临床质量语言 (CQL) 库，可作为 FHIR CDS Hooks 服务部署。专门构建的代理将流程图结构提取到类型化有向图中，执行确定性路径枚举，对每个节点的可计算性进行结构化语义审计，生成由官方 Java CQL-to-ELM 编译器验证的术语约束的 CQL 定义，并生成覆盖 100% 枚举患者旅程的路由逻辑。 PathWISE 在英国 NHS 的五个癌症路径（结直肠癌、肺癌、皮肤癌、上消化道和乳腺癌）中进行了演示，审核了多达 183 个节点（混合配置下为 182 个），识别了四个问题类别的 544 个结构化治理发现，实现了 100% 句法编译成功，其中 UNCOMPUTABLE 节点接收虚假占位符，保留了可编译性，同时为临床审查提供了治理差距，并为以下问题生成了零幻觉术语代码：字典涵盖的概念。至关重要的是，PathWISE 将非确定性 LLM 推理限制在知识提取中，而确定性图数学和标准编译器支撑着每个验证步骤。

</details>

---

## 2. RAPTOR+: A Visually Grounded Vision-Language Framework to Improve Clinical Trust and Auditability in Automated Cancer Referral Processing / RAPTOR+：基于视觉的视觉语言框架，可提高自动化癌症转诊处理中的临床信任和可审核性

**Date**: 2026-05-25 | **arXiv**: [2605.25956v1](http://arxiv.org/abs/2605.25956v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.25956v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Urgent suspected colorectal cancer (CRC) referrals create operational bottlenecks because semi-structured clinical documents often require manual review and transcription. The original RAPTOR system used Large Language Models for structured extraction but relied on a separate OCR stage, making it vulnerable to handwriting, layout variation, and loss of visual evidence linkage. We present RAPTOR+, a multimodal extension that uses Vision-Language Models (VLMs) for end-to-end referral understanding. We evaluate fine-tuned VLMs, commercial and open-source zero-shot VLMs, and the original OCR-based pipeline on 223 clinically curated CRC urgent referral forms. We also introduce a grounding-aware evaluation framework that measures both extraction accuracy and evidence localisation. Results show a clear grounding gap in zero-shot models. Gemini 2.5 Flash achieved 92.6% Reading Accuracy but only 1.2% Strict Safety. In contrast, fine-tuned Qwen3-VL-8B achieved 96.1% Reading Accuracy and 60.6% Strict Safety, substantially improving verifiable evidence grounding. These findings show that task-specific fine-tuning is essential for reliable, auditable clinical document understanding. RAPTOR+ enables extracted referral decisions to be linked to visual evidence, supporting safer and more efficient cancer referral triage.

紧急疑似结直肠癌 (CRC) 转诊会造成运营瓶颈，因为半结构化临床文件通常需要人工审核和转录。最初的 RAPTOR 系统使用大型语言模型进行结构化提取，但依赖于单独的 OCR 阶段，使其容易受到手写、布局变化和视觉证据链接丢失的影响。我们推出了 RAPTOR+，这是一种多模式扩展，它使用视觉语言模型 (VLM) 进行端到端的转诊理解。我们在 223 个临床策划的 CRC 紧急转诊表格上评估了微调的 VLM、商业和开源零样本 VLM 以及基于 OCR 的原始管道。我们还引入了一个接地感知评估框架，可以衡量提取准确性和证据本地化。结果显示零样本模型中存在明显的接地差距。 Gemini 2.5 Flash 的读取准确度达到 92.6%，但严格安全性仅为 1.2%。相比之下，经过微调的 Qwen3-VL-8B 实现了 96.1% 的读取准确度和 60.6% 的严格安全性，大大提高了可验证的证据基础。这些发现表明，针对特定任务的微调对于可靠、可审核的临床文档理解至关重要。 RAPTOR+ 能够将提取的转诊决策与视觉证据联系起来，支持更安全、更高效的癌症转诊分类。

</details>

---

## 3. AgentGrounder: Zero-Shot 3D Visual Pointcloud Grounding using Multimodal Language Models / AgentGrounder：使用多模态语言模型的零射击 3D 视觉点云接地

**Date**: 2026-05-25 | **arXiv**: [2605.25901v1](http://arxiv.org/abs/2605.25901v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.25901v1)

**Categories**: cs.CV, cs.RO

**Code**: https://github.com/be2rlab/AgentGrounder.

<details><summary><b>Abstract / 摘要</b></summary>

3D Visual Grounding (3DVG) is an essential capability for embodied AI, requiring agents to localize objects in 3D scenes based on natural language descriptions. Recent zero-shot methods leverage 2D vision-language models (LVLMs). However, they often rely on existing sets of multi-view images and struggle with the limited semantic and spatial details provided by standard 3D segmentation tools. We present $\textbf{AgentGrounder}$, a zero-shot 3D visual grounding framework that operates directly on colored point clouds without task-specific 3D training. Our approach follows a two-stage design: (1) an offline stage that applies 3D model to build an Object Lookup Table (OLT) with instance IDs, semantic labels, 3D bounding boxes; and (2) an online tool-driven agent that decomposes each query, retrieves only relevant candidates from the OLT, performs geometric scoring, and triggers image rendering on demand when additional visual evidence (e.g., color, material, or viewpoint-sensitive cues) is required. Compared with fixed anchor-target matching pipelines, this design reduces cascading matching errors and improves context-window efficiency by avoiding prompts overloaded with irrelevant objects. We evaluate on ScanRefer and Nr3D under a zero-shot setting and observe consistent improvements over SeeGround in our setup, including +2.5% Acc@0.5 on ScanRefer and +6.3% on Nr3D, with a notable +6.3% gain on Nr3D view-independent queries. These results show that combining selective retrieval, geometric reasoning, and adaptive visual inspection yields a practical and robust foundation for open-vocabulary 3D grounding. Our code is available at https://github.com/be2rlab/AgentGrounder.

3D 视觉基础 (3DVG) 是嵌入式 AI 的一项基本功能，要求智能体根据自然语言描述定位 3D 场景中的对象。最近的零样本方法利用了 2D 视觉语言模型 (LVLM)。然而，它们通常依赖于现有的多视图图像集，并且难以应对标准 3D 分割工具提供的有限语义和空间细节。我们提出了 $\textbf{AgentGrounder}$，一个零样本 3D 视觉基础框架，可以直接在彩色点云上运行，无需特定于任务的 3D 训练。我们的方法遵循两阶段设计：(1) 离线阶段，应用 3D 模型构建包含实例 ID、语义标签、3D 边界框的对象查找表 (OLT)； (2) 在线工具驱动代理，分解每个查询，仅从 OLT 检索相关候选者，执行几何评分，并在需要额外视觉证据（例如颜色、材料或视点敏感线索）时按需触发图像渲染。与固定的锚点-目标匹配管道相比，这种设计通过避免提示因不相关的对象而过载，减少了级联匹配错误并提高了上下文窗口效率。我们在零样本设置下对 ScanRefer 和 Nr3D 进行评估，并在我们的设置中观察到相对于 SeeGround 的一致改进，包括 ScanRefer 上 +2.5% Acc@0.5 和 Nr3D 上 +6.3%，其中 Nr3D 独立于视图的查询显着增加 6.3%。这些结果表明，选择性检索、几何推理和自适应视觉检查的结合为开放词汇 3D 基础奠定了实用且坚实的基础。我们的代码可在 https://github.com/be2rlab/AgentGrounder 获取。

</details>

---

## 4. Insuring Every Action: An Authority Frontier Framework for Runtime Actuarial Control of Autonomous AI Agents / 确保每一个行动：自主人工智能代理运行时精算控制的权威前沿框架

**Date**: 2026-05-25 | **arXiv**: [2605.25632v1](http://arxiv.org/abs/2605.25632v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.25632v1)

**Categories**: cs.AI, cs.LG, q-fin.RM

<details><summary><b>Abstract / 摘要</b></summary>

Autonomous AI agents increasingly issue side-effect-bearing actions: database mutations, refunds, payments, external commitments. We propose the Actuarial Action Interface (AAI), a deterministic runtime contract that prices each such action against a contractually fixed safe default under a time-consistent risk mapping, and gates execution against a per-boundary reserve capital budget. We then develop the Authority Frontier, an evaluation primitive measuring how much autonomous authority the runtime releases at each level of reserve capital. The framework provides (i) a deterministic quote-bind-commit protocol with toll-bounded capability tokens; (ii) a universal seven-class action taxonomy mapping heterogeneous tool calls to comparable authority units; (iii) replay determinism and pathwise reserve coverage under alpha-spending; (iv) cross-domain normalization via full reserve demand C_full and capital metrics Capital@k. We instantiate AAI across four agentic environments (database mutation, customer-service refund, and the public tau-bench retail and airline tool-use traces) and report a live Postgres panel in which three Azure-hosted models propose actions through the same contract. The frontier exhibits a common low-reserve refusal and intermediate-release pattern across domains, with saturation only where the budget grid reaches full reserve demand; required reserve capital varies by 22x (Capital@50 from 289 to 6457). The framework does not force domains into the same shape; it surfaces each domain's actuarial geometry. In the live panel the contract prevents realized loss across all three models at low budget while differing in underwriting persistence under denial: model identity is an actuarial underwriting variable. The contribution is a benchmark-ready evaluation framework for runtime actuarial control of autonomous-agent side effects.

自主人工智能代理越来越多地发出带有副作用的操作：数据库突变、退款、付款、外部承诺。我们提出了精算操作接口（AAI），这是一种确定性的运行时合约，它在时间一致的风险映射下根据合约固定的安全默认值对每个此类操作进行定价，并根据每个边界的储备资本预算来控制执行。然后，我们开发了权威前沿，这是一种评估原语，用于测量运行时在每个储备资本级别释放多少自治权威。该框架提供了（i）具有收费限制功能令牌的确定性引用-绑定-提交协议； (ii) 通用七类行动分类法，将异构工具调用映射到可比较的权威单位； (iii) 阿尔法支出下的重播决定论和路径储备覆盖率； (iv) 通过全额准备金需求 C_full 和资本指标 Capital@k 进行跨域标准化。我们跨四个代理环境（数据库突变、客户服务退款以及公共 tau 工作台零售和航空公司工具使用跟踪）实例化 AAI，并报告实时 Postgres 面板，其中三个 Azure 托管模型通过同一合约提出操作建议。前沿呈现出跨领域常见的低储备拒绝和中间释放模式，只有在预算网格达到全部储备需求时才会饱和；所需储备资本变化 22 倍（Capital@50 从 289 到 6457）。该框架不会强制域具有相同的形状；它展示了每个域的精算几何。在实时面板中，合约可以防止所有三种模型以低预算实现的损失，同时在拒绝承保的情况下持续性有所不同：模型身份是一个精算承保变量。该贡献是一个用于自主代理副作用的运行时精算控制的基准就绪评估框架。

</details>

---

## 5. Metric--Phase Fields: Decoupling Distance and Sign for Thin-Structure Reconstruction from Unoriented Point Clouds / 度量相场：从无向点云进行薄结构重建的解耦距离和符号

**Date**: 2026-05-25 | **arXiv**: [2605.25503v1](http://arxiv.org/abs/2605.25503v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.25503v1)

**Categories**: cs.CV

**Code**: https://github.com/JIAYI-Scarlett/ICML2026-MPF

<details><summary><b>Abstract / 摘要</b></summary>

Neural Signed Distance Functions (SDFs) excel at reconstructing watertight manifolds but fail on thin structures and open boundaries due to strict inside--outside constraints. Conversely, Unsigned Distance Fields (UDFs) accommodate general geometries but suffer from gradient singularities at the zero-level set, hindering optimization and extraction. We introduce Metric--Phase Fields (MPFs), a decoupled implicit representation that separates metric proximity from topological phase. Given an unoriented point cloud, MPFs learn (i) an unsigned metric field $r$ and (ii) a smooth phase field $θ$, for which we derive a bounded phase indicator $P=\tanh(βθ)$ that provides soft inside--outside cues where they are meaningful. We couple the two fields via a gated-metric formulation with a residual phase injection to obtain a signed implicit function with stable near-surface gradients. The phase coefficient $β$ is learnable, allowing MPFs to adaptively control the sharpness of the phase transition and the degree of saturation of the soft sign indicator. Experiments on both synthetic and scanned thin-shell and thin-plate shapes demonstrate that MPFs preserve thin and layered structures more faithfully than recent SDF-based methods, while also enabling more robust training and more reliable surface extraction than UDF-based approaches. Check out \href{https://github.com/JIAYI-Scarlett/ICML2026-MPF}{MPFs-GitHub} for source code and test models.

神经符号距离函数 (SDF) 擅长重建无懈可击的流形，但由于严格的内外约束，在薄结构和开放边界上失败。相反，无符号距离场（UDF）适应一般几何形状，但在零水平集处存在梯度奇点，阻碍了优化和提取。我们引入度量相域（MPF），一种解耦的隐式表示，它将度量邻近度与拓扑相位分开。给定一个无方向的点云，MPF 学习（i）一个无符号度量场 $r$ 和（ii）一个平滑相位场 $θ$，为此我们推导出一个有界相位指示器 $P=\tanh(βθ)$，它提供了有意义的软内部-外部线索。我们通过门控度量公式和残余相注入将两个场耦合起来，以获得具有稳定近表面梯度的带符号隐函数。相位系数$β$是可学习的，允许MPF自适应地控制相变的锐度和软符号指示器的饱和度。对合成和扫描薄壳和薄板形状的实验表明，MPF 比最近基于 SDF 的方法更忠实地保留薄层结构，同时还比基于 UDF 的方法实现更稳健的训练和更可靠的表面提取。查看 \href{https://github.com/JIAYI-Scarlett/ICML2026-MPF}{MPFs-GitHub} 以获取源代码和测试模型。

</details>

---

## 6. Physics-Aware 3D Gaussian Editing for Driving Scene Generation / 用于生成驾驶场景的物理感知 3D 高斯编辑

**Date**: 2026-05-25 | **arXiv**: [2605.25373v1](http://arxiv.org/abs/2605.25373v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.25373v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

3D Gaussian Splatting (3DGS) has shown great potential in autonomous driving simulation and data generation, enabling photorealistic reconstruction and flexible scene manipulation. However, existing 3DGS scene editing methods have limited support for road geometry editing (e.g., inserting speed humps or sunken roads), and generally do not couple such edits with plausible vehicle-road interaction dynamics. Such editing is essential for generating training data under extreme driving scenarios or evaluating system reliability under these road irregularities. Moreover, many optimization-based methods require minutes of per-edit refinement, while existing efficient alternatives mainly focus on appearance-level or object-level manipulation rather than physics-aware road irregularity editing. To address these limitations, we propose RoVES, a Road-and-Vehicle Editing System for physics-aware 3D Gaussian editing in driving scenes. RoVES enables single-image-driven road geometry insertion and couples the edited road profile with a 4-DOF half-car vehicle dynamics model to achieve physics-aware vehicle pose correction in vertical displacement and pitch. RoVES inserts road elements in a one-shot, optimization-free pipeline (1.84s), and the full pipeline (including color transfer and vehicle-dynamics-based pose correction) completes in 6.24s; it edits dynamic vehicles via pose editing and corrects poses frame-by-frame to approximate dynamics-consistent vertical displacement and pitch responses. Experiments on the Waymo dataset show that RoVES provides practical efficiency and competitive visual consistency for physics-aware driving scene generation.

3D 高斯溅射 (3DGS) 在自动驾驶仿真和数据生成方面显示出巨大潜力，可实现逼真的重建和灵活的场景操作。然而，现有的 3DGS 场景编辑方法对道路几何编辑（例如，插入减速带或下沉道路）的支持有限，并且通常不会将此类编辑与合理的车路交互动力学结合起来。这种编辑对于在极端驾驶场景下生成训练数据或评估这些道路不规则情况下的系统可靠性至关重要。此外，许多基于优化的方法需要几分钟的每次编辑细化，而现有的有效替代方法主要关注外观级或对象级操作，而不是物理感知的道路不规则编辑。为了解决这些限制，我们提出了 RoVES，一种道路和车辆编辑系统，用于在驾驶场景中进行物理感知 3D 高斯编辑。 RoVES 支持单图像驱动的道路几何形状插入，并将编辑后的道路轮廓与 4-DOF 半车车辆动力学模型相结合，以实现垂直位移和俯仰方面的物理感知车辆姿态校正。 RoVES 将道路元素插入一次性、免优化的管道（1.84 秒）中，整个管道（包括颜色传输和基于车辆动力学的姿态校正）在 6.24 秒内完成；它通过姿势编辑来编辑动态车辆，并逐帧校正姿势以近似动态一致的垂直位移和俯仰响应。 Waymo 数据集上的实验表明，RoVES 为物理感知驾驶场景生成提供了实用效率和有竞争力的视觉一致性。

</details>

---

## 7. Depth Peeling for High-Fidelity Gaussian-Enhanced Surfel Rendering / 高保真高斯增强面元渲染的深度剥离

**Date**: 2026-05-25 | **arXiv**: [2605.25345v1](http://arxiv.org/abs/2605.25345v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.25345v1)

**Categories**: cs.GR, cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Novel view synthesis has been significantly advanced by NeRFs and 3D Gaussian Splatting (3DGS), which require ordering volumetric samples or primitives for correct color blending. While the recent Gaussian-Enhanced Surfels (GES) enable high-performance, sort-free rendering, they suffer from aliasing artifacts and suboptimal reconstruction. To address these limitations, we propose DP-GES, a novel representation that augments opaque surfels with semi-transparent boundaries and leverages Depth Peeling to establish accurate per-pixel ordering. This design enables sort-free Gaussian splatting with correct transmittance modulation, effectively eliminating aliasing and popping artifacts while facilitating a fully differentiable joint optimization. Extensive experiments demonstrate that our method achieves superior reconstruction quality and compares favorably against state-of-the-art techniques across a wide range of scenes.

NeRF 和 3D 高斯溅射 (3DGS) 显着推进了新颖的视图合成，它们需要订购体积样本或基元以进行正确的颜色混合。虽然最近的高斯增强面元 (GES) 能够实现高性能、无排序渲染，但它们存在锯齿伪影和次优重建的问题。为了解决这些限制，我们提出了 DP-GES，这是一种新颖的表示形式，可以通过半透明边界增强不透明面元，并利用深度剥离来建立准确的每像素排序。该设计可通过正确的透射率调制实现无排序高斯泼溅，有效消除混叠和爆裂伪影，同时促进完全可微分的联合优化。大量的实验表明，我们的方法实现了卓越的重建质量，并且在各种场景中与最先进的技术相媲美。

</details>

---

## 8. Recursive Class Connectivity Classification (R3C) Applied to Binary Image Segmentation for Improved Infant Fingerprint Enhancement / 递归类连接分类 (R3C) 应用于二值图像分割以改进婴儿指纹增强

**Date**: 2026-05-25 | **arXiv**: [2605.25307v1](http://arxiv.org/abs/2605.25307v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.25307v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Image enhancement plays a crucial role in infant fingerprint matching, as child-specific characteristics such as smaller finger dimensions and thinner ridge structures often degrade image quality during acquisition. To address these limitations, enrollment typically depends on specialized highresolution scanners, which most existing enhancement methods are not designed to support. Consequently, identification rates for children remain significantly lower than those achieved with adult fingerprints. This study introduces Recursive Class Connectivity Classification (R3C), a novel framework that iteratively refines binary segmentation outputs from existing enhancement methods by extending ridge structures. R3C does not require modifications to the underlying classifier and operates without training data, which is not currently available for infant fingerprints. Instead, the method improves segmentation by repeatedly feeding the classified image back into the classification process, while combining each intermediate segmentation with the original input image. Experiments conducted on three fingerprint datasets using four different enhancement classifiers show that R3C can increase the True Acceptance Rate (TAR) by up to 4% for children and over 40% for newborns, compared to using the enhancement methods alone. A qualitative analysis further demonstrates that R3C reconnects fragmented ridge patterns, improving the visual quality of segmentation. Because it functions independently of the enhancement method used, R3C provides a flexible and broadly applicable solution for improving binary segmentation.

图像增强在婴儿指纹匹配中起着至关重要的作用，因为较小的手指尺寸和较薄的脊结构等儿童特定特征通常会在采集过程中降低图像质量。为了解决这些限制，注册通常依赖于专门的高分辨率扫描仪，而大多数现有的增强方法并不支持这种扫描仪。因此，儿童的识别率仍然明显低于成人指纹的识别率。本研究引入了递归类连通性分类（R3C），这是一种新颖的框架，可通过扩展脊结构来迭代地细化现有增强方法的二进制分割输出。 R3C 不需要修改底层分类器，并且无需训练数据即可运行，而目前婴儿指纹还无法使用训练数据。相反，该方法通过将分类图像重复反馈回分类过程，同时将每个中间分割与原始输入图像相结合来改进分割。使用四种不同的增强分类器对三个指纹数据集进行的实验表明，与单独使用增强方法相比，R3C 可以将儿童的真实接受率 (TAR) 提高多达 4%，将新生儿的真实接受率提高 40% 以上。定性分析进一步表明，R3C 重新连接了碎片化的脊线图案，提高了分割的视觉质量。由于其功能独立于所使用的增强方法，R3C 为改进二进制分段提供了灵活且广泛适用的解决方案。

</details>

---

## 9. DeltaCam: Differential Intrinsic Camera Modeling for Video Generation / DeltaCam：用于视频生成的差分本征相机建模

**Date**: 2026-05-24 | **arXiv**: [2605.25266v1](http://arxiv.org/abs/2605.25266v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.25266v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Incorporating camera intrinsics into video generation models offers a principled way to control not only scene dynamics but also the imaging process that governs visual appearance. Prior work has primarily focused on extrinsic control, such as camera pose and motion, while treating intrinsic camera parameters as implicit or fixed. A key bottleneck is the lack of large-scale video datasets with accurate and diverse temporally varying camera metadata, which makes learning absolute camera parameterizations difficult. As a result, current models struggle to incorporate photographic camera behavior, including depth-of-field transitions, exposure variations, lens distortions, and color processing, in a controllable and temporally consistent manner. We introduce DeltaCam, a video diffusion framework that models camera behavior through $Δ$-parameterized neural camera adaptors, operating on relative changes in camera motion and intrinsics instead of absolute states. By learning this differential formulation from synthetic video data, we mitigate reliance on precise real-world camera labels and enable smooth, consistent control over imaging factors such as focal length, aperture, ISO, color temperature, and lens distortion. We extend this framework to real-world footage through two mechanisms: finetuning the controls on real image-metadata pairs for precise shot matching, and extracting disentangled embeddings for implicit video-to-video style transfer without requiring explicit camera parameters. By effectively separating scene content from intrinsic imaging behavior, DeltaCam enables camera-consistent video generation and editing operations that are difficult to achieve with existing models. Ultimately, our results establish a practical and scalable approach for bridging synthetic control and real-world photographic emulation.

将相机内在功能纳入视频生成模型提供了一种原则性的方法，不仅可以控制场景动态，还可以控制控制视觉外观的成像过程。之前的工作主要集中在外部控制，例如相机姿势和运动，同时将内部相机参数视为隐式或固定的。一个关键瓶颈是缺乏具有准确且多样化的随时间变化的相机元数据的大规模视频数据集，这使得学习绝对相机参数化变得困难。因此，当前的模型很难以可控且时间一致的方式整合摄影机行为，包括景深过渡、曝光变化、镜头畸变和颜色处理。我们引入了 DeltaCam，这是一种视频扩散框架，它通过 $Δ$ 参数化神经相机适配器对相机行为进行建模，对相机运动和内在函数的相对变化而不是绝对状态进行操作。通过从合成视频数据中学习这种微分公式，我们减轻了对精确的现实世界相机标签的依赖，并能够对焦距、光圈、ISO、色温和镜头畸变等成像因素进行平滑、一致的控制。我们通过两种机制将该框架扩展到现实世界的镜头：微调真实图像元数据对的控制以实现精确的镜头匹配，以及提取解缠结的嵌入以实现隐式视频到视频风格的传输，而不需要显式的相机参数。通过有效地将场景内容与固有成像行为分离，DeltaCam 能够实现相机一致的视频生成和编辑操作，而这是现有模型难以实现的。最终，我们的结果建立了一种实用且可扩展的方法，用于桥接合成控制和现实世界的摄影模拟。

</details>

---

## 10. Injecting Image Guidance into Text-Conditioned Diffusion Models at Inference / 在推理时将图像引导注入文本条件扩散模型

**Date**: 2026-05-24 | **arXiv**: [2605.25191v1](http://arxiv.org/abs/2605.25191v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.25191v1)

**Categories**: cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Text-to-image diffusion models like Stable Diffusion generate high-quality images from text, but lack a way to inject visual guidance (e.g. sketches, styles) at inference without retraining. Existing methods either require computationally expensive fine-tuning or rely on style transfer techniques that risk semantic misalignment with textual prompts. We introduce Visual Concept Fusion (VCF), the first method offering dual conditioning on both an image and text prompt at inference time without any concept-specific training. VCF enables visual concept injection into Stable Diffusion by aligning CLIP image features with the text embedding space. VCF consists of three components: (1) a lightweight aligner that maps image tokens to the text embedding manifold using InfoNCE and cross-attention reconstruction losses, (2) a fusion strategy that preserves both textual and visual semantics, and (3) an optional Prompt-Noise Optimization (PNO) module for test-time refinement. Our experiments demonstrate that VCF successfully transfers visual attributes including style, composition, and color palette from reference images while maintaining prompt adherence. Quantitative results show a trade-off between text alignment (CLIP score) and visual correspondence (LPIPS), with VCF outperforming baselines in reference fidelity.

像稳定扩散这样的文本到图像扩散模型可以从文本生成高质量图像，但缺乏一种在推理时注入视觉指导（例如草图、样式）而无需重新训练的方法。现有方法要么需要计算量大的微调，要么依赖风格转换技术，这些技术可能会导致语义与文本提示不一致。我们引入了视觉概念融合（VCF），这是第一种在推理时对图像和文本提示提供双重调节的方法，无需任何特定于概念的训练。 VCF 通过将 CLIP 图像特征与文本嵌入空间对齐，实现将视觉概念注入到稳定扩散中。 VCF 由三个组件组成：(1) 一个轻量级对齐器，使用 InfoNCE 和交叉注意重建损失将图像标记映射到文本嵌入流形；(2) 保留文本和视觉语义的融合策略；(3) 用于测试时细化的可选提示噪声优化 (PNO) 模块。我们的实验表明，VCF 成功地从参考图像中转移了视觉属性，包括风格、构图和调色板，同时保持了及时的依从性。定量结果显示文本对齐（CLIP 分数）和视觉对应（LPIPS）之间存在权衡，VCF 在参考保真度方面优于基线。

</details>

---

## 11. ROI Extraction in Thermographic Breast Images Using Genetic Algorithms / 使用遗传算法提取热成像乳腺图像中的 ROI

**Date**: 2026-05-21 | **arXiv**: [2605.22899v1](http://arxiv.org/abs/2605.22899v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.22899v1)

**Categories**: q-bio.TO, eess.IV

<details><summary><b>Abstract / 摘要</b></summary>

This work proposes the use of Genetic Algorithms (GA) to identify the area of the breast from the background in thermographic breast images. The proposed method uses color information, a fitness function based on cardioids, and GA. This is the first work in the literature to propose a Region of Interest (ROI) extraction based on GA and cariods. ROI extraction can improve the accuracy of cancer detection and assist with the standardization of acquisition protocols. The method is able to successfully separate the breast region in 52 out of 58 images, while being fully automatic, and not requiring manual selection of seed points.

这项工作建议使用遗传算法 (GA) 从热成像乳房图像的背景中识别乳房区域。所提出的方法使用颜色信息、基于心形的适应度函数和 GA。这是文献中第一个提出基于 GA 和 cariod 的感兴趣区域 (ROI) 提取的工作。 ROI 提取可以提高癌症检测的准确性，并有助于采集协议的标准化。该方法能够成功地从 58 幅图像中的 52 幅图像中分离出乳房区域，同时是全自动的，不需要手动选择种子点。

</details>

---

## 12. Time-varying rPPG signal separation via block-sparse signal model / 通过块稀疏信号模型进行时变 rPPG 信号分离

**Date**: 2026-05-21 | **arXiv**: [2605.22425v1](http://arxiv.org/abs/2605.22425v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.22425v1)

**Categories**: eess.IV, cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Remote photoplethysmography (rPPG) enables non-contact measurement of cardiac pulse signals by analyzing subtle color changes in facial videos. Nevertheless, extracting rPPG signals remains challenging because of their extremely weak signal strength and susceptibility to illumination noise. In this paper, we propose an rPPG signal extraction method that exploits the quasi-periodic characteristics of rPPG signals. Our approach models quasi-periodicity of the rPPG signal, which arises from the stable cardiac cycle, as a block-sparse structure in the time-frequency domain. To incorporate a block-sparse model and enable adaptive signal separation under illumination fluctuations, we construct a time-varying signal separation framework. Experiments using a public dataset demonstrate the effectiveness of our method.

远程光电体积描记法 (rPPG) 通过分析面部视频中细微的颜色变化，实现心脏脉冲信号的非接触式测量。然而，提取 rPPG 信号仍然具有挑战性，因为它们的信号强度极弱并且容易受到照明噪声的影响。在本文中，我们提出了一种利用 rPPG 信号的准周期特性的 rPPG 信号提取方法。我们的方法将 rPPG 信号的准周期性建模为时频域中的块稀疏结构，该信号源自稳定的心动周期。为了合并块稀疏模型并在光照波动下实现自适应信号分离，我们构建了一个时变信号分离框架。使用公共数据集的实验证明了我们方法的有效性。

</details>

---

## 13. Probability-Conserving Flow Guidance / 概率守恒流程指导

**Date**: 2026-05-19 | **arXiv**: [2605.20079v1](http://arxiv.org/abs/2605.20079v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.20079v1)

**Categories**: cs.CV, cs.AI, cs.LG, eess.IV

<details><summary><b>Abstract / 摘要</b></summary>

Diffusion and flow-based generative models dominate visual synthesis, with guidance aligning samples to user input and improving perceptual quality. However, Classifier-Free Guidance (CFG) and extrapolation-based methods are heuristic linear combinations of velocities/scores that ignore the generative manifold geometry, breaking probability conservation and driving samples off the learned manifold under strong guidance. We analyse guidance through the continuity equation and show its effect decomposes into a divergence term and a score-parallel term defined invariantly across parameterisations. We prove the divergence term blows up structurally as sampling approaches the data manifold, motivating a time-dependent schedule alongside score-parallel attenuation. The resulting plug-and-play rule, Adaptive Manifold Guidance (AdaMaG), bounds both terms at no additional inference cost. Finally, we show that most empirical heuristics for reducing saturation or improving generation quality correspond directly to the two terms in our decomposition. Across image generation benchmarks, AdaMaG improves realism, reduces hallucinations, and induces controlled desaturation in high-guidance regimes.

基于扩散和流的生成模型主导着视觉合成，指导将样本与用户输入对齐并提高感知质量。然而，无分类器指导（CFG）和基于外推的方法是速度/分数的启发式线性组合，忽略了生成流形几何，打破了概率守恒并在强指导下将样本从学习流形中驱动出来。我们通过连续性方程分析指导，并表明其效果分解为散度项和在参数化过程中不变定义的分数平行项。我们证明，当采样接近数据流形时，发散项会在结构上爆炸，从而激发时间相关的时间表以及分数平行衰减。由此产生的即插即用规则，自适应流形指导（AdaMaG），在没有额外推理成本的情况下限制了这两个项。最后，我们表明，大多数用于降低饱和度或提高发电质量的经验启发法直接对应于我们分解中的这两项。在图像生成基准中，AdaMaG 提高了真实感，减少了幻觉，并在高指导制度下诱导受控去饱和。

</details>

---

## 14. GLUT: 3D Gaussian Lookup Table for Continuous Color Transformation / GLUT：用于连续颜色变换的 3D 高斯查找表

**Date**: 2026-05-19 | **arXiv**: [2605.19889v1](http://arxiv.org/abs/2605.19889v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.19889v1)

**Categories**: cs.GR, cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

3D Lookup Tables (3D LUTs) are widely used for color mapping, but their grid-based representation requires discretizing the RGB space, leading to a capacity-memory trade-off that becomes prohibitive when storing large numbers of LUTs. Recent approaches adopt implicit neural representations to improve scalability, yet their black-box nature limits interpretability and hinders intuitive, localized editing. In this paper, we propose Gaussian LUT (GLUT), a continuous and explicit color representation that models color transformations using a set of learnable 3D Gaussian primitives. By avoiding fixed-resolution grids, GLUT achieves flexible representational capacity while maintaining a compact memory footprint. Its explicit, spatially localized formulation further enables both accurate modeling and interpretability. Building on this representation, we introduce a compact conditional generator (CGLUT) that predicts GLUT parameters for multiple LUT instances, encoding diverse color styles in a single framework to enable smooth and controllable LUT style blending. Moreover, GLUT supports efficient, user-friendly editing by allowing localized adjustments to specific color regions without global retraining. Experimental results demonstrate that our approach outperforms prior neural LUT representations in both accuracy and efficiency, while offering improved interpretability and interactive control.

3D 查找表 (3D LUT) 广泛用于颜色映射，但其基于网格的表示需要离散化 RGB 空间，导致容量与内存之间的权衡，在存储大量 LUT 时，这种权衡变​​得令人望而却步。最近的方法采用隐式神经表示来提高可扩展性，但它们的黑盒性质限制了可解释性并阻碍了直观的本地化编辑。在本文中，我们提出了高斯 LUT (GLUT)，这是一种连续且显式的颜色表示，它使用一组可学习的 3D 高斯基元对颜色变换进行建模。通过避免固定分辨率网格，GLU​​T 实现了灵活的表示能力，同时保持紧凑的内存占用。其明确的、空间局部化的公式进一步实现了精确的建模和可解释性。在此表示的基础上，我们引入了一个紧凑的条件生成器 (CGLUT)，它可以预测多个 LUT 实例的 GLUT 参数，在单个框架中编码不同的颜色样式，以实现平滑且可控的 LUT 样式混合。此外，GLUT 支持高效、用户友好的编辑，允许对特定颜色区域进行本地调整，而无需全局重新训练。实验结果表明，我们的方法在准确性和效率方面优于先前的神经 LUT 表示，同时提供改进的可解释性和交互控制。

</details>

---

## 15. 3D Skew Gaussian Splatting with Any Camera Trajectory Visualization Engine / 使用任何相机轨迹可视化引擎进行 3D 倾斜高斯泼溅

**Date**: 2026-05-18 | **arXiv**: [2605.18334v1](http://arxiv.org/abs/2605.18334v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.18334v1)

**Categories**: cs.CV, cs.GR

<details><summary><b>Abstract / 摘要</b></summary>

While 3D Gaussian Splatting (3DGS) has revolutionized real-time photorealistic view synthesis, its fundamental reliance on symmetric Gaussian distributions introduces visual artifacts that hinder accurate spatial data exploration. Specifically, symmetric kernels struggle to capture shape and color discontinuities , which cause blurriness and primitive redundancy that mislead human perception during visual analysis. To address these visualization barriers, we introduce 3D Skew Gaussian Splatting (3DSGS), a novel framework that significantly enhances the structural fidelity and compactness of explicit scene representations. Our key insight lies in extending the standard primitive to a general Skew Gaussian counterpart. This generalized primitive inherits the highly efficient rasterization properties of standard Gaussians while gaining intrinsic asymmetric modeling capabilities. We couple this with an enhanced opacity representation to better handle complex transparency, alongside a depth-aware densification strategy that intelligently manages primitive allocation. Furthermore, to make these advancements actionable for real-world visual analytics, we re-derive the CUDA rasterization pipeline to universally support both symmetric and skew Gaussians, integrating it into a decoupled, free-camera interactive visualization engine. Extensive experiments demonstrate that 3DSGS achieves superior rendering quality and structural compactness, particularly in regions with intricate details, while maintaining the real-time frame rates necessary for fluid interactive exploration. Supplementary derivations and visual results are available at \textbf{\textit{https://3d-skew-gs.github.io/}}.

虽然 3D 高斯分布 (3DGS) 彻底改变了实时真实感视图合成，但其对对称高斯分布的根本依赖引入了视觉伪影，阻碍了准确的空间数据探索。具体来说，对称内核很难捕获形状和颜色的不连续性，这会导致模糊和原始冗余，从而在视觉分析过程中误导人类的感知。为了解决这些可视化障碍，我们引入了 3D Skew Gaussian Splatting (3DSGS)，这是一种新颖的框架，可以显着增强显式场景表示的结构保真度和紧凑性。我们的关键见解在于将标准原语扩展到一般的倾斜高斯对应物。这种广义基元继承了标准高斯的高效光栅化属性，同时获得了内在的非对称建模功能。我们将其与增强的不透明度表示相结合，以更好地处理复杂的透明度，以及智能管理原始分配的深度感知致密化策略。此外，为了使这些进步可用于现实世界的视觉分析，我们重新推导了 CUDA 光栅化管道以普遍支持对称和倾斜高斯，并将其集成到解耦的、免费相机的交互式可视化引擎中。大量实验表明，3DSGS 实现了卓越的渲染质量和结构紧凑性，特别是在具有复杂细节的区域，同时保持了流体交互探索所需的实时帧速率。补充推导和可视化结果可在 \textbf{\textit{https://3d-skew-gs.github.io/}} 获得。

</details>

---

## 16. LUMEN: Low-light Unified Multi-stage Enhancement Network using depth-guided flash, clustering, and attention-based Transformers / LUMEN：使用深度引导闪存、集群和基于注意力的 Transformer 的低光统一多级增强网络

**Date**: 2026-05-18 | **arXiv**: [2605.17893v1](http://arxiv.org/abs/2605.17893v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.17893v1)

**Categories**: eess.IV

<details><summary><b>Abstract / 摘要</b></summary>

Low-light image enhancement remains a challenging problem due to severe noise, color distortion, contrast degradation, and loss of structural details under insufficient illumination. Existing methods typically apply uniform enhancement without considering the depth-dependent nature of light attenuation and sensor noise in real-world scenes. To address this limitation, we propose LUMEN, a multi-stage enhancement framework that integrates virtual flash simulation with transformer-based feature fusion. The proposed framework first estimates scene depth from low-light inputs using a dedicated encoder-decoder network, after which a soft clustering module partitions pixels into depth-aware regions, enabling depth-dependent flash simulation. The simulated flash features, together with depth representations, are fused with image features through efficient attention-based fusion blocks to enhance global context while preserving fine details. A composite loss function combining reconstruction, perceptual, structural, color, edge, and depth consistency objectives ensures both visual fidelity and perceptual quality. Extensive experiments on LOL-v1 and LOL-v2 benchmarks demonstrate that LUMEN achieves state-of-the-art performance and produces visually natural results compared with several state-of-the-art methods.

由于光照不足下严重的噪声、颜色失真、对比度下降和结构细节丢失，低光图像增强仍然是一个具有挑战性的问题。现有方法通常应用均匀增强，而不考虑现实场景中光衰减和传感器噪声的深度相关性质。为了解决这个限制，我们提出了 LUMEN，这是一个多阶段增强框架，它将虚拟闪存模拟与基于变压器的特征融合相集成。所提出的框架首先使用专用的编码器-解码器网络从低光输入估计场景深度，然后软聚类模块将像素划分为深度感知区域，从而实现与深度相关的闪光模拟。模拟的闪光特征与深度表示一起，通过有效的基于注意力的融合块与图像特征融合，以增强全局上下文，同时保留精细细节。结合了重建、感知、结构、颜色、边缘和深度一致性目标的复合损失函数确保了视觉保真度和感知质量。对 LOL-v1 和 LOL-v2 基准的大量实验表明，与几种最先进的方法相比，LUMEN 实现了最先进的性能，并产生了视觉上自然的结果。

</details>

---

## 17. An Underwater Dehazing Network with Implicit Transmission Estimation / 具有隐式传输估计的水下除雾网络

**Date**: 2026-05-13 | **arXiv**: [2605.13720v1](http://arxiv.org/abs/2605.13720v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.13720v1)

**Categories**: eess.IV

<details><summary><b>Abstract / 摘要</b></summary>

Underwater images suffer from wavelength-dependent light absorption and scattering, which reduces visual quality. This phenomenon could limit the operational reliability of autonomous underwater vehicles, marine surveys, and offshore inspection systems. Purely classical methods often achieve suboptimal performance in real-world datasets, while purely data-driven methods lack physical interpretability. In this letter, we propose UDehaze-iT, a deep network for underwater image enhancement that estimates scene depth implicitly and derives per-channel transmission through the Beer-Lambert law with learnable attenuation coefficients. We estimate atmospheric light as a semi-classical per-channel scalar, and a zero-initialized residual refiner corrects remaining artefacts after dehazing. To effectively train our method, we apply a composite loss function consisting of five key terms: a L1 loss, a multi-scale patchwise DCT loss, a forward model reconstruction loss, and two regularization terms. With ~0.9M parameters, UDehaze-iT achieves competitive performance on UIEB and UFO-120 datasets.

水下图像会受到波长相关的光吸收和散射的影响，从而降低视觉质量。这种现象可能会限制自主水下航行器、海洋调查和近海检查系统的运行可靠性。纯粹的经典方法通常在现实世界的数据集中实现次优性能，而纯粹的数据驱动方法缺乏物理可解释性。在这封信中，我们提出了 UDehaze-iT，这是一种用于水下图像增强的深度网络，它隐式估计场景深度，并通过具有可学习衰减系数的比尔-朗伯定律导出每通道传输。我们将大气光估计为半经典的每通道标量，并且零初始化的残差细化器在去雾后校正剩余的伪影。为了有效地训练我们的方法，我们应用了由五个关键项组成的复合损失函数：L1 损失、多尺度补丁 DCT 损失、前向模型重建损失和两个正则化项。 UDehaze-iT 凭借约 0.9M 参数，在 UIEB 和 UFO-120 数据集上实现了具有竞争力的性能。

</details>

---

## 18. Physics-Grounded Adversarial Stain Augmentation with Calibrated Coverage Guarantees / 基于物理的对抗性染色增强和校准覆盖保证

**Date**: 2026-05-12 | **arXiv**: [2605.13889v1](http://arxiv.org/abs/2605.13889v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.13889v1)

**Categories**: eess.IV, cs.CV, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Stain variation across hospitals degrades histopathology models at deployment. Existing augmentation methods perturb color spaces with arbitrary hyperparameters, lacking both a principled budget and coverage guarantees for unseen centers. We propose \textbf{C}alibrated \textbf{A}dversarial \textbf{S}tain \textbf{A}ugmentation (\textbf{CASA}), which performs adversarial augmentation in the Macenko stain parameter space with a budget calibrated from multi-center statistics via the DKW inequality. On Camelyon17-WILDS (5 seeds), CASA achieves $93.9\% \pm 1.6\%$ slide-level accuracy -- outperforming HED-strong ($88.4\% \pm 7.3\%$), RandStainNA ($85.2\% \pm 6.7\%$), and ERM ($63.9\% \pm 11.3\%$) -- with the highest worst-group accuracy ($84.9\% \pm 0.9\%$) among all 10 compared methods.

各医院的染色差异会降低部署时的组织病理学模型。现有的增强方法用任意超参数扰乱色彩空间，缺乏原则性的预算和对看不见的中心的覆盖保证。我们提出 \textbf{C}alibated \textbf{A}dversarial \textbf{S}tain \textbf{A}ugmentation (\textbf{CASA})，它在 Macenko 染色参数空间中执行对抗性增强，其预算通过 DKW 不等式从多中心统计数据校准。在 Camelyon17-WILDS（5 颗种子）上，CASA 达到了 $93.9\% \pm 1.6\%$ 滑动精度 - 优于 HED-strong ($88.4\% \pm 7.3\%$)、RandStainNA ($85.2\% \pm 6.7\%$) 和 ERM ($63.9\% \pm 11.3\%$) - 最高所有 10 种比较方法中最差的组准确度 ($84.9\% \pm 0.9\%$)。

</details>

---

## 19. Are Compact Rationales Free? Measuring Tile Selection Headroom in Frozen WSI-MIL / 紧凑基本原理是免费的吗？测量冻结 WSI-MIL 中的瓷砖选择余量

**Date**: 2026-05-12 | **arXiv**: [2605.12575v1](http://arxiv.org/abs/2605.12575v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.12575v1)

**Categories**: eess.IV, cs.AI, cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Whole-slide image (WSI) multiple instance learning (MIL) classifiers can achieve strong slide-level AUC while leaving the full-bag prediction opaque. Attention scores are widely reused as post-hoc explanations, but high attention can reflect aggregation preference rather than a compact, model-sufficient rationale. We study post-hoc rationale highlighting for frozen WSI-MIL: given a trained classifier, can its slide-level prediction be recovered from a compact, output-consistent tile subset without retraining the backbone? We instantiate this with Finding Optimal Contextual Instances (FOCI), a lightweight rationale-readout layer over a frozen MIL backbone. FOCI is trained with model-output sufficiency and exclusion objectives over keep/drop tile subsets, evaluated with an insertion-style Sequential Reveal Protocol (SRP) adapted to WSI-MIL, and summarized by the Selection Headroom Index (SHI). Across three WSI benchmarks and seven MIL backbones, FOCI reveals that compact rationales are selection-headroom dependent: transformer and multi-branch attention aggregators can admit compact rationales, near-minimal attention-pooling baselines enter a selection-saturation regime, and hard-selection backbones can conflict with an external readout. For TransMIL, relative to its documented CLS-proxy ranking, FOCI reduces the Minimum Sufficient K (MSK) tile count by 32-56% across benchmarks, while ACMIL+FOCI attains the highest mean SHI (+0.465). Deletion-based perturbation and selected-only downstream evaluation provide complementary checks. These results position FOCI as a model-level interpretability and audit layer: selected tiles are not claims of clinical or pathologist-level diagnostic sufficiency, but candidate rationales that offer a compact, reviewable view of when a frozen MIL prediction can be localized to a small output-consistent subset.

全幻灯片图像 (WSI) 多实例学习 (MIL) 分类器可以实现强大的幻灯片级 AUC，同时使整袋预测不透明。注意力分数被广泛用作事后解释，但高注意力可以反映聚合偏好，而不是紧凑的、模型充足的基本原理。我们研究了冻结 WSI-MIL 的事后基本原理突出显示：给定一个经过训练的分类器，是否可以从紧凑的、输出一致的图块子集中恢复其滑动级预测，而无需重新训练主干网？我们通过寻找最佳上下文实例 (FOCI) 来实例化这一点，FOCI 是冻结 MIL 主干上的轻量级原理读出层。 FOCI 使用模型输出充分性和保留/丢弃图块子集的排除目标进行训练，使用适合 WSI-MIL 的插入式顺序显示协议 (SRP) 进行评估，并通过选择余量指数 (SHI) 进行总结。在三个 WSI 基准和七个 MIL 主干中，FOCI 揭示了紧凑的基本原理是选择余量依赖的：变压器和多分支注意力聚合器可以承认紧凑的基本原理，接近最小的注意力池基线进入选择饱和状态，而硬选择主干可能与外部读数发生冲突。对于 TransMIL，相对于其记录的 CLS 代理排名，FOCI 在基准测试中将最小足够 K (MSK) 切片数量减少了 32-56%，而 ACMIL+FOCI 获得了最高的平均 SHI (+0.465)。基于删除的扰动和仅选择的下游评估提供了补充检查。这些结果将 FOCI 定位为模型级可解释性和审核层：选定的图块并不是临床或病理学家级诊断充分性的声明，而是候选原理，它们提供了一个紧凑的、可审查的视图，说明何时可以将冻结的 MIL 预测本地化到一个小的输出一致子集。

</details>

---

## 20. ALGOGEN: Tool-Generated Verifiable Traces for Reliable Algorithm Visualization / ALGOGEN：工具生成的可验证跟踪，用于可靠的算法可视化

**Date**: 2026-05-12 | **arXiv**: [2605.12159v1](http://arxiv.org/abs/2605.12159v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.12159v1)

**Categories**: cs.AI, cs.GR

<details><summary><b>Abstract / 摘要</b></summary>

Algorithm Visualization (AV) helps students build mental models by animating algorithm execution states. Recent LLM-based systems such as CODE2VIDEO generate AV videos in an end-to-end manner. However, this paradigm requires the system to simultaneously simulate algorithm flow and satisfy video rendering constraints, such as element layout and color schemes. This complex task induces LLM hallucinations, resulting in reduced execution success rates, element overlap, and inter-frame inconsistencies.   To address these challenges, we propose ALGOGEN, a novel paradigm that decouples algorithm execution from rendering. We first introduce Visualization Trace Algebra (VTA), a monoid over algorithm visual states and operations. The LLM then generates a Python tracker that simulates algorithm flow and outputs VTA-JSON traces, a JSON encoding of VTA. For rendering, we define a Rendering Style Language (RSL) to templatize algorithm layouts. A deterministic renderer then compiles algorithm traces with RSL into Manim, LaTeX/TikZ, or Three.js outputs.   Evaluated on a LeetCode AV benchmark of 200 tasks, ALGOGEN achieves an average success rate improvement of 17.3% compared to end-to-end methods, with 99.8% versus 82.5%. These results demonstrate that our decoupling paradigm effectively mitigates LLM hallucinations in complex AV tasks, providing a more reliable solution for automated generation of high-quality algorithm visualizations. Demo videos and code are available in the project repository.

算法可视化 (AV) 通过动画算法执行状态帮助学生构建心理模型。最近基于 LLM 的系统（例如 CODE2VIDEO）以端到端方式生成 AV 视频。然而，这种范例要求系统同时模拟算法流程并满足视频渲染约束，例如元素布局和配色方案。这项复杂的任务会引发 LLM 幻觉，导致执行成功率降低、元素重叠和帧间不一致。   为了应对这些挑战，我们提出了 ALGOGEN，一种将算法执行与渲染分离的新颖范例。我们首先介绍可视化追踪代数（VTA），这是一种算法视觉状态和操作的幺半群。然后，LLM 生成一个 Python 跟踪器，用于模拟算法流程并输出 VTA-JSON 跟踪（VTA 的 JSON 编码）。对于渲染，我们定义了渲染风格语言（RSL）来模板化算法布局。然后，确定性渲染器使用 RSL 将算法跟踪编译为 Manim、LaTeX/TikZ 或 Three.js 输出。   在 200 个任务的 LeetCode AV 基准上进行评估，与端到端方法相比，ALGOGEN 的平均成功率提高了 17.3%，分别为 99.8% 和 82.5%。这些结果表明，我们的解耦范式有效地减轻了复杂 AV 任务中的 LLM 幻觉，为自动生成高质量算法可视化提供了更可靠的解决方案。项目存储库中提供了演示视频和代码。

</details>

---

## 21. Kelvin v1.0: A Neural Pre-Encoder for H.264: A standards-compliant learned preprocessor with -27.62% BD-VMAF on UVG / Kelvin v1.0：H.264 的神经预编码器：符合标准的学习预处理器，UVG 上的 BD-VMAF 为 -27.62%

**Date**: 2026-05-10 | **arXiv**: [2605.16376v1](http://arxiv.org/abs/2605.16376v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.16376v1)

**Categories**: eess.IV, cs.CV, cs.DC, cs.LG, cs.MM

<details><summary><b>Abstract / 摘要</b></summary>

Kelvin is a lightweight learned pre-encoder that sits in front of an unmodified libx264 encoder. It applies content-adaptive pixel adjustments, bounded at +/-1/255 per channel, so that the encoder allocates bits where they matter most perceptually, while emitting a standard H.264 bitstream compatible with every existing decoder, player, and CDN. On the seven-sequence 1080p UVG benchmark, Kelvin v1.0 achieves a mean BD-VMAF of -27.62% (7 of 7 wins) and BD-VMAF-NEG of -5.18% (6 of 7 wins) relative to baseline libx264 at preset medium. On the 30-sequence MCL-JCV public set (28 unseen by training), the same checkpoint wins on 28 of 30 clips by BD-VMAF; with the two diagnosable failures removed the mean is -27.70% BD-VMAF and -5.37% BD-VMAF-NEG, consistent with UVG to within one percentage point. A central engineering challenge is the non-differentiability of H.264: we describe a hybrid codec proxy that combines a calibrated differentiable rate estimator (Spearman rho = 0.986 vs. real libx264 bits-per-pixel) with a U-Net distortion proxy trained on real encoder outputs. We publish full per-sequence rate-distortion data, a named failure-mode taxonomy on MCL-JCV (rate-floor violation, distribution shift, metric saturation), a five-baseline sanity panel (hqdn3d, unsharp, -tune psnr, -tune ssim, x265 medium), and honest positioning: x265 medium beats Kelvin on every metric on the same corpus. Kelvin is therefore designed for workloads where remaining on H.264 is a constraint rather than a choice.

Kelvin 是一个轻量级学习预编码器，位于未修改的 libx264 编码器前面。它应用内容自适应像素调整，每个通道的范围为 +/-1/255，以便编码器在感知最重要的位置分配比特，同时发出与每个现有解码器、播放器和 CDN 兼容的标准 H.264 比特流。在七序列 1080p UVG 基准测试中，相对于预设介质下的基线 libx264，Kelvin v1.0 的平均 BD-VMAF 为 -27.62%（7 胜中的 7 胜），BD-VMAF-NEG 为 -5.18%（7 胜中的 6 胜）。在 30 序列 MCL-JCV 公共集（训练中未看到 28 个）上，同一检查点在 BD-VMAF 的 30 个片段中的 28 个中获胜；除去两个可诊断故障后，平均值为 -27.70% BD-VMAF 和 -5.37% BD-VMAF-NEG，与 UVG 一致，误差在 1 个百分点以内。一个核心的工程挑战是 H.264 的不可微性：我们描述了一种混合编解码器代理，它将校准的可微速率估计器（Spearman rho = 0.986 对比真实的 libx264 位/像素）与在真实编码器输出上训练的 U-Net 失真代理相结合。我们发布了完整的每序列率失真数据、MCL-JCV 上的命名故障模式分类法（速率下限违规、分布偏移、指标饱和度）、五基线健全性面板（hqdn3d、unsharp、-tune psnr、-tune ssim、x265medium）以及诚实定位：x265medium 在同一语料库上的每个指标上都击败了 Kelvin。因此，Kelvin 专为保留 H.264 是一种限制而不是一种选择的工作负载而设计。

</details>

---

## 22. CAGS: Color-Adaptive Volumetric Video Streaming with Dynamic 3D Gaussian Splatting / CAGS：具有动态 3D 高斯分布的颜色自适应体积视频流

**Date**: 2026-05-10 | **arXiv**: [2605.09279v1](http://arxiv.org/abs/2605.09279v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.09279v1)

**Categories**: cs.GR, cs.CV, cs.MM, cs.NI, eess.IV

<details><summary><b>Abstract / 摘要</b></summary>

Volumetric video (VV) streaming enables real-time, immersive access to remote 3D environments, powering telepresence, ecological monitoring, and robotic teleoperation. These applications turn VV streaming into a real-time interface to remote physical environments, imposing new system-level demands for photorealistic scene representation, low-latency interaction, and robust performance under heterogeneous networks. 3D Gaussian Splatting (3DGS) has been widely used for real-time photorealistic rendering, offering superior visual quality and rendering performance, but it faces challenges due to bandwidth consumption. Furthermore, as the foundation of adaptive VV streaming, existing Levels of Detail (LoD) methods based on density are not well-suited to Gaussian representations, leading to visible gaps and severe quality degradation. Recent studies have also explored attribute compression techniques to reduce bandwidth consumption. Our preliminary studies reveal that aggressive attribute compression primarily causes color distortion, which can be effectively corrected in the rendered image using a reference image. Motivated by these findings, we propose a novel Color-Adaptive scheme for adaptive VV streaming that uses vector quantization (VQ) to establish LoDs and correct color distortions with low-resolution reference images. We further present CAGS, an adaptive VV streaming system compatible with diverse Gaussian representations, which integrates the Color-Adaptive scheme by rendering reference images on the streaming server and performing color restoration on the client. Extensive experiments on our prototype system demonstrate that CAGS outperforms the existing adaptive streaming systems in PSNR by 5$\sim$20 dB under fluctuating bandwidth, operates significantly faster than existing scalable Gaussian compression methods, and generalizes across different Gaussian representations.

体积视频 (VV) 流媒体可实现对远程 3D 环境的实时、沉浸式访问，为远程呈现、生态监测和机器人远程操作提供支持。这些应用程序将 VV 流转为远程物理环境的实时接口，对异构网络下的逼真场景表示、低延迟交互和鲁棒性能提出了新的系统级需求。 3D高斯溅射（3DGS）已广泛用于实时真实感渲染，提供卓越的视觉质量和渲染性能，但由于带宽消耗而面临挑战。此外，作为自适应 VV 流的基础，现有的基于密度的细节级别 (LoD) 方法不太适合高斯表示，导致可见的间隙和严重的质量下降。最近的研究还探索了属性压缩技术来减少带宽消耗。我们的初步研究表明，激进的属性压缩主要会导致颜色失真，可以使用参考图像在渲染图像中有效地校正颜色失真。受这些发现的启发，我们提出了一种用于自适应 VV 流的新颖颜色自适应方案，该方案使用矢量量化 (VQ) 来建立 LoD 并使用低分辨率参考图像校正颜色失真。我们进一步提出了 CAGS，一种与多种高斯表示兼容的自适应 VV 流媒体系统，它通过在流媒体服务器上渲染参考图像并在客户端上执行颜色恢复来集成颜色自适应方案。在我们的原型系统上进行的大量实验表明，CAGS 在波动带宽下的 PSNR 性能优于现有的自适应流系统 5$\sim$20 dB，运行速度明显快于现有的可扩展高斯压缩方法，并且可以推广到不同的高斯表示。

</details>

---

## 23. Relightable Gaussian Splatting for Virtual Production Using Image-Based Illumination / 使用基于图像的照明进行虚拟生产的可重新照明高斯溅射

**Date**: 2026-05-09 | **arXiv**: [2605.09024v1](http://arxiv.org/abs/2605.09024v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.09024v1)

**Categories**: cs.CV, cs.GR, cs.MM, eess.IV

<details><summary><b>Abstract / 摘要</b></summary>

Virtual production (VP) use LED walls to provide both background imagery and image-based lighting. While this enables on-set compositing, it couples lighting to background and scene appearance, limiting flexibility for downstream editing. In addition, inverse rendering conventionally relies on physically-based rendering to estimates 3D geometry and lighting, using environment maps. However, these maps are typically low-resolution and assume far-field lighting. In VP, with near-field and high-resolution image-based lighting, this can lead to inaccuracies and introduce complexities when editing. Addressing this, we propose a VP-specific framework for 3D reconstruction and relighting using Gaussian Splatting. This uses the known background imagery to condition the relighting process. This avoids relying on environment maps and reduces compositing to a background-image editing task. To realize our framework, we introduce a process (and associated dataset) that captures real VP scenes under varying background content and illumination conditions. This data is used to decompose a 3D scene into fixed appearance and variable lighting components. The variable lighting process simulates light transport by parameterizing each primitive with a UV coordinate, intensity value and resolution modifier. Using mipmaps, these directly sample the background texture in image space - implicitly capturing reflections and refractions without physically-based rendering. Combined with the fixed appearance component, this allows us to render relit scenes using a Gaussian Splatting rasterizer. Compared to baselines, our approach achieves higher-quality 3D reconstruction and controllable relighting. The method is efficient (<3 GB RAM, <5 GB VRAM, <2 hours training, ~35 FPS) and supports rendering useful arbitrary output variables including depth, lighting intensity, lighting color, and unlit renders.

虚拟制作 (VP) 使用 LED 墙提供背景图像和基于图像的照明。虽然这可以实现现场合成，但它将照明与背景和场景外观耦合在一起，限制了下游编辑的灵活性。此外，逆向渲染通常依赖于基于物理的渲染，使用环境贴图来估计 3D 几何和照明。然而，这些地图通常是低分辨率的并且假设远场照明。在 VP 中，使用近场和基于高分辨率图像的照明，这可能会导致编辑时不准确并带来复杂性。为了解决这个问题，我们提出了一个特定于 VP 的框架，用于使用高斯分布进行 3D 重建和重新照明。这使用已知的背景图像来调节重新照明过程。这避免了对环境贴图的依赖，并减少了背景图像编辑任务的合成。为了实现我们的框架，我们引入了一个过程（以及相关的数据集），该过程可以在不同的背景内容和照明条件下捕获真实的 VP 场景。该数据用于将 3D 场景分解为固定外观和可变照明组件。可变光照过程通过使用 UV 坐标、强度值和分辨率修改器对每个基元进行参数化来模拟光传输。使用 mipmap，这些直接对图像空间中的背景纹理进行采样 - 隐式捕获反射和折射，而无需基于物理的渲染。与固定外观组件相结合，这使我们能够使用高斯泼溅光栅器渲染重新照明的场景。与基线相比，我们的方法实现了更高质量的 3D 重建和可控重新照明。该方法非常高效（<3 GB RAM、<5 GB VRAM、<2 小时训练、~35 FPS），并且支持渲染有用的任意输出变量，包括深度、光照强度、光照颜色和无光照渲染。

</details>

---

## 24. Stage Light is Sequence$^2$: Multi-Light Control via Imitation Learning / 舞台灯光是序列$^2$：通过模仿学习进行多灯控制

**Date**: 2026-05-05 | **arXiv**: [2605.03660v1](http://arxiv.org/abs/2605.03660v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.03660v1)

**Categories**: cs.MM, cs.AI

**Code**: https://github.com/RS2002/SeqLight

<details><summary><b>Abstract / 摘要</b></summary>

Music-inspired Automatic Stage Lighting Control (ASLC) has gained increasing attention in recent years due to the substantial time and financial costs associated with hiring and training professional lighting engineers. However, existing methods suffer from several notable limitations: the low interpretability of rule-based approaches, the restriction to single-primary-light control in music-to-color-space methods, and the limited transferability of music-to-controlling-parameter frameworks. To address these gaps, we propose SeqLight, a hierarchical deep learning framework that maps music to multi-light Hue-Saturation-Value (HSV) space. Our approach first customizes SkipBART, an end-to-end single primary light generation model, to predict the full light color distribution for each frame, followed by hybrid Imitation Learning (IL) techniques to derive an effective decomposition strategy that distributes the global color distribution among individual lights. Notably, the light decomposition module can be trained under varying venue-specific lighting configurations using only mixed light data and no professional demonstrations, thereby flexibly adapting across diverse venues. In this stage, we formulate the light decomposition task as a Goal-Conditioned Markov Decision Process (GCMDP), construct an expert demonstration set inspired by Hindsight Experience Replay (HER), and introduce a three-phase IL training pipeline, achieving strong generalization capability. To validate our IL solution for the proposed GCMDP, we conduct a series of quantitative analysis and human study. The code and trained models are provided at https://github.com/RS2002/SeqLight .

近年来，受音乐启发的自动舞台灯光控制 (ASLC) 受到越来越多的关注，因为雇用和培训专业灯光工程师需要大量的时间和财务成本。然而，现有方法存在几个显着的局限性：基于规则的方法的可解释性低、音乐到色彩空间方法中对单基色光控制的限制以及音乐到控制参数框架的可转移性有限。为了解决这些差距，我们提出了 SeqLight，这是一种分层深度学习框架，可将音乐映射到多光色调-饱和度-值 (HSV) 空间。我们的方法首先定制 SkipBART（一种端到端单基色光生成模型）来预测每个帧的完整光颜色分布，然后采用混合模仿学习 (IL) 技术来导出有效的分解策略，在各个光之间分配全局颜色分布。值得注意的是，光分解模块可以仅使用混合光数据而无需专业演示，在不同的特定场地照明配置下进行训练，从而灵活地适应不同的场地。在这个阶段，我们将轻分解任务制定为目标条件马尔可夫决策过程（GCMDP），构建受后见之明经验回放（HER）启发的专家演示集，并引入三阶段IL训练管道，实现了强大的泛化能力。为了验证我们针对拟议的 GCMDP 的 IL 解决方案，我们进行了一系列定量分析和人体研究。代码和经过训练的模型位于 https://github.com/RS2002/SeqLight 。

</details>

---

## 25. EMOVIS: Emotion-Optimized Image Processing / EMOVIS：情感优化的图像处理

**Date**: 2026-05-04 | **arXiv**: [2605.03131v1](http://arxiv.org/abs/2605.03131v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.03131v1)

**Categories**: eess.IV, cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

In cinematography, visual attributes such as color grading, contrast, and brightness are manipulated to reinforce the emotional narrative of a scene. However, conventional Image Signal Processors (ISPs) prioritize scene fidelity, effectively neglecting this expressive dimension. To bring this cinematic capability to real-time camera pipelines during video capture, we introduce EMOVIS (EMotion-Optimized VISual processing). We establish a systematic mapping between a compact set of high-level emotional states (Happy, Calm, Angry, Sad) and low-level ISP controls - including color saturation, local tone mapping, and sharpness - supported by a calibration user study with statistically significant effects across parameters. We propose a control framework that integrates these emotion-driven adjustments into standard ISP hardware without altering the underlying processing stages. Validation via blind A/B testing shows that viewers prefer the emotion-optimized rendering in 87% of trials when the target emotion matches the scene context, indicating that emotion-aligned ISP control improves perceived suitability for expressive visual content.

在电影摄影中，通过操纵色彩分级、对比度和亮度等视觉属性来强化场景的情感叙事。然而，传统的图像信号处理器 (ISP) 优先考虑场景保真度，实际上忽略了这一表达维度。为了在视频捕获期间将这种电影功能引入实时摄像机管道，我们引入了 EMOVIS（EMotion 优化的视觉处理）。我们在一组紧凑的高级情绪状态（快乐、平静、愤怒、悲伤）和低级 ISP 控制（包括色彩饱和度、局部色调映射和锐度）之间建立了系统映射，并由校准用户研究支持，对参数具有统计上的显着影响。我们提出了一个控制框架，将这些情感驱动的调整集成到标准 ISP 硬件中，而不改变底层处理阶段。通过盲 A/B 测试进行的验证表明，当目标情感与场景上下文相匹配时，观看者在 87% 的试验中更喜欢情感优化渲染，这表明情感一致的 ISP 控制提高了对富有表现力的视觉内容的感知适合性。

</details>

---

## 26. Development and Validation of an Integrated LiDAR-Camera System for Real-Time Monitoring of Underground Longwall Operations / 用于实时监控地下长壁作业的集成激光雷达相机系统的开发和验证

**Date**: 2026-05-04 | **arXiv**: [2605.02516v1](http://arxiv.org/abs/2605.02516v1) | **PDF**: [Link](http://arxiv.org/pdf/2605.02516v1)

**Categories**: eess.IV

<details><summary><b>Abstract / 摘要</b></summary>

Real-time spatial monitoring in underground longwall operations is challenging due to methane-related safety risks, poor visibility, elevated thermal loads, spatial confinement, and bandwidth-limited communications. Currently available camera-based monitoring provides visual context but lacks direct depth information, while standalone underground LiDAR scanners are limited to monochromatic or periodic 3D mapping. This paper presents the design, integration, and experimental validation of a LiDAR-camera monitoring system built around a certified flameproof enclosure that prevents flame propagation into the surrounding atmosphere. The system combines a solid-state LiDAR, an industrial RGB camera, and an onboard processor within a compact hardware assembly, supporting LiDAR-camera fusion, low-light image enhancement, and real-time processing. Laboratory experiments evaluated LiDAR and camera performance through the protective polycarbonate dome and quantified optical and geometric distortions introduced by the enclosure. Thermal testing showed that iterative component placement, heat sinking, and passive conduction reduced peak surface temperature from 106 °C to 70 °C, with internal temperature stabilising at 57 °C. Furthermore, a representative longwall simulation was created to evaluate the complete sensing, fusion, and transmission workflow under controlled geometric and low-light conditions. In the final configuration, more than 97% of LiDAR points fell within the camera field of view, supporting reliable colourisation. Enclosure-aware calibration and correction maintained geometric accuracy, while processed colourised point clouds were transmitted at up to 10 Hz with sustained bandwidth below 25 Mb/s.

由于甲烷相关的安全风险、能见度差、热负荷升高、空间限制和带宽有限的通信，地下长壁作业中的实时空间监测具有挑战性。目前可用的基于摄像头的监控提供视觉背景，但缺乏直接的深度信息，而独立的地下 LiDAR 扫描仪仅限于单色或周期性 3D 测绘。本文介绍了围绕经过认证的防火外壳构建的激光雷达摄像头监控系统的设计、集成和实验验证，该外壳可防止火焰传播到周围大气中。该系统在紧凑的硬件组件中结合了固态 LiDAR、工业 RGB 相机和板载处理器，支持 LiDAR-相机融合、低光图像增强和实时处理。实验室实验通过保护性聚碳酸酯圆顶以及量化外壳引入的光学和几何变形来评估激光雷达和相机的性能。热测试表明，迭代组件放置、散热和无源传导将峰值表面温度从 106 °C 降低至 70 °C，内部温度稳定在 57 °C。此外，还创建了具有代表性的长壁模拟，以评估受控几何和低光条件下的完整传感、融合和传输工作流程。在最终配置中，超过 97% 的 LiDAR 点落在相机视野内，支持可靠的彩色化。外壳感知校准和校正保持了几何精度，同时处理后的彩色点云以高达 10 Hz 的频率传输，持续带宽低于 25 Mb/s。

</details>

---

## 27. Colorful-Noise: Training-Free Low-Frequency Noise Manipulation for Color-Based Conditional Image Generation / 彩色噪声：用于基于颜色的条件图像生成的免训练低频噪声处理

**Date**: 2026-05-01 | **arXiv**: [2605.00548v2](http://arxiv.org/abs/2605.00548v2) | **PDF**: [Link](http://arxiv.org/pdf/2605.00548v2)

**Categories**: cs.CV, cs.GR

<details><summary><b>Abstract / 摘要</b></summary>

Text-to-image diffusion models generate images by gradually converting white Gaussian noise into a natural image. White Gaussian noise is well suited for producing diverse outputs from a single text prompt due to its absence of structure. However, this very property limits control over, and predictability of, specific visual attributes, as the noise is not human-interpretable. In this work, we investigate the characteristics of the input noise in diffusion models. We show that, although all frequencies in white Gaussian noise have comparable statistical energy, low-frequency components primarily determine the images global structure and color composition, while high-frequency components control finer details. Building on this observation, we demonstrate that simple manipulations of the low-frequency noise using low-frequency image priors can effectively condition the generation process to reconstruct these low-frequency visual cues. This allows us to define a simple, training-free method with minimal overhead that steers overall image structure and color, while letting high-frequency components freely emerge as fine details, enabling variability across generated outputs.

文本到图像扩散模型通过逐渐将高斯白噪声转换为自然图像来生成图像。由于缺乏结构，高斯白噪声非常适合从单个文本提示生成不同的输出。然而，这种特性限制了对特定视觉属性的控制和可预测性，因为噪声是人类无法解释的。在这项工作中，我们研究了扩散模型中输入噪声的特征。我们表明，尽管高斯白噪声中的所有频率都具有可比较的统计能量，但低频分量主要决定图像的全局结构和颜色组成，而高频分量控制更精细的细节。基于这一观察，我们证明使用低频图像先验对低频噪声进行简单操作可以有效地调节生成过程以重建这些低频视觉线索。这使我们能够定义一种简单、免训练的方法，以最小的开销控制整体图像结构和颜色，同时让高频分量自由地以精细细节的形式出现，从而实现生成的输出的可变性。

</details>

---

## 28. FASH-iCNN: Making Editorial Fashion Identity Inspectable Through Multimodal CNN Probing / FASH-iCNN：通过多模态 CNN 探测使编辑时尚身份可检查

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
