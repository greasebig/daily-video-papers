# Agent Papers

Daily updates of agent-related arXiv papers.

## Papers Index

<!-- PAPERS_INDEX_START -->
- [2026-02-20](papers/2026-02-20.md) - 36 papers
- [2026-02-19](papers/2026-02-19.md) - 25 papers
- [2026-02-18](papers/2026-02-18.md) - 32 papers
- [2026-02-17](papers/2026-02-17.md) - 18 papers
- [2026-02-16](papers/2026-02-16.md) - 7 papers
- [2026-02-14](papers/2026-02-14.md) - 53 papers
<!-- PAPERS_INDEX_END -->

## Daily Papers

<!-- PAPERS_CONTENT_START -->
<details><summary><b>2026-02-20 (36 papers)</b></summary>

# arXiv Agent Papers - 2026-02-20

**Paper Count**: 36

---

## 1. FAMOSE: A ReAct Approach to Automated Feature Discovery / FAMASE：自动化特征发现的 ReAct 方法

**Date**: 2026-02-19 | **arXiv**: [2602.17641v1](http://arxiv.org/abs/2602.17641v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.17641v1)

**Categories**: cs.LG, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Feature engineering remains a critical yet challenging bottleneck in machine learning, particularly for tabular data, as identifying optimal features from an exponentially large feature space traditionally demands substantial domain expertise. To address this challenge, we introduce FAMOSE (Feature AugMentation and Optimal Selection agEnt), a novel framework that leverages the ReAct paradigm to autonomously explore, generate, and refine features while integrating feature selection and evaluation tools within an agent architecture. To our knowledge, FAMOSE represents the first application of an agentic ReAct framework to automated feature engineering, especially for both regression and classification tasks. Extensive experiments demonstrate that FAMOSE is at or near the state-of-the-art on classification tasks (especially tasks with more than 10K instances, where ROC-AUC increases 0.23% on average), and achieves the state-of-the-art for regression tasks by reducing RMSE by 2.0% on average, while remaining more robust to errors than other algorithms. We hypothesize that FAMOSE's strong performance is because ReAct allows the LLM context window to record (via iterative feature discovery and evaluation steps) what features did or did not work. This is similar to a few-shot prompt and guides the LLM to invent better, more innovative features. Our work offers evidence that AI agents are remarkably effective in solving problems that require highly inventive solutions, such as feature engineering.

特征工程仍然是机器学习中一个关键但具有挑战性的瓶颈，特别是对于表格数据，因为从指数级大的特征空间中识别最佳特征传统上需要大量的领域专业知识。为了应对这一挑战，我们引入了 FAMOSE（特征增强和最佳选择代理），这是一种新颖的框架，利用 ReAct 范式自主探索、生成和细化特征，同时将特征选择和评估工具集成到代理架构中。据我们所知，FAMOSE 代表了代理 ReAct 框架在自动化特征工程中的首次应用，特别是对于回归和分类任务。大量实验表明，FAMOSE 在分类任务（尤其是超过 10K 实例的任务，其中 ROC-AUC 平均增加 0.23%）上达到或接近最先进的水平，并且通过将 RMSE 平均降低 2.0% 实现了回归任务的最先进水平，同时比其他算法对错误具有更强的鲁棒性。我们假设 FAMOSE 的强大性能是因为 ReAct 允许 LLM 上下文窗口记录（通过迭代特征发现和评估步骤）哪些功能有效或无效。这类似于几次提示，指导法学硕士发明更好、更具创新性的功能。我们的工作证明，人工智能代理在解决需要高度创造性解决方案的问题（例如特征工程）方面非常有效。

</details>

---

## 2. AutoNumerics: An Autonomous, PDE-Agnostic Multi-Agent Pipeline for Scientific Computing / AutoNumerics：用于科学计算的自治、与偏微分方程无关的多代理管道

**Date**: 2026-02-19 | **arXiv**: [2602.17607v1](http://arxiv.org/abs/2602.17607v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.17607v1)

**Categories**: cs.AI, cs.LG, math.NA

<details><summary><b>Abstract / 摘要</b></summary>

PDEs are central to scientific and engineering modeling, yet designing accurate numerical solvers typically requires substantial mathematical expertise and manual tuning. Recent neural network-based approaches improve flexibility but often demand high computational cost and suffer from limited interpretability. We introduce \texttt{AutoNumerics}, a multi-agent framework that autonomously designs, implements, debugs, and verifies numerical solvers for general PDEs directly from natural language descriptions. Unlike black-box neural solvers, our framework generates transparent solvers grounded in classical numerical analysis. We introduce a coarse-to-fine execution strategy and a residual-based self-verification mechanism. Experiments on 24 canonical and real-world PDE problems demonstrate that \texttt{AutoNumerics} achieves competitive or superior accuracy compared to existing neural and LLM-based baselines, and correctly selects numerical schemes based on PDE structural properties, suggesting its viability as an accessible paradigm for automated PDE solving.

偏微分方程是科学和工程建模的核心，但设计精确的数值求解器通常需要大量的数学专业知识和手动调整。最近基于神经网络的方法提高了灵活性，但通常需要较高的计算成本，并且可解释性有限。我们引入了 \texttt{AutoNumerics}，这是一个多智能体框架，可以直接根据自然语言描述自主设计、实现、调试和验证通用偏微分方程的数值求解器。与黑盒神经求解器不同，我们的框架生成基于经典数值分析的透明求解器。我们引入了从粗到细的执行策略和基于残差的自验证机制。对 24 个典型和现实世界 PDE 问题的实验表明，与现有的神经和基于 LLM 的基线相比，\texttt{AutoNumerics} 实现了具有竞争力或更高的精度，并根据 PDE 结构属性正确选择数值方案，这表明它作为自动 PDE 求解的可访问范例的可行性。

</details>

---

## 3. KLong: Training LLM Agent for Extremely Long-horizon Tasks / KLong：训练 LLM 代理执行超长期任务

**Date**: 2026-02-19 | **arXiv**: [2602.17547v1](http://arxiv.org/abs/2602.17547v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.17547v1)

**Categories**: cs.AI, cs.CL

<details><summary><b>Abstract / 摘要</b></summary>

This paper introduces KLong, an open-source LLM agent trained to solve extremely long-horizon tasks. The principle is to first cold-start the model via trajectory-splitting SFT, then scale it via progressive RL training. Specifically, we first activate basic agentic abilities of a base model with a comprehensive SFT recipe. Then, we introduce Research-Factory, an automated pipeline that generates high-quality training data by collecting research papers and constructing evaluation rubrics. Using this pipeline, we build thousands of long-horizon trajectories distilled from Claude 4.5 Sonnet (Thinking). To train with these extremely long trajectories, we propose a new trajectory-splitting SFT, which preserves early context, progressively truncates later context, and maintains overlap between sub-trajectories. In addition, to further improve long-horizon task-solving capability, we propose a novel progressive RL, which schedules training into multiple stages with progressively extended timeouts. Experiments demonstrate the superiority and generalization of KLong, as shown in Figure 1. Notably, our proposed KLong (106B) surpasses Kimi K2 Thinking (1T) by 11.28% on PaperBench, and the performance improvement generalizes to other coding benchmarks like SWE-bench Verified and MLE-bench.

本文介绍了 KLong，这是一种开源 LLM 代理，经过训练可以解决超长期任务。其原理是首先通过轨迹分割 SFT 冷启动模型，然后通过渐进式 RL 训练对其进行扩展。具体来说，我们首先使用综合的 SFT 配方激活基本模型的基本代理能力。然后，我们介绍 Research-Factory，这是一个自动化管道，通过收集研究论文和构建评估标准来生成高质量的训练数据。使用这个管道，我们构建了数千条从 Claude 4.5 Sonnet（思考）中提炼出来的长视野轨迹。为了使用这些极长的轨迹进行训练，我们提出了一种新的轨迹分割 SFT，它保留早期上下文，逐步截断后来的上下文，并保持子轨迹之间的重叠。此外，为了进一步提高长期任务解决能力，我们提出了一种新颖的渐进式强化学习，它将训练安排为多个阶段，并逐步延长超时时间。实验证明了 KLong 的优越性和泛化性，如图 1 所示。值得注意的是，我们提出的 KLong (106B) 在 PaperBench 上超越 Kimi K2 Thinking (1T) 11.28%，并且性能改进推广到其他编码基准，如 SWE-bench Verified 和 MLE-bench。

</details>

---

## 4. Evaluating Chain-of-Thought Reasoning through Reusability and Verifiability / 通过可重用性和可验证性评估思想链推理

**Date**: 2026-02-19 | **arXiv**: [2602.17544v1](http://arxiv.org/abs/2602.17544v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.17544v1)

**Categories**: cs.AI, cs.CL, cs.IR

<details><summary><b>Abstract / 摘要</b></summary>

In multi-agent IR pipelines for tasks such as search and ranking, LLM-based agents exchange intermediate reasoning in terms of Chain-of-Thought (CoT) with each other. Current CoT evaluation narrowly focuses on target task accuracy. However, this metric fails to assess the quality or utility of the reasoning process itself. To address this limitation, we introduce two novel measures: reusability and verifiability. We decouple CoT generation from execution using a Thinker-Executor framework. Reusability measures how easily an Executor can reuse the Thinker's CoT. Verifiability measures how frequently an Executor can match the Thinker's answer using the CoT. We evaluated four Thinker models against a committee of ten Executor models across five benchmarks. Our results reveal that reusability and verifiability do not correlate with standard accuracy, exposing a blind spot in current accuracy-based leaderboards for reasoning capability. Surprisingly, we find that CoTs from specialized reasoning models are not consistently more reusable or verifiable than those from general-purpose LLMs like Llama and Gemma.

在用于搜索和排名等任务的多智能体 IR 管道中，基于 LLM 的智能体根据思想链 (CoT) 相互交换中间推理。目前的 CoT 评估主要关注目标任务的准确性。然而，这个指标无法评估推理过程本身的质量或效用。为了解决这个限制，我们引入了两种新颖的措施：可重用性和可验证性。我们使用 Thinker-Executor 框架将 CoT 生成与执行分离。可重用性衡量执行者重用思考者 CoT 的容易程度。可验证性衡量执行者使用 CoT 匹配思考者答案的频率。我们通过五个基准评估了四个 Thinker 模型与由十个 Executor 模型组成的委员会。我们的结果表明，可重用性和可验证性与标准准确性无关，暴露了当前基于准确性的排行榜中推理能力的盲点。令人惊讶的是，我们发现来自专门推理模型的 CoT 并不总是比来自 Llama 和 Gemma 等通用 LLM 的 CoT 更具可重用性或可验证性。

</details>

---

## 5. Toward a Fully Autonomous, AI-Native Particle Accelerator / 迈向完全自主的人工智能原生粒子加速器

**Date**: 2026-02-19 | **arXiv**: [2602.17536v1](http://arxiv.org/abs/2602.17536v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.17536v1)

**Categories**: physics.acc-ph, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

This position paper presents a vision for self-driving particle accelerators that operate autonomously with minimal human intervention. We propose that future facilities be designed through artificial intelligence (AI) co-design, where AI jointly optimizes the accelerator lattice, diagnostics, and science application from inception to maximize performance while enabling autonomous operation. Rather than retrofitting AI onto human-centric systems, we envision facilities designed from the ground up as AI-native platforms. We outline nine critical research thrusts spanning agentic control architectures, knowledge integration, adaptive learning, digital twins, health monitoring, safety frameworks, modular hardware design, multimodal data fusion, and cross-domain collaboration. This roadmap aims to guide the accelerator community toward a future where AI-driven design and operation deliver unprecedented science output and reliability.

本立场文件提出了自动驾驶粒子加速器的愿景，该加速器可以在最少的人为干预下自主运行。我们建议通过人工智能（AI）协同设计来设计未来的设施，其中人工智能从一开始就联合优化加速器晶格、诊断和科学应用，以最大限度地提高性能，同时实现自主操作。我们不是将人工智能改造到以人为中心的系统上，而是设想将设施从头开始设计为人工智能原生平台。我们概述了九个关键研究方向，涵盖代理控制架构、知识集成、自适应学习、数字孪生、健康监测、安全框架、模块化硬件设计、多模式数据融合和跨领域协作。该路线图旨在引导加速器社区走向未来，让人工智能驱动的设计和操作提供前所未有的科学产出和可靠性。

</details>

---

## 6. Jolt Atlas: Verifiable Inference via Lookup Arguments in Zero Knowledge / Jolt Atlas：通过零知识中的查找参数进行可验证的推理

**Date**: 2026-02-19 | **arXiv**: [2602.17452v1](http://arxiv.org/abs/2602.17452v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.17452v1)

**Categories**: cs.CR, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

We present Jolt Atlas, a zero-knowledge machine learning (zkML) framework that extends the Jolt proving system to model inference. Unlike zkVMs (zero-knowledge virtual machines), which emulate CPU instruction execution, Jolt Atlas adapts Jolt's lookup-centric approach and applies it directly to ONNX tensor operations. The ONNX computational model eliminates the need for CPU registers and simplifies memory consistency verification. In addition, ONNX is an open-source, portable format, which makes it easy to share and deploy models across different frameworks, hardware platforms, and runtime environments without requiring framework-specific conversions.   Our lookup arguments, which use sumcheck protocol, are well-suited for non-linear functions -- key building blocks in modern ML. We apply optimisations such as neural teleportation to reduce the size of lookup tables while preserving model accuracy, as well as several tensor-level verification optimisations detailed in this paper. We demonstrate that Jolt Atlas can prove model inference in memory-constrained environments -- a prover property commonly referred to as \textit{streaming}. Furthermore, we discuss how Jolt Atlas achieves zero-knowledge through the BlindFold technique, as introduced in Vega. In contrast to existing zkML frameworks, we show practical proving times for classification, embedding, automated reasoning, and small language models.   Jolt Atlas enables cryptographic verification that can be run on-device, without specialised hardware. The resulting proofs are succinctly verifiable. This makes Jolt Atlas well-suited for privacy-centric and adversarial environments. In a companion work, we outline various use cases of Jolt Atlas, including how it serves as guardrails in agentic commerce and for trustless AI context (often referred to as \textit{AI memory}).

我们推出了 Jolt Atlas，这是一个零知识机器学习 (zkML) 框架，它将 Jolt 证明系统扩展到模型推理。与模拟 CPU 指令执行的 zkVM（零知识虚拟机）不同，Jolt Atlas 采用 Jolt 以查找为中心的方法，并将其直接应用于 ONNX 张量操作。 ONNX 计算模型消除了对 CPU 寄存器的需求，并简化了内存一致性验证。此外，ONNX 是一种开源、可移植的格式，可以轻松地在不同框架、硬件平台和运行时环境之间共享和部署模型，而无需进行特定于框架的转换。   我们的查找参数使用 sumcheck 协议，非常适合非线性函数——现代机器学习中的关键构建块。我们应用神经隐形传态等优化来减小查找表的大小，同时保持模型精度，以及本文中详细介绍的几个张量级验证优化。我们证明 Jolt Atlas 可以在内存受限的环境中证明模型推理——一个通常称为 \textit{streaming} 的证明者属性。此外，我们还讨论了 Jolt Atlas 如何通过 Vega 中介绍的 BlindFold 技术实现零知识。与现有的 zkML 框架相比，我们展示了分类、嵌入、自动推理和小型语言模型的实际证明时间。   Jolt Atlas 支持在设备上运行加密验证，无需专门的硬件。由此产生的证明是可以简洁验证的。这使得 Jolt Atlas 非常适合以隐私为中心的对抗性环境。在配套工作中，我们概述了 Jolt Atlas 的各种用例，包括它如何充当代理商务中的护栏和无需信任的 AI 环境（通常称为 \textit{AI 内存}）。

</details>

---

## 7. WarpRec: Unifying Academic Rigor and Industrial Scale for Responsible, Reproducible, and Efficient Recommendation / WarpRec：将学术严谨性与工业规模相结合，实现负责任、可重复且高效的推荐

**Date**: 2026-02-19 | **arXiv**: [2602.17442v1](http://arxiv.org/abs/2602.17442v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.17442v1)

**Categories**: cs.AI, cs.IR

**Code**: https://github.com/sisinflab/warprec/

<details><summary><b>Abstract / 摘要</b></summary>

Innovation in Recommender Systems is currently impeded by a fractured ecosystem, where researchers must choose between the ease of in-memory experimentation and the costly, complex rewriting required for distributed industrial engines. To bridge this gap, we present WarpRec, a high-performance framework that eliminates this trade-off through a novel, backend-agnostic architecture. It includes 50+ state-of-the-art algorithms, 40 metrics, and 19 filtering and splitting strategies that seamlessly transition from local execution to distributed training and optimization. The framework enforces ecological responsibility by integrating CodeCarbon for real-time energy tracking, showing that scalability need not come at the cost of scientific integrity or sustainability. Furthermore, WarpRec anticipates the shift toward Agentic AI, leading Recommender Systems to evolve from static ranking engines into interactive tools within the Generative AI ecosystem. In summary, WarpRec not only bridges the gap between academia and industry but also can serve as the architectural backbone for the next generation of sustainable, agent-ready Recommender Systems. Code is available at https://github.com/sisinflab/warprec/

目前，推荐系统的创新受到支离破碎的生态系统的阻碍，研究人员必须在内存实验的简易性和分布式工业引擎所需的昂贵且复杂的重写之间做出选择。为了弥补这一差距，我们推出了 WarpRec，这是一个高性能框架，它通过一种新颖的、与后端无关的架构消除了这种权衡。它包括 50 多个最先进的算法、40 个指标以及 19 种过滤和分割策略，可从本地执行无缝过渡到分布式训练和优化。该框架通过集成 CodeCarbon 进行实时能源跟踪来强化生态责任，这表明可扩展性不必以牺牲科学完整性或可持续性为代价。此外，WarpRec 预计向 Agentic AI 的转变，引导推荐系统从静态排名引擎发展为生成 AI 生态系统中的交互式工具。总之，WarpRec 不仅弥合了学术界和工业界之间的差距，而且可以作为下一代可持续的、代理就绪的推荐系统的架构支柱。代码可在 https://github.com/sisinflab/warprec/ 获取

</details>

---

## 8. MedClarify: An information-seeking AI agent for medical diagnosis with case-specific follow-up questions / MedClarify：一种用于医疗诊断的信息搜索人工智能代理，并提供针对具体病例的后续问题

**Date**: 2026-02-19 | **arXiv**: [2602.17308v1](http://arxiv.org/abs/2602.17308v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.17308v1)

**Categories**: cs.AI, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Large language models (LLMs) are increasingly used for diagnostic tasks in medicine. In clinical practice, the correct diagnosis can rarely be immediately inferred from the initial patient presentation alone. Rather, reaching a diagnosis often involves systematic history taking, during which clinicians reason over multiple potential conditions through iterative questioning to resolve uncertainty. This process requires considering differential diagnoses and actively excluding emergencies that demand immediate intervention. Yet, the ability of medical LLMs to generate informative follow-up questions and thus reason over differential diagnoses remains underexplored. Here, we introduce MedClarify, an AI agent for information-seeking that can generate follow-up questions for iterative reasoning to support diagnostic decision-making. Specifically, MedClarify computes a list of candidate diagnoses analogous to a differential diagnosis, and then proactively generates follow-up questions aimed at reducing diagnostic uncertainty. By selecting the question with the highest expected information gain, MedClarify enables targeted, uncertainty-aware reasoning to improve diagnostic performance. In our experiments, we first demonstrate the limitations of current LLMs in medical reasoning, which often yield multiple, similarly likely diagnoses, especially when patient cases are incomplete or relevant information for diagnosis is missing. We then show that our information-theoretic reasoning approach can generate effective follow-up questioning and thereby reduces diagnostic errors by ~27 percentage points (p.p.) compared to a standard single-shot LLM baseline. Altogether, MedClarify offers a path to improve medical LLMs through agentic information-seeking and to thus promote effective dialogues with medical LLMs that reflect the iterative and uncertain nature of real-world clinical reasoning.

大语言模型 (LLM) 越来越多地用于医学诊断任务。在临床实践中，仅根据患者的初始表现很少能够立即推断出正确的诊断。相反，做出诊断通常需要系统地采集病史，在此期间临床医生通过反复询问来推理多种潜在的情况，以解决不确定性。这个过程需要考虑鉴别诊断并积极排除需要立即干预的紧急情况。然而，医学法学硕士提出信息丰富的后续问题并由此推理鉴别诊断的能力仍未得到充分探索。在这里，我们介绍 MedClarify，这是一种用于信息搜索的人工智能代理，可以生成后续问题以进行迭代推理，以支持诊断决策。具体来说，MedClarify 计算类似于鉴别诊断的候选诊断列表，然后主动生成旨在减少诊断不确定性的后续问题。通过选择预期信息增益最高的问题，MedClarify 能够进行有针对性的、不确定性感知的推理，以提高诊断性能。在我们的实验中，我们首先证明了当前法学硕士在医学推理方面的局限性，这通常会产生多种类似的诊断，特别是当患者病例不完整或诊断的相关信息缺失时。然后，我们表明，与标准单次 LLM 基线相比，我们的信息论推理方法可以产生有效的后续提问，从而将诊断错误减少约 27 个百分点 (p.p.)。总而言之，MedClarify 提供了一条通过代理信息寻求来改进医学法学硕士的途径，从而促进与医学法学硕士的有效对话，反映现实世界临床推理的迭代和不确定性。

</details>

---

## 9. Federated Latent Space Alignment for Multi-user Semantic Communications / 多用户语义通信的联合潜在空间对齐

**Date**: 2026-02-19 | **arXiv**: [2602.17271v1](http://arxiv.org/abs/2602.17271v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.17271v1)

**Categories**: cs.IT, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Semantic communication aims to convey meaning for effective task execution, but differing latent representations in AI-native devices can cause semantic mismatches that hinder mutual understanding. This paper introduces a novel approach to mitigating latent space misalignment in multi-agent AI- native semantic communications. In a downlink scenario, we consider an access point (AP) communicating with multiple users to accomplish a specific AI-driven task. Our method implements a protocol that shares a semantic pre-equalizer at the AP and local semantic equalizers at user devices, fostering mutual understanding and task-oriented communication while considering power and complexity constraints. To achieve this, we employ a federated optimization for the decentralized training of the semantic equalizers at the AP and user sides. Numerical results validate the proposed approach in goal-oriented semantic communication, revealing key trade-offs among accuracy, com- munication overhead, complexity, and the semantic proximity of AI-native communication devices.

语义通信旨在传达有效任务执行的含义，但人工智能本机设备中不同的潜在表示可能会导致语义不匹配，从而阻碍相互理解。本文介绍了一种减轻多智能体人工智能语义通信中潜在空间错位的新方法。在下行链路场景中，我们考虑一个接入点 (AP) 与多个用户通信以完成特定的人工智能驱动任务。我们的方法实现了一种协议，该协议在 AP 处共享语义预均衡器，在用户设备处共享本地语义均衡器，从而促进相互理解和面向任务的通信，同时考虑功率和复杂性限制。为了实现这一目标，我们采用联合优化来对 AP 和用户侧的语义均衡器进行分散训练。数值结果验证了所提出的面向目标的语义通信方法，揭示了人工智能原生通信设备的准确性、通信开销、复杂性和语义接近性之间的关键权衡。

</details>

---

## 10. Web Verbs: Typed Abstractions for Reliable Task Composition on the Agentic Web / Web 动词：代理 Web 上可靠任务组合的类型化抽象

**Date**: 2026-02-19 | **arXiv**: [2602.17245v1](http://arxiv.org/abs/2602.17245v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.17245v1)

**Categories**: cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

The Web is evolving from a medium that humans browse to an environment where software agents act on behalf of users. Advances in large language models (LLMs) make natural language a practical interface for goal-directed tasks, yet most current web agents operate on low-level primitives such as clicks and keystrokes. These operations are brittle, inefficient, and difficult to verify. Complementing content-oriented efforts such as NLWeb's semantic layer for retrieval, we argue that the agentic web also requires a semantic layer for web actions. We propose \textbf{Web Verbs}, a web-scale set of typed, semantically documented functions that expose site capabilities through a uniform interface, whether implemented through APIs or robust client-side workflows. These verbs serve as stable and composable units that agents can discover, select, and synthesize into concise programs. This abstraction unifies API-based and browser-based paradigms, enabling LLMs to synthesize reliable and auditable workflows with explicit control and data flow. Verbs can carry preconditions, postconditions, policy tags, and logging support, which improves \textbf{reliability} by providing stable interfaces, \textbf{efficiency} by reducing dozens of steps into a few function calls, and \textbf{verifiability} through typed contracts and checkable traces. We present our vision, a proof-of-concept implementation, and representative case studies that demonstrate concise and robust execution compared to existing agents. Finally, we outline a roadmap for standardization to make verbs deployable and trustworthy at web scale.

Web 正在从人类浏览的媒介发展到软件代理代表用户行事的环境。大语言模型 (LLM) 的进步使自然语言成为目标导向任务的实用界面，但当前大多数 Web 代理都在低级原语（例如点击和击键）上运行。这些操作脆弱、低效且难以验证。作为对面向内容的工作（例如用于检索的 NLWeb 语义层）的补充，我们认为代理 Web 还需要一个用于 Web 操作的语义层。我们提出 \textbf{Web Verbs}，这是一组网络规模的类型化、语义记录的函数，通过统一的接口公开站点功能，无论是通过 API 还是强大的客户端工作流程实现。这些动词作为稳定且可组合的单元，代理可以发现、选择并将其合成为简洁的程序。这种抽象统一了基于 API 和基于浏览器的范例，使法学硕士能够通过显式控制和数据流综合可靠且可审计的工作流程。动词可以携带前置条件、后置条件、策略标签和日志记录支持，通过提供稳定的接口来提高 \textbf{可靠性}，通过将数十个步骤减少为几个函数调用来提高 \textbf{效率}，并通过类型化契约和可检查跟踪来提高 \textbf{可验证性}。我们展示了我们的愿景、概念验证实施以及代表性案例研究，与现有代理相比，这些案例展示了简洁而强大的执行力。最后，我们概述了标准化路线图，以使动词在网络规模上可部署且值得信赖。

</details>

---

## 11. From Labor to Collaboration: A Methodological Experiment Using AI Agents to Augment Research Perspectives in Taiwan's Humanities and Social Sciences / 从劳动到协作：利用人工智能代理增强台湾人文社会科学研究视角的方法论实验

**Date**: 2026-02-19 | **arXiv**: [2602.17221v1](http://arxiv.org/abs/2602.17221v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.17221v1)

**Categories**: cs.AI, cs.CL, cs.CY

<details><summary><b>Abstract / 摘要</b></summary>

Generative AI is reshaping knowledge work, yet existing research focuses predominantly on software engineering and the natural sciences, with limited methodological exploration for the humanities and social sciences. Positioned as a "methodological experiment," this study proposes an AI Agent-based collaborative research workflow (Agentic Workflow) for humanities and social science research. Taiwan's Claude.ai usage data (N = 7,729 conversations, November 2025) from the Anthropic Economic Index (AEI) serves as the empirical vehicle for validating the feasibility of this methodology.   This study operates on two levels: the primary level is the design and validation of a methodological framework - a seven-stage modular workflow grounded in three principles: task modularization, human-AI division of labor, and verifiability, with each stage delineating clear roles for human researchers (research judgment and ethical decisions) and AI Agents (information retrieval and text generation); the secondary level is the empirical analysis of AEI Taiwan data - serving as an operational demonstration of the workflow's application to secondary data research, showcasing both the process and output quality (see Appendix A).   This study contributes by proposing a replicable AI collaboration framework for humanities and social science researchers, and identifying three operational modes of human-AI collaboration - direct execution, iterative refinement, and human-led - through reflexive documentation of the operational process. This taxonomy reveals the irreplaceability of human judgment in research question formulation, theoretical interpretation, contextualized reasoning, and ethical reflection. Limitations including single-platform data, cross-sectional design, and AI reliability risks are acknowledged.

生成式人工智能正在重塑知识工作，但现有的研究主要集中在软件工程和自然科学上，对人文和社会科学的方法论探索有限。本研究定位为“方法论实验”，提出了一种基于AI Agent的人文社会科学研究协作研究工作流程（Agentic Workflow）。来自人类经济指数 (AEI) 的台湾 Claude.ai 使用数据（N = 7,729 次对话，2025 年 11 月）作为验证该方法可行性的实证工具。   这项研究在两个层面上进行：第一层是方法框架的设计和验证——一个七阶段的模块化工作流程，基于三个原则：任务模块化、人类与人工智能的分工和可验证性，每个阶段都为人类研究人员（研究判断和道德决策）和人工智能代理（信息检索和文本生成）描绘了明确的角色；第二层是AEI台湾数据的实证分析——作为工作流程应用于二手数据研究的操作演示，展示过程和输出质量（见附录A）。   这项研究为人文和社会科学研究人员提出了一个可复制的人工智能协作框架，并通过对操作过程的反思性记录，确定了人类与人工智能协作的三种操作模式——直接执行、迭代细化和人类主导。这种分类法揭示了人类判断在研究问题提出、理论解释、情境推理和伦理反思中的不可替代性。承认单一平台数据、横截面设计和人工智能可靠性风险等局限性。

</details>

---

## 12. Agentic Wireless Communication for 6G: Intent-Aware and Continuously Evolving Physical-Layer Intelligence / 6G 代理无线通信：意图感知和不断发展的物理层智能

**Date**: 2026-02-19 | **arXiv**: [2602.17096v1](http://arxiv.org/abs/2602.17096v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.17096v1)

**Categories**: cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

As 6G wireless systems evolve, growing functional complexity and diverse service demands are driving a shift from rule-based control to intent-driven autonomous intelligence. User requirements are no longer captured by a single metric (e.g., throughput or reliability), but by multi-dimensional objectives such as latency sensitivity, energy preference, computational constraints, and service-level requirements. These objectives may also change over time due to environmental dynamics and user-network interactions. Therefore, accurate understanding of both the communication environment and user intent is critical for autonomous and sustainably evolving 6G communications.   Large language models (LLMs), with strong contextual understanding and cross-modal reasoning, provide a promising foundation for intent-aware network agents. Compared with rule-driven or centrally optimized designs, LLM-based agents can integrate heterogeneous information and translate natural-language intents into executable control and configuration decisions.   Focusing on a closed-loop pipeline of intent perception, autonomous decision making, and network execution, this paper investigates agentic AI for the 6G physical layer and its realization pathways. We review representative physical-layer tasks and their limitations in supporting intent awareness and autonomy, identify application scenarios where agentic AI is advantageous, and discuss key challenges and enabling technologies in multimodal perception, cross-layer decision making, and sustainable optimization. Finally, we present a case study of an intent-driven link decision agent, termed AgenCom, which adaptively constructs communication links under diverse user preferences and channel conditions.

随着 6G 无线系统的发展，不断增长的功能复杂性和多样化的服务需求正在推动从基于规则的控制向意图驱动的自主智能的转变。用户需求不再由单一指标（例如吞吐量或可靠性）捕获，而是由多维目标捕获，例如延迟敏感性、能源偏好、计算约束和服务级别要求。由于环境动态和用户网络交互，这些目标也可能随着时间的推移而改变。因此，准确理解通信环境和用户意图对于自主和可持续发展的 6G 通信至关重要。   大型语言模型（LLM）具有强大的上下文理解和跨模式推理能力，为意图感知网络代理提供了有希望的基础。与规则驱动或集中优化的设计相比，基于LLM的代理可以集成异构信息并将自然语言意图转换为可执行的控制和配置决策。   本文聚焦于意图感知、自主决策和网络执行的闭环管道，研究了 6G 物理层的代理人工智能及其实现路径。我们回顾了代表性的物理层任务及其在支持意图感知和自主方面的局限性，确定了代理人工智能具有优势的应用场景，并讨论了多模态感知、跨层决策和可持续优化方面的关键挑战和支持技术。最后，我们提出了一个名为 AgenCom 的意图驱动链路决策代理的案例研究，它可以在不同的用户偏好和信道条件下自适应地构建通信链路。

</details>

---

## 13. Retaining Suboptimal Actions to Follow Shifting Optima in Multi-Agent Reinforcement Learning / 保留次优动作以遵循多智能体强化学习中不断变化的最优值

**Date**: 2026-02-19 | **arXiv**: [2602.17062v1](http://arxiv.org/abs/2602.17062v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.17062v1)

**Categories**: cs.AI

**Code**: https://github.com/hyeon1996/S2Q.

<details><summary><b>Abstract / 摘要</b></summary>

Value decomposition is a core approach for cooperative multi-agent reinforcement learning (MARL). However, existing methods still rely on a single optimal action and struggle to adapt when the underlying value function shifts during training, often converging to suboptimal policies. To address this limitation, we propose Successive Sub-value Q-learning (S2Q), which learns multiple sub-value functions to retain alternative high-value actions. Incorporating these sub-value functions into a Softmax-based behavior policy, S2Q encourages persistent exploration and enables $Q^{\text{tot}}$ to adjust quickly to the changing optima. Experiments on challenging MARL benchmarks confirm that S2Q consistently outperforms various MARL algorithms, demonstrating improved adaptability and overall performance. Our code is available at https://github.com/hyeon1996/S2Q.

价值分解是协作多智能体强化学习（MARL）的核心方法。然而，现有的方法仍然依赖于单一的最优行动，并且当潜在价值函数在训练过程中发生变化时难以适应，通常会收敛到次优策略。为了解决这个限制，我们提出了连续子值 Q 学习（S2Q），它学习多个子值函数以保留替代的高价值动作。 S2Q 将这些子价值函数纳入基于 Softmax 的行为策略中，鼓励持续探索，并使 $Q^{\text{tot}}$ 能够快速调整以适应不断变化的最优值。在具有挑战性的 MARL 基准上进行的实验证实，S2Q 始终优于各种 MARL 算法，展示了改进的适应性和整体性能。我们的代码可在 https://github.com/hyeon1996/S2Q 获取。

</details>

---

## 14. IntentCUA: Learning Intent-level Representations for Skill Abstraction and Multi-Agent Planning in Computer-Use Agents / IntentCUA：学习计算机使用代理中技能抽象和多代理规划的意图级表示

**Date**: 2026-02-19 | **arXiv**: [2602.17049v1](http://arxiv.org/abs/2602.17049v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.17049v1)

**Categories**: cs.AI, cs.HC, cs.RO

<details><summary><b>Abstract / 摘要</b></summary>

Computer-use agents operate over long horizons under noisy perception, multi-window contexts, evolving environment states. Existing approaches, from RL-based planners to trajectory retrieval, often drift from user intent and repeatedly solve routine subproblems, leading to error accumulation and inefficiency. We present IntentCUA, a multi-agent computer-use framework designed to stabilize long-horizon execution through intent-aligned plan memory. A Planner, Plan-Optimizer, and Critic coordinate over shared memory that abstracts raw interaction traces into multi-view intent representations and reusable skills. At runtime, intent prototypes retrieve subgroup-aligned skills and inject them into partial plans, reducing redundant re-planning and mitigating error propagation across desktop applications. In end-to-end evaluations, IntentCUA achieved a 74.83% task success rate with a Step Efficiency Ratio of 0.91, outperforming RL-based and trajectory-centric baselines. Ablations show that multi-view intent abstraction and shared plan memory jointly improve execution stability, with the cooperative multi-agent loop providing the largest gains on long-horizon tasks. These results highlight that system-level intent abstraction and memory-grounded coordination are key to reliable and efficient desktop automation in large, dynamic environments.

计算机使用代理在嘈杂的感知、多窗口环境、不断变化的环境状态下长期运行。现有的方法，从基于强化学习的规划器到轨迹检索，经常偏离用户意图并重复解决常规子问题，导致错误累积和效率低下。我们推出了 IntentCUA，这是一种多代理计算机使用框架，旨在通过意图对齐的计划内存来稳定长期执行。规划器、规划优化器和批评者通过共享内存进行协调，将原始交互跟踪抽象为多视图意图表示和可重用技能。在运行时，意图原型检索子组对齐的技能并将其注入部分计划中，从而减少冗余的重新规划并减轻桌面应用程序之间的错误传播。在端到端评估中，IntentCUA 的任务成功率为 74.83%，步长效率比为 0.91，优于基于 RL 和以轨迹为中心的基线。消融表明，多视图意图抽象和共享计划内存共同提高了执行稳定性，协作多智能体循环为长期任务提供了最大的收益。这些结果强调，系统级意图抽象和基于内存的协调是大型动态环境中可靠且高效的桌面自动化的关键。

</details>

---

## 15. Phase-Aware Mixture of Experts for Agentic Reinforcement Learning / 用于代理强化学习的阶段感知专家组合

**Date**: 2026-02-19 | **arXiv**: [2602.17038v1](http://arxiv.org/abs/2602.17038v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.17038v1)

**Categories**: cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Reinforcement learning (RL) has equipped LLM agents with a strong ability to solve complex tasks. However, existing RL methods normally use a \emph{single} policy network, causing \emph{simplicity bias} where simple tasks occupy most parameters and dominate gradient updates, leaving insufficient capacity for complex tasks. A plausible remedy could be employing the Mixture-of-Experts (MoE) architecture in the policy network, as MoE allows different parameters (experts) to specialize in different tasks, preventing simple tasks from dominating all parameters. However, a key limitation of traditional MoE is its token-level routing, where the router assigns each token to specialized experts, which fragments phase-consistent patterns into scattered expert assignments and thus undermines expert specialization. In this paper, we propose \textbf{Phase-Aware Mixture of Experts (PA-MoE)}. It first features a lightweight \emph{phase router} that learns latent phase boundaries directly from the RL objective without pre-defining phase categories. Then, the phase router allocates temporally consistent assignments to the same expert, allowing experts to preserve phase-specific expertise. Experimental results demonstrate the effectiveness of our proposed PA-MoE.

强化学习（RL）使LLM智能体具备了解决复杂任务的强大能力。然而，现有的强化学习方法通​​常使用 emph{single} 策略网络，导致 emph{simplicitybias}，即简单任务占据大部分参数并主导梯度更新，从而导致复杂任务的容量不足。一种可行的补救措施可能是在策略网络中采用专家混合（MoE）架构，因为 MoE 允许不同的参数（专家）专门从事不同的任务，从而防止简单的任务主导所有参数。然而，传统 MoE 的一个关键限制是其令牌级路由，其中​​路由器将每个令牌分配给专门的专家，这将阶段一致的模式分解为分散的专家分配，从而破坏了专家的专业化。在本文中，我们提出\textbf{相位感知专家混合（PA-MoE）}。它首先具有一个轻量级的 \emph{phase router}，可以直接从 RL 目标中学习潜在的阶段边界，而无需预先定义阶段类别。然后，阶段路由器将时间一致的任务分配给同一专家，从而允许专家保留特定阶段的专业知识。实验结果证明了我们提出的 PA-MoE 的有效性。

</details>

---

## 16. Wink: Recovering from Misbehaviors in Coding Agents / Wink：从编码代理的不当行为中恢复

**Date**: 2026-02-19 | **arXiv**: [2602.17037v1](http://arxiv.org/abs/2602.17037v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.17037v1)

**Categories**: cs.SE, cs.AI, cs.HC, cs.PL

<details><summary><b>Abstract / 摘要</b></summary>

Autonomous coding agents, powered by large language models (LLMs), are increasingly being adopted in the software industry to automate complex engineering tasks. However, these agents are prone to a wide range of misbehaviors, such as deviating from the user's instructions, getting stuck in repetitive loops, or failing to use tools correctly. These failures disrupt the development workflow and often require resource-intensive manual intervention. In this paper, we present a system for automatically recovering from agentic misbehaviors at scale. We first introduce a taxonomy of misbehaviors grounded in an analysis of production traffic, identifying three primary categories: Specification Drift, Reasoning Problems, and Tool Call Failures, which we find occur in about 30% of all agent trajectories.   To address these issues, we developed a lightweight, asynchronous self-intervention system named Wink. Wink observes agent trajectories and provides targeted course-correction guidance to nudge the agent back to a productive path. We evaluated our system on over 10,000 real world agent trajectories and found that it successfully resolves 90% of the misbehaviors that require a single intervention. Furthermore, a live A/B test in our production environment demonstrated that our system leads to a statistically significant reduction in Tool Call Failures, Tokens per Session and Engineer Interventions per Session. We present our experience designing and deploying this system, offering insights into the challenges of building resilient agentic systems at scale.

软件行业越来越多地采用由大型语言模型 (LLM) 提供支持的自主编码代理来自动执行复杂的工程任务。然而，这些代理很容易出现各种不当行为，例如偏离用户的指令、陷入重复循环或无法正确使用工具。这些故障会扰乱开发工作流程，并且通常需要占用大量资源的手动干预。在本文中，我们提出了一种自动从大规模代理不当行为中恢复的系统。我们首先介绍基于生产流量分析的不当行为分类，确定三个主要类别：规范漂移、推理问题和工具调用失败，我们发现这些错误发生在所有代理轨迹的 30% 左右。   为了解决这些问题，我们开发了一个轻量级、异步的自我干预系统，名为 Wink。 Wink 观察代理轨迹并提供有针对性的路线修正指导，以推动代理回到高效的路径。我们根据 10,000 多个现实世界代理轨迹评估了我们的系统，发现它成功解决了 90% 需要单次干预的不当行为。此外，我们生产环境中的实时 A/B 测试表明，我们的系统在统计上显着减少了工具调用失败、每个会话的令牌和每个会话的工程师干预。我们展示了设计和部署该系统的经验，提供了对大规模构建弹性代理系统挑战的见解。

</details>

---

## 17. M2F: Automated Formalization of Mathematical Literature at Scale / M2F：大规模数学文献的自动形式化

**Date**: 2026-02-19 | **arXiv**: [2602.17016v1](http://arxiv.org/abs/2602.17016v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.17016v1)

**Categories**: cs.AI

**Code**: https://github.com/optsuite/ReasBook.git.

<details><summary><b>Abstract / 摘要</b></summary>

Automated formalization of mathematics enables mechanical verification but remains limited to isolated theorems and short snippets. Scaling to textbooks and research papers is largely unaddressed, as it requires managing cross-file dependencies, resolving imports, and ensuring that entire projects compile end-to-end. We present M2F (Math-to-Formal), the first agentic framework for end-to-end, project-scale autoformalization in Lean. The framework operates in two stages. The statement compilation stage splits the document into atomic blocks, orders them via inferred dependencies, and repairs declaration skeletons until the project compiles, allowing placeholders in proofs. The proof repair stage closes these holes under fixed signatures using goal-conditioned local edits. Throughout both stages, M2F keeps the verifier in the loop, committing edits only when toolchain feedback confirms improvement. In approximately three weeks, M2F converts long-form mathematical sources into a project-scale Lean library of 153,853 lines from 479 pages textbooks on real analysis and convex analysis, fully formalized as Lean declarations with accompanying proofs. This represents textbook-scale formalization at a pace that would typically require months or years of expert effort. On FATE-H, we achieve $96\%$ proof success (vs.\ $80\%$ for a strong baseline). Together, these results demonstrate that practical, large-scale automated formalization of mathematical literature is within reach. The full generated Lean code from our runs is available at https://github.com/optsuite/ReasBook.git.

数学的自动形式化可以实现机械验证，但仍然仅限于孤立的定理和简短的片段。扩展到教科书和研究论文基本上没有得到解决，因为它需要管理跨文件依赖关系、解决导入问题并确保整个项目端到端编译。我们提出了 M2F（数学到形式），这是精益中第一个端到端、项目规模自动形式化的代理框架。该框架分两个阶段运行。语句编译阶段将文档拆分为原子块，通过推断的依赖关系对它们进行排序，并修复声明骨架，直到项目编译为止，从而允许在证明中使用占位符。证明修复阶段使用目标条件本地编辑在固定签名下封闭这些漏洞。在这两个阶段中，M2F 都会让验证者随时了解情况，仅在工具链反馈确认改进时才提交编辑。在大约三周内，M2F 将长篇数学源转换为项目规模的精益库，其中包含来自 479 页实分析和凸分析教科书的 153,853 行内容，完全形式化为精益声明并附有证明。这代表了教科书规模的形式化，其速度通常需要数月或数年的专家努力。在 FATE-H 上，我们取得了 96\%$ 的成功证明（对比 80\%$ 的强基线）。总之，这些结果表明数学文献的实用、大规模自动化形式化是可以实现的。从我们的运行中生成的完整精益代码可在 https://github.com/optsuite/ReasBook.git 上找到。

</details>

---

## 18. A Unified Framework for Locality in Scalable MARL / 可扩展 MARL 中局部性的统一框架

**Date**: 2026-02-19 | **arXiv**: [2602.16966v1](http://arxiv.org/abs/2602.16966v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16966v1)

**Categories**: cs.LG, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Scalable Multi-Agent Reinforcement Learning (MARL) is fundamentally challenged by the curse of dimensionality. A common solution is to exploit locality, which hinges on an Exponential Decay Property (EDP) of the value function. However, existing conditions that guarantee the EDP are often conservative, as they are based on worst-case, environment-only bounds (e.g., supremums over actions) and fail to capture the regularizing effect of the policy itself. In this work, we establish that locality can also be a \emph{policy-dependent} phenomenon. Our central contribution is a novel decomposition of the policy-induced interdependence matrix, $H^π$, which decouples the environment's sensitivity to state ($E^{\mathrm{s}}$) and action ($E^{\mathrm{a}}$) from the policy's sensitivity to state ($Π(π)$). This decomposition reveals that locality can be induced by a smooth policy (small $Π(π)$) even when the environment is strongly action-coupled, exposing a fundamental locality-optimality tradeoff. We use this framework to derive a general spectral condition $ρ(E^{\mathrm{s}}+E^{\mathrm{a}}Π(π)) < 1$ for exponential decay, which is strictly tighter than prior norm-based conditions. Finally, we leverage this theory to analyze a provably-sound localized block-coordinate policy improvement framework with guarantees tied directly to this spectral radius.

可扩展多智能体强化学习（MARL）从根本上受到维度灾难的挑战。常见的解决方案是利用局部性，这取决于值函数的指数衰减特性 (EDP)。然而，保证 EDP 的现有条件通常是保守的，因为它们基于最坏情况、仅限环境的界限（例如，对行动的上限），并且无法捕捉政策本身的规范化效果。在这项工作中，我们确定局部性也可以是一种\emph{政策依赖}现象。我们的核心贡献是对策略引发的相互依赖矩阵 $H^π$ 进行新颖的分解，它将环境对状态的敏感性 ($E^{\mathrm{s}}$) 和行动 ($E^{\mathrm{a}}$) 与策略对状态的敏感性 ($Π(π)$) 解耦。这种分解揭示了即使环境是强行动耦合的，局部性也可以通过平滑的策略（小$Π(π)$）来诱导，从而暴露出基本的局部性-最优性权衡。我们使用这个框架导出指数衰减的一般谱条件 $ρ(E^{\mathrm{s}}+E^{\mathrm{a}}Π(π)) < 1$，这比先前基于范数的条件严格严格。最后，我们利用这一理论来分析一个可证明合理的局部块坐标策略改进框架，并提供与该频谱半径直接相关的保证。

</details>

---

## 19. Modeling Distinct Human Interaction in Web Agents / 在 Web 代理中建模独特的人类交互

**Date**: 2026-02-19 | **arXiv**: [2602.17588v1](http://arxiv.org/abs/2602.17588v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.17588v1)

**Categories**: cs.CL, cs.HC

<details><summary><b>Abstract / 摘要</b></summary>

Despite rapid progress in autonomous web agents, human involvement remains essential for shaping preferences and correcting agent behavior as tasks unfold. However, current agentic systems lack a principled understanding of when and why humans intervene, often proceeding autonomously past critical decision points or requesting unnecessary confirmation. In this work, we introduce the task of modeling human intervention to support collaborative web task execution. We collect CowCorpus, a dataset of 400 real-user web navigation trajectories containing over 4,200 interleaved human and agent actions. We identify four distinct patterns of user interaction with agents -- hands-off supervision, hands-on oversight, collaborative task-solving, and full user takeover. Leveraging these insights, we train language models (LMs) to anticipate when users are likely to intervene based on their interaction styles, yielding a 61.4-63.4% improvement in intervention prediction accuracy over base LMs. Finally, we deploy these intervention-aware models in live web navigation agents and evaluate them in a user study, finding a 26.5% increase in user-rated agent usefulness. Together, our results show structured modeling of human intervention leads to more adaptive, collaborative agents.

尽管自主网络代理取得了快速进展，但随着任务的展开，人类的参与对于塑造偏好和纠正代理行为仍然至关重要。然而，当前的代理系统缺乏对人类何时以及为何进行干预的原则性理解，通常会自主地越过关键决策点或请求不必要的确认。在这项工作中，我们介绍了对人工干预进行建模以支持协作 Web 任务执行的任务。我们收集了 CowCorpus，这是一个包含 400 个真实用户 Web 导航轨迹的数据集，其中包含超过 4,200 个交错的人类和代理动作。我们确定了用户与代理交互的四种不同模式——不干涉监督、亲自监督、协作解决任务和完全用户接管。利用这些见解，我们训练语言模型 (LM) 来根据用户的交互风格预测用户何时可能进行干预，与基础 LM 相比，干预预测准确性提高了 61.4-63.4%。最后，我们将这些干预感知模型部署在实时 Web 导航代理中，并在用户研究中对其进行评估，发现用户评价的代理有用性提高了 26.5%。总之，我们的结果表明，人类干预的结构化模型可以产生更具适应性、协作性的代理。

</details>

---

## 20. The Emergence of Lab-Driven Alignment Signatures: A Psychometric Framework for Auditing Latent Bias and Compounding Risk in Generative AI / 实验室驱动的对齐签名的出现：用于审计生成人工智能中潜在偏差和复合风险的心理测量框架

**Date**: 2026-02-19 | **arXiv**: [2602.17127v1](http://arxiv.org/abs/2602.17127v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.17127v1)

**Categories**: cs.CL

<details><summary><b>Abstract / 摘要</b></summary>

As Large Language Models (LLMs) transition from standalone chat interfaces to foundational reasoning layers in multi-agent systems and recursive evaluation loops (LLM-as-a-judge), the detection of durable, provider-level behavioral signatures becomes a critical requirement for safety and governance. Traditional benchmarks measure transient task accuracy but fail to capture stable, latent response policies -- the ``prevailing mindsets'' embedded during training and alignment that outlive individual model versions.   This paper introduces a novel auditing framework that utilizes psychometric measurement theory -- specifically latent trait estimation under ordinal uncertainty -- to quantify these tendencies without relying on ground-truth labels. Utilizing forced-choice ordinal vignettes masked by semantically orthogonal decoys and governed by cryptographic permutation-invariance, the research audits nine leading models across dimensions including Optimization Bias, Sycophancy, and Status-Quo Legitimization.   Using Mixed Linear Models (MixedLM) and Intraclass Correlation Coefficient (ICC) analysis, the research identifies that while item-level framing drives high variance, a persistent ``lab signal'' accounts for significant behavioral clustering. These findings demonstrate that in ``locked-in'' provider ecosystems, latent biases are not merely static errors but compounding variables that risk creating recursive ideological echo chambers in multi-layered AI architectures.

随着大型语言模型 (LLM) 从独立聊天界面过渡到多代理系统和递归评估循环（LLM 作为法官）中的基础推理层，对持久的、提供者级行为签名的检测成为安全和治理的关键要求。传统的基准测试衡量瞬态任务的准确性，但无法捕捉稳定的、潜在的响应策略——训练和调整过程中嵌入的“普遍心态”，比单个模型版本的寿命更长。   本文介绍了一种新颖的审计框架，该框架利用心理测量理论（特别是序数不确定性下的潜在特质估计）来量化这些趋势，而不依赖于真实标签。该研究利用由语义正交诱饵掩盖并受密码排列不变性控制的强制选择序数小插图，审核了跨维度的九个领先模型，包括优化偏差、阿谀奉承和现状合法化。   通过使用混合线性模型 (MixedLM) 和类内相关系数 (ICC) 分析，该研究发现，虽然项目级框架会导致高方差，但持久的“实验室信号”会导致显着的行为聚类。这些发现表明，在“锁定”的提供商生态系统中，潜在偏差不仅仅是静态错误，而且是复合变量，有可能在多层人工智能架构中产生递归意识形态回音室。

</details>

---

## 21. Linear Convergence in Games with Delayed Feedback via Extra Prediction / 通过额外预测实现延迟反馈的游戏中的线性收敛

**Date**: 2026-02-19 | **arXiv**: [2602.17486v1](http://arxiv.org/abs/2602.17486v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.17486v1)

**Categories**: cs.LG, cs.GT, cs.MA, math.OC

<details><summary><b>Abstract / 摘要</b></summary>

Feedback delays are inevitable in real-world multi-agent learning. They are known to severely degrade performance, and the convergence rate under delayed feedback is still unclear, even for bilinear games. This paper derives the rate of linear convergence of Weighted Optimistic Gradient Descent-Ascent (WOGDA), which predicts future rewards with extra optimism, in unconstrained bilinear games. To analyze the algorithm, we interpret it as an approximation of the Extra Proximal Point (EPP), which is updated based on farther future rewards than the classical Proximal Point (PP). Our theorems show that standard optimism (predicting the next-step reward) achieves linear convergence to the equilibrium at a rate $\exp(-Θ(t/m^{5}))$ after $t$ iterations for delay $m$. Moreover, employing extra optimism (predicting farther future reward) tolerates a larger step size and significantly accelerates the rate to $\exp(-Θ(t/(m^{2}\log m)))$. Our experiments also show accelerated convergence driven by the extra optimism and are qualitatively consistent with our theorems. In summary, this paper validates that extra optimism is a promising countermeasure against performance degradation caused by feedback delays.

在现实世界的多智能体学习中，反馈延迟是不可避免的。众所周知，它们会严重降低性能，并且延迟反馈下的收敛速度仍不清楚，即使对于双线性游戏也是如此。本文推导了加权乐观梯度下降-上升（WOGDA）的线性收敛率，它在无约束双线性游戏中以额外乐观的方式预测未来的奖励。为了分析该算法，我们将其解释为额外近点（EPP）的近似值，它是根据比经典近点（PP）更远的未来奖励进行更新的。我们的定理表明，标准乐观主义（预测下一步奖励）在延迟 $m$ 的 $t$ 迭代后以 $\exp(-θ(t/m^{5}))$ 的速率线性收敛到均衡。此外，采用额外的乐观主义（预测更远的未来奖励）可以容忍更大的步长，并显着加快$\exp(-θ(t/(m^{2}\log m)))$的速度。我们的实验还表明，由额外的乐观情绪驱动的加速收敛，并且在质量上与我们的定理一致。总之，本文验证了额外的乐观情绪是针对反馈延迟引起的性能下降的一种有前途的对策。

</details>

---

## 22. Spatio-temporal dual-stage hypergraph MARL for human-centric multimodal corridor traffic signal control / 用于以人为中心的多模式走廊交通信号控制的时空双阶段超图 MARL

**Date**: 2026-02-19 | **arXiv**: [2602.17068v1](http://arxiv.org/abs/2602.17068v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.17068v1)

**Categories**: cs.LG, eess.SY

<details><summary><b>Abstract / 摘要</b></summary>

Human-centric traffic signal control in corridor networks must increasingly account for multimodal travelers, particularly high-occupancy public transportation, rather than focusing solely on vehicle-centric performance. This paper proposes STDSH-MARL (Spatio-Temporal Dual-Stage Hypergraph based Multi-Agent Reinforcement Learning), a scalable multi-agent deep reinforcement learning framework that follows a centralized training and decentralized execution paradigm. The proposed method captures spatio-temporal dependencies through a novel dual-stage hypergraph attention mechanism that models interactions across both spatial and temporal hyperedges. In addition, a hybrid discrete action space is introduced to jointly determine the next signal phase configuration and its corresponding green duration, enabling more adaptive signal timing decisions. Experiments conducted on a corridor network under five traffic scenarios demonstrate that STDSH-MARL consistently improves multimodal performance and provides clear benefits for public transportation priority. Compared with state-of-the-art baseline methods, the proposed approach achieves superior overall performance. Further ablation studies confirm the contribution of each component of STDSH-MARL, with temporal hyperedges identified as the most influential factor driving the observed performance gains.

走廊网络中以人为本的交通信号控制必须越来越多地考虑多式联运旅客，特别是高载客量的公共交通，而不是仅仅关注以车辆为中心的性能。本文提出了 STDSH-MARL（基于时空双阶段超图的多智能体强化学习），这是一种可扩展的多智能体深度强化学习框架，遵循集中式训练和分散式执行范式。所提出的方法通过一种新颖的双阶段超图注意机制来捕获时空依赖性，该机制对空间和时间超边之间的交互进行建模。此外，引入混合离散动作空间来共同确定下一个信号相位配置及其相应的绿灯持续时间，从而实现更自适应的信号定时决策。在五种交通场景下的走廊网络上进行的实验表明，STDSH-MARL 持续提高了多式联运性能，并为公共交通优先提供了明显的好处。与最先进的基线方法相比，所提出的方法实现了优越的整体性能。进一步的消融研究证实了 STDSH-MARL 每个组成部分的贡献，其中时间超边缘被确定为驱动观察到的性能增益的最有影响力的因素。

</details>

---

## 23. Action-Graph Policies: Learning Action Co-dependencies in Multi-Agent Reinforcement Learning / 动作图策略：学习多智能体强化学习中的动作相互依赖性

**Date**: 2026-02-19 | **arXiv**: [2602.17009v1](http://arxiv.org/abs/2602.17009v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.17009v1)

**Categories**: cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Coordinating actions is the most fundamental form of cooperation in multi-agent reinforcement learning (MARL). Successful decentralized decision-making often depends not only on good individual actions, but on selecting compatible actions across agents to synchronize behavior, avoid conflicts, and satisfy global constraints. In this paper, we propose Action Graph Policies (AGP), that model dependencies among agents' available action choices. It constructs, what we call, \textit{coordination contexts}, that enable agents to condition their decisions on global action dependencies. Theoretically, we show that AGPs induce a strictly more expressive joint policy compared to fully independent policies and can realize coordinated joint actions that are provably more optimal than greedy execution even from centralized value-decomposition methods. Empirically, we show that AGP achieves 80-95\% success on canonical coordination tasks with partial observability and anti-coordination penalties, where other MARL methods reach only 10-25\%. We further demonstrate that AGP consistently outperforms these baselines in diverse multi-agent environments.

协调动作是多智能体强化学习（MARL）中最基本的合作形式。成功的去中心化决策通常不仅取决于良好的个体行动，还取决于在代理之间选择兼容的行动以同步行为、避免冲突并满足全局约束。在本文中，我们提出了动作图策略（AGP），它对代理可用动作选择之间的依赖关系进行建模。它构造了我们所说的 \textit{协调上下文}，使代理能够根据全局动作依赖性来调整其决策。从理论上讲，我们表明，与完全独立的策略相比，AGP 会产生严格更具表现力的联合策略，并且可以实现协调的联合行动，即使通过集中的价值分解方法，这些行动也比贪婪执行更优化。根据经验，我们表明 AGP 在具有部分可观察性和反协调惩罚的规范协调任务上取得了 80-95% 的成功，而其他 MARL 方法仅达到 10-25%。我们进一步证明，AGP 在不同的多代理环境中始终优于这些基线。

</details>

---

## 24. Patch-Based Spatial Authorship Attribution in Human-Robot Collaborative Paintings / 人机协作绘画中基于补丁的空间作者归属

**Date**: 2026-02-19 | **arXiv**: [2602.17030v1](http://arxiv.org/abs/2602.17030v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.17030v1)

**Categories**: cs.CV, cs.RO

<details><summary><b>Abstract / 摘要</b></summary>

As agentic AI becomes increasingly involved in creative production, documenting authorship has become critical for artists, collectors, and legal contexts. We present a patch-based framework for spatial authorship attribution within human-robot collaborative painting practice, demonstrated through a forensic case study of one human artist and one robotic system across 15 abstract paintings. Using commodity flatbed scanners and leave-one-painting-out cross-validation, the approach achieves 88.8% patch-level accuracy (86.7% painting-level via majority vote), outperforming texture-based and pretrained-feature baselines (68.0%-84.7%). For collaborative artworks, where ground truth is inherently ambiguous, we use conditional Shannon entropy to quantify stylistic overlap; manually annotated hybrid regions exhibit 64% higher uncertainty than pure paintings (p=0.003), suggesting the model detects mixed authorship rather than classification failure. The trained model is specific to this human-robot pair but provides a methodological grounding for sample-efficient attribution in data-scarce human-AI creative workflows that, in the future, has the potential to extend authorship attribution to any human-robot collaborative painting.

随着代理人工智能越来越多地参与创意生产，记录作者身份对于艺术家、收藏家和法律环境变得至关重要。我们提出了一种基于补丁的框架，用于人机协作绘画实践中的空间作者归属，并通过对一位人类艺术家和一个机器人系统在 15 幅抽象绘画中的法证案例研究进行了演示。使用商用平板扫描仪和留一绘制交叉验证，该方法实现了 88.8% 的补丁级准确率（通过多数投票达到绘画级的 86.7%），优于基于纹理和预训练特征的基线 (68.0%-84.7%)。对于协作艺术作品，其基本事实本质上是不明确的，我们使用条件香农熵来量化风格重叠；手动注释的混合区域表现出比纯绘画高 64% 的不确定性 (p=0.003)，这表明该模型检测的是混合作者身份而不是分类失败。经过训练的模型特定于这对人类-机器人对，但为数据稀缺的人类-人工智能创意工作流程中的样本有效归因提供了方法论基础，在未来，该工作流程有可能将作者归属扩展到任何人类-机器人协作绘画。

</details>

---

## 25. Automating Agent Hijacking via Structural Template Injection / 通过结构模板注入自动化代理劫持

**Date**: 2026-02-18 | **arXiv**: [2602.16958v1](http://arxiv.org/abs/2602.16958v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16958v1)

**Categories**: cs.AI, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Agent hijacking, highlighted by OWASP as a critical threat to the Large Language Model (LLM) ecosystem, enables adversaries to manipulate execution by injecting malicious instructions into retrieved content. Most existing attacks rely on manually crafted, semantics-driven prompt manipulation, which often yields low attack success rates and limited transferability to closed-source commercial models. In this paper, we propose Phantom, an automated agent hijacking framework built upon Structured Template Injection that targets the fundamental architectural mechanisms of LLM agents. Our key insight is that agents rely on specific chat template tokens to separate system, user, assistant, and tool instructions. By injecting optimized structured templates into the retrieved context, we induce role confusion and cause the agent to misinterpret the injected content as legitimate user instructions or prior tool outputs. To enhance attack transferability against black-box agents, Phantom introduces a novel attack template search framework. We first perform multi-level template augmentation to increase structural diversity and then train a Template Autoencoder (TAE) to embed discrete templates into a continuous, searchable latent space. Subsequently, we apply Bayesian optimization to efficiently identify optimal adversarial vectors that are decoded into high-potency structured templates. Extensive experiments on Qwen, GPT, and Gemini demonstrate that our framework significantly outperforms existing baselines in both Attack Success Rate (ASR) and query efficiency. Moreover, we identified over 70 vulnerabilities in real-world commercial products that have been confirmed by vendors, underscoring the practical severity of structured template-based hijacking and providing an empirical foundation for securing next-generation agentic systems.

OWASP 强调代理劫持是对大型语言模型 (LLM) 生态系统的严重威胁，它使攻击者能够通过将恶意指令注入检索到的内容来操纵执行。大多数现有攻击依赖于手动设计、语义驱动的提示操作，这通常会导致攻击成功率较低，并且向闭源商业模型的可移植性有限。在本文中，我们提出了 Phantom，一个基于结构化模板注入构建的自动代理劫持框架，其目标是 LLM 代理的基本架构机制。我们的主要见解是，代理依赖特定的聊天模板令牌来分隔系统、用户、助手和工具指令。通过将优化的结构化模板注入到检索到的上下文中，我们会引起角色混淆，并导致代理将注入的内容误解为合法的用户指令或先前的工具输出。为了增强针对黑盒代理的攻击可转移性，Phantom 引入了一种新颖的攻击模板搜索框架。我们首先执行多级模板增强以增加结构多样性，然后训练模板自动编码器（TAE）将离散模板嵌入到连续的、可搜索的潜在空间中。随后，我们应用贝叶斯优化来有效地识别被解码为高效结构化模板的最佳对抗向量。在 Qwen、GPT 和 Gemini 上进行的大量实验表明，我们的框架在攻击成功率 (ASR) 和查询效率方面均显着优于现有基线。此外，我们还发现了现实商业产品中的 70 多个漏洞，这些漏洞已得到供应商的确认，强调了基于结构化模板的劫持的实际严重性，并为保护下一代代理系统提供了经验基础。

</details>

---

## 26. LLM4Cov: Execution-Aware Agentic Learning for High-coverage Testbench Generation / LLM4Cov：用于高覆盖率测试平台生成的执行感知代理学习

**Date**: 2026-02-18 | **arXiv**: [2602.16953v1](http://arxiv.org/abs/2602.16953v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16953v1)

**Categories**: cs.AI, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Execution-aware LLM agents offer a promising paradigm for learning from tool feedback, but such feedback is often expensive and slow to obtain, making online reinforcement learning (RL) impractical. High-coverage hardware verification exemplifies this challenge due to its reliance on industrial simulators and non-differentiable execution signals. We propose LLM4Cov, an offline agent-learning framework that models verification as memoryless state transitions guided by deterministic evaluators. Building on this formulation, we introduce execution-validated data curation, policy-aware agentic data synthesis, and worst-state-prioritized sampling to enable scalable learning under execution constraints. We further curate a reality-aligned benchmark adapted from an existing verification suite through a revised evaluation protocol. Using the proposed pipeline, a compact 4B-parameter model achieves 69.2% coverage pass rate under agentic evaluation, outperforming its teacher by 5.3% and demonstrating competitive performance against models an order of magnitude larger.

执行感知的 LLM 代理为从工具反馈中学习提供了一种有前景的范例，但此类反馈通常成本高昂且获取缓慢，使得在线强化学习 (RL) 不切实际。高覆盖率硬件验证体现了这一挑战，因为它依赖工业模拟器和不可微分的执行信号。我们提出了 LLM4Cov，一种离线代理学习框架，它将验证建模为由确定性评估器引导的无记忆状态转换。在此基础上，我们引入了执行验证的数据管理、策略感知的代理数据合成和最坏状态优先采样，以在执行约束下实现可扩展的学习。我们通过修订的评估协议进一步策划了一个根据现有验证套件改编的符合现实的基准。使用所提出的管道，紧凑的 4B 参数模型在代理评估下实现了 69.2% 的覆盖通过率，比其老师高出 5.3%，并展示了与大一个数量级的模型相比的竞争性能。

</details>

---

## 27. Discovering Multiagent Learning Algorithms with Large Language Models / 发现具有大型语言模型的多智能体学习算法

**Date**: 2026-02-18 | **arXiv**: [2602.16928v1](http://arxiv.org/abs/2602.16928v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16928v1)

**Categories**: cs.GT, cs.AI, cs.MA

<details><summary><b>Abstract / 摘要</b></summary>

Much of the advancement of Multi-Agent Reinforcement Learning (MARL) in imperfect-information games has historically depended on manual iterative refinement of baselines. While foundational families like Counterfactual Regret Minimization (CFR) and Policy Space Response Oracles (PSRO) rest on solid theoretical ground, the design of their most effective variants often relies on human intuition to navigate a vast algorithmic design space. In this work, we propose the use of AlphaEvolve, an evolutionary coding agent powered by large language models, to automatically discover new multiagent learning algorithms. We demonstrate the generality of this framework by evolving novel variants for two distinct paradigms of game-theoretic learning. First, in the domain of iterative regret minimization, we evolve the logic governing regret accumulation and policy derivation, discovering a new algorithm, Volatility-Adaptive Discounted (VAD-)CFR. VAD-CFR employs novel, non-intuitive mechanisms-including volatility-sensitive discounting, consistency-enforced optimism, and a hard warm-start policy accumulation schedule-to outperform state-of-the-art baselines like Discounted Predictive CFR+. Second, in the regime of population based training algorithms, we evolve training-time and evaluation-time meta strategy solvers for PSRO, discovering a new variant, Smoothed Hybrid Optimistic Regret (SHOR-)PSRO. SHOR-PSRO introduces a hybrid meta-solver that linearly blends Optimistic Regret Matching with a smoothed, temperature-controlled distribution over best pure strategies. By dynamically annealing this blending factor and diversity bonuses during training, the algorithm automates the transition from population diversity to rigorous equilibrium finding, yielding superior empirical convergence compared to standard static meta-solvers.

多智能体强化学习（MARL）在不完美信息游戏中的大部分进步历来依赖于基线的手动迭代细化。虽然像反事实遗憾最小化（CFR）和政策空间响应预言机（PSRO）这样的基础系列有坚实的理论基础，但其最有效变体的设计通常依赖于人类的直觉来驾驭广阔的算法设计空间。在这项工作中，我们建议使用 AlphaEvolve（一种由大型语言模型支持的进化编码代理）来自动发现新的多代理学习算法。我们通过为博弈论学习的两种不同范式发展新的变体来证明该框架的通用性。首先，在迭代后悔最小化领域，我们发展了控制后悔积累和策略推导的逻辑，发现了一种新算法，波动性自适应贴现（VAD-）CFR。 VAD-CFR 采用新颖的、非直观的机制（包括波动敏感贴现、一致性强制乐观主义和硬热启动政策积累计划）来超越贴现预测 CFR+ 等最先进的基准。其次，在基于群体的训练算法体系中，我们改进了 PSRO 的训练时间和评估时间元策略求解器，发现了一个新的变体，平滑混合乐观遗憾 (SHOR-)PSRO。 SHOR-PSRO 引入了一种混合元求解器，可将乐观遗憾匹配与最佳纯策略上的平滑、温度控制分布线性混合。通过在训练期间动态退火该混合因子和多样性奖励，该算法自动从种群多样性过渡到严格的均衡发现，与标准静态元求解器相比，产生卓越的经验收敛性。

</details>

---

## 28. AgentLAB: Benchmarking LLM Agents against Long-Horizon Attacks / AgentLAB：LLM 代理针对长期攻击的基准测试

**Date**: 2026-02-18 | **arXiv**: [2602.16901v1](http://arxiv.org/abs/2602.16901v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16901v1)

**Categories**: cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

LLM agents are increasingly deployed in long-horizon, complex environments to solve challenging problems, but this expansion exposes them to long-horizon attacks that exploit multi-turn user-agent-environment interactions to achieve objectives infeasible in single-turn settings. To measure agent vulnerabilities to such risks, we present AgentLAB, the first benchmark dedicated to evaluating LLM agent susceptibility to adaptive, long-horizon attacks. Currently, AgentLAB supports five novel attack types including intent hijacking, tool chaining, task injection, objective drifting, and memory poisoning, spanning 28 realistic agentic environments, and 644 security test cases. Leveraging AgentLAB, we evaluate representative LLM agents and find that they remain highly susceptible to long-horizon attacks; moreover, defenses designed for single-turn interactions fail to reliably mitigate long-horizon threats. We anticipate that AgentLAB will serve as a valuable benchmark for tracking progress on securing LLM agents in practical settings. The benchmark is publicly available at https://tanqiujiang.github.io/AgentLAB_main.

LLM 代理越来越多地部署在长期、复杂的环境中，以解决具有挑战性的问题，但这种扩展使它们面临长期攻击，这些攻击利用多轮用户-代理-环境交互来实现单轮设置中不可行的目标。为了衡量代理对此类风险的脆弱性，我们推出了 AgentLAB，这是第一个专门用于评估 LLM 代理对自适应、长范围攻击的敏感性的基准。目前，AgentLAB 支持五种新颖的攻击类型，包括意图劫持、工具链、任务注入、目标漂移和内存中毒，涵盖 28 个真实的代理环境和 644 个安全测试用例。利用 AgentLAB，我们评估了具有代表性的 LLM 代理，发现它们仍然非常容易受到长期攻击；此外，为单轮交互设计的防御无法可靠地缓解长期威胁。我们预计 AgentLAB 将成为跟踪实际环境中保护 LLM 代理进展情况的宝贵基准。该基准测试可在 https://tanqiujian.github.io/AgentLAB_main 上公开获取。

</details>

---

## 29. AdaptOrch: Task-Adaptive Multi-Agent Orchestration in the Era of LLM Performance Convergence / AdaptOrch：LLM性能融合时代的任务自适应多代理编排

**Date**: 2026-02-18 | **arXiv**: [2602.16873v1](http://arxiv.org/abs/2602.16873v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16873v1)

**Categories**: cs.MA, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

As large language models from diverse providers converge toward comparable benchmark performance, the traditional paradigm of selecting a single best model per task yields diminishing returns. We argue that orchestration topology -- the structural composition of how multiple agents are coordinated, parallelized, and synthesized -- now dominates system-level performance over individual model capability. We present AdaptOrch, a formal framework for task-adaptive multi-agent orchestration that dynamically selects among four canonical topologies (parallel, sequential, hierarchical, and hybrid) based on task dependency graphs and empirically derived domain characteristics. Our framework introduces three key contributions: (1) a Performance Convergence Scaling Law, formalizing conditions under which orchestration selection outweighs model selection; (2) a Topology Routing Algorithm that maps task decomposition DAGs to optimal orchestration patterns in O(|V| + |E|) time; and (3) an Adaptive Synthesis Protocol with provable termination guarantees and heuristic consistency scoring for parallel agent outputs. We validate AdaptOrch across coding (SWE-bench), reasoning (GPQA), and retrieval-augmented generation tasks, demonstrating that topology-aware orchestration achieves 12-23% improvement over static single-topology baselines, even when using identical underlying models. Our results establish orchestration design as a first-class optimization target independent of model scaling.

随着来自不同提供商的大型语言模型趋向于可比较的基准性能，为每个任务选择单个最佳模型的传统范例会产生收益递减。我们认为，编排拓扑（多个代理如何协调、并行化和综合的结构组成）现在主导着系统级性能，而不是单个模型的能力。我们提出了 AdaptOrch，一个用于任务自适应多代理编排的正式框架，它根据任务依赖图和经验得出的域特征动态地在四种规范拓扑（并行、顺序、分层和混合）中进行选择。我们的框架引入了三个关键贡献：（1）性能收敛缩放法则，正式化编排选择优于模型选择的条件； (2) 拓扑路由算法，在 O(|V| + |E|) 时间内将任务分解 DAG 映射到最佳编排模式； (3) 自适应综合协议，具有可证明的终止保证和并行代理输出的启发式一致性评分。我们跨编码（SWE-bench）、推理（GPQA）和检索增强生成任务验证了 AdaptOrch，证明即使使用相同的底层模型，拓扑感知编排也比静态单拓扑基线实现了 12-23% 的改进。我们的结果将编排设计确立为独立于模型扩展的一流优化目标。

</details>

---

## 30. Overseeing Agents Without Constant Oversight: Challenges and Opportunities / 在没有持续监督的情况下监督代理人：挑战和机遇

**Date**: 2026-02-18 | **arXiv**: [2602.16844v1](http://arxiv.org/abs/2602.16844v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16844v1)

**Categories**: cs.HC, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

To enable human oversight, agentic AI systems often provide a trace of reasoning and action steps. Designing traces to have an informative, but not overwhelming, level of detail remains a critical challenge. In three user studies on a Computer User Agent, we investigate the utility of basic action traces for verification, explore three alternatives via design probes, and test a novel interface's impact on error finding in question-answering tasks. As expected, we find that current practices are cumbersome, limiting their efficacy. Conversely, our proposed design reduced the time participants spent finding errors. However, although participants reported higher levels of confidence in their decisions, their final accuracy was not meaningfully improved. To this end, our study surfaces challenges for human verification of agentic systems, including managing built-in assumptions, users' subjective and changing correctness criteria, and the shortcomings, yet importance, of communicating the agent's process.

为了实现人类监督，代理人工智能系统通常会提供推理和行动步骤的痕迹。设计具有信息丰富但又不至于压倒性的详细程度的迹线仍然是一个严峻的挑战。在关于计算机用户代理的三项用户研究中，我们研究了用于验证的基本动作跟踪的实用性，通过设计探针探索了三种替代方案，并测试了新颖的界面对问答任务中错误发现的影响。正如预期的那样，我们发现当前的做法很麻烦，限制了它们的功效。相反，我们提出的设计减少了参与者查找错误所花费的时间。然而，尽管参与者对自己的决定表现出更高的信心，但他们的最终准确性并没有显着提高。为此，我们的研究提出了对代理系统进行人工验证的挑战，包括管理内置假设、用户的主观和不断变化的正确性标准，以及传达代理过程的缺点和重要性。

</details>

---

## 31. NeuDiff Agent: A Governed AI Workflow for Single-Crystal Neutron Crystallography / NeuDiff Agent：单晶中子晶体学的受控 AI 工作流程

**Date**: 2026-02-18 | **arXiv**: [2602.16812v1](http://arxiv.org/abs/2602.16812v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16812v1)

**Categories**: cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Large-scale facilities increasingly face analysis and reporting latency as the limiting step in scientific throughput, particularly for structurally and magnetically complex samples that require iterative reduction, integration, refinement, and validation. To improve time-to-result and analysis efficiency, NeuDiff Agent is introduced as a governed, tool-using AI workflow for TOPAZ at the Spallation Neutron Source that takes instrument data products through reduction, integration, refinement, and validation to a validated crystal structure and a publication-ready CIF. NeuDiff Agent executes this established pipeline under explicit governance by restricting actions to allowlisted tools, enforcing fail-closed verification gates at key workflow boundaries, and capturing complete provenance for inspection, auditing, and controlled replay. Performance is assessed using a fixed prompt protocol and repeated end-to-end runs with two large language model backends, with user and machine time partitioned and intervention burden and recovery behaviors quantified under gating. In a reference-case benchmark, NeuDiff Agent reduces wall time from 435 minutes (manual) to 86.5(4.7) to 94.4(3.5) minutes (4.6-5.0x faster) while producing a validated CIF with no checkCIF level A or B alerts. These results establish a practical route to deploy agentic AI in facility crystallography while preserving traceability and publication-facing validation requirements.

大型设施越来越多地面临分析和报告延迟作为科学吞吐量的限制步骤，特别是对于需要迭代减少、集成、细化和验证的结构和磁性复杂的样品。为了提高获得结果的时间和分析效率，NeuDiff Agent 被引入作为散裂中子源 TOPAZ 的受控、使用工具的 AI 工作流程，该工作流程通过减少、集成、细化和验证仪器数据产品，得到经过验证的晶体结构和可发布的 CIF。 NeuDiff Agent 在显式治理下执行此已建立的管道，方法是将操作限制在许可名单上的工具，在关键工作流程边界强制执行故障关闭验证门，并捕获完整的来源以进行检查、审计和受控重放。使用固定的提示协议和两个大型语言模型后端的重复端到端运行来评估性能，并划分用户和机器时间，并在门控下量化干预负担和恢复行为。在参考案例基准中，NeuDiff Agent 将挂起时间从 435 分钟（手动）减少到 86.5(4.7) 至 94.4(3.5) 分钟（快 4.6-5.0 倍），同时生成经过验证的 CIF，且没有 checkCIF A 或 B 级警报。这些结果建立了一条在设施晶体学中部署代理人工智能的实用途径，同时保留可追溯性和面向出版物的验证要求。

</details>

---

## 32. Simple Baselines are Competitive with Code Evolution / 简单的基线与代码演进具有竞争力

**Date**: 2026-02-18 | **arXiv**: [2602.16805v1](http://arxiv.org/abs/2602.16805v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16805v1)

**Categories**: cs.AI, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Code evolution is a family of techniques that rely on large language models to search through possible computer programs by evolving or mutating existing code. Many proposed code evolution pipelines show impressive performance but are often not compared to simpler baselines. We test how well two simple baselines do over three domains: finding better mathematical bounds, designing agentic scaffolds, and machine learning competitions. We find that simple baselines match or exceed much more sophisticated methods in all three. By analyzing these results we find various shortcomings in how code evolution is both developed and used. For the mathematical bounds, a problem's search space and domain knowledge in the prompt are chiefly what dictate a search's performance ceiling and efficiency, with the code evolution pipeline being secondary. Thus, the primary challenge in finding improved bounds is designing good search spaces, which is done by domain experts, and not the search itself. When designing agentic scaffolds we find that high variance in the scaffolds coupled with small datasets leads to suboptimal scaffolds being selected, resulting in hand-designed majority vote scaffolds performing best. We propose better evaluation methods that reduce evaluation stochasticity while keeping the code evolution economically feasible. We finish with a discussion of avenues and best practices to enable more rigorous code evolution in future work.

代码演化是一系列技术，依靠大型语言模型通过演化或变异现有代码来搜索可能的计算机程序。许多提出的代码演化管道显示出令人印象深刻的性能，但通常不与更简单的基线进行比较。我们测试两个简单基线在三个领域的表现：寻找更好的数学界限、设计代理支架和机器学习竞赛。我们发现简单的基线在这三种方法中都匹配或超过了更复杂的方法。通过分析这些结果，我们发现代码演化的开发和使用方式存在各种缺陷。对于数学界限，问题的搜索空间和提示中的领域知识主要决定搜索的性能上限和效率，代码演化管道是次要的。因此，寻找改进边界的主要挑战是设计良好的搜索空间，这是由领域专家完成的，而不是搜索本身。在设计代理支架时，我们发现支架的高方差加上小数据集会导致选择次优支架，从而导致手工设计的多数投票支架表现最佳。我们提出了更好的评估方法，可以减少评估随机性，同时保持代码演化在经济上可行。最后，我们讨论了在未来的工作中实现更严格的代码演变的途径和最佳实践。

</details>

---

## 33. Policy Compiler for Secure Agentic Systems / 安全代理系统的策略编译器

**Date**: 2026-02-18 | **arXiv**: [2602.16708v2](http://arxiv.org/abs/2602.16708v2) | **PDF**: [Link](http://arxiv.org/pdf/2602.16708v2)

**Categories**: cs.CR, cs.AI, cs.MA

<details><summary><b>Abstract / 摘要</b></summary>

LLM-based agents are increasingly being deployed in contexts requiring complex authorization policies: customer service protocols, approval workflows, data access restrictions, and regulatory compliance. Embedding these policies in prompts provides no enforcement guarantees. We present PCAS, a Policy Compiler for Agentic Systems that provides deterministic policy enforcement.   Enforcing such policies requires tracking information flow across agents, which linear message histories cannot capture. Instead, PCAS models the agentic system state as a dependency graph capturing causal relationships among events such as tool calls, tool results, and messages. Policies are expressed in a Datalog-derived language, as declarative rules that account for transitive information flow and cross-agent provenance. A reference monitor intercepts all actions and blocks violations before execution, providing deterministic enforcement independent of model reasoning.   PCAS takes an existing agent implementation and a policy specification, and compiles them into an instrumented system that is policy-compliant by construction, with no security-specific restructuring required. We evaluate PCAS on three case studies: information flow policies for prompt injection defense, approval workflows in a multi-agent pharmacovigilance system, and organizational policies for customer service. On customer service tasks, PCAS improves policy compliance from 48% to 93% across frontier models, with zero policy violations in instrumented runs.

基于 LLM 的代理越来越多地部署在需要复杂授权策略的环境中：客户服务协议、审批工作流程、数据访问限制和法规遵从性。将这些策略嵌入提示中并不能提供强制执行保证。我们推出了 PCAS，一种用于代理系统的策略编译器，可提供确定性策略执行。   执行此类策略需要跟踪代理之间的信息流，这是线性消息历史无法捕获的。相反，PCAS 将代理系统状态建模为依赖图，捕获工具调用、工具结果和消息等事件之间的因果关系。策略以 Datalog 派生语言表达，作为说明传递信息流和跨代理来源的声明性规则。参考监视器拦截所有操作并在执行前阻止违规行为，从而提供独立于模型推理的确定性执行。   PCAS 采用现有的代理实现和策略规范，并将它们编译成一个通过构建符合策略的仪表化系统，无需进行特定于安全的重组。我们通过三个案例研究评估 PCAS：即时注射防御的信息流策略、多代理药物警戒系统中的审批工作流程以及客户服务的组织策略。在客户服务任务中，PCAS 将前沿模型的策略合规性从 48% 提高到 93%，并且在仪器化运行中实现了零策略违规。

</details>

---

## 34. RoboGene: Boosting VLA Pre-training via Diversity-Driven Agentic Framework for Real-World Task Generation / RoboGene：通过多样性驱动的代理框架促进真实世界任务生成的 VLA 预训练

**Date**: 2026-02-18 | **arXiv**: [2602.16444v2](http://arxiv.org/abs/2602.16444v2) | **PDF**: [Link](http://arxiv.org/pdf/2602.16444v2)

**Categories**: cs.RO, cs.AI, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

The pursuit of general-purpose robotic manipulation is hindered by the scarcity of diverse, real-world interaction data. Unlike data collection from web in vision or language, robotic data collection is an active process incurring prohibitive physical costs. Consequently, automated task curation to maximize data value remains a critical yet under-explored challenge. Existing manual methods are unscalable and biased toward common tasks, while off-the-shelf foundation models often hallucinate physically infeasible instructions. To address this, we introduce RoboGene, an agentic framework designed to automate the generation of diverse, physically plausible manipulation tasks across single-arm, dual-arm, and mobile robots. RoboGene integrates three core components: diversity-driven sampling for broad task coverage, self-reflection mechanisms to enforce physical constraints, and human-in-the-loop refinement for continuous improvement. We conduct extensive quantitative analysis and large-scale real-world experiments, collecting datasets of 18k trajectories and introducing novel metrics to assess task quality, feasibility, and diversity. Results demonstrate that RoboGene significantly outperforms state-of-the-art foundation models (e.g., GPT-4o, Gemini 2.5 Pro). Furthermore, real-world experiments show that VLA models pre-trained with RoboGene achieve higher success rates and superior generalization, underscoring the importance of high-quality task generation. Our project is available at https://robogene-boost-vla.github.io.

对通用机器人操作的追求因缺乏多样化的现实世界交互数据而受到阻碍。与通过视觉或语言从网络收集数据不同，机器人数据收集是一个主动过程，会产生高昂的物理成本。因此，实现数据价值最大化的自动化任务管理仍然是一个关键但尚未充分探索的挑战。现有的手动方法不可扩展并且偏向于常见任务，而现成的基础模型通常会产生物理上不可行的指令。为了解决这个问题，我们引入了 RoboGene，这是一个代理框架，旨在自动生成单臂、双臂和移动机器人上多样化的、物理上合理的操作任务。 RoboGene 集成了三个核心组件：用于广泛任务覆盖的多样性驱动采样、用于强制物理约束的自我反思机制，以及用于持续改进的人机循环细化。我们进行了广泛的定量分析和大规模的现实世界实验，收集 18k 轨迹的数据集并引入新颖的指标来评估任务质量、可行性和多样性。结果表明，RoboGene 的性能显着优于最先进的基础模型（例如 GPT-4o、Gemini 2.5 Pro）。此外，现实世界的实验表明，使用 RoboGene 预训练的 VLA 模型实现了更高的成功率和卓越的泛化能力，强调了高质量任务生成的重要性。我们的项目可在 https://robogene-boost-vla.github.io 上获取。

</details>

---

## 35. Multi-Agent Lipschitz Bandits / 多代理 Lipschitz Bandits

**Date**: 2026-02-18 | **arXiv**: [2602.16965v1](http://arxiv.org/abs/2602.16965v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16965v1)

**Categories**: cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

We study the decentralized multi-player stochastic bandit problem over a continuous, Lipschitz-structured action space where hard collisions yield zero reward. Our objective is to design a communication-free policy that maximizes collective reward, with coordination costs that are independent of the time horizon $T$. We propose a modular protocol that first solves the multi-agent coordination problem -- identifying and seating players on distinct high-value regions via a novel maxima-directed search -- and then decouples the problem into $N$ independent single-player Lipschitz bandits. We establish a near-optimal regret bound of $\tilde{O}(T^{(d+1)/(d+2)})$ plus a $T$-independent coordination cost, matching the single-player rate. To our knowledge, this is the first framework providing such guarantees, and it extends to general distance-threshold collision models.

我们研究了连续的、Lipschitz 结构的动作空间中的分散式多人随机强盗问题，其中硬碰撞产生零奖励。我们的目标是设计一种无需沟通的政策，最大限度地提高集体奖励，协调成本与时间范围 $T$ 无关。我们提出了一个模块化协议，首先解决多智能体协调问题——通过新颖的最大值定向搜索来识别不同高价值区域的玩家并就座——然后将问题分解为$N$独立的单人Lipschitz bandits。我们建立了一个近乎最优的后悔界限 $\tilde{O}(T^{(d+1)/(d+2)})$ 加上与单人游戏率相匹配的与 $T$ 无关的协调成本。据我们所知，这是第一个提供此类保证的框架，并且它扩展到一般距离阈值碰撞模型。

</details>

---

## 36. Smooth trajectory generation and hybrid B-splines-Quaternions based tool path interpolation for a 3T1R parallel kinematic milling robot / 3T1R 并联运动铣削机器人的平滑轨迹生成和基于混合 B 样条四元数的刀具路径插补

**Date**: 2026-02-18 | **arXiv**: [2602.16758v1](http://arxiv.org/abs/2602.16758v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16758v1)

**Categories**: cs.RO, eess.SY

<details><summary><b>Abstract / 摘要</b></summary>

This paper presents a smooth trajectory generation method for a four-degree-of-freedom parallel kinematic milling robot. The proposed approach integrates B-spline and Quaternion interpolation techniques to manage decoupled position and orientation data points. The synchronization of orientation and arc-length-parameterized position data is achieved through the fitting of smooth piece-wise Bezier curves, which describe the non-linear relationship between path length and tool orientation, solved via sequential quadratic programming. By leveraging the convex hull properties of Bezier curves, the method ensures spatial and temporal separation constraints for multi-agent trajectory generation. Unit quaternions are employed for orientation interpolation, providing a robust and efficient representation that avoids gimbal lock and facilitates smooth, continuous rotation. Modifier polynomials are used for position interpolation. Temporal trajectories are optimized using minimum jerk, time-optimal piece-wise Bezier curves in two stages: task space followed by joint space, implemented on a low-cost microcontroller. Experimental results demonstrate that the proposed method offers enhanced accuracy, reduced velocity fluctuations, and computational efficiency compared to conventional interpolation methods.

本文提出了一种四自由度并联运动铣削机器人的平滑轨迹生成方法。所提出的方法集成了 B 样条和四元数插值技术来管理解耦的位置和方向数据点。方向和弧长参数化位置数据的同步是通过平滑分段贝塞尔曲线的拟合来实现的，该曲线描述了路径长度和工具方向之间的非线性关系，并通过顺序二次规划求解。通过利用贝塞尔曲线的凸包特性，该方法确保了多智能体轨迹生成的空间和时间分离约束。采用单位四元数进行方向插值，提供稳健且高效的表示，避免万向节锁定并促进平滑、连续的旋转。修正多项式用于位置插值。使用最小加加速度、时间最优分段贝塞尔曲线分两个阶段对时间轨迹进行优化：任务空间，然后是关节空间，在低成本微控制器上实现。实验结果表明，与传统插值方法相比，所提出的方法具有更高的精度、更低的速度波动和计算效率。

</details>

---



</details>

<details><summary><b>2026-02-19 (25 papers)</b></summary>

# arXiv Agent Papers - 2026-02-19

**Paper Count**: 25

---

## 1. Policy Compiler for Secure Agentic Systems / 安全代理系统的策略编译器

**Date**: 2026-02-18 | **arXiv**: [2602.16708v1](http://arxiv.org/abs/2602.16708v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16708v1)

**Categories**: cs.CR, cs.AI, cs.MA

<details><summary><b>Abstract / 摘要</b></summary>

LLM-based agents are increasingly being deployed in contexts requiring complex authorization policies: customer service protocols, approval workflows, data access restrictions, and regulatory compliance. Embedding these policies in prompts provides no enforcement guarantees. We present PCAS, a Policy Compiler for Agentic Systems that provides deterministic policy enforcement.   Enforcing such policies requires tracking information flow across agents, which linear message histories cannot capture. Instead, PCAS models the agentic system state as a dependency graph capturing causal relationships among events such as tool calls, tool results, and messages. Policies are expressed in a Datalog-derived language, as declarative rules that account for transitive information flow and cross-agent provenance. A reference monitor intercepts all actions and blocks violations before execution, providing deterministic enforcement independent of model reasoning.   PCAS takes an existing agent implementation and a policy specification, and compiles them into an instrumented system that is policy-compliant by construction, with no security-specific restructuring required. We evaluate PCAS on three case studies: information flow policies for prompt injection defense, approval workflows in a multi-agent pharmacovigilance system, and organizational policies for customer service. On customer service tasks, PCAS improves policy compliance from 48% to 93% across frontier models, with zero policy violations in instrumented runs.

基于 LLM 的代理越来越多地部署在需要复杂授权策略的环境中：客户服务协议、审批工作流程、数据访问限制和法规遵从性。将这些策略嵌入提示中并不能提供强制执行保证。我们推出了 PCAS，一种用于代理系统的策略编译器，可提供确定性策略执行。   执行此类策略需要跟踪代理之间的信息流，这是线性消息历史无法捕获的。相反，PCAS 将代理系统状态建模为依赖图，捕获工具调用、工具结果和消息等事件之间的因果关系。策略以 Datalog 派生语言表达，作为说明传递信息流和跨代理来源的声明性规则。参考监视器拦截所有操作并在执行前阻止违规行为，从而提供独立于模型推理的确定性执行。   PCAS 采用现有的代理实现和策略规范，并将它们编译成一个通过构建符合策略的仪表化系统，无需进行特定于安全的重组。我们通过三个案例研究评估 PCAS：即时注射防御的信息流策略、多代理药物警戒系统中的审批工作流程以及客户服务的组织策略。在客户服务任务中，PCAS 将前沿模型的策略合规性从 48% 提高到 93%，并且在仪器化运行中实现了零策略违规。

</details>

---

## 2. Towards a Science of AI Agent Reliability / 迈向人工智能代理可靠性科学

**Date**: 2026-02-18 | **arXiv**: [2602.16666v1](http://arxiv.org/abs/2602.16666v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16666v1)

**Categories**: cs.AI, cs.CY, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

AI agents are increasingly deployed to execute important tasks. While rising accuracy scores on standard benchmarks suggest rapid progress, many agents still continue to fail in practice. This discrepancy highlights a fundamental limitation of current evaluations: compressing agent behavior into a single success metric obscures critical operational flaws. Notably, it ignores whether agents behave consistently across runs, withstand perturbations, fail predictably, or have bounded error severity. Grounded in safety-critical engineering, we provide a holistic performance profile by proposing twelve concrete metrics that decompose agent reliability along four key dimensions: consistency, robustness, predictability, and safety. Evaluating 14 agentic models across two complementary benchmarks, we find that recent capability gains have only yielded small improvements in reliability. By exposing these persistent limitations, our metrics complement traditional evaluations while offering tools for reasoning about how agents perform, degrade, and fail.

人工智能代理越来越多地被部署来执行重要任务。虽然标准基准的准确性分数不断上升表明进步很快，但许多代理在实践中仍然失败。这种差异凸显了当前评估的根本局限性：将代理行为压缩为单个成功指标会掩盖关键的操作缺陷。值得注意的是，它忽略了代理在运行中的行为是否一致、是否能承受扰动、是否可预测地失败或是否具有有限的错误严重性。基于安全关键工程，我们提出了十二个具体指标，将代理可靠性分解为四个关键维度：一致性、稳健性、可预测性和安全性，从而提供整体性能概况。通过两个互补基准评估 14 个代理模型，我们发现最近的能力提升仅在可靠性方面带来了微小的改进。通过揭示这些持续存在的局限性，我们的指标补充了传统的评估，同时提供了推理代理如何执行、降级和失败的工具。

</details>

---

## 3. DataJoint 2.0: A Computational Substrate for Agentic Scientific Workflows / DataJoint 2.0：代理科学工作流程的计算基础

**Date**: 2026-02-18 | **arXiv**: [2602.16585v1](http://arxiv.org/abs/2602.16585v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16585v1)

**Categories**: cs.DB, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Operational rigor determines whether human-agent collaboration succeeds or fails. Scientific data pipelines need the equivalent of DevOps -- SciOps -- yet common approaches fragment provenance across disconnected systems without transactional guarantees. DataJoint 2.0 addresses this gap through the relational workflow model: tables represent workflow steps, rows represent artifacts, foreign keys prescribe execution order. The schema specifies not only what data exists but how it is derived -- a single formal system where data structure, computational dependencies, and integrity constraints are all queryable, enforceable, and machine-readable. Four technical innovations extend this foundation: object-augmented schemas integrating relational metadata with scalable object storage, semantic matching using attribute lineage to prevent erroneous joins, an extensible type system for domain-specific formats, and distributed job coordination designed for composability with external orchestration. By unifying data structure, data, and computational transformations, DataJoint creates a substrate for SciOps where agents can participate in scientific workflows without risking data corruption.

操作的严谨性决定了人机协作的成败。科学数据管道需要相当于 DevOps（SciOps）的东西，但常见的方法会在没有事务保证的情况下跨断开连接的系统碎片化来源。 DataJoint 2.0 通过关系工作流模型解决了这一差距：表代表工作流步骤，行代表工件，外键规定执行顺序。该模式不仅指定存在哪些数据，还指定数据的派生方式——一个单一的正式系统，其中数据结构、计算依赖性和完整性约束都是可查询、可执行和机器可读的。四项技术创新扩展了这一基础：将关系元数据与可扩展对象存储集成的对象增强模式、使用属性沿袭来防止错误连接的语义匹配、用于特定于域的格式的可扩展类型系统以及为与外部编排的可组合性而设计的分布式作业协调。通过统一数据结构、数据和计算转换，DataJoint 为 SciOps 创建了一个基础，代理可以在其中参与科学工作流程，而不会冒数据损坏的风险。

</details>

---

## 4. MerLean: An Agentic Framework for Autoformalization in Quantum Computation / MerLean：量子计算中自动形式化的代理框架

**Date**: 2026-02-18 | **arXiv**: [2602.16554v1](http://arxiv.org/abs/2602.16554v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16554v1)

**Categories**: cs.LO, cs.AI, cs.ET, quant-ph

<details><summary><b>Abstract / 摘要</b></summary>

We introduce MerLean, a fully automated agentic framework for autoformalization in quantum computation. MerLean extracts mathematical statements from \LaTeX{} source files, formalizes them into verified Lean~4 code built on Mathlib, and translates the result back into human-readable \LaTeX{} for semantic review. We evaluate MerLean on three theoretical quantum computing papers producing 2,050 Lean declarations from 114 statements in total. MerLean achieves end-to-end formalization on all three papers, reducing the verification burden to only the newly introduced definitions and axioms. Our results demonstrate that agentic autoformalization can scale to frontier research, offering both a practical tool for machine-verified peer review and a scalable engine for mining high-quality synthetic data to train future reasoning models. Our approach can also be generalized to any other rigorous research in mathematics and theoretical physics.

我们介绍 MerLean，一个用于量子计算中自动形式化的全自动代理框架。 MerLean 从 \LaTeX{} 源文件中提取数学语句，将其形式化为基于 Mathlib 构建的经过验证的 Lean~4 代码，并将结果转换回人类可读的 \LaTeX{} 进行语义审查。我们根据三篇理论量子计算论文对 MerLean 进行了评估，总共从 114 条声明中生成了 2,050 条精益声明。 MerLean 在所有三篇论文上实现了端到端形式化，将验证负担减少到仅新引入的定义和公理。我们的结果表明，代理自动形式化可以扩展到前沿研究，既提供了用于机器验证同行评审的实用工具，也提供了用于挖掘高质量合成数据以训练未来推理模型的可扩展引擎。我们的方法也可以推广到数学和理论物理领域的任何其他严格研究。

</details>

---

## 5. Recursive language models for jailbreak detection: a procedural defense for tool-augmented agents / 用于越狱检测的递归语言模型：工具增强代理的程序防御

**Date**: 2026-02-18 | **arXiv**: [2602.16520v1](http://arxiv.org/abs/2602.16520v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16520v1)

**Categories**: cs.CR, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Jailbreak prompts are a practical and evolving threat to large language models (LLMs), particularly in agentic systems that execute tools over untrusted content. Many attacks exploit long-context hiding, semantic camouflage, and lightweight obfuscations that can evade single-pass guardrails. We present RLM-JB, an end-to-end jailbreak detection framework built on Recursive Language Models (RLMs), in which a root model orchestrates a bounded analysis program that transforms the input, queries worker models over covered segments, and aggregates evidence into an auditable decision. RLM-JB treats detection as a procedure rather than a one-shot classification: it normalizes and de-obfuscates suspicious inputs, chunks text to reduce context dilution and guarantee coverage, performs parallel chunk screening, and composes cross-chunk signals to recover split-payload attacks. On AutoDAN-style adversarial inputs, RLM-JB achieves high detection effectiveness across three LLM backends (ASR/Recall 92.5-98.0%) while maintaining very high precision (98.99-100%) and low false positive rates (0.0-2.0%), highlighting a practical sensitivity-specificity trade-off as the screening backend changes.

越狱提示对于大型语言模型 (LLM) 来说是一种实际且不断演变的威胁，特别是在对不受信任的内容执行工具的代理系统中。许多攻击利用长上下文隐藏、语义伪装和轻量级混淆来逃避单通道护栏。我们提出了 RLM-JB，这是一个基于递归语言模型 (RLM) 构建的端到端越狱检测框架，其中根模型编排了一个有界分析程序，该程序可以转换输入、在覆盖的部分上查询工作模型，并将证据聚合成可审计的决策。 RLM-JB 将检测视为一个过程而不是一次性分类：它对可疑输入进行标准化和反混淆，对文本进行分块以减少上下文稀释并保证覆盖范围，执行并行块筛选，并组合跨块信号以恢复拆分有效负载攻击。在 AutoDAN 式的对抗性输入上，RLM-JB 在三个 LLM 后端实现了高检测效率（ASR/召回率 92.5-98.0%），同时保持非常高的精度（98.99-100%）和低误报率（0.0-2.0%），突出了随着筛选后端变化的实际灵敏度-特异性权衡。

</details>

---

## 6. Team of Thoughts: Efficient Test-time Scaling of Agentic Systems through Orchestrated Tool Calling / 思想团队：通过协调的工具调用有效扩展代理系统的测试时间

**Date**: 2026-02-18 | **arXiv**: [2602.16485v1](http://arxiv.org/abs/2602.16485v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16485v1)

**Categories**: cs.CL, cs.AI, cs.MA

<details><summary><b>Abstract / 摘要</b></summary>

Existing Multi-Agent Systems (MAS) typically rely on static, homogeneous model configurations, limiting their ability to exploit the distinct strengths of differently post-trained models. To address this, we introduce Team-of-Thoughts, a novel MAS architecture that leverages the complementary capabilities of heterogeneous agents via an orchestrator-tool paradigm. Our framework introduces two key mechanisms to optimize performance: (1) an orchestrator calibration scheme that identifies models with superior coordination capabilities, and (2) a self-assessment protocol where tool agents profile their own domain expertise to account for variations in post-training skills. During inference, the orchestrator dynamically activates the most suitable tool agents based on these proficiency profiles. Experiments on five reasoning and code generation benchmarks show that Team-of-Thoughts delivers consistently superior task performance. Notably, on AIME24 and LiveCodeBench, our approach achieves accuracies of 96.67% and 72.53%, respectively, substantially outperforming homogeneous role-play baselines, which score 80% and 65.93%.

现有的多智能体系统（MAS）通常依赖于静态、同质的模型配置，限制了它们利用不同后训练模型的独特优势的能力。为了解决这个问题，我们引入了 Team-of-Thoughts，这是一种新颖的 MAS 架构，它通过协调器工具范例利用异构代理的互补功能。我们的框架引入了两个关键机制来优化性能：(1) 协调器校准方案，用于识别具有卓越协调能力的模型；(2) 自我评估协议，工具代理可以在其中描述自己的领域专业知识，以考虑培训后技能的变化。在推理过程中，协调器根据这些熟练程度动态激活最合适的工具代理。对五个推理和代码生成基准的实验表明，Team-of-Thoughts 始终提供卓越的任务性能。值得注意的是，在 AIME24 和 LiveCodeBench 上，我们的方法分别实现了 96.67% 和 72.53% 的准确率，大大优于同质角色扮演基线（得分为 80% 和 65.93%）。

</details>

---

## 7. RoboGene: Boosting VLA Pre-training via Diversity-Driven Agentic Framework for Real-World Task Generation / RoboGene：通过多样性驱动的代理框架促进真实世界任务生成的 VLA 预训练

**Date**: 2026-02-18 | **arXiv**: [2602.16444v1](http://arxiv.org/abs/2602.16444v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16444v1)

**Categories**: cs.RO, cs.AI, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

The pursuit of general-purpose robotic manipulation is hindered by the scarcity of diverse, real-world interaction data. Unlike data collection from web in vision or language, robotic data collection is an active process incurring prohibitive physical costs. Consequently, automated task curation to maximize data value remains a critical yet under-explored challenge. Existing manual methods are unscalable and biased toward common tasks, while off-the-shelf foundation models often hallucinate physically infeasible instructions. To address this, we introduce RoboGene, an agentic framework designed to automate the generation of diverse, physically plausible manipulation tasks across single-arm, dual-arm, and mobile robots. RoboGene integrates three core components: diversity-driven sampling for broad task coverage, self-reflection mechanisms to enforce physical constraints, and human-in-the-loop refinement for continuous improvement. We conduct extensive quantitative analysis and large-scale real-world experiments, collecting datasets of 18k trajectories and introducing novel metrics to assess task quality, feasibility, and diversity. Results demonstrate that RoboGene significantly outperforms state-of-the-art foundation models (e.g., GPT-4o, Gemini 2.5 Pro). Furthermore, real-world experiments show that VLA models pre-trained with RoboGene achieve higher success rates and superior generalization, underscoring the importance of high-quality task generation. Our project is available at https://robogene-boost-vla.github.io.

对通用机器人操作的追求因缺乏多样化的现实世界交互数据而受到阻碍。与通过视觉或语言从网络收集数据不同，机器人数据收集是一个主动过程，会产生高昂的物理成本。因此，实现数据价值最大化的自动化任务管理仍然是一个关键但尚未充分探索的挑战。现有的手动方法不可扩展并且偏向于常见任务，而现成的基础模型通常会产生物理上不可行的指令。为了解决这个问题，我们引入了 RoboGene，这是一个代理框架，旨在自动生成单臂、双臂和移动机器人上多样化的、物理上合理的操作任务。 RoboGene 集成了三个核心组件：用于广泛任务覆盖的多样性驱动采样、用于强制物理约束的自我反思机制，以及用于持续改进的人机循环细化。我们进行了广泛的定量分析和大规模的现实世界实验，收集 18k 轨迹的数据集并引入新颖的指标来评估任务质量、可行性和多样性。结果表明，RoboGene 的性能显着优于最先进的基础模型（例如 GPT-4o、Gemini 2.5 Pro）。此外，现实世界的实验表明，使用 RoboGene 预训练的 VLA 模型实现了更高的成功率和卓越的泛化能力，强调了高质量任务生成的重要性。我们的项目可在 https://robogene-boost-vla.github.io 上获取。

</details>

---

## 8. Causally-Guided Automated Feature Engineering with Multi-Agent Reinforcement Learning / 具有多智能体强化学习的因果引导自动化特征工程

**Date**: 2026-02-18 | **arXiv**: [2602.16435v1](http://arxiv.org/abs/2602.16435v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16435v1)

**Categories**: cs.AI, cs.LG, cs.MA

<details><summary><b>Abstract / 摘要</b></summary>

Automated feature engineering (AFE) enables AI systems to autonomously construct high-utility representations from raw tabular data. However, existing AFE methods rely on statistical heuristics, yielding brittle features that fail under distribution shift. We introduce CAFE, a framework that reformulates AFE as a causally-guided sequential decision process, bridging causal discovery with reinforcement learning-driven feature construction. Phase I learns a sparse directed acyclic graph over features and the target to obtain soft causal priors, grouping features as direct, indirect, or other based on their causal influence with respect to the target. Phase II uses a cascading multi-agent deep Q-learning architecture to select causal groups and transformation operators, with hierarchical reward shaping and causal group-level exploration strategies that favor causally plausible transformations while controlling feature complexity. Across 15 public benchmarks (classification with macro-F1; regression with inverse relative absolute error), CAFE achieves up to 7% improvement over strong AFE baselines, reduces episodes-to-convergence, and delivers competitive time-to-target. Under controlled covariate shifts, CAFE reduces performance drop by ~4x relative to a non-causal multi-agent baseline, and produces more compact feature sets with more stable post-hoc attributions. These findings underscore that causal structure, used as a soft inductive prior rather than a rigid constraint, can substantially improve the robustness and efficiency of automated feature engineering.

自动化特征工程 (AFE) 使 AI 系统能够从原始表格数据自主构建高实用性表示。然而，现有的 AFE 方法依赖于统计启发法，产生在分布偏移下失败的脆弱特征。我们引入了 CAFE，这是一个框架，它将 AFE 重新表述为因果引导的顺序决策过程，将因果发现与强化学习驱动的特征构建联系起来。第一阶段学习特征和目标上的稀疏有向无环图，以获得软因果先验，根据特征对目标的因果影响将特征分组为直接、间接或其他。第二阶段使用级联多智能体深度 Q 学习架构来选择因果组和转换算子，并采用分层奖励塑造和因果组级探索策略，有利于因果合理的转换，同时控制特征复杂性。在 15 个公共基准（宏观 F1 分类；逆相对绝对误差回归）中，CAFE 比强大的 AFE 基线提高了 7%，减少了收敛事件，并提供了具有竞争力的目标时间。在受控协变量变化下，CAFE 相对于非因果多智能体基线将性能下降减少了约 4 倍，并生成具有更稳定的事后归因的更紧凑的特征集。这些发现强调，因果结构用作软归纳先验而不是刚性约束，可以显着提高自动化特征工程的稳健性和效率。

</details>

---

## 9. Multi-agent cooperation through in-context co-player inference / 通过上下文中的共同玩家推理进行多智能体合作

**Date**: 2026-02-18 | **arXiv**: [2602.16301v1](http://arxiv.org/abs/2602.16301v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16301v1)

**Categories**: cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Achieving cooperation among self-interested agents remains a fundamental challenge in multi-agent reinforcement learning. Recent work showed that mutual cooperation can be induced between "learning-aware" agents that account for and shape the learning dynamics of their co-players. However, existing approaches typically rely on hardcoded, often inconsistent, assumptions about co-player learning rules or enforce a strict separation between "naive learners" updating on fast timescales and "meta-learners" observing these updates. Here, we demonstrate that the in-context learning capabilities of sequence models allow for co-player learning awareness without requiring hardcoded assumptions or explicit timescale separation. We show that training sequence model agents against a diverse distribution of co-players naturally induces in-context best-response strategies, effectively functioning as learning algorithms on the fast intra-episode timescale. We find that the cooperative mechanism identified in prior work-where vulnerability to extortion drives mutual shaping-emerges naturally in this setting: in-context adaptation renders agents vulnerable to extortion, and the resulting mutual pressure to shape the opponent's in-context learning dynamics resolves into the learning of cooperative behavior. Our results suggest that standard decentralized reinforcement learning on sequence models combined with co-player diversity provides a scalable path to learning cooperative behaviors.

在自利智能体之间实现合作仍然是多智能体强化学习中的一个基本挑战。最近的研究表明，“有学习意识”的智能体之间可以引发相互合作，这些智能体解释并塑造了共同参与者的学习动态。然而，现有的方法通常依赖于关于共同玩家学习规则的硬编码的、通常不一致的假设，或者强制严格区分快速更新的“天真的学习者”和观察这些更新的“元学习者”。在这里，我们证明了序列模型的上下文学习能力允许共同玩家学习意识，而不需要硬编码假设或明确的时间尺度分离。我们表明，针对不同分布的共同玩家训练序列模型智能体自然会产生上下文中的最佳响应策略，在快速的情节时间尺度上有效地发挥学习算法的作用。我们发现，先前工作中确定的合作机制（易受勒索影响而导致相互塑造）在这种情况下自然出现：上下文适应使智能体容易受到敲诈勒索，而由此产生的塑造对手上下文学习动态的相互压力则分解为合作行为的学习。我们的结果表明，序列模型上的标准去中心化强化学习与合作者多样性相结合，为学习合作行为提供了一条可扩展的路径。

</details>

---

## 10. Toward Scalable Verifiable Reward: Proxy State-Based Evaluation for Multi-turn Tool-Calling LLM Agents / 迈向可扩展的可验证奖励：多轮工具调用 LLM 代理的基于代理状态的评估

**Date**: 2026-02-18 | **arXiv**: [2602.16246v1](http://arxiv.org/abs/2602.16246v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16246v1)

**Categories**: cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Interactive large language model (LLM) agents operating via multi-turn dialogue and multi-step tool calling are increasingly used in production. Benchmarks for these agents must both reliably compare models and yield on-policy training data. Prior agentic benchmarks (e.g., tau-bench, tau2-bench, AppWorld) rely on fully deterministic backends, which are costly to build and iterate. We propose Proxy State-Based Evaluation, an LLM-driven simulation framework that preserves final state-based evaluation without a deterministic database. Specifically, a scenario specifies the user goal, user/system facts, expected final state, and expected agent behavior, and an LLM state tracker infers a structured proxy state from the full interaction trace. LLM judges then verify goal completion and detect tool/user hallucinations against scenario constraints. Empirically, our benchmark produces stable, model-differentiating rankings across families and inference-time reasoning efforts, and its on-/off-policy rollouts provide supervision that transfers to unseen scenarios. Careful scenario specification yields near-zero simulator hallucination rates as supported by ablation studies. The framework also supports sensitivity analyses over user personas. Human-LLM judge agreement exceeds 90%, indicating reliable automated evaluation. Overall, proxy state-based evaluation offers a practical, scalable alternative to deterministic agentic benchmarks for industrial LLM agents.

通过多轮对话和多步工具调用进行操作的交互式大语言模型（LLM）代理在生产中越来越多地使用。这些代理的基准必须可靠地比较模型并产生符合策略的训练数据。之前的代理基准（例如 tau-bench、tau2-bench、AppWorld）依赖于完全确定性的后端，而构建和迭代的成本很高。我们提出基于代理状态的评估，这是一种法学硕士驱动的模拟框架，可以在没有确定性数据库的情况下保留最终的基于状态的评估。具体来说，场景指定用户目标、用户/系统事实、预期最终状态和预期代理行为，LLM 状态跟踪器从完整交互跟踪中推断结构化代理状态。然后，法学硕士法官会验证目标完成情况，并根据场景限制检测工具/用户的幻觉。根据经验，我们的基准测试可以在不同的家庭和推理时间推理工作中产生稳定的、模型差异化的排名，并且其在策略/策略外的推出提供了转移到未见过的场景的监督。正如消融研究所支持的，仔细的场景规范产生了接近于零的模拟器幻觉率。该框架还支持对用户角色的敏感性分析。人类与法学硕士的判断一致性超过 90%，表明自动化评估可靠。总体而言，基于代理状态的评估为工业法学硕士代理提供了一种实用的、可扩展的替代确定性代理基准的方法。

</details>

---

## 11. Graphon Mean-Field Subsampling for Cooperative Heterogeneous Multi-Agent Reinforcement Learning / 用于协作异构多智能体强化学习的图形平均场子采样

**Date**: 2026-02-18 | **arXiv**: [2602.16196v1](http://arxiv.org/abs/2602.16196v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16196v1)

**Categories**: cs.LG, cs.AI, cs.MA

<details><summary><b>Abstract / 摘要</b></summary>

Coordinating large populations of interacting agents is a central challenge in multi-agent reinforcement learning (MARL), where the size of the joint state-action space scales exponentially with the number of agents. Mean-field methods alleviate this burden by aggregating agent interactions, but these approaches assume homogeneous interactions. Recent graphon-based frameworks capture heterogeneity, but are computationally expensive as the number of agents grows. Therefore, we introduce $\texttt{GMFS}$, a $\textbf{G}$raphon $\textbf{M}$ean-$\textbf{F}$ield $\textbf{S}$ubsampling framework for scalable cooperative MARL with heterogeneous agent interactions. By subsampling $κ$ agents according to interaction strength, we approximate the graphon-weighted mean-field and learn a policy with sample complexity $\mathrm{poly}(κ)$ and optimality gap $O(1/\sqrtκ)$. We verify our theory with numerical simulations in robotic coordination, showing that $\texttt{GMFS}$ achieves near-optimal performance.

协调大量相互作用的智能体是多智能体强化学习（MARL）中的一个核心挑战，其中联合状态-动作空间的大小随着智能体的数量呈指数级增长。平均场方法通过聚合代理交互来减轻这种负担，但这些方法假设同质交互。最近基于图基的框架捕获了异质性，但随着代理数量的增长，计算成本很高。因此，我们引入了 $\texttt{GMFS}$，一个 $\textbf{G}$raphon $\textbf{M}$ean-$\textbf{F}$ield $\textbf{S}$ubsampling 框架，用于具有异构代理交互的可扩展协作 MARL。通过根据交互强度对 $κ$ 代理进行二次采样，我们近似图子加权平均场并学习具有样本复杂度 $\mathrm{poly}(κ)$ 和最优性差距 $O(1/\sqrtκ)$ 的策略。我们通过机器人协调的数值模拟验证了我们的理论，表明 $\texttt{GMFS}$ 实现了接近最佳的性能。

</details>

---

## 12. EnterpriseGym Corecraft: Training Generalizable Agents on High-Fidelity RL Environments / EnterpriseGym Corecraft：在高保真 RL 环境中训练通用智能体

**Date**: 2026-02-18 | **arXiv**: [2602.16179v1](http://arxiv.org/abs/2602.16179v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16179v1)

**Categories**: cs.AI, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

We show that training AI agents on high-fidelity reinforcement learning environments produces capabilities that generalize beyond the training distribution. We introduce \corecraft{}, the first environment in \textsc{EnterpriseGym}, Surge AI's suite of agentic RL environments. \corecraft{} is a fully operational enterprise simulation of a customer support organization, comprising over 2,500 entities across 14 entity types with 23 unique tools, designed to measure whether AI agents can perform the multi-step, domain-specific work that real jobs demand. Frontier models such as GPT-5.2 and Claude Opus 4.6 solve fewer than 30\% of tasks when all expert-authored rubric criteria must be satisfied. Using this environment, we train GLM~4.6 with Group Relative Policy Optimization (GRPO) and adaptive clipping. After a single epoch of training, the model improves from 25.37\% to 36.76\% task pass rate on held-out evaluation tasks. More importantly, these gains transfer to out-of-distribution benchmarks: +4.5\% on BFCL Parallel, +7.4\% on $τ^2$-Bench Retail, and +6.8\% on Toolathlon (Pass@1). We believe three environment properties are consistent with the observed transfer: task-centric world building that optimizes for diverse, challenging tasks; expert-authored rubrics enabling reliable reward computation; and enterprise workflows that reflect realistic professional patterns. Our results suggest that environment quality, diversity, and realism are key factors enabling generalizable agent capabilities.

我们证明，在高保真强化学习环境中训练人工智能代理可以产生超越训练分布的泛化能力。我们引入 \corecraft{}，它是 Surge AI 代理 RL 环境套件 \textsc{EnterpriseGym} 中的第一个环境。 \corecraft{} 是一个完全可操作的客户支持组织企业模拟，由 14 种实体类型的 2,500 多个实体组成，具有 23 种独特的工具，旨在衡量 AI 代理是否可以执行实际工作所需的多步骤、特定领域的工作。当必须满足所有专家编写的标准标准时，GPT-5.2 和 Claude Opus 4.6 等前沿模型只能解决不到 30% 的任务。使用此环境，我们使用组相对策略优化 (GRPO) 和自适应裁剪来训练 GLM~4.6。经过单轮训练后，模型在保留评估任务上的任务通过率从 25.37% 提高到 36.76%。更重要的是，这些收益转移到了分配外基准：BFCL Parallel 上 +4.5\%、$τ^2$-Bench Retail 上 +7.4\%、Toolathlon (Pass@1) 上 +6.8\%。我们相信三个环境属性与观察到的迁移一致：以任务为中心的世界构建，针对多样化、具有挑战性的任务进行优化；专家撰写的评分标准可实现可靠的奖励计算；以及反映现实专业模式的企业工作流程。我们的结果表明，环境质量、多样性和真实性是实现泛化智能体能力的关键因素。

</details>

---

## 13. GPSBench: Do Large Language Models Understand GPS Coordinates? / GPSBench：大型语言模型能理解 GPS 坐标吗？

**Date**: 2026-02-18 | **arXiv**: [2602.16105v1](http://arxiv.org/abs/2602.16105v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16105v1)

**Categories**: cs.AI

**Code**: https://github.com/joey234/gpsbench

<details><summary><b>Abstract / 摘要</b></summary>

Large Language Models (LLMs) are increasingly deployed in applications that interact with the physical world, such as navigation, robotics, or mapping, making robust geospatial reasoning a critical capability. Despite that, LLMs' ability to reason about GPS coordinates and real-world geography remains underexplored. We introduce GPSBench, a dataset of 57,800 samples across 17 tasks for evaluating geospatial reasoning in LLMs, spanning geometric coordinate operations (e.g., distance and bearing computation) and reasoning that integrates coordinates with world knowledge. Focusing on intrinsic model capabilities rather than tool use, we evaluate 14 state-of-the-art LLMs and find that GPS reasoning remains challenging, with substantial variation across tasks: models are generally more reliable at real-world geographic reasoning than at geometric computations. Geographic knowledge degrades hierarchically, with strong country-level performance but weak city-level localization, while robustness to coordinate noise suggests genuine coordinate understanding rather than memorization. We further show that GPS-coordinate augmentation can improve in downstream geospatial tasks, and that finetuning induces trade-offs between gains in geometric computation and degradation in world knowledge. Our dataset and reproducible code are available at https://github.com/joey234/gpsbench

大型语言模型 (LLM) 越来越多地部署在与物理世界交互的应用程序中，例如导航、机器人或地图，这使得强大的地理空间推理成为一项关键功能。尽管如此，法学硕士推理 GPS 坐标和现实世界地理的能力仍未得到充分探索。我们引入了 GPSBench，这是一个包含 17 个任务的 57,800 个样本的数据集，用于评估法学硕士中的地理空间推理，涵盖几何坐标操作（例如距离和方位计算）以及将坐标与世界知识相结合的推理。我们专注于内在模型功能而不是工具使用，评估了 14 个最先进的法学硕士，发现 GPS 推理仍然具有挑战性，不同任务之间存在很大差异：模型在现实世界地理推理中通常比在几何计算中更可靠。地理知识按层次退化，国家层面的表现较强，但城市层面的本地化较弱，而对坐标噪声的鲁棒性表明真正的坐标理解而不是记忆。我们进一步表明，GPS 坐标增强可以改善下游地理空间任务，并且微调会导致几何计算增益和世界知识退化之间的权衡。我们的数据集和可重现的代码可在 https://github.com/joey234/gpsbench 获取

</details>

---

## 14. TabAgent: A Framework for Replacing Agentic Generative Components with Tabular-Textual Classifiers / TabAgent：用表格文本分类器替换代理生成组件的框架

**Date**: 2026-02-18 | **arXiv**: [2602.16429v1](http://arxiv.org/abs/2602.16429v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16429v1)

**Categories**: cs.CL

<details><summary><b>Abstract / 摘要</b></summary>

Agentic systems, AI architectures that autonomously execute multi-step workflows to achieve complex goals, are often built using repeated large language model (LLM) calls for closed-set decision tasks such as routing, shortlisting, gating, and verification. While convenient, this design makes deployments slow and expensive due to cumulative latency and token usage. We propose TabAgent, a framework for replacing generative decision components in closed-set selection tasks with a compact textual-tabular classifier trained on execution traces. TabAgent (i) extracts structured schema, state, and dependency features from trajectories (TabSchema), (ii) augments coverage with schema-aligned synthetic supervision (TabSynth), and (iii) scores candidates with a lightweight classifier (TabHead). On the long-horizon AppWorld benchmark, TabAgent maintains task-level success while eliminating shortlist-time LLM calls, reducing latency by approximately 95% and inference cost by 85-91%. Beyond tool shortlisting, TabAgent generalizes to other agentic decision heads, establishing a paradigm for learned discriminative replacements of generative bottlenecks in production agent architectures.

代理系统是自动执行多步骤工作流程以实现复杂目标的人工智能架构，通常使用重复的大语言模型 (LLM) 调用来构建封闭集决策任务，例如路由、入围、门控和验证。虽然方便，但由于累积延迟和令牌使用，这种设计使部署变得缓慢且昂贵。我们提出了 TabAgent，这是一个框架，用于用在执行跟踪上训练的紧凑文本表格分类器替换封闭集选择任务中的生成决策组件。 TabAgent (i) 从轨迹中提取结构化模式、状态和依赖性特征 (TabSchema)，(ii) 通过模式对齐的综合监督 (TabSynth) 扩大覆盖范围，(iii) 使用轻量级分类器 (TabHead) 对候选者进行评分。在长期 AppWorld 基准测试中，TabAgent 保持了任务级成功，同时消除了入围时间的 LLM 调用，将延迟减少了约 95%，推理成本减少了 85-91%。除了工具入围之外，TabAgent 还可以推广到其他代理决策头，为生产代理架构中的生成瓶颈的学习判别性替换建立一个范例。

</details>

---

## 15. Label-Consistent Data Generation for Aspect-Based Sentiment Analysis Using LLM Agents / 使用 LLM 代理为基于方面的情感分析生成标签一致的数据

**Date**: 2026-02-18 | **arXiv**: [2602.16379v1](http://arxiv.org/abs/2602.16379v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16379v1)

**Categories**: cs.CL

<details><summary><b>Abstract / 摘要</b></summary>

We propose an agentic data augmentation method for Aspect-Based Sentiment Analysis (ABSA) that uses iterative generation and verification to produce high quality synthetic training examples. To isolate the effect of agentic structure, we also develop a closely matched prompting-based baseline using the same model and instructions. Both methods are evaluated across three ABSA subtasks (Aspect Term Extraction (ATE), Aspect Sentiment Classification (ATSC), and Aspect Sentiment Pair Extraction (ASPE)), four SemEval datasets, and two encoder-decoder models: T5-Base and Tk-Instruct. Our results show that the agentic augmentation outperforms raw prompting in label preservation of the augmented data, especially when the tasks require aspect term generation. In addition, when combined with real data, agentic augmentation provides higher gains, consistently outperforming prompting-based generation. These benefits are most pronounced for T5-Base, while the more heavily pretrained Tk-Instruct exhibits smaller improvements. As a result, augmented data helps T5-Base achieve comparable performance with its counterpart.

我们提出了一种用于基于方面的情感分析（ABSA）的代理数据增强方法，该方法使用迭代生成和验证来生成高质量的合成训练示例。为了隔离主体结构的影响，我们还使用相同的模型和指令开发了一个紧密匹配的基于提示的基线。这两种方法都通过三个 ABSA 子任务（方面术语提取 (ATE)、方面情感分类 (ATSC) 和方面情感对提取 (ASPE)）、四个 SemEval 数据集和两个编码器-解码器模型：T5-Base 和 Tk-Instruct 进行评估。我们的结果表明，代理增强在增强数据的标签保存方面优于原始提示，特别是当任务需要方面术语生成时。此外，当与真实数据结合时，代理增强可以提供更高的收益，始终优于基于提示的生成。这些优势对于 T5-Base 最为明显，而经过更严格的预训练的 Tk-Instruct 则表现出较小的改进。因此，增强数据有助于 T5-Base 实现与其对应产品相当的性能。

</details>

---

## 16. MemoryArena: Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks / MemoryArena：在相互依赖的多会话代理任务中对代理内存进行基准测试

**Date**: 2026-02-18 | **arXiv**: [2602.16313v1](http://arxiv.org/abs/2602.16313v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16313v1)

**Categories**: cs.CL

<details><summary><b>Abstract / 摘要</b></summary>

Existing evaluations of agents with memory typically assess memorization and action in isolation. One class of benchmarks evaluates memorization by testing recall of past conversations or text but fails to capture how memory is used to guide future decisions. Another class focuses on agents acting in single-session tasks without the need for long-term memory. However, in realistic settings, memorization and action are tightly coupled: agents acquire memory while interacting with the environment, and subsequently rely on that memory to solve future tasks. To capture this setting, we introduce MemoryArena, a unified evaluation gym for benchmarking agent memory in multi-session Memory-Agent-Environment loops. The benchmark consists of human-crafted agentic tasks with explicitly interdependent subtasks, where agents must learn from earlier actions and feedback by distilling experiences into memory, and subsequently use that memory to guide later actions to solve the overall task. MemoryArena supports evaluation across web navigation, preference-constrained planning, progressive information search, and sequential formal reasoning, and reveals that agents with near-saturated performance on existing long-context memory benchmarks like LoCoMo perform poorly in our agentic setting, exposing a gap in current evaluations for agents with memory.

现有的对记忆主体的评估通常是孤立地评估记忆和行动。一类基准通过测试对过去对话或文本的回忆来评估记忆力，但无法捕捉记忆如何用于指导未来的决策。另一类侧重于在不需要长期记忆的情况下执行单会话任务的代理。然而，在现实环境中，记忆和行动是紧密耦合的：智能体在与环境交互时获取记忆，然后依靠该记忆来解决未来的任务。为了捕获此设置，我们引入了 MemoryArena，这是一个统一的评估工具，用于在多会话内存-代理-环境循环中对代理内存进行基准测试。该基准由人工设计的代理任务和明确相互依赖的子任务组成，其中代理必须通过将经验提取到内存中来从早期的操作和反馈中学习，然后使用该内存来指导后续的操作来解决整个任务。 MemoryArena 支持跨网络导航、偏好约束规划、渐进式信息搜索和顺序形式推理的评估，并揭示了在现有长上下文记忆基准（如 LoCoMo）上性能接近饱和的智能体在我们的智能体环境中表现不佳，暴露了当前对具有记忆的智能体评估中的差距。

</details>

---

## 17. A Scalable Approach to Solving Simulation-Based Network Security Games / 解决基于模拟的网络安全游戏的可扩展方法

**Date**: 2026-02-18 | **arXiv**: [2602.16564v1](http://arxiv.org/abs/2602.16564v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16564v1)

**Categories**: cs.LG, cs.CR

<details><summary><b>Abstract / 摘要</b></summary>

We introduce MetaDOAR, a lightweight meta-controller that augments the Double Oracle / PSRO paradigm with a learned, partition-aware filtering layer and Q-value caching to enable scalable multi-agent reinforcement learning on very large cyber-network environments. MetaDOAR learns a compact state projection from per node structural embeddings to rapidly score and select a small subset of devices (a top-k partition) on which a conventional low-level actor performs focused beam search utilizing a critic agent. Selected candidate actions are evaluated with batched critic forwards and stored in an LRU cache keyed by a quantized state projection and local action identifiers, dramatically reducing redundant critic computation while preserving decision quality via conservative k-hop cache invalidation. Empirically, MetaDOAR attains higher player payoffs than SOTA baselines on large network topologies, without significant scaling issues in terms of memory usage or training time. This contribution provide a practical, theoretically motivated path to efficient hierarchical policy learning for large-scale networked decision problems.

我们引入了 MetaDOAR，这是一种轻量级元控制器，它通过学习的、分区感知的过滤层和 Q 值缓存增强了 Double Oracle / PSRO 范式，从而在非常大的网络环境中实现可扩展的多代理强化学习。 MetaDOAR 从每个节点的结构嵌入中学习紧凑的状态投影，以快速评分并选择一小部分设备（top-k 分区），传统的低级参与者在其上利用批评代理执行聚焦波束搜索。选定的候选动作通过批量批评家前向进行评估，并存储在由量化状态投影和本地动作标识符键控的 LRU 缓存中，从而显着减少冗余批评家计算，同时通过保守的 k 跳缓存失效来保持决策质量。根据经验，MetaDOAR 在大型网络拓扑上比 SOTA 基线获得了更高的玩家回报，并且在内存使用或训练时间方面没有显着的扩展问题。这一贡献为大规模网络决策问题的高效分层策略学习提供了一条实用的、理论上的路径。

</details>

---

## 18. Multi-Agent Combinatorial-Multi-Armed-Bandit framework for the Submodular Welfare Problem under Bandit Feedback / Bandit反馈下子模福利问题的多智能体组合-多臂-老虎机框架

**Date**: 2026-02-18 | **arXiv**: [2602.16183v1](http://arxiv.org/abs/2602.16183v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16183v1)

**Categories**: cs.GT, cs.LG, stat.ML

<details><summary><b>Abstract / 摘要</b></summary>

We study the \emph{Submodular Welfare Problem} (SWP), where items are partitioned among agents with monotone submodular utilities to maximize the total welfare under \emph{bandit feedback}. Classical SWP assumes full value-oracle access, achieving $(1-1/e)$ approximations via continuous-greedy algorithms. We extend this to a \emph{multi-agent combinatorial bandit} framework (\textsc{MA-CMAB}), where actions are partitions under full-bandit feedback with non-communicating agents. Unlike prior single-agent or separable multi-agent CMAB models, our setting couples agents through shared allocation constraints. We propose an explore-then-commit strategy with randomized assignments, achieving $\tilde{\mathcal{O}}(T^{2/3})$ regret against a $(1-1/e)$ benchmark, the first such guarantee for partition-based submodular welfare problem under bandit feedback.

我们研究\emph{子模福利问题}（SWP），其中项目在具有单调子模效用的代理之间进行划分，以最大化\emph{强盗反馈}下的总福利。经典 SWP 假设完全价值预言机访问，通过连续贪婪算法实现 $(1-1/e)$ 近似值。我们将其扩展到\emph{多代理组合强盗}框架（\textsc{MA-CMAB}），其中动作是与非通信代理的全强盗反馈下的分区。与之前的单代理或可分离多代理 CMAB 模型不同，我们的设置通过共享分配约束来耦合代理。我们提出了一种随机分配的探索然后提交策略，实现了对 $(1-1/e)$ 基准的 $\tilde{\mathcal{O}}(T^{2/3})$ 遗憾，这是在强盗反馈下基于分区的子模福利问题的第一个此类保证。

</details>

---

## 19. Markerless Robot Detection and 6D Pose Estimation for Multi-Agent SLAM / 多智能体 SLAM 的无标记机器人检测和 6D 姿态估计

**Date**: 2026-02-18 | **arXiv**: [2602.16308v1](http://arxiv.org/abs/2602.16308v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16308v1)

**Categories**: cs.RO

<details><summary><b>Abstract / 摘要</b></summary>

The capability of multi-robot SLAM approaches to merge localization history and maps from different observers is often challenged by the difficulty in establishing data association. Loop closure detection between perceptual inputs of different robotic agents is easily compromised in the context of perceptual aliasing, or when perspectives differ significantly. For this reason, direct mutual observation among robots is a powerful way to connect partial SLAM graphs, but often relies on the presence of calibrated arrays of fiducial markers (e.g., AprilTag arrays), which severely limits the range of observations and frequently fails under sharp lighting conditions, e.g., reflections or overexposure. In this work, we propose a novel solution to this problem leveraging recent advances in Deep-Learning-based 6D pose estimation. We feature markerless pose estimation as part of a decentralized multi-robot SLAM system and demonstrate the benefit to the relative localization accuracy among the robotic team. The solution is validated experimentally on data recorded in a test field campaign on a planetary analogous environment.

多机器人 SLAM 方法合并来自不同观察者的定位历史和地图的能力经常受到建立数据关联的困难的挑战。在感知混叠的情况下，或者当视角显着不同时，不同机器人代理的感知输入之间的闭环检测很容易受到损害。因此，机器人之间的直接相互观察是连接部分 SLAM 图的有效方法，但通常依赖于基准标记校准阵列（例如 AprilTag 阵列）的存在，这严重限制了观察范围，并且在强光条件下（例如反射或过度曝光）经常失败。在这项工作中，我们利用基于深度学习的 6D 姿态估计的最新进展，提出了一种解决该问题的新颖解决方案。我们将无标记姿态估计作为分散式多机器人 SLAM 系统的一部分，并展示了机器人团队之间相对定位精度的好处。该解决方案根据在行星类似环境中的测试现场活动中记录的数据进行了实验验证。

</details>

---

## 20. Optimization Instability in Autonomous Agentic Workflows for Clinical Symptom Detection / 优化临床症状检测自主代理工作流程的不稳定性

**Date**: 2026-02-17 | **arXiv**: [2602.16037v1](http://arxiv.org/abs/2602.16037v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16037v1)

**Categories**: cs.AI, cs.MA

<details><summary><b>Abstract / 摘要</b></summary>

Autonomous agentic workflows that iteratively refine their own behavior hold considerable promise, yet their failure modes remain poorly characterized. We investigate optimization instability, a phenomenon in which continued autonomous improvement paradoxically degrades classifier performance, using Pythia, an open-source framework for automated prompt optimization. Evaluating three clinical symptoms with varying prevalence (shortness of breath at 23%, chest pain at 12%, and Long COVID brain fog at 3%), we observed that validation sensitivity oscillated between 1.0 and 0.0 across iterations, with severity inversely proportional to class prevalence. At 3% prevalence, the system achieved 95% accuracy while detecting zero positive cases, a failure mode obscured by standard evaluation metrics. We evaluated two interventions: a guiding agent that actively redirected optimization, amplifying overfitting rather than correcting it, and a selector agent that retrospectively identified the best-performing iteration successfully prevented catastrophic failure. With selector agent oversight, the system outperformed expert-curated lexicons on brain fog detection by 331% (F1) and chest pain by 7%, despite requiring only a single natural language term as input. These findings characterize a critical failure mode of autonomous AI systems and demonstrate that retrospective selection outperforms active intervention for stabilization in low-prevalence classification tasks.

迭代地完善自身行为的自主代理工作流程具有相当大的前景，但其故障模式的特征仍然很差。我们使用 Pythia（一种用于自动提示优化的开源框架）研究优化不稳定性，这是一种持续自主改进反而降低分类器性能的现象。通过评估三种不同患病率的临床症状（23% 的呼吸短促、12% 的胸痛和 3% 的长期新冠脑雾），我们观察到验证敏感性在迭代中在 1.0 和 0.0 之间波动，其严重程度与类别患病率成反比。在 3% 的患病率下，系统在检测到零阳性病例时达到了 95% 的准确率，这是一种被标准评估指标掩盖的故障模式。我们评估了两种干预措施：主动重定向优化、放大过度拟合而不是纠正过度拟合的引导代理，以及回顾性识别最佳性能迭代的选择代理，成功防止了灾难性失败。在选择器代理的监督下，尽管只需要一个自然语言术语作为输入，但该系统在脑雾检测方面比专家策划的词典高出 331% (F1)，在胸痛检测方面比专家策划的词典高出 7%。这些发现描述了自主人工智能系统的关键故障模式，并证明在低流行分类任务中，回顾性选择优于主动干预以实现稳定。

</details>

---

## 21. EarthSpatialBench: Benchmarking Spatial Reasoning Capabilities of Multimodal LLMs on Earth Imagery / EarthSpatialBench：地球图像上多模态法学硕士的空间推理能力基准测试

**Date**: 2026-02-17 | **arXiv**: [2602.15918v1](http://arxiv.org/abs/2602.15918v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.15918v1)

**Categories**: cs.CV, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Benchmarking spatial reasoning in multimodal large language models (MLLMs) has attracted growing interest in computer vision due to its importance for embodied AI and other agentic systems that require precise interaction with the physical world. However, spatial reasoning on Earth imagery has lagged behind, as it uniquely involves grounding objects in georeferenced images and quantitatively reasoning about distances, directions, and topological relations using both visual cues and vector geometry coordinates (e.g., 2D bounding boxes, polylines, and polygons). Existing benchmarks for Earth imagery primarily focus on 2D spatial grounding, image captioning, and coarse spatial relations (e.g., simple directional or proximity cues). They lack support for quantitative direction and distance reasoning, systematic topological relations, and complex object geometries beyond bounding boxes. To fill this gap, we propose \textbf{EarthSpatialBench}, a comprehensive benchmark for evaluating spatial reasoning in MLLMs on Earth imagery. The benchmark contains over 325K question-answer pairs spanning: (1) qualitative and quantitative reasoning about spatial distance and direction; (2) systematic topological relations; (3) single-object queries, object-pair queries, and compositional aggregate group queries; and (4) object references expressed via textual descriptions, visual overlays, and explicit geometry coordinates, including 2D bounding boxes, polylines, and polygons. We conducted extensive experiments on both open-source and proprietary models to identify limitations in the spatial reasoning of MLLMs.

多模态大语言模型 (MLLM) 中的空间推理基准测试越来越引起人们对计算机视觉的兴趣，因为它对于具体人工智能和其他需要与物理世界进行精确交互的代理系统非常重要。然而，地球图像的空间推理已经落后，因为它独特地涉及地理参考图像中的接地对象，并使用视觉线索和矢量几何坐标（例如，2D 边界框、折线和多边形）对距离、方向和拓扑关系进行定量推理。地球图像的现有基准主要集中在 2D 空间基础、图像字幕和粗略空间关系（例如，简单的方向或邻近线索）。它们缺乏对定量方向和距离推理、系统拓扑关系以及边界框之外的复杂对象几何形状的支持。为了填补这一空白，我们提出了 \textbf{EarthSpatialBench}，这是一个用于评估地球图像上 MLLM 空间推理的综合基准。该基准包含超过 325K 个问答对，涵盖：(1) 关于空间距离和方向的定性和定量推理； (2)系统的拓扑关系； (3)单对象查询、对象对查询、组合聚合组查询； (4) 通过文本描述、视觉覆盖和显式几何坐标表达的对象引用，包括 2D 边界框、折线和多边形。我们对开源和专有模型进行了广泛的实验，以确定 MLLM 空间推理的局限性。

</details>

---

## 22. The Limits of Long-Context Reasoning in Automated Bug Fixing / 长上下文推理在自动错误修复中的局限性

**Date**: 2026-02-17 | **arXiv**: [2602.16069v1](http://arxiv.org/abs/2602.16069v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16069v1)

**Categories**: cs.SE, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Rapidly increasing context lengths have led to the assumption that large language models (LLMs) can directly reason over entire codebases. Concurrently, recent advances in LLMs have enabled strong performance on software engineering benchmarks, particularly when paired with agentic workflows. In this work, we systematically evaluate whether current LLMs can reliably perform long-context code debugging and patch generation. Using SWE-bench Verified as a controlled experimental setting, we first evaluate state-of-the-art models within an agentic harness (mini-SWE-agent), where performance improves substantially: GPT-5-nano achieves up to a 31\% resolve rate on 100 samples, and open-source models such as Deepseek-R1-0528 obtain competitive results. However, token-level analysis shows that successful agentic trajectories typically remain under 20k tokens, and that longer accumulated contexts correlate with lower success rates, indicating that agentic success primarily arises from task decomposition into short-context steps rather than effective long-context reasoning. To directly test long-context capability, we construct a data pipeline where we artificially inflate the context length of the input by placing the relevant files into the context (ensuring perfect retrieval recall); we then study single-shot patch generation under genuinely long contexts (64k-128k tokens). Despite this setup, performance degrades sharply: Qwen3-Coder-30B-A3B achieves only a 7\% resolve rate at 64k context, while GPT-5-nano solves none of the tasks. Qualitative analysis reveals systematic failure modes, including hallucinated diffs, incorrect file targets, and malformed patch headers. Overall, our findings highlight a significant gap between nominal context length and usable context capacity in current LLMs, and suggest that existing agentic coding benchmarks do not meaningfully evaluate long-context reasoning.

上下文长度的快速增加导致人们假设大型语言模型（LLM）可以直接对整个代码库进行推理。与此同时，法学硕士的最新进展使得软件工程基准测试具有强劲的性能，特别是与代理工作流程配合使用时。在这项工作中，我们系统地评估了当前的法学硕士是否能够可靠地执行长上下文代码调试和补丁生成。使用 SWE-bench Verified 作为受控实验设置，我们首先评估代理工具（mini-SWE-agent）中最先进的模型，其性能显着提高：GPT-5-nano 在 100 个样本上实现了高达 31% 的解析率，而 Deepseek-R1-0528 等开源模型获得了有竞争力的结果。然而，令牌级分析表明，成功的代理轨迹通常保持在 20k 个令牌以下，并且较长的累积上下文与较低的成功率相关，这表明代理的成功主要来自任务分解为短上下文步骤，而不是有效的长上下文推理。为了直接测试长上下文能力，我们构建了一个数据管道，通过将相关文件放入上下文中来人为地增加输入的上下文长度（确保完美的检索召回）；然后，我们研究真正长上下文（64k-128k 标记）下的单次补丁生成。尽管这样设置，性能还是急剧下降：Qwen3-Coder-30B-A3B 在 64k 上下文中仅实现 7% 的解析率，而 GPT-5-nano 则无法解决任何任务。定性分析揭示了系统故障模式，包括幻觉差异、不正确的文件目标和格式错误的补丁头。总体而言，我们的研究结果强调了当前法学硕士中名义上下文长度和可用上下文容量之间的显着差距，并表明现有的代理编码基准并不能有意义地评估长上下文推理。

</details>

---

## 23. MARLEM: A Multi-Agent Reinforcement Learning Simulation Framework for Implicit Cooperation in Decentralized Local Energy Markets / MARLEM：用于去中心化本地能源市场隐式合作的多智能体强化学习模拟框架

**Date**: 2026-02-17 | **arXiv**: [2602.16063v1](http://arxiv.org/abs/2602.16063v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16063v1)

**Categories**: eess.SY, cs.CE, cs.ET, cs.LG, stat.CO

**Code**: https://github.com/salazarna/marlem,

<details><summary><b>Abstract / 摘要</b></summary>

This paper introduces a novel, open-source MARL simulation framework for studying implicit cooperation in LEMs, modeled as a decentralized partially observable Markov decision process and implemented as a Gymnasium environment for MARL. Our framework features a modular market platform with plug-and-play clearing mechanisms, physically constrained agent models (including battery storage), a realistic grid network, and a comprehensive analytics suite to evaluate emergent coordination. The main contribution is a novel method to foster implicit cooperation, where agents' observations and rewards are enhanced with system-level key performance indicators to enable them to independently learn strategies that benefit the entire system and aim for collectively beneficial outcomes without explicit communication. Through representative case studies (available in a dedicated GitHub repository in https://github.com/salazarna/marlem, we show the framework's ability to analyze how different market configurations (such as varying storage deployment) impact system performance. This illustrates its potential to facilitate emergent coordination, improve market efficiency, and strengthen grid stability. The proposed simulation framework is a flexible, extensible, and reproducible tool for researchers and practitioners to design, test, and validate strategies for future intelligent, decentralized energy systems.

本文介绍了一种新颖的开源 MARL 模拟框架，用于研究 LEM 中的隐式合作，该框架被建模为分散的部分可观察马尔可夫决策过程，并作为 MARL 的 Gymnasium 环境实现。我们的框架具有模块化市场平台，具有即插即用的清算机制、物理约束的代理模型（包括电池存储）、现实的网格网络以及用于评估紧急协调的综合分析套件。主要贡献是一种促进隐性合作的新方法，通过系统级关键绩效指标增强智能体的观察和奖励，使他们能够独立学习有利于整个系统的策略，并在没有明确沟通的情况下实现集体受益的结果。通过代表性案例研究（可在 https://github.com/salazarna/marlem 的专用 GitHub 存储库中找到），我们展示了该框架分析不同市场配置（例如不同存储部署）如何影响系统性能的能力。这说明了其促进紧急协调、提高市场效率和加强电网稳定性的潜力。所提出的模拟框架是一种灵活、可扩展和可重复的工具，可供研究人员和从业人员设计、测试和验证未来智能、去中心化能源系统的策略。

</details>

---

## 24. Harnessing Implicit Cooperation: A Multi-Agent Reinforcement Learning Approach Towards Decentralized Local Energy Markets / 利用隐式合作：实现去中心化本地能源市场的多智能体强化学习方法

**Date**: 2026-02-17 | **arXiv**: [2602.16062v1](http://arxiv.org/abs/2602.16062v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.16062v1)

**Categories**: eess.SY, cs.CE, cs.LG, cs.MA, stat.AP

<details><summary><b>Abstract / 摘要</b></summary>

This paper proposes implicit cooperation, a framework enabling decentralized agents to approximate optimal coordination in local energy markets without explicit peer-to-peer communication. We formulate the problem as a decentralized partially observable Markov decision problem that is solved through a multi-agent reinforcement learning task in which agents use stigmergic signals (key performance indicators at the system level) to infer and react to global states. Through a 3x3 factorial design on an IEEE 34-node topology, we evaluated three training paradigms (CTCE, CTDE, DTDE) and three algorithms (PPO, APPO, SAC). Results identify APPO-DTDE as the optimal configuration, achieving a coordination score of 91.7% relative to the theoretical centralized benchmark (CTCE). However, a critical trade-off emerges between efficiency and stability: while the centralized benchmark maximizes allocative efficiency with a peer-to-peer trade ratio of 0.6, the fully decentralized approach (DTDE) demonstrates superior physical stability. Specifically, DTDE reduces the variance of grid balance by 31% compared to hybrid architectures, establishing a highly predictable, import-biased load profile that simplifies grid regulation. Furthermore, topological analysis reveals emergent spatial clustering, where decentralized agents self-organize into stable trading communities to minimize congestion penalties. While SAC excelled in hybrid settings, it failed in decentralized environments due to entropy-driven instability. This research proves that stigmergic signaling provides sufficient context for complex grid coordination, offering a robust, privacy-preserving alternative to expensive centralized communication infrastructure.

本文提出了隐性合作，这是一个框架，使分散的代理能够在当地能源市场上实现近似最佳协调，而无需明确的点对点通信。我们将该问题表述为分散的部分可观察马尔可夫决策问题，该问题通过多智能体强化学习任务来解决，其中智能体使用 stigmergic 信号（系统级别的关键性能指标）来推断全局状态并做出反应。通过 IEEE 34 节点拓扑上的 3x3 因子设计，我们评估了三种训练范例（CTCE、CTDE、DTDE）和三种算法（PPO、APPO、SAC）。结果将 APPO-DTDE 确定为最佳配置，相对于理论集中基准（CTCE），协调得分为 91.7%。然而，效率和稳定性之间出现了一个关键的权衡：虽然中心化基准以 0.6 的点对点交易比率最大化配置效率，但完全去中心化方法 (DTDE) 表现出卓越的物理稳定性。具体来说，与混合架构相比，DTDE 将电网平衡的方差降低了 31%，建立了高度可预测的、进口偏向的负载曲线，从而简化了电网调节。此外，拓扑分析揭示了新兴的空间集群，分散的代理自我组织成稳定的交易社区，以最大限度地减少拥堵惩罚。虽然 SAC 在混合环境中表现出色，但由于熵驱动的不稳定性，它在去中心化环境中失败了。这项研究证明，stigmergic 信号为复杂的网格协调提供了足够的背景，为昂贵的集中式通信基础设施提供了强大的、保护隐私的替代方案。

</details>

---

## 25. A Geometric Analysis of Small-sized Language Model Hallucinations / 小规模语言模型幻象的几何分析

**Date**: 2026-02-16 | **arXiv**: [2602.14778v2](http://arxiv.org/abs/2602.14778v2) | **PDF**: [Link](http://arxiv.org/pdf/2602.14778v2)

**Categories**: cs.CL, cs.AI, cs.CY

<details><summary><b>Abstract / 摘要</b></summary>

Hallucinations -- fluent but factually incorrect responses -- pose a major challenge to the reliability of language models, especially in multi-step or agentic settings.   This work investigates hallucinations in small-sized LLMs through a geometric perspective, starting from the hypothesis that when models generate multiple responses to the same prompt, genuine ones exhibit tighter clustering in the embedding space, we prove this hypothesis and, leveraging this geometrical insight, we also show that it is possible to achieve a consistent level of separability. This latter result is used to introduce a label-efficient propagation method that classifies large collections of responses from just 30-50 annotations, achieving F1 scores above 90%.   Our findings, framing hallucinations from a geometric perspective in the embedding space, complement traditional knowledge-centric and single-response evaluation paradigms, paving the way for further research.

幻觉——流畅但实际上不正确的反应——对语言模型的可靠性构成了重大挑战，特别是在多步骤或代理环境中。   这项工作通过几何角度研究小型法学硕士中的幻觉，从假设开始，当模型对同一提示生成多个响应时，真实的响应在嵌入空间中表现出更紧密的聚类，我们证明了这一假设，并利用这种几何洞察力，我们还表明有可能实现一致的可分离性水平。后一个结果用于引入一种标签高效的传播方法，该方法可以对仅 30-50 个注释的大量响应集合进行分类，实现 F1 分数高于 90%。   我们的研究结果从嵌入空间的几何角度构建幻觉，补充了传统的以知识为中心和单一响应的评估范式，为进一步的研究铺平了道路。

</details>

---



</details>

<details><summary><b>2026-02-18 (32 papers)</b></summary>

# arXiv Agent Papers - 2026-02-18

**Paper Count**: 32

---

## 1. GlobeDiff: State Diffusion Process for Partial Observability in Multi-Agent Systems / GlobeDiff：多智能体系统中部分可观测性的状态扩散过程

**Date**: 2026-02-17 | **arXiv**: [2602.15776v1](http://arxiv.org/abs/2602.15776v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.15776v1)

**Categories**: cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

In the realm of multi-agent systems, the challenge of \emph{partial observability} is a critical barrier to effective coordination and decision-making. Existing approaches, such as belief state estimation and inter-agent communication, often fall short. Belief-based methods are limited by their focus on past experiences without fully leveraging global information, while communication methods often lack a robust model to effectively utilize the auxiliary information they provide. To solve this issue, we propose Global State Diffusion Algorithm~(GlobeDiff) to infer the global state based on the local observations. By formulating the state inference process as a multi-modal diffusion process, GlobeDiff overcomes ambiguities in state estimation while simultaneously inferring the global state with high fidelity. We prove that the estimation error of GlobeDiff under both unimodal and multi-modal distributions can be bounded. Extensive experimental results demonstrate that GlobeDiff achieves superior performance and is capable of accurately inferring the global state.

在多智能体系统领域，\emph{部分可观察性}的挑战是有效协调和决策的关键障碍。现有的方法，例如信念状态估计和智能体间通信，常常存在不足。基于信念的方法因其关注过去的经验而没有充分利用全球信息而受到限制，而沟通方法通常缺乏强大的模型来有效利用它们提供的辅助信息。为了解决这个问题，我们提出了全局状态扩散算法~（GlobeDiff）来根据局部观察来推断全局状态。通过将状态推断过程表述为多模态扩散过程，GlobeDiff 克服了状态估计中的模糊性，同时以高保真度推断全局状态。我们证明了 GlobeDiff 在单峰和多峰分布下的估计误差都是有界的。大量的实验结果表明，GlobeDiff 具有卓越的性能，并且能够准确推断全局状态。

</details>

---

## 2. Lifelong Scalable Multi-Agent Realistic Testbed and A Comprehensive Study on Design Choices in Lifelong AGV Fleet Management Systems / 终身可扩展多代理现实测试台和终身 AGV 车队管理系统设计选择的综合研究

**Date**: 2026-02-17 | **arXiv**: [2602.15721v1](http://arxiv.org/abs/2602.15721v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.15721v1)

**Categories**: cs.RO, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

We present Lifelong Scalable Multi-Agent Realistic Testbed (LSMART), an open-source simulator to evaluate any Multi-Agent Path Finding (MAPF) algorithm in a Fleet Management System (FMS) with Automated Guided Vehicles (AGVs). MAPF aims to move a group of agents from their corresponding starting locations to their goals. Lifelong MAPF (LMAPF) is a variant of MAPF that continuously assigns new goals for agents to reach. LMAPF applications, such as autonomous warehouses, often require a centralized, lifelong system to coordinate the movement of a fleet of robots, typically AGVs. However, existing works on MAPF and LMAPF often assume simplified kinodynamic models, such as pebble motion, as well as perfect execution and communication for AGVs. Prior work has presented SMART, a software capable of evaluating any MAPF algorithms while considering agent kinodynamics, communication delays, and execution uncertainties. However, SMART is designed for MAPF, not LMAPF. Generalizing SMART to an FMS requires many more design choices. First, an FMS parallelizes planning and execution, raising the question of when to plan. Second, given planners with varying optimality and differing agent-model assumptions, one must decide how to plan. Third, when the planner fails to return valid solutions, the system must determine how to recover. In this paper, we first present LSMART, an open-source simulator that incorporates all these considerations to evaluate any MAPF algorithms in an FMS. We then provide experiment results based on state-of-the-art methods for each design choice, offering guidance on how to effectively design centralized lifelong AGV Fleet Management Systems. LSMART is available at https://smart-mapf.github.io/lifelong-smart.

我们推出了终身可扩展多智能体现实测试台 (LSMART)，这是一个开源模拟器，用于评估具有自动导引车 (AGV) 的车队管理系统 (FMS) 中的任何多智能体路径查找 (MAPF) 算法。 MAPF 的目标是将一组智能体从相应的起始位置移动到目标位置。终身 MAPF (LMAPF) 是 MAPF 的一个变体，它不断地为智能体分配新的目标以达到。 LMAPF 应用（例如自主仓库）通常需要一个集中的、终身的系统来协调机器人车队（通常是 AGV）的移动。然而，现有的 MAPF 和 LMAPF 工作通常假设简化的运动动力学模型，例如卵石运动，以及 AGV 的完美执行和通信。之前的工作已经提出了 SMART，这是一种能够评估任何 MAPF 算法的软件，同时考虑代理运动动力学、通信延迟和执行不确定性。然而，SMART 是为 MAPF 而设计的，而不是 LMAPF。将 SMART 推广到 FMS 需要更多的设计选择。首先，FMS 并行规划和执行，提出了何时规划的问题。其次，考虑到规划者具有不同的最优性和不同的代理模型假设，人们必须决定如何规划。第三，当规划器未能返回有效的解决方案时，系统必须确定如何恢复。在本文中，我们首先介绍 LSMART，这是一个开源模拟器，它结合了所有这些考虑因素来评估 FMS 中的任何 MAPF 算法。然后，我们根据每种设计选择的最先进方法提供实验结果，为如何有效设计集中式终身 AGV 车队管理系统提供指导。 LSMART 可在 https://smart-mapf.github.io/lifelong-smart 上获取。

</details>

---

## 3. World-Model-Augmented Web Agents with Action Correction / 具有动作校正功能的世界模型增强网络代理

**Date**: 2026-02-17 | **arXiv**: [2602.15384v1](http://arxiv.org/abs/2602.15384v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.15384v1)

**Categories**: cs.AI, cs.CL

<details><summary><b>Abstract / 摘要</b></summary>

Web agents based on large language models have demonstrated promising capability in automating web tasks. However, current web agents struggle to reason out sensible actions due to the limitations of predicting environment changes, and might not possess comprehensive awareness of execution risks, prematurely performing risky actions that cause losses and lead to task failure. To address these challenges, we propose WAC, a web agent that integrates model collaboration, consequence simulation, and feedback-driven action refinement. To overcome the cognitive isolation of individual models, we introduce a multi-agent collaboration process that enables an action model to consult a world model as a web-environment expert for strategic guidance; the action model then grounds these suggestions into executable actions, leveraging prior knowledge of environmental state transition dynamics to enhance candidate action proposal. To achieve risk-aware resilient task execution, we introduce a two-stage deduction chain. A world model, specialized in environmental state transitions, simulates action outcomes, which a judge model then scrutinizes to trigger action corrective feedback when necessary. Experiments show that WAC achieves absolute gains of 1.8% on VisualWebArena and 1.3% on Online-Mind2Web.

基于大型语言模型的 Web 代理在自动化 Web 任务方面表现出了良好的能力。然而，当前的网络代理由于预测环境变化的局限性，难以推理出合理的行动，并且可能不具备全面的执行风险意识，过早地执行风险行动，造成损失并导致任务失败。为了应对这些挑战，我们提出了 WAC，这是一种集成了模型协作、结果模拟和反馈驱动的动作细化的网络代理。为了克服各个模型的认知隔离，我们引入了多智能体协作流程，使行动模型能够作为网络环境专家咨询世界模型以获取战略指导；然后，行动模型将这些建议转化为可执行的行动，利用环境状态转换动态的先验知识来增强候选行动建议。为了实现风险感知的弹性任务执行，我们引入了两阶段推论链。专门研究环境状态转换的世界模型会模拟行动结果，然后法官模型会对其进行仔细检查，以在必要时触发行动纠正反馈。实验表明，WAC 在 VisualWebArena 上获得了 1.8% 的绝对收益，在 Online-Mind2Web 上获得了 1.3% 的绝对收益。

</details>

---

## 4. AgriWorld:A World Tools Protocol Framework for Verifiable Agricultural Reasoning with Code-Executing LLM Agents / AgriWorld：使用代码执行 LLM 代理进行可验证农业推理的世界工具协议框架

**Date**: 2026-02-17 | **arXiv**: [2602.15325v1](http://arxiv.org/abs/2602.15325v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.15325v1)

**Categories**: cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Foundation models for agriculture are increasingly trained on massive spatiotemporal data (e.g., multi-spectral remote sensing, soil grids, and field-level management logs) and achieve strong performance on forecasting and monitoring. However, these models lack language-based reasoning and interactive capabilities, limiting their usefulness in real-world agronomic workflows. Meanwhile, large language models (LLMs) excel at interpreting and generating text, but cannot directly reason over high-dimensional, heterogeneous agricultural datasets. We bridge this gap with an agentic framework for agricultural science. It provides a Python execution environment, AgriWorld, exposing unified tools for geospatial queries over field parcels, remote-sensing time-series analytics, crop growth simulation, and task-specific predictors (e.g., yield, stress, and disease risk). On top of this environment, we design a multi-turn LLM agent, Agro-Reflective, that iteratively writes code, observes execution results, and refines its analysis via an execute-observe-refine loop. We introduce AgroBench, with scalable data generation for diverse agricultural QA spanning lookups, forecasting, anomaly detection, and counterfactual "what-if" analysis. Experiments outperform text-only and direct tool-use baselines, validating execution-driven reflection for reliable agricultural reasoning.

农业基础模型越来越多地接受海量时空数据（例如多光谱遥感、土壤网格和田间管理日志）的训练，并在预测和监测方面取得了良好的表现。然而，这些模型缺乏基于语言的推理和交互功能，限制了它们在现实世界农艺工作流程中的实用性。与此同时，大型语言模型（LLM）擅长解释和生成文本，但无法直接对高维、异构农业数据集进行推理。我们通过农业科学的代理框架来弥合这一差距。它提供了一个 Python 执行环境 AgriWorld，提供统一的工具用于田间地块的地理空间查询、遥感时间序列分析、作物生长模拟和特定任务的预测器（例如产量、压力和疾病风险）。在此环境之上，我们设计了一个多轮LLM代理Agro-Reflective，它迭代地编写代码，观察执行结果，并通过执行-观察-细化循环来细化其分析。我们推出了 AgroBench，它具有可扩展的数据生成功能，适用于各种农业 QA，涵盖查找、预测、异常检测和反事实“假设”分析。实验优于纯文本和直接工具使用基线，验证了执行驱动的反射是否可靠的农业推理。

</details>

---

## 5. EAA: Automating materials characterization with vision language model agents / EAA：使用视觉语言模型代理自动进行材料表征

**Date**: 2026-02-17 | **arXiv**: [2602.15294v1](http://arxiv.org/abs/2602.15294v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.15294v1)

**Categories**: cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

We present Experiment Automation Agents (EAA), a vision-language-model-driven agentic system designed to automate complex experimental microscopy workflows. EAA integrates multimodal reasoning, tool-augmented action, and optional long-term memory to support both autonomous procedures and interactive user-guided measurements. Built on a flexible task-manager architecture, the system enables workflows ranging from fully agent-driven automation to logic-defined routines that embed localized LLM queries. EAA further provides a modern tool ecosystem with two-way compatibility for Model Context Protocol (MCP), allowing instrument-control tools to be consumed or served across applications. We demonstrate EAA at an imaging beamline at the Advanced Photon Source, including automated zone plate focusing, natural language-described feature search, and interactive data acquisition. These results illustrate how vision-capable agents can enhance beamline efficiency, reduce operational burden, and lower the expertise barrier for users.

我们提出了实验自动化代理（EAA），这是一种视觉语言模型驱动的代理系统，旨在自动化复杂的实验显微镜工作流程。 EAA 集成了多模态推理、工具增强操作和可选的长期记忆，以支持自主程序和交互式用户引导测量。该系统建立在灵活的任务管理器架构之上，支持从完全代理驱动的自动化到嵌入本地化 LLM 查询的逻辑定义例程的工作流程。 EAA 进一步提供了一个具有模型上下文协议 (MCP) 双向兼容性的现代工具生态系统，允许跨应用程序使用或提供仪器控制工具。我们在高级光子源的成像光束线上演示了 EAA，包括自动波带板聚焦、自然语言描述的特征搜索和交互式数据采集。这些结果说明了具有视觉能力的代理如何提高光束线效率、减轻操作负担并降低用户的专业知识障碍。

</details>

---

## 6. GLM-5: from Vibe Coding to Agentic Engineering / GLM-5：从 Vibe 编码到代理工程

**Date**: 2026-02-17 | **arXiv**: [2602.15763v1](http://arxiv.org/abs/2602.15763v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.15763v1)

**Categories**: cs.LG, cs.CL

**Code**: https://github.com/zai-org/GLM-5.

<details><summary><b>Abstract / 摘要</b></summary>

We present GLM-5, a next-generation foundation model designed to transition the paradigm of vibe coding to agentic engineering. Building upon the agentic, reasoning, and coding (ARC) capabilities of its predecessor, GLM-5 adopts DSA to significantly reduce training and inference costs while maintaining long-context fidelity. To advance model alignment and autonomy, we implement a new asynchronous reinforcement learning infrastructure that drastically improves post-training efficiency by decoupling generation from training. Furthermore, we propose novel asynchronous agent RL algorithms that further improve RL quality, enabling the model to learn from complex, long-horizon interactions more effectively. Through these innovations, GLM-5 achieves state-of-the-art performance on major open benchmarks. Most critically, GLM-5 demonstrates unprecedented capability in real-world coding tasks, surpassing previous baselines in handling end-to-end software engineering challenges. Code, models, and more information are available at https://github.com/zai-org/GLM-5.

我们提出了 GLM-5，这是一种下一代基础模型，旨在将振动编码范式转变为代理工程。 GLM-5 以其前身的代理、推理和编码 (ARC) 功能为基础，采用 DSA 来显着降低训练和推理成本，同时保持长上下文保真度。为了推进模型对齐和自治，我们实施了一个新的异步强化学习基础设施，通过将生成与训练解耦，极大地提高了训练后的效率。此外，我们提出了新颖的异步代理强化学习算法，可以进一步提高强化学习质量，使模型能够更有效地从复杂的长范围交互中学习。通过这些创新，GLM-5 在主要开放基准测试中实现了最先进的性能。最关键的是，GLM-5 在实际编码任务中展示了前所未有的能力，在处理端到端软件工程挑战方面超越了以前的基线。代码、模型和更多信息请访问 https://github.com/zai-org/GLM-5。

</details>

---

## 7. The Vision Wormhole: Latent-Space Communication in Heterogeneous Multi-Agent Systems / 视觉虫洞：异构多智能体系统中的潜在空间通信

**Date**: 2026-02-17 | **arXiv**: [2602.15382v1](http://arxiv.org/abs/2602.15382v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.15382v1)

**Categories**: cs.CL, cs.CV, cs.LG

**Code**: https://github.com/xz-liu/heterogeneous-latent-mas

<details><summary><b>Abstract / 摘要</b></summary>

Multi-Agent Systems (MAS) powered by Large Language Models have unlocked advanced collaborative reasoning, yet they remain shackled by the inefficiency of discrete text communication, which imposes significant runtime overhead and information quantization loss. While latent state transfer offers a high-bandwidth alternative, existing approaches either assume homogeneous sender-receiver architectures or rely on pair-specific learned translators, limiting scalability and modularity across diverse model families with disjoint manifolds. In this work, we propose the Vision Wormhole, a novel framework that repurposes the visual interface of Vision-Language Models (VLMs) to enable model-agnostic, text-free communication. By introducing a Universal Visual Codec, we map heterogeneous reasoning traces into a shared continuous latent space and inject them directly into the receiver's visual pathway, effectively treating the vision encoder as a universal port for inter-agent telepathy. Our framework adopts a hub-and-spoke topology to reduce pairwise alignment complexity from O(N^2) to O(N) and leverages a label-free, teacher-student distillation objective to align the high-speed visual channel with the robust reasoning patterns of the text pathway. Extensive experiments across heterogeneous model families (e.g., Qwen-VL, Gemma) demonstrate that the Vision Wormhole reduces end-to-end wall-clock time in controlled comparisons while maintaining reasoning fidelity comparable to standard text-based MAS. Code is available at https://github.com/xz-liu/heterogeneous-latent-mas

由大型语言模型支持的多代理系统（MAS）已经解锁了高级协作推理，但它们仍然受到离散文本通信效率低下的束缚，这会带来巨大的运行时开销和信息量化损失。虽然潜在状态传输提供了高带宽替代方案，但现有方法要么采用同质发送器-接收器架构，要么依赖于特定对的学习转换器，从而限制了具有不相交流形的不同模型系列的可扩展性和模块化性。在这项工作中，我们提出了 Vision Wormhole，这是一种新颖的框架，它重新利用视觉语言模型 (VLM) 的视觉界面，以实现与模型无关的、无文本的通信。通过引入通用视觉编解码器，我们将异构推理轨迹映射到共享的连续潜在空间中，并将它们直接注入接收者的视觉路径中，有效地将视觉编码器视为代理间心灵感应的通用端口。我们的框架采用中心辐射型拓扑结构，将成对对齐复杂度从 O(N^2) 降低到 O(N)，并利用无标签、师生蒸馏目标将高速视觉通道与文本路径的稳健推理模式对齐。跨异构模型系列（例如 Qwen-VL、Gemma）的大量实验表明，Vision Wormhole 在受控比较中减少了端到端挂钟时间，同时保持了与标准基于文本的 MAS 相当的推理保真度。代码可在 https://github.com/xz-liu/heterogeneous-latent-mas 获取

</details>

---

## 8. Fairness over Equality: Correcting Social Incentives in Asymmetric Sequential Social Dilemmas / 公平高于平等：纠正不对称序列社会困境中的社会激励

**Date**: 2026-02-17 | **arXiv**: [2602.15407v1](http://arxiv.org/abs/2602.15407v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.15407v1)

**Categories**: cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Sequential Social Dilemmas (SSDs) provide a key framework for studying how cooperation emerges when individual incentives conflict with collective welfare. In Multi-Agent Reinforcement Learning, these problems are often addressed by incorporating intrinsic drives that encourage prosocial or fair behavior. However, most existing methods assume that agents face identical incentives in the dilemma and require continuous access to global information about other agents to assess fairness. In this work, we introduce asymmetric variants of well-known SSD environments and examine how natural differences between agents influence cooperation dynamics. Our findings reveal that existing fairness-based methods struggle to adapt under asymmetric conditions by enforcing raw equality that wrongfully incentivize defection. To address this, we propose three modifications: (i) redefining fairness by accounting for agents' reward ranges, (ii) introducing an agent-based weighting mechanism to better handle inherent asymmetries, and (iii) localizing social feedback to make the methods effective under partial observability without requiring global information sharing. Experimental results show that in asymmetric scenarios, our method fosters faster emergence of cooperative policies compared to existing approaches, without sacrificing scalability or practicality.

序列社会困境（SSD）为研究当个人激励与集体福利发生冲突时合作如何出现提供了一个关键框架。在多智能体强化学习中，这些问题通常通过纳入鼓励亲社会或公平行为的内在驱动力来解决。然而，大多数现有方法假设代理人在困境中面临相同的激励，并且需要持续访问有关其他代理人的全局信息来评估公平性。在这项工作中，我们介绍了著名 SSD 环境的非对称变体，并研究了代理之间的自然差异如何影响合作动态。我们的研究结果表明，现有的基于公平的方法很难适应不对称条件，因为强制执行原始平等会错误地激励叛逃。为了解决这个问题，我们提出了三个修改：（i）通过考虑代理的奖励范围来重新定义公平性，（ii）引入基于代理的加权机制以更好地处理固有的不对称性，以及（iii）本地化社会反馈以使方法在部分可观察性下有效，而不需要全局信息共享。实验结果表明，在不对称场景中，与现有方法相比，我们的方法可以更快地出现合作策略，而不会牺牲可扩展性或实用性。

</details>

---

## 9. Secure and Energy-Efficient Wireless Agentic AI Networks / 安全且节能的无线代理人工智能网络

**Date**: 2026-02-16 | **arXiv**: [2602.15212v1](http://arxiv.org/abs/2602.15212v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.15212v1)

**Categories**: cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

In this paper, we introduce a secure wireless agentic AI network comprising one supervisor AI agent and multiple other AI agents to provision quality of service (QoS) for users' reasoning tasks while ensuring confidentiality of private knowledge and reasoning outcomes. Specifically, the supervisor AI agent can dynamically assign other AI agents to participate in cooperative reasoning, while the unselected AI agents act as friendly jammers to degrade the eavesdropper's interception performance. To extend the service duration of AI agents, an energy minimization problem is formulated that jointly optimizes AI agent selection, base station (BS) beamforming, and AI agent transmission power, subject to latency and reasoning accuracy constraints. To address the formulated problem, we propose two resource allocation schemes, ASC and LAW, which first decompose it into three sub-problems. Specifically, ASC optimizes each sub-problem iteratively using the proposed alternating direction method of multipliers (ADMM)-based algorithm, semi-definite relaxation (SDR), and successive convex approximation (SCA), while LAW tackles each sub-problem using the proposed large language model (LLM) optimizer within an agentic workflow. The experimental results show that the proposed solutions can reduce network energy consumption by up to 59.1% compared to other benchmark schemes. Furthermore, the proposed schemes are validated using a practical agentic AI system based on Qwen, demonstrating satisfactory reasoning accuracy across various public benchmarks.

在本文中，我们介绍了一种安全的无线代理人工智能网络，由一个主管人工智能代理和多个其他人工智能代理组成，为用户的推理任务提供服务质量（QoS），同时确保私有知识和推理结果的机密性。具体来说，监督AI代理可以动态分配其他AI代理参与协作推理，而未选择的AI代理则充当友好干扰器，降低窃听者的拦截性能。为了延长人工智能代理的服务持续时间，提出了一个能量最小化问题，在延迟和推理精度约束下，联合优化人工智能代理选择、基站（BS）波束成形和人工智能代理传输功率。为了解决所提出的问题，我们提出了两种资源分配方案：ASC 和 LAW，它们首先将其分解为三个子问题。具体来说，ASC 使用所提出的基于乘子交替方向法 (ADMM) 的算法、半定松弛 (SDR) 和逐次凸逼近 (SCA) 迭代优化每个子问题，而 LAW 在代理工作流程中使用所提出的大语言模型 (LLM) 优化器来解决每个子问题。实验结果表明，与其他基准方案相比，所提出的解决方案可以降低网络能耗高达59.1%。此外，所提出的方案使用基于 Qwen 的实用代理人工智能系统进行了验证，在各种公共基准测试中表现出了令人满意的推理准确性。

</details>

---

## 10. Colosseum: Auditing Collusion in Cooperative Multi-Agent Systems / Colosseum：协作多代理系统中的审计共谋

**Date**: 2026-02-16 | **arXiv**: [2602.15198v1](http://arxiv.org/abs/2602.15198v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.15198v1)

**Categories**: cs.MA, cs.AI, cs.CL

<details><summary><b>Abstract / 摘要</b></summary>

Multi-agent systems, where LLM agents communicate through free-form language, enable sophisticated coordination for solving complex cooperative tasks. This surfaces a unique safety problem when individual agents form a coalition and \emph{collude} to pursue secondary goals and degrade the joint objective. In this paper, we present Colosseum, a framework for auditing LLM agents' collusive behavior in multi-agent settings. We ground how agents cooperate through a Distributed Constraint Optimization Problem (DCOP) and measure collusion via regret relative to the cooperative optimum. Colosseum tests each LLM for collusion under different objectives, persuasion tactics, and network topologies. Through our audit, we show that most out-of-the-box models exhibited a propensity to collude when a secret communication channel was artificially formed. Furthermore, we discover ``collusion on paper'' when agents plan to collude in text but would often pick non-collusive actions, thus providing little effect on the joint task. Colosseum provides a new way to study collusion by measuring communications and actions in rich yet verifiable environments.

在多代理系统中，LLM 代理通过自由格式语言进行通信，可以实现复杂的协调来解决复杂的协作任务。当个体代理形成联盟并\emph{共谋}以追求次要目标并降低联合目标时，这就出现了一个独特的安全问题。在本文中，我们提出了 Colosseum，一个用于审核多代理环境中 LLM 代理合谋行为的框架。我们通过分布式约束优化问题（DCOP）来研究代理如何合作，并通过相对于合作最优的遗憾来衡量共谋。 Colosseum 测试每个法学硕士在不同目标、说服策略和网络拓扑下的串通行为。通过我们的审计，我们发现大多数开箱即用的模型在人为形成秘密通信渠道时表现出共谋的倾向。此外，当智能体计划在文本中共谋但通常会选择非共谋行为时，我们会发现“纸面上的共谋”，从而对联合任务几乎没有影响。斗兽场通过测量丰富但可验证的环境中的通信和行为，提供了一种研究共谋的新方法。

</details>

---

## 11. OpaqueToolsBench: Learning Nuances of Tool Behavior Through Interaction / OpaqueToolsBench：通过交互学习工具行为的细微差别

**Date**: 2026-02-16 | **arXiv**: [2602.15197v1](http://arxiv.org/abs/2602.15197v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.15197v1)

**Categories**: cs.CL, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Tool-calling is essential for Large Language Model (LLM) agents to complete real-world tasks. While most existing benchmarks assume simple, perfectly documented tools, real-world tools (e.g., general "search" APIs) are often opaque, lacking clear best practices or failure modes. Can LLM agents improve their performance in environments with opaque tools by interacting and subsequently improving documentation? To study this, we create OpaqueToolsBench, a benchmark consisting of three distinct task-oriented environments: general function calling, interactive chess playing, and long-trajectory agentic search. Each environment provides underspecified tools that models must learn to use effectively to complete the task. Results on OpaqueToolsBench suggest existing methods for automatically documenting tools are expensive and unreliable when tools are opaque. To address this, we propose a simple framework, ToolObserver, that iteratively refines tool documentation by observing execution feedback from tool-calling trajectories. Our approach outperforms existing methods on OpaqueToolsBench across datasets, even in relatively hard settings. Furthermore, for test-time tool exploration settings, our method is also efficient, consuming 3.5-7.5x fewer total tokens than the best baseline.

工具调用对于大型语言模型 (LLM) 代理完成实际任务至关重要。虽然大多数现有基准测试都假设使用简单、记录完善的工具，但现实世界的工具（例如通用“搜索”API）通常是不透明的，缺乏明确的最佳实践或故障模式。 LLM 代理能否通过交互并随后改进文档来提高使用不透明工具的环境中的性能？为了研究这个问题，我们创建了 OpaqueToolsBench，这是一个由三个不同的面向任务的环境组成的基准测试：通用函数调用、交互式下棋和长轨迹代理搜索。每个环境都提供了未指定的工具，模型必须学会有效地使用这些工具来完成任务。 OpaqueToolsBench 的结果表明，当工具不透明时，自动记录工具的现有方法既昂贵又不可靠。为了解决这个问题，我们提出了一个简单的框架 ToolObserver，它通过观察工具调用轨迹的执行反馈来迭代地完善工具文档。即使在相对困难的设置中，我们的方法也优于 OpaqueToolsBench 上跨数据集的现有方法。此外，对于测试时工具探索设置，我们的方法也很高效，消耗的总令牌比最佳基线少 3.5-7.5 倍。

</details>

---

## 12. Mind the (DH) Gap! A Contrast in Risky Choices Between Reasoning and Conversational LLMs / 注意（DH）差距！推理法学硕士和会话法学硕士之间的风险选择对比

**Date**: 2026-02-16 | **arXiv**: [2602.15173v1](http://arxiv.org/abs/2602.15173v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.15173v1)

**Categories**: cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

The use of large language models either as decision support systems, or in agentic workflows, is rapidly transforming the digital ecosystem. However, the understanding of LLM decision-making under uncertainty remains limited. We initiate a comparative study of LLM risky choices along two dimensions: (1) prospect representation (explicit vs. experience based) and (2) decision rationale (explanation). Our study, which involves 20 frontier and open LLMs, is complemented by a matched human subjects experiment, which provides one reference point, while an expected payoff maximizing rational agent model provides another. We find that LLMs cluster into two categories: reasoning models (RMs) and conversational models (CMs). RMs tend towards rational behavior, are insensitive to the order of prospects, gain/loss framing, and explanations, and behave similarly whether prospects are explicit or presented via experience history. CMs are significantly less rational, slightly more human-like, sensitive to prospect ordering, framing, and explanation, and exhibit a large description-history gap. Paired comparisons of open LLMs suggest that a key factor differentiating RMs and CMs is training for mathematical reasoning.

使用大型语言模型作为决策支持系统或在代理工作流程中正在迅速改变数字生态系统。然而，人们对不确定性下的法学硕士决策的理解仍然有限。我们从两个维度启动了法学硕士风险选择的比较研究：（1）前景表征（明确的与基于经验的）和（2）决策理由（解释）。我们的研究涉及 20 名前沿和开放的法学硕士，并辅以匹配的人类受试者实验，该实验提供了一个参考点，而预期回报最大化理性代理模型则提供了另一个参考点。我们发现法学硕士分为两类：推理模型（RM）和会话模型（CM）。 RM 倾向于理性行为，对前景的顺序、收益/损失框架和解释不敏感，并且无论前景是明确的还是通过经验历史呈现的，其行为都相似。 CM 明显不那么理性，稍微更像人类，对前景排序、框架和解释敏感，并且表现出巨大的描述历史差距。开放式法学硕士的配对比较表明，区分 RM 和 CM 的关键因素是数学推理培训。

</details>

---

## 13. Hunt Globally: Wide Search AI Agents for Drug Asset Scouting in Investing, Business Development, and Competitive Intelligence / 全球狩猎：广泛搜索人工智能代理，用于投资、业务开发和竞争情报领域的药物资产搜寻

**Date**: 2026-02-16 | **arXiv**: [2602.15019v2](http://arxiv.org/abs/2602.15019v2) | **PDF**: [Link](http://arxiv.org/pdf/2602.15019v2)

**Categories**: cs.AI, cs.IR

<details><summary><b>Abstract / 摘要</b></summary>

Bio-pharmaceutical innovation has shifted: many new drug assets now originate outside the United States and are disclosed primarily via regional, non-English channels. Recent data suggests that over 85% of patent filings originate outside the U.S., with China accounting for nearly half of the global total. A growing share of scholarly output is also non-U.S. Industry estimates put China at 30% of global drug development, spanning 1,200+ novel candidates. In this high-stakes environment, failing to surface "under-the-radar" assets creates multi-billion-dollar risk for investors and business development teams, making asset scouting a coverage-critical competition where speed and completeness drive value. Yet today's Deep Research AI agents still lag human experts in achieving high recall discovery across heterogeneous, multilingual sources without hallucination. We propose a benchmarking methodology for drug asset scouting and a tuned, tree-based self-learning Bioptic Agent aimed at complete, non-hallucinated scouting. We construct a challenging completeness benchmark using a multilingual multi-agent pipeline: complex user queries paired with ground-truth assets that are largely outside U.S.-centric radar. To reflect real-deal complexity, we collected screening queries from expert investors, BD, and VC professionals and used them as priors to conditionally generate benchmark queries. For grading, we use LLM-as-judge evaluation calibrated to expert opinions. On this benchmark, our Bioptic Agent achieves 79.7% F1 score, outperforming Claude Opus 4.6 (56.2%), Gemini 3 Pro + Deep Research (50.6%), OpenAI GPT-5.2 Pro (46.6%), Perplexity Deep Research (44.2%), and Exa Websets (26.9%). Performance improves steeply with additional compute, supporting the view that more compute yields better results.

生物制药创新已经发生转变：许多新药资产现在源自美国境外，并主要通过地区性非英语渠道披露。最近的数据显示，超过 85% 的专利申请来自美国以外，其中中国占全球总量的近一半。非美国学术成果的份额越来越大。行业估计，中国的药物开发占全球药物开发的 30%，涉及 1,200 多种新候选药物。在这种高风险环境中，未能揭露“不为人知”的资产会给投资者和业务开发团队带来数十亿美元的风险，从而使资产搜寻成为覆盖范围至关重要的竞争，速度和完整性驱动价值。然而，当今的深度研究人工智能代理在跨异构、多语言来源实现高回忆发现而没有幻觉方面仍然落后于人类专家。我们提出了一种药物资产侦察的基准方法和一个经过调整的、基于树的自学习生物光学代理，旨在实现完整的、非幻觉的侦察。我们使用多语言多代理管道构建了一个具有挑战性的完整性基准：复杂的用户查询与基本位于以美国为中心的雷达之外的真实资产相结合。为了反映真实交易的复杂性，我们收集了来自专家投资者、BD 和 VC 专业人士的筛选查询，并将它们用作先验条件来有条件地生成基准查询。对于评分，我们使用法学硕士作为评判评估，并根据专家意见进行校准。在此基准测试中，我们的 Bioptic Agent 获得了 79.7% 的 F1 分数，优于 Claude Opus 4.6 (56.2%)、Gemini 3 Pro + Deep Research (50.6%)、OpenAI GPT-5.2 Pro (46.6%)、Perplexity Deep Research (44.2%) 和 Exa Websets (26.9%)。通过额外的计算，性能急剧提高，支持了更多计算产生更好结果的观点。

</details>

---

## 14. MAC-AMP: A Closed-Loop Multi-Agent Collaboration System for Multi-Objective Antimicrobial Peptide Design / MAC-AMP：用于多目标抗菌肽设计的闭环多代理协作系统

**Date**: 2026-02-16 | **arXiv**: [2602.14926v1](http://arxiv.org/abs/2602.14926v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14926v1)

**Categories**: cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

To address the global health threat of antimicrobial resistance, antimicrobial peptides (AMP) are being explored for their potent and promising ability to fight resistant pathogens. While artificial intelligence (AI) is being employed to advance AMP discovery and design, most AMP design models struggle to balance key goals like activity, toxicity, and novelty, using rigid or unclear scoring methods that make results hard to interpret and optimize. As the capabilities of Large Language Models (LLM) advance and evolve swiftly, we turn to AI multi-agent collaboration based on such models (multi-agent LLMs), which show rapidly rising potential in complex scientific design scenarios. Based on this, we introduce MAC-AMP, a closed-loop multi-agent collaboration (MAC) system for multi-objective AMP design. The system implements a fully autonomous simulated peer review-adaptive reinforcement learning framework that requires only a task description and example dataset to design novel AMPs. The novelty of our work lies in introducing a closed-loop multi-agent system for AMP design, with cross-domain transferability, that supports multi-objective optimization while remaining explainable rather than a 'black box'. Experiments show that MAC-AMP outperforms other AMP generative models by effectively optimizing AMP generation for multiple key molecular properties, demonstrating exceptional results in antibacterial activity, AMP likeliness, toxicity compliance, and structural reliability.

为了解决抗菌素耐药性对全球健康的威胁，人们正在探索抗菌肽 (AMP)，因为它们具有对抗耐药性病原体的强大且有前景的能力。虽然人工智能 (AI) 被用来推进 AMP 发现和设计，但大多数 AMP 设计模型都在努力平衡活性、毒性和新颖性等关键目标，使用严格或不明确的评分方法，导致结果难以解释和优化。随着大型语言模型（LLM）能力的快速进步和发展，我们转向基于此类模型的人工智能多智能体协作（多智能体LLM），它在复杂的科学设计场景中显示出快速增长的潜力。基于此，我们引入了MAC-AMP，一种用于多目标AMP设计的闭环多智能体协作（MAC）系统。该系统实现了一个完全自主的模拟同行评审自适应强化学习框架，只需任务描述和示例数据集即可设计新颖的 AMP。我们工作的新颖之处在于引入了用于 AMP 设计的闭环多智能体系统，具有跨域可转移性，支持多目标优化，同时保持可解释性，而不是“黑匣子”。实验表明，MAC-AMP 通过有效优化多个关键分子特性的 AMP 生成，优于其他 AMP 生成模型，在抗菌活性、AMP 可能性、毒性合规性和结构可靠性方面表现出优异的结果。

</details>

---

## 15. ReusStdFlow: A Standardized Reusability Framework for Dynamic Workflow Construction in Agentic AI / ReusStdFlow：Agentic AI 中动态工作流构建的标准化可重用性框架

**Date**: 2026-02-16 | **arXiv**: [2602.14922v1](http://arxiv.org/abs/2602.14922v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14922v1)

**Categories**: cs.AI, cs.SE

<details><summary><b>Abstract / 摘要</b></summary>

To address the ``reusability dilemma'' and structural hallucinations in enterprise Agentic AI,this paper proposes ReusStdFlow, a framework centered on a novel ``Extraction-Storage-Construction'' paradigm. The framework deconstructs heterogeneous, platform-specific Domain Specific Languages (DSLs) into standardized, modular workflow segments. It employs a dual knowledge architecture-integrating graph and vector databases-to facilitate synergistic retrieval of both topological structures and functional semantics. Finally, workflows are intelligently assembled using a retrieval-augmented generation (RAG) strategy. Tested on 200 real-world n8n workflows, the system achieves over 90% accuracy in both extraction and construction. This framework provides a standardized solution for the automated reorganization and efficient reuse of enterprise digital assets.

为了解决企业代理人工智能中的“可重用性困境”和结构幻觉，本文提出了 ReusStdFlow，一个以新颖的“提取-存储-构造”范式为中心的框架。该框架将异构的、特定于平台的领域特定语言 (DSL) 解构为标准化、模块化的工作流程段。它采用双知识架构——集成图和向量数据库——以促进拓扑结构和功能语义的协同检索。最后，使用检索增强生成（RAG）策略智能地组装工作流程。该系统在 200 个真实的 n8n 工作流程上进行了测试，在提取和构建方面都达到了 90% 以上的准确率。该框架为企业数字资产的自动化重组和高效复用提供了标准化的解决方案。

</details>

---

## 16. Picking the Right Specialist: Attentive Neural Process-based Selection of Task-Specialized Models as Tools for Agentic Healthcare Systems / 选择合适的专家：基于神经过程的仔细选择任务专用模型作为代理医疗保健系统的工具

**Date**: 2026-02-16 | **arXiv**: [2602.14901v1](http://arxiv.org/abs/2602.14901v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14901v1)

**Categories**: cs.LG, cs.AI, cs.CV, cs.MA

<details><summary><b>Abstract / 摘要</b></summary>

Task-specialized models form the backbone of agentic healthcare systems, enabling the agents to answer clinical queries across tasks such as disease diagnosis, localization, and report generation. Yet, for a given task, a single "best" model rarely exists. In practice, each task is better served by multiple competing specialist models where different models excel on different data samples. As a result, for any given query, agents must reliably select the right specialist model from a heterogeneous pool of tool candidates. To this end, we introduce ToolSelect, which adaptively learns model selection for tools by minimizing a population risk over sampled specialist tool candidates using a consistent surrogate of the task-conditional selection loss. Concretely, we propose an Attentive Neural Process-based selector conditioned on the query and per-model behavioral summaries to choose among the specialist models. Motivated by the absence of any established testbed, we, for the first time, introduce an agentic Chest X-ray environment equipped with a diverse suite of task-specialized models (17 disease detection, 19 report generation, 6 visual grounding, and 13 VQA) and develop ToolSelectBench, a benchmark of 1448 queries. Our results demonstrate that ToolSelect consistently outperforms 10 SOTA methods across four different task families.

任务专用模型构成了代理医疗保健系统的支柱，使代理能够跨疾病诊断、定位和报告生成等任务回答临床查询。然而，对于给定的任务，很少存在单一的“最佳”模型。在实践中，每个任务都可以通过多个相互竞争的专业模型更好地完成，其中不同的模型在不同的数据样本上表现出色。因此，对于任何给定的查询，代理必须从异构的候选工具池中可靠地选择正确的专家模型。为此，我们引入了 ToolSelect，它通过使用任务条件选择损失的一致替代来最小化采样的专业工具候选者的总体风险，从而自适应地学习工具的模型选择。具体来说，我们提出了一种基于注意力神经过程的选择器，以查询和每个模型的行为摘要为条件，以在专业模型中进行选择。由于缺乏任何已建立的测试平台，我们首次引入了一种代理胸部 X 射线环境，配备了多种任务专用模型（17 个疾病检测、19 个报告生成、6 个视觉基础和 13 个 VQA），并开发了 ToolSelectBench（1448 个查询的基准）。我们的结果表明，ToolSelect 在四个不同的任务系列中始终优于 10 个 SOTA 方法。

</details>

---

## 17. Atomix: Timely, Transactional Tool Use for Reliable Agentic Workflows / Atomix：及时的事务性工具用于可靠的代理工作流程

**Date**: 2026-02-16 | **arXiv**: [2602.14849v1](http://arxiv.org/abs/2602.14849v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14849v1)

**Categories**: cs.LG, cs.AI, cs.DC, cs.MA

<details><summary><b>Abstract / 摘要</b></summary>

LLM agents increasingly act on external systems, yet tool effects are immediate. Under failures, speculation, or contention, losing branches can leak unintended side effects with no safe rollback. We introduce Atomix, a runtime that provides progress-aware transactional semantics for agent tool calls. Atomix tags each call with an epoch, tracks per-resource frontiers, and commits only when progress predicates indicate safety; bufferable effects can be delayed, while externalized effects are tracked and compensated on abort. Across real workloads with fault injection, transactional retry improves task success, while frontier-gated commit strengthens isolation under speculation and contention.

LLM 代理越来越多地作用于外部系统，但工具效果是立竿见影的。在失败、猜测或争用的情况下，丢失分支可能会泄漏意想不到的副作用，并且无法安全回滚。我们引入了 Atomix，一个为代理工具调用提供进度感知事务语义的运行时。 Atomix 用纪元标记每个调用，跟踪每个资源的边界，并仅在进度谓词表明安全时才提交；可缓冲的效果可以被延迟，而外部化的效果在中止时被跟踪和补偿。在具有故障注入的实际工作负载中，事务性重试可以提高任务成功率，而边界门控提交则可以加强猜测和争用下的隔离。

</details>

---

## 18. A Geometric Analysis of Small-sized Language Model Hallucinations / 小规模语言模型幻象的几何分析

**Date**: 2026-02-16 | **arXiv**: [2602.14778v1](http://arxiv.org/abs/2602.14778v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14778v1)

**Categories**: cs.CL, cs.AI, cs.CY

<details><summary><b>Abstract / 摘要</b></summary>

Hallucinations -- fluent but factually incorrect responses -- pose a major challenge to the reliability of language models, especially in multi-step or agentic settings.   This work investigates hallucinations in small-sized LLMs through a geometric perspective, starting from the hypothesis that when models generate multiple responses to the same prompt, genuine ones exhibit tighter clustering in the embedding space, we prove this hypothesis and, leveraging this geometrical insight, we also show that it is possible to achieve a consistent level of separability. This latter result is used to introduce a label-efficient propagation method that classifies large collections of responses from just 30-50 annotations, achieving F1 scores above 90%.   Our findings, framing hallucinations from a geometric perspective in the embedding space, complement traditional knowledge-centric and single-response evaluation paradigms, paving the way for further research.

幻觉——流畅但实际上不正确的反应——对语言模型的可靠性构成了重大挑战，特别是在多步骤或代理环境中。   这项工作通过几何角度研究小型法学硕士中的幻觉，从假设开始，当模型对同一提示生成多个响应时，真实的响应在嵌入空间中表现出更紧密的聚类，我们证明了这一假设，并利用这种几何洞察力，我们还表明有可能实现一致的可分离性水平。后一个结果用于引入一种标签高效的传播方法，该方法可以对仅 30-50 个注释的大量响应集合进行分类，实现 F1 分数高于 90%。   我们的研究结果从嵌入空间的几何角度构建幻觉，补充了传统的以知识为中心和单一响应的评估范式，为进一步的研究铺平了道路。

</details>

---

## 19. Multi-Agent Comedy Club: Investigating Community Discussion Effects on LLM Humor Generation / 多代理喜剧俱乐部：调查社区讨论对法学硕士幽默生成的影响

**Date**: 2026-02-16 | **arXiv**: [2602.14770v2](http://arxiv.org/abs/2602.14770v2) | **PDF**: [Link](http://arxiv.org/pdf/2602.14770v2)

**Categories**: cs.CL, cs.AI, cs.CY, cs.HC

<details><summary><b>Abstract / 摘要</b></summary>

Prior work has explored multi-turn interaction and feedback for LLM writing, but evaluations still largely center on prompts and localized feedback, leaving persistent public reception in online communities underexamined. We test whether broadcast community discussion improves stand-up comedy writing in a controlled multi-agent sandbox: in the discussion condition, critic and audience threads are recorded, filtered, stored as social memory, and later retrieved to condition subsequent generations, whereas the baseline omits discussion. Across 50 rounds (250 paired monologues) judged by five expert annotators using A/B preference and a 15-item rubric, discussion wins 75.6% of instances and improves Craft/Clarity (Δ = 0.440) and Social Response (Δ = 0.422), with occasional increases in aggressive humor.

之前的工作已经探索了法学硕士写作的多轮互动和反馈，但评估仍然主要集中在提示和本地化反馈上，导致在线社区中持续的公众接受度未被充分审查。我们测试广播社区讨论是否可以在受控的多代理沙箱中改善单口喜剧写作：在讨论条件下，评论家和观众的线索被记录、过滤、存储为社会记忆，然后检索以调节后代，而基线则忽略讨论。在由五位专家注释者使用 A/B 偏好和 15 项评分标准进行的 50 轮（250 配对独白）中，讨论赢得了 75.6% 的实例，并提高了工艺/清晰度 (Δ = 0.440) 和社交反应 (Δ = 0.422)，偶尔会增加攻击性幽默。

</details>

---

## 20. Evolutionary System Prompt Learning can Facilitate Reinforcement Learning for LLMs / 进化系统即时学习可以促进法学硕士的强化学习

**Date**: 2026-02-16 | **arXiv**: [2602.14697v1](http://arxiv.org/abs/2602.14697v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14697v1)

**Categories**: cs.AI, cs.LG

**Code**: https://github.com/LunjunZhang/E-SPL

<details><summary><b>Abstract / 摘要</b></summary>

Building agentic systems that can autonomously self-improve from experience is a longstanding goal of AI. Large language models (LLMs) today primarily self-improve via two mechanisms: self-reflection for context updates, and reinforcement learning (RL) for weight updates. In this work, we propose Evolutionary System Prompt Learning (E-SPL), a method for jointly improving model contexts and model weights. In each RL iteration, E-SPL selects multiple system prompts and runs rollouts with each in parallel. It applies RL updates to model weights conditioned on each system prompt, and evolutionary updates to the system prompt population via LLM-driven mutation and crossover. Each system prompt has a TrueSkill rating for evolutionary selection, updated from relative performance within each RL iteration batch. E-SPL encourages a natural division between declarative knowledge encoded in prompts and procedural knowledge encoded in weights, resulting in improved performance across reasoning and agentic tasks. For instance, in an easy-to-hard (AIME $\rightarrow$ BeyondAIME) generalization setting, E-SPL improves RL success rate from 38.8% $\rightarrow$ 45.1% while also outperforming reflective prompt evolution (40.0%). Overall, our results show that coupling reinforcement learning with system prompt evolution yields consistent gains in sample efficiency and generalization. Code: https://github.com/LunjunZhang/E-SPL

构建能够根据经验自主自我改进的代理系统是人工智能的长期目标。如今，大型语言模型 (LLM) 主要通过两种机制进行自我改进：用于上下文更新的自我反思和用于权重更新的强化学习 (RL)。在这项工作中，我们提出了进化系统即时学习（E-SPL），这是一种联合改进模型上下文和模型权重的方法。在每次 RL 迭代中，E-SPL 选择多个系统提示并并行运行每个提示。它将 RL 更新应用于以每个系统提示为条件的模型权重，并通过 LLM 驱动的突变和交叉对系统提示群体进行进化更新。每个系统提示都有一个用于进化选择的 TrueSkill 评级，根据每个 RL 迭代批次内的相对表现进行更新。 E-SPL 鼓励在提示中编码的声明性知识和在权重中编码的程序性知识之间进行自然划分，从而提高推理和代理任务的性能。例如，在从易到难（AIME $\rightarrow$ BeyondAIME）泛化设置中，E-SPL 将 RL 成功率从 38.8% $\rightarrow$ 提高到 45.1%，同时也优于反射提示进化 (40.0%)。总的来说，我们的结果表明，将强化学习与系统即时进化相结合可以在样本效率和泛化方面产生一致的收益。代码：https://github.com/LunjunZhang/E-SPL

</details>

---

## 21. ST-EVO: Towards Generative Spatio-Temporal Evolution of Multi-Agent Communication Topologies / ST-EVO：迈向多智能体通信拓扑的生成时空演化

**Date**: 2026-02-16 | **arXiv**: [2602.14681v1](http://arxiv.org/abs/2602.14681v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14681v1)

**Categories**: cs.MA, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

LLM-powered Multi-Agent Systems (MAS) have emerged as an effective approach towards collaborative intelligence, and have attracted wide research interests. Among them, ``self-evolving'' MAS, treated as a more flexible and powerful technical route, can construct task-adaptive workflows or communication topologies, instead of relying on a predefined static structue template. Current self-evolving MAS mainly focus on Spatial Evolving or Temporal Evolving paradigm, which only considers the single dimension of evolution and does not fully incentivize LLMs' collaborative capability. In this work, we start from a novel Spatio-Temporal perspective by proposing ST-EVO, which supports dialogue-wise communication scheduling with a compact yet powerful flow-matching based Scheduler. To make precise Spatio-Temporal scheduling, ST-EVO can also perceive the uncertainty of MAS, and possesses self-feedback ability to learn from accumulated experience. Extensive experiments on nine benchmarks demonstrate the state-of-the-art performance of ST-EVO, achieving about 5%--25% accuracy improvement.

由法学硕士支持的多智能体系统（MAS）已成为实现协作智能的有效方法，并吸引了广泛的研究兴趣。其中，“自我进化”的MAS被视为一种更加灵活和强大的技术路线，可以构建任务自适应的工作流或通信拓扑，而不是依赖于预定义的静态结构模板。目前的自进化MAS主要集中在空间进化或时间进化范式，仅考虑进化的单一维度，并没有充分激励LLM的协作能力。在这项工作中，我们从新颖的时空角度出发，提出了 ST-EVO，它通过紧凑而强大的基于流匹配的调度器支持对话式通信调度。为了进行精确的时空调度，ST-EVO还可以感知MAS的不确定性，并具有从积累的经验中学习的自我反馈能力。对九个基准的大量实验证明了 ST-EVO 的最先进性能，实现了约 5%--25% 的精度提升。

</details>

---

## 22. Towards Selection as Power: Bounding Decision Authority in Autonomous Agents / 走向选择作为权力：限制自治代理的决策权

**Date**: 2026-02-16 | **arXiv**: [2602.14606v1](http://arxiv.org/abs/2602.14606v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14606v1)

**Categories**: cs.MA, cs.AI, cs.CE

<details><summary><b>Abstract / 摘要</b></summary>

Autonomous agentic systems are increasingly deployed in regulated, high-stakes domains where decisions may be irreversible and institutionally constrained. Existing safety approaches emphasize alignment, interpretability, or action-level filtering. We argue that these mechanisms are necessary but insufficient because they do not directly govern selection power: the authority to determine which options are generated, surfaced, and framed for decision. We propose a governance architecture that separates cognition, selection, and action into distinct domains and models autonomy as a vector of sovereignty. Cognitive autonomy remains unconstrained, while selection and action autonomy are bounded through mechanically enforced primitives operating outside the agent's optimization space. The architecture integrates external candidate generation (CEFL), a governed reducer, commit-reveal entropy isolation, rationale validation, and fail-loud circuit breakers. We evaluate the system across multiple regulated financial scenarios under adversarial stress targeting variance manipulation, threshold gaming, framing skew, ordering effects, and entropy probing. Metrics quantify selection concentration, narrative diversity, governance activation cost, and failure visibility. Results show that mechanical selection governance is implementable, auditable, and prevents deterministic outcome capture while preserving reasoning capacity. Although probabilistic concentration remains, the architecture measurably bounds selection authority relative to conventional scalar pipelines. This work reframes governance as bounded causal power rather than internal intent alignment, offering a foundation for deploying autonomous agents where silent failure is unacceptable.

自主代理系统越来越多地部署在受监管的高风险领域，这些领域的决策可能是不可逆转的且受到制度限制。现有的安全方法强调一致性、可解释性或操作级别过滤。我们认为，这些机制是必要的，但还不够，因为它们不直接支配选择权：决定生成、浮现和制定哪些选项以供决策的权力。我们提出了一种治理架构，将认知、选择和行动分为不同的领域，并将自治建模为主权的载体。认知自主权仍然不受约束，而选择和行动自主权则通过在代理优化空间之外运行的机械强制原语来限制。该架构集成了外部候选生成 (CEFL)、受控减速器、提交-显示熵隔离、基本原理验证和故障大声断路器。我们在对抗性压力目标方差操纵、阈值博弈、框架倾斜、排序效应和熵探测等多种受监管的金融场景下评估系统。指标量化选择集中度、叙述多样性、治理激活成本和失败可见性。结果表明，机械选择治理是可实施的、可审计的，并且可以防止确定性结果捕获，同时保留推理能力。尽管概率集中仍然存在，但相对于传统的标量管道，该架构明显限制了选择权限。这项工作将治理重新定义为有限的因果力量，而不是内部意图一致性，为在不可接受的无声故障中部署自主代理提供了基础。

</details>

---

## 23. Fluid-Agent Reinforcement Learning / 流体剂强化学习

**Date**: 2026-02-16 | **arXiv**: [2602.14559v1](http://arxiv.org/abs/2602.14559v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14559v1)

**Categories**: cs.LG, cs.AI, cs.MA

<details><summary><b>Abstract / 摘要</b></summary>

The primary focus of multi-agent reinforcement learning (MARL) has been to study interactions among a fixed number of agents embedded in an environment. However, in the real world, the number of agents is neither fixed nor known a priori. Moreover, an agent can decide to create other agents (for example, a cell may divide, or a company may spin off a division). In this paper, we propose a framework that allows agents to create other agents; we call this a fluid-agent environment. We present game-theoretic solution concepts for fluid-agent games and empirically evaluate the performance of several MARL algorithms within this framework. Our experiments include fluid variants of established benchmarks such as Predator-Prey and Level-Based Foraging, where agents can dynamically spawn, as well as a new environment we introduce that highlights how fluidity can unlock novel solution strategies beyond those observed in fixed-population settings. We demonstrate that this framework yields agent teams that adjust their size dynamically to match environmental demands.

多智能体强化学习 (MARL) 的主要焦点是研究嵌入环境中的固定数量智能体之间的交互。然而，在现实世界中，代理的数量既不是固定的，也不是先验已知的。此外，一个代理可以决定创建其他代理（例如，一个细胞可以分裂，或者一个公司可以分拆一个部门）。在本文中，我们提出了一个允许代理创建其他代理的框架；我们称之为流体剂环境。我们提出了流体代理博弈的博弈论解决方案概念，并根据经验评估了该框架内几种 MARL 算法的性能。我们的实验包括既定基准的流体变体，例如捕食者-猎物和基于水平的觅食，其中代理可以动态生成，以及我们引入的新环境，该环境强调流动性如何能够解锁超出固定种群设置中观察到的新颖解决方案策略。我们证明，该框架产生的代理团队可以动态调整其规模以适应环境需求。

</details>

---

## 24. Socially-Weighted Alignment: A Game-Theoretic Framework for Multi-Agent LLM Systems / 社会加权对齐：多代理法学硕士系统的博弈论框架

**Date**: 2026-02-16 | **arXiv**: [2602.14471v1](http://arxiv.org/abs/2602.14471v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14471v1)

**Categories**: cs.MA, cs.AI, cs.GT, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Deploying large language model (LLM) agents in shared environments introduces a fundamental tension between individual alignment and collective stability: locally rational decisions can impose negative externalities that degrade system-level performance. We propose Socially-Weighted Alignment (SWA), a game-theoretic framework that modifies inference-time decision making by interpolating between an agent's private objective and an estimate of group welfare via a social weight $λ\in[0,1]$. In a shared-resource congestion game with $n$ agents and congestion severity $β$, we show that SWA induces a critical threshold $λ^*=(n-β)/(n-1)$ above which agents no longer have marginal incentive to increase demand under overload, yielding a phase transition from persistent congestion to stable operation near capacity. We further provide an inference-time algorithmic instantiation of SWA that does not require parameter updates or multi-agent reinforcement learning, and use a multi-agent simulation to empirically validate the predicted threshold behavior.

在共享环境中部署大型语言模型（LLM）代理会在个体一致性和集体稳定性之间引入根本性的紧张关系：局部理性决策可能会带来负面外部性，从而降低系统级性能。我们提出了社会加权对齐（SWA），这是一种博弈论框架，它通过社会权重 $λ\in[0,1]$ 在代理的私人目标和群体福利估计之间进行插值，从而修改推理时间决策。在具有 $n$ 代理和拥塞严重程度 $β$ 的共享资源拥塞博弈中，我们表明 SWA 引入了一个临界阈值 $λ^*=(n-β)/(n-1)$，高于该阈值代理不再有边际动机在过载情况下增加需求，从而产生从持续拥塞到接近容量的稳定运行的阶段过渡。我们进一步提供了 SWA 的推理时间算法实例，不需要参数更新或多代理强化学习，并使用多代理模拟来凭经验验证预测的阈值行为。

</details>

---

## 25. Frontier AI Risk Management Framework in Practice: A Risk Analysis Technical Report v1.5 / 实践中的前沿人工智能风险管理框架：风险分析技术报告 v1.5

**Date**: 2026-02-16 | **arXiv**: [2602.14457v1](http://arxiv.org/abs/2602.14457v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14457v1)

**Categories**: cs.AI, cs.CL, cs.CV, cs.CY, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

To understand and identify the unprecedented risks posed by rapidly advancing artificial intelligence (AI) models, Frontier AI Risk Management Framework in Practice presents a comprehensive assessment of their frontier risks. As Large Language Models (LLMs) general capabilities rapidly evolve and the proliferation of agentic AI, this version of the risk analysis technical report presents an updated and granular assessment of five critical dimensions: cyber offense, persuasion and manipulation, strategic deception, uncontrolled AI R\&D, and self-replication. Specifically, we introduce more complex scenarios for cyber offense. For persuasion and manipulation, we evaluate the risk of LLM-to-LLM persuasion on newly released LLMs. For strategic deception and scheming, we add the new experiment with respect to emergent misalignment. For uncontrolled AI R\&D, we focus on the ``mis-evolution'' of agents as they autonomously expand their memory substrates and toolsets. Besides, we also monitor and evaluate the safety performance of OpenClaw during the interaction on the Moltbook. For self-replication, we introduce a new resource-constrained scenario. More importantly, we propose and validate a series of robust mitigation strategies to address these emerging threats, providing a preliminary technical and actionable pathway for the secure deployment of frontier AI. This work reflects our current understanding of AI frontier risks and urges collective action to mitigate these challenges.

为了了解和识别快速发展的人工智能（AI）模型所带来的前所未有的风险，实践中的前沿人工智能风险管理框架对其前沿风险进行了全面评估。随着大型语言模型（LLM）通用能力的快速发展和代理人工智能的扩散，此版本的风险分析技术报告对五个关键维度进行了更新和精细的评估：网络攻击、说服和操纵、战略欺骗、不受控制的人工智能研发和自我复制。具体来说，我们引入了更复杂的网络攻击场景。对于说服和操纵，我们评估了新发布的法学硕士的法学硕士到法学硕士的说服风险。对于战略欺骗和阴谋，我们添加了关于紧急错位的新实验。对于不受控制的人工智能研发，我们重点关注代理自主扩展其内存基质和工具集时的“错误进化”。此外，我们还对OpenClaw在Moltbook上交互过程中的安全性能进行监控和评估。对于自我复制，我们引入了一种新的资源受限场景。更重要的是，我们提出并验证了一系列强有力的缓解策略来应对这些新兴威胁，为前沿人工智能的安全部署提供了初步的技术和可操作的途径。这项工作反映了我们目前对人工智能前沿风险的理解，并敦促采取集体行动来缓解这些挑战。

</details>

---

## 26. Tool-Aware Planning in Contact Center AI: Evaluating LLMs through Lineage-Guided Query Decomposition / 联络中心人工智能中的工具感知规划：通过沿袭引导的查询分解评估法学硕士

**Date**: 2026-02-16 | **arXiv**: [2602.14955v1](http://arxiv.org/abs/2602.14955v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14955v1)

**Categories**: cs.CL, cs.SE

<details><summary><b>Abstract / 摘要</b></summary>

We present a domain-grounded framework and benchmark for tool-aware plan generation in contact centers, where answering a query for business insights, our target use case, requires decomposing it into executable steps over structured tools (Text2SQL (T2S)/Snowflake) and unstructured tools (RAG/transcripts) with explicit depends_on for parallelism. Our contributions are threefold: (i) a reference-based plan evaluation framework operating in two modes - a metric-wise evaluator spanning seven dimensions (e.g., tool-prompt alignment, query adherence) and a one-shot evaluator; (ii) a data curation methodology that iteratively refines plans via an evaluator->optimizer loop to produce high-quality plan lineages (ordered plan revisions) while reducing manual effort; and (iii) a large-scale study of 14 LLMs across sizes and families for their ability to decompose queries into step-by-step, executable, and tool-assigned plans, evaluated under prompts with and without lineage. Empirically, LLMs struggle on compound queries and on plans exceeding 4 steps (typically 5-15); the best total metric score reaches 84.8% (Claude-3-7-Sonnet), while the strongest one-shot match rate at the "A+" tier (Extremely Good, Very Good) is only 49.75% (o3-mini). Plan lineage yields mixed gains overall but benefits several top models and improves step executability for many. Our results highlight persistent gaps in tool-understanding, especially in tool-prompt alignment and tool-usage completeness, and show that shorter, simpler plans are markedly easier. The framework and findings provide a reproducible path for assessing and improving agentic planning with tools for answering data-analysis queries in contact-center settings.

我们提出了一个基于领域的框架和基准，用于联络中心的工具感知计划生成，其中回答业务洞察查询（我们的目标用例）需要将其分解为结构化工具（Text2SQL（T2S）/Snowflake）和非结构化工具（RAG/transcripts）上的可执行步骤，并具有显式的depends_on以实现并行性。我们的贡献有三个方面：（i）一个以两种模式运行的基于参考的计划评估框架——一个跨越七个维度（例如，工具提示对齐、查询遵守）的度量评估器和一个一次性评估器； (ii) 一种数据管理方法，通过评估器->优化器循环迭代地完善计划，以生成高质量的计划谱系（有序的计划修订），同时减少手动工作； (iii) 对 14 名不同规模和家庭的法学硕士进行了大规模研究，研究他们将查询分解为逐步的、可执行的和工具分配的计划的能力，并在有或没有血统的提示下进行评估。根据经验，法学硕士在复合查询和超过 4 个步骤（通常为 5-15 个）的计划上遇到了困难；最好的总指标达到84.8%（Claude-3-7-Sonnet），而“A+”级别的最强单次匹配率（Extremely Good、Very Good）仅为49.75%（o3-mini）。计划沿袭总体上带来了好坏参半的收益，但有利于几个顶级模型，并提高了许多步骤的可执行性。我们的结果凸显了工具理解方面持续存在的差距，特别是在工具提示对齐和工具使用完整性方面，并表明更短、更简单的计划明显更容易。该框架和研究结果提供了一条可重复的路径，用于通过在联络中心设置中回答数据分析查询的工具来评估和改进代理规划。

</details>

---

## 27. Distributed Quantum Gaussian Processes for Multi-Agent Systems / 多智能体系统的分布式量子高斯过程

**Date**: 2026-02-16 | **arXiv**: [2602.15006v1](http://arxiv.org/abs/2602.15006v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.15006v1)

**Categories**: cs.MA, cs.LG, math.DG

<details><summary><b>Abstract / 摘要</b></summary>

Gaussian Processes (GPs) are a powerful tool for probabilistic modeling, but their performance is often constrained in complex, largescale real-world domains due to the limited expressivity of classical kernels. Quantum computing offers the potential to overcome this limitation by embedding data into exponentially large Hilbert spaces, capturing complex correlations that remain inaccessible to classical computing approaches. In this paper, we propose a Distributed Quantum Gaussian Process (DQGP) method in a multiagent setting to enhance modeling capabilities and scalability. To address the challenging non-Euclidean optimization problem, we develop a Distributed consensus Riemannian Alternating Direction Method of Multipliers (DR-ADMM) algorithm that aggregates local agent models into a global model. We evaluate the efficacy of our method through numerical experiments conducted on a quantum simulator in classical hardware. We use real-world, non-stationary elevation datasets of NASA's Shuttle Radar Topography Mission and synthetic datasets generated by Quantum Gaussian Processes. Beyond modeling advantages, our framework highlights potential computational speedups that quantum hardware may provide, particularly in Gaussian processes and distributed optimization.

高斯过程（GP）是概率建模的强大工具，但由于经典核的表达能力有限，其性能通常在复杂、大规模的现实世界领域受到限制。量子计算通过将数据嵌入到指数级大的希尔伯特空间中，捕获传统计算方法无法访问的复杂相关性，提供了克服这一限制的潜力。在本文中，我们提出了多智能体设置中的分布式量子高斯过程（DQGP）方法，以增强建模能力和可扩展性。为了解决具有挑战性的非欧几里得优化问题，我们开发了一种分布式共识黎曼交替方向乘子法 (DR-ADMM) 算法，该算法将局部代理模型聚合为全局模型。我们通过在经典硬件中的量子模拟器上进行的数值实验来评估我们方法的有效性。我们使用美国宇航局航天飞机雷达地形任务的真实世界非平稳高程数据集以及量子高斯过程生成的合成数据集。除了建模优势之外，我们的框架还强调了量子硬件可能提供的潜在计算加速，特别是在高斯过程和分布式优化中。

</details>

---

## 28. Scalable Multi-Robot Path Planning via Quadratic Unconstrained Binary Optimization / 通过二次无约束二元优化的可扩展多机器人路径规划

**Date**: 2026-02-16 | **arXiv**: [2602.14799v1](http://arxiv.org/abs/2602.14799v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14799v1)

**Categories**: cs.RO, quant-ph

<details><summary><b>Abstract / 摘要</b></summary>

Multi-Agent Path Finding (MAPF) remains a fundamental challenge in robotics, where classical centralized approaches exhibit exponential growth in joint-state complexity as the number of agents increases. This paper investigates Quadratic Unconstrained Binary Optimization (QUBO) as a structurally scalable alternative for simultaneous multi-robot path planning. This approach is a robotics-oriented QUBO formulation incorporating BFS-based logical pre-processing (achieving over 95% variable reduction), adaptive penalty design for collision and constraint enforcement, and a time-windowed decomposition strategy that enables execution within current hardware limitations. An experimental evaluation in grid environments with up to four robots demonstrated near-optimal solutions in dense scenarios and favorable scaling behavior compared to sequential classical planning. These results establish a practical and reproducible baseline for future quantum and quantum-inspired multi-robot coordinations.

多智能体路径查找（MAPF）仍然是机器人技术中的一个基本挑战，随着智能体数量的增加，经典的集中式方法的联合状态复杂性呈现指数增长。本文研究了二次无约束二元优化（QUBO）作为同步多机器人路径规划的结构可扩展替代方案。该方法是一种面向机器人的 QUBO 公式，结合了基于 BFS 的逻辑预处理（实现超过 95% 的变量减少）、碰撞和约束执行的自适应惩罚设计，以及能够在当前硬件限制内执行的时间窗口分解策略。在最多四个机器人的网格环境中进行的实验评估表明，与顺序经典规划相比，密集场景中的解决方案接近最优，并且具有良好的扩展行为。这些结果为未来的量子和量子启发的多机器人协调建立了实用且可重复的基线。

</details>

---

## 29. ROSA: Roundabout Optimized Speed Advisory with Multi-Agent Trajectory Prediction in Multimodal Traffic / ROSA：多式联运中具有多智能体轨迹预测的环岛优化速度咨询

**Date**: 2026-02-16 | **arXiv**: [2602.14780v1](http://arxiv.org/abs/2602.14780v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14780v1)

**Categories**: cs.MA, cs.CY, cs.RO, eess.SY

<details><summary><b>Abstract / 摘要</b></summary>

We present ROSA -- Roundabout Optimized Speed Advisory -- a system that combines multi-agent trajectory prediction with coordinated speed guidance for multimodal, mixed traffic at roundabouts. Using a Transformer-based model, ROSA jointly predicts the future trajectories of vehicles and Vulnerable Road Users (VRUs) at roundabouts. Trained for single-step prediction and deployed autoregressively, it generates deterministic outputs, enabling actionable speed advisories. Incorporating motion dynamics, the model achieves high accuracy (ADE: 1.29m, FDE: 2.99m at a five-second prediction horizon), surpassing prior work. Adding route intention further improves performance (ADE: 1.10m, FDE: 2.36m), demonstrating the value of connected vehicle data. Based on predicted conflicts with VRUs and circulating vehicles, ROSA provides real-time, proactive speed advisories for approaching and entering the roundabout. Despite prediction uncertainty, ROSA significantly improves vehicle efficiency and safety, with positive effects even on perceived safety from a VRU perspective. The source code of this work is available under: github.com/urbanAIthi/ROSA.

我们推出了 ROSA（环岛优化速度咨询）系统，该系统将多智能体轨迹预测与环岛多式联运混合交通的协调速度指导相结合。 ROSA 使用基于 Transformer 的模型联合预测环岛处车辆和弱势道路使用者 (VRU) 的未来轨迹。它经过单步预测训练并以自回归方式部署，可生成确定性输出，从而实现可操作的速度建议。结合运动动力学，该模型实现了高精度（ADE：1.29m，FDE：5 秒预测范围内的 2.99m），超越了之前的工作。添加路线意​​图进一步提高性能（ADE：1.10m，FDE：2.36m），展示了联网车辆数据的价值。根据与 VRU 和流通车辆的预测冲突，ROSA 为接近和进入环岛提供实时、主动的速度建议。尽管预测存在不确定性，ROSA 仍显着提高了车辆效率和安全性，甚至从 VRU 角度来看对感知安全性也产生了积极影响。这项工作的源代码位于：github.com/urbanAIthi/ROSA。

</details>

---

## 30. RoboSolver: A Multi-Agent Large Language Model Framework for Solving Robotic Arm Problems / RoboSolver：解决机械臂问题的多智能体大语言模型框架

**Date**: 2026-02-16 | **arXiv**: [2602.14438v1](http://arxiv.org/abs/2602.14438v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14438v1)

**Categories**: cs.RO, cs.MA

<details><summary><b>Abstract / 摘要</b></summary>

This study proposes an intelligent multi-agent framework built on LLMs and VLMs and specifically tailored to robotics. The goal is to integrate the strengths of LLMs and VLMs with computational tools to automatically analyze and solve problems related to robotic manipulators. Our developed framework accepts both textual and visual inputs and can automatically perform forward and inverse kinematics, compute velocities and accelerations of key points, generate 3D simulations of the robot, and ultimately execute motion control within the simulated environment, all according to the user's query. To evaluate the framework, three benchmark tests were designed, each consisting of ten questions. In the first benchmark test, the framework was evaluated while connected to GPT-4o, DeepSeek-V3.2, and Claude-Sonnet-4.5, as well as their corresponding raw models. The objective was to extract the forward kinematics of robots directly from textual descriptions. The results showed that the framework integrated with GPT-4o achieved the highest accuracy, reaching 0.97 in computing the final solution, whereas the raw model alone attained an accuracy of only 0.30 for the same task. Similarly, for the other two models, the framework consistently outperformed the corresponding raw models in terms of accuracy. The second benchmark test was identical to the first, except that the input was provided in visual form. In this test, the GPT-4o LLM was used alongside the Gemini 2.5 Pro VLM. The results showed that the framework achieved an accuracy of 0.93 in obtaining the final answer, which is approximately 20% higher than that of the corresponding raw model. The third benchmark test encompassed a range of robotic tasks, including simulation, control, velocity and acceleration computation, as well as inverse kinematics and Jacobian calculation, for which the framework achieved an accuracy of 0.97.

本研究提出了一种基于 LLM 和 VLM 且专门针对机器人技术定制的智能多代理框架。目标是将 LLM 和 VLM 的优势与计算工具相结合，以自动分析和解决与机器人操纵器相关的问题。我们开发的框架接受文本和视觉输入，可以自动执行正向和反向运动学，计算关键点的速度和加速度，生成机器人的 3D 模拟，并最终在模拟环境中执行运动控制，所有这些都根据用户的查询。为了评估该框架，设计了三个基准测试，每个测试包含十个问题。在第一个基准测试中，框架在连接到 GPT-4o、DeepSeek-V3.2 和 Claude-Sonnet-4.5 及其相应的原始模型时进行了评估。目的是直接从文本描述中提取机器人的正向运动学。结果表明，与 GPT-4o 集成的框架实现了最高的准确度，在计算最终解决方案时达到了 0.97，而单独的原始模型对于相同任务的准确度仅为 0.30。同样，对于其他两个模型，该框架在准确性方面始终优于相应的原始模型。第二个基准测试与第一个相同，只是输入以视觉形式提供。在此测试中，GPT-4o LLM 与 Gemini 2.5 Pro VLM 一起使用。结果表明，该框架在获得最终答案时的准确率达到了0.93，比相应的原始模型高出约20%。第三个基准测试涵盖了一系列机器人任务，包括模拟、控制、速度和加速度计算，以及逆运动学和雅可比计算，该框架的精度达到了0.97。

</details>

---

## 31. MCPShield: A Security Cognition Layer for Adaptive Trust Calibration in Model Context Protocol Agents / MCPShield：用于模型上下文协议代理中自适应信任校准的安全认知层

**Date**: 2026-02-15 | **arXiv**: [2602.14281v1](http://arxiv.org/abs/2602.14281v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14281v1)

**Categories**: cs.CR, cs.CL

<details><summary><b>Abstract / 摘要</b></summary>

The Model Context Protocol (MCP) standardizes tool use for LLM-based agents and enable third-party servers. This openness introduces a security misalignment: agents implicitly trust tools exposed by potentially untrusted MCP servers. However, despite its excellent utility, existing agents typically offer limited validation for third-party MCP servers. As a result, agents remain vulnerable to MCP-based attacks that exploit the misalignment between agents and servers throughout the tool invocation lifecycle. In this paper, we propose MCPShield as a plug-in security cognition layer that mitigates this misalignment and ensures agent security when invoking MCP-based tools. Drawing inspiration from human experience-driven tool validation, MCPShield assists agent forms security cognition with metadata-guided probing before invocation. Our method constrains execution within controlled boundaries while cognizing runtime events, and subsequently updates security cognition by reasoning over historical traces after invocation, building on human post-use reflection on tool behavior. Experiments demonstrate that MCPShield exhibits strong generalization in defending against six novel MCP-based attack scenarios across six widely used agentic LLMs, while avoiding false positives on benign servers and incurring low deployment overhead. Overall, our work provides a practical and robust security safeguard for MCP-based tool invocation in open agent ecosystems.

模型上下文协议 (MCP) 标准化了基于 LLM 的代理的工具使用并支持第三方服务器。这种开放性引入了安全错位：代理隐式信任潜在不受信任的 MCP 服务器公开的工具。然而，尽管其实用性极佳，现有代理通常为第三方 MCP 服务器提供有限的验证。因此，代理仍然容易受到基于 MCP 的攻击，这些攻击在整个工具调用生命周期中利用代理和服务器之间的不一致。在本文中，我们提出 MCPShield 作为插件安全认知层，可以减轻这种不一致并确保代理在调用基于 MCP 的工具时的安全性。 MCPShield 从人类经验驱动的工具验证中汲取灵感，通过元数据引导的探测在调用前帮助代理形成安全认知。我们的方法在认知运行时事件的同时将执行限制在受控边界内，随后通过对调用后的历史跟踪进行推理来更新安全认知，建立在人类对工具行为的使用后反思的基础上。实验表明，MCPShield 在防御六种广泛使用的代理 LLM 中的六种基于 MCP 的新型攻击场景方面表现出很强的通用性，同时避免良性服务器上的误报并产生较低的部署开销。总的来说，我们的工作为开放代理生态系统中基于 MCP 的工具调用提供了实用且强大的安全保障。

</details>

---

## 32. REDSearcher: A Scalable and Cost-Efficient Framework for Long-Horizon Search Agents / REDSearcher：用于长视野搜索代理的可扩展且经济高效的框架

**Date**: 2026-02-15 | **arXiv**: [2602.14234v1](http://arxiv.org/abs/2602.14234v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14234v1)

**Categories**: cs.AI, cs.CL

<details><summary><b>Abstract / 摘要</b></summary>

Large language models are transitioning from generalpurpose knowledge engines to realworld problem solvers, yet optimizing them for deep search tasks remains challenging. The central bottleneck lies in the extreme sparsity of highquality search trajectories and reward signals, arising from the difficulty of scalable longhorizon task construction and the high cost of interactionheavy rollouts involving external tool calls. To address these challenges, we propose REDSearcher, a unified framework that codesigns complex task synthesis, midtraining, and posttraining for scalable searchagent optimization. Specifically, REDSearcher introduces the following improvements: (1) We frame task synthesis as a dualconstrained optimization, where task difficulty is precisely governed by graph topology and evidence dispersion, allowing scalable generation of complex, highquality tasks. (2) We introduce toolaugmented queries to encourage proactive tool use rather than passive recall.(3) During midtraining, we strengthen core atomic capabilities knowledge, planning, and function calling substantially reducing the cost of collecting highquality trajectories for downstream training. (4) We build a local simulated environment that enables rapid, lowcost algorithmic iteration for reinforcement learning experiments. Across both textonly and multimodal searchagent benchmarks, our approach achieves stateoftheart performance. To facilitate future research on longhorizon search agents, we will release 10K highquality complex text search trajectories, 5K multimodal trajectories and 1K text RL query set, and together with code and model checkpoints.

大型语言模型正在从通用知识引擎过渡到现实世界的问题解决器，但针对深度搜索任务优化它们仍然具有挑战性。中心瓶颈在于高质量搜索轨迹和奖励信号的极度稀疏，这是由于可扩展的长期任务构建的难度以及涉及外部工具调用的交互重推出的高成本而引起的。为了应对这些挑战，我们提出了 REDSearcher，这是一个统一的框架，可以对复杂的任务合成、训练中和训练后进行协同设计，以实现可扩展的搜索代理优化。具体来说，REDSearcher 引入了以下改进：（1）我们将任务合成构建为双重约束优化，其中任务难度由图拓扑和证据分散精确控制，从而允许可扩展地生成复杂的高质量任务。 (2)我们引入了增强查询来鼓励主动使用工具而不是被动回忆。(3)在训练中期，我们加强了核心原子能力知识、规划和函数调用，大大降低了为下游训练收集高质量轨迹的成本。 (4) 我们构建了一个本地模拟环境，可以为强化学习实验提供快速、低成本的算法迭代。在纯文本和多模式搜索代理基准测试中，我们的方法实现了最先进的性能。为了促进未来对长视野搜索代理的研究，我们将发布 10K 高质量复杂文本搜索轨迹、5K 多模态轨迹和 1K 文本 RL 查询集，以及代码和模型检查点。

</details>

---



</details>

<details><summary><b>2026-02-17 (18 papers)</b></summary>

# arXiv Agent Papers - 2026-02-17

**Paper Count**: 18

---

## 1. Process-Supervised Multi-Agent Reinforcement Learning for Reliable Clinical Reasoning / 用于可靠临床推理的过程监督多智能体强化学习

**Date**: 2026-02-15 | **arXiv**: [2602.14160v1](http://arxiv.org/abs/2602.14160v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14160v1)

**Categories**: cs.AI

**Code**: https://github.com/chaeeunlee-io/GeneDiseaseCurationAgents.

<details><summary><b>Abstract / 摘要</b></summary>

Clinical decision-making requires nuanced reasoning over heterogeneous evidence and traceable justifications. While recent LLM multi-agent systems (MAS) show promise, they largely optimise for outcome accuracy while overlooking process-grounded reasoning aligned with clinical standards. One critical real-world case of this is gene-disease validity curation, where experts must determine whether a gene is causally implicated in a disease by synthesising diverse biomedical evidence. We introduce an agent-as-tool reinforcement learning framework for this task with two objectives: (i) process-level supervision to ensure reasoning follows valid clinical pathways, and (ii) efficient coordination via a hierarchical multi-agent system. Our evaluation on the ClinGen dataset shows that with outcome-only rewards, MAS with a GRPO-trained Qwen3-4B supervisor agent substantially improves final outcome accuracy from 0.195 with a base model supervisor to 0.732, but results in poor process alignment (0.392 F1). Conversely, with process + outcome rewards, MAS with GRPO-trained supervisor achieves higher outcome accuracy (0.750) while significantly improving process fidelity to 0.520 F1. Our code is available at https://github.com/chaeeunlee-io/GeneDiseaseCurationAgents.

临床决策需要对异质证据和可追溯的理由进行细致入微的推理。虽然最近的法学硕士多智能体系统（MAS）显示出了希望，但它们在很大程度上优化了结果的准确性，同时忽视了与临床标准一致的基于流程的推理。现实世界的一个关键案例是基因疾病有效性管理，专家必须通过综合不同的生物医学证据来确定基因是否与疾病存在因果关系。我们为此任务引入了一个代理工具强化学习框架，其目标有两个：（i）过程级监督，以确保推理遵循有效的临床路径，以及（ii）通过分层多代理系统进行有效协调。我们对 ClinGen 数据集的评估表明，在仅结果奖励的情况下，具有经过 GRPO 训练的 Qwen3-4B 监督代理的 MAS 大大提高了最终结果的准确性，从基本模型监​​督的 0.195 提高到 0.732，但导致过程一致性较差 (0.392 F1)。相反，通过过程 + 结果奖励，MAS 与经过 GRPO 培训的主管实现了更高的结果准确性 (0.750)，同时将过程保真度显着提高到 0.520 F1。我们的代码可在 https://github.com/chaeeunlee-io/GeneDiseaseCurationAgents 获取。

</details>

---

## 2. A Multi-Agent Framework for Medical AI: Leveraging Fine-Tuned GPT, LLaMA, and DeepSeek R1 for Evidence-Based and Bias-Aware Clinical Query Processing / 医疗 AI 的多代理框架：利用微调的 GPT、LLaMA 和 DeepSeek R1 进行基于证据和偏差感知的临床查询处理

**Date**: 2026-02-15 | **arXiv**: [2602.14158v1](http://arxiv.org/abs/2602.14158v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14158v1)

**Categories**: cs.CL, cs.AI, cs.MA

<details><summary><b>Abstract / 摘要</b></summary>

Large language models (LLMs) show promise for healthcare question answering, but clinical use is limited by weak verification, insufficient evidence grounding, and unreliable confidence signalling. We propose a multi-agent medical QA framework that combines complementary LLMs with evidence retrieval, uncertainty estimation, and bias checks to improve answer reliability. Our approach has two phases. First, we fine-tune three representative LLM families (GPT, LLaMA, and DeepSeek R1) on MedQuAD-derived medical QA data (20k+ question-answer pairs across multiple NIH domains) and benchmark generation quality. DeepSeek R1 achieves the strongest scores (ROUGE-1 0.536 +- 0.04; ROUGE-2 0.226 +-0.03; BLEU 0.098 -+ 0.018) and substantially outperforms the specialised biomedical baseline BioGPT in zero-shot evaluation. Second, we implement a modular multi-agent pipeline in which a Clinical Reasoning agent (fine-tuned LLaMA) produces structured explanations, an Evidence Retrieval agent queries PubMed to ground responses in recent literature, and a Refinement agent (DeepSeek R1) improves clarity and factual consistency; an optional human validation path is triggered for high-risk or high-uncertainty cases. Safety mechanisms include Monte Carlo dropout and perplexity-based uncertainty scoring, plus lexical and sentiment-based bias detection supported by LIME/SHAP-based analyses. In evaluation, the full system achieves 87% accuracy with relevance around 0.80, and evidence augmentation reduces uncertainty (perplexity 4.13) compared to base responses, with mean end-to-end latency of 36.5 seconds under the reported configuration. Overall, the results indicate that agent specialisation and verification layers can mitigate key single-model limitations and provide a practical, extensible design for evidence-based and bias-aware medical AI.

大型语言模型 (LLM) 在医疗保健问题解答方面显示出良好的前景，但临床应用因验证薄弱、证据不足和置信信号不可靠而受到限制。我们提出了一个多智能体医学 QA 框架，它将补充法学硕士与证据检索、不确定性估计和偏差检查相结合，以提高答案的可靠性。我们的方法有两个阶段。首先，我们根据 MedQuAD 衍生的医学 QA 数据（跨多个 NIH 领域的 20k 多个问答对）和基准生成质量对三个代表性的 LLM 系列（GPT、LLaMA 和 DeepSeek R1）进行微调。 DeepSeek R1 取得了最高的分数（ROUGE-1 0.536 +- 0.04；ROUGE-2 0.226 +-0.03；BLEU 0.098 -+ 0.018），并且在零样本评估中大大优于专业生物医学基线 BioGPT。其次，我们实现了一个模块化的多智能体管道，其中临床推理智能体（经过微调的 LLaMA）产生结构化解释，证据检索智能体查询 PubMed 以了解最近文献中的地面响应，而细化智能体（DeepSeek R1）则提高清晰度和事实一致性；对于高风险或高不确定性情况，会触发可选的人工验证路径。安全机制包括蒙特卡洛退出和基于困惑的不确定性评分，以及基于 LIME/SHAP 分析支持的词汇和基于情感的偏差检测。在评估中，整个系统的准确度达到 87%，相关性约为 0.80，与基础响应相比，证据增强降低了不确定性（困惑度 4.13），在报告的配置下平均端到端延迟为 36.5 秒。总体而言，结果表明代理专业化和验证层可以减轻关键的单一模型限制，并为基于证据和偏差感知的医疗人工智能提供实用的、可扩展的设计。

</details>

---

## 3. ForesightSafety Bench: A Frontier Risk Evaluation and Governance Framework towards Safe AI / ForesightSafety Bench：安全人工智能的前沿风险评估和治理框架

**Date**: 2026-02-15 | **arXiv**: [2602.14135v1](http://arxiv.org/abs/2602.14135v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14135v1)

**Categories**: cs.AI, cs.CR, cs.CY

**Code**: https://github.com/Beijing-AISI/ForesightSafety-Bench.

<details><summary><b>Abstract / 摘要</b></summary>

Rapidly evolving AI exhibits increasingly strong autonomy and goal-directed capabilities, accompanied by derivative systemic risks that are more unpredictable, difficult to control, and potentially irreversible. However, current AI safety evaluation systems suffer from critical limitations such as restricted risk dimensions and failed frontier risk detection. The lagging safety benchmarks and alignment technologies can hardly address the complex challenges posed by cutting-edge AI models. To bridge this gap, we propose the "ForesightSafety Bench" AI Safety Evaluation Framework, beginning with 7 major Fundamental Safety pillars and progressively extends to advanced Embodied AI Safety, AI4Science Safety, Social and Environmental AI risks, Catastrophic and Existential Risks, as well as 8 critical industrial safety domains, forming a total of 94 refined risk dimensions. To date, the benchmark has accumulated tens of thousands of structured risk data points and assessment results, establishing a widely encompassing, hierarchically clear, and dynamically evolving AI safety evaluation framework. Based on this benchmark, we conduct systematic evaluation and in-depth analysis of over twenty mainstream advanced large models, identifying key risk patterns and their capability boundaries. The safety capability evaluation results reveals the widespread safety vulnerabilities of frontier AI across multiple pillars, particularly focusing on Risky Agentic Autonomy, AI4Science Safety, Embodied AI Safety, Social AI Safety and Catastrophic and Existential Risks. Our benchmark is released at https://github.com/Beijing-AISI/ForesightSafety-Bench. The project website is available at https://foresightsafety-bench.beijing-aisi.ac.cn/.

快速发展的人工智能展现出越来越强的自主性和目标导向能力，同时也衍生出更加不可预测、难以控制且可能不可逆转的系统性风险。然而，当前的人工智能安全评估系统存在风险维度有限、前沿风险检测失败等严重局限性。落后的安全基准和对位技术很难应对尖端人工智能模型带来的复杂挑战。为了弥补这一差距，我们提出了“ForesightSafety Bench”人工智能安全评估框架，从7大基本安全支柱开始，逐步延伸到先进的体现人工智能安全、AI4科学安全、社会和环境人工智能风险、灾难性和存在风险以及8个关键工业安全领域，总共形成94个细化风险维度。截至目前，该基准已积累数万个结构化风险数据点和评估结果，建立了覆盖面广、层次清晰、动态演进的人工智能安全评估框架。以此基准为基础，我们对二十多个主流先进大型模型进行系统评估和深入分析，识别关键风险模式及其能力边界。安全能力评估结果揭示了前沿人工智能在多个支柱上普遍存在的安全漏洞，特别关注风险智能自主、人工智能科学安全、具体人工智能安全、社会人工智能安全以及灾难性和存在性风险。我们的基准测试发布于 https://github.com/Beijing-AISI/ForesightSafety-Bench。项目网站：https://foresightsafety-bench.beijing-aisi.ac.cn/。

</details>

---

## 4. Toward Autonomous O-RAN: A Multi-Scale Agentic AI Framework for Real-Time Network Control and Management / 迈向自主 O-RAN：用于实时网络控制和管理的多尺度代理 AI 框架

**Date**: 2026-02-15 | **arXiv**: [2602.14117v1](http://arxiv.org/abs/2602.14117v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14117v1)

**Categories**: cs.NI, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Open Radio Access Networks (O-RAN) promise flexible 6G network access through disaggregated, software-driven components and open interfaces, but this programmability also increases operational complexity. Multiple control loops coexist across the service management layer and RAN Intelligent Controller (RIC), while independently developed control applications can interact in unintended ways. In parallel, recent advances in generative Artificial Intelligence (AI) are enabling a shift from isolated AI models toward agentic AI systems that can interpret goals, coordinate multiple models and control functions, and adapt their behavior over time. This article proposes a multi-scale agentic AI framework for O-RAN that organizes RAN intelligence as a coordinated hierarchy across the Non-Real-Time (Non-RT), Near-Real-Time (Near-RT), and Real-Time (RT) control loops: (i) A Large Language Model (LLM) agent in the Non-RT RIC translates operator intent into policies and governs model lifecycles. (ii) Small Language Model (SLM) agents in the Near-RT RIC execute low-latency optimization and can activate, tune, or disable existing control applications; and (iii) Wireless Physical-layer Foundation Model (WPFM) agents near the distributed unit provide fast inference close to the air interface. We describe how these agents cooperate through standardized O-RAN interfaces and telemetry. Using a proof-of-concept implementation built on open-source models, software, and datasets, we demonstrate the proposed agentic approach in two representative scenarios: robust operation under non-stationary conditions and intent-driven slice resource control.

开放无线接入网络 (O-RAN) 承诺通过分解的软件驱动组件和开放接口实现灵活的 6G 网络接入，但这种可编程性也增加了操作复杂性。多个控制环路在服务管理层和 RAN 智能控制器 (RIC) 中共存，而独立开发的控制应用程序可能会以意想不到的方式进行交互。与此同时，生成人工智能 (AI) 的最新进展正在实现从孤立的 AI 模型向代理 AI 系统的转变，该系统可以解释目标、协调多个模型和控制功能，并随着时间的推移调整其行为。本文提出了一种用于 O-RAN 的多尺度代理 AI 框架，它将 RAN 智能组织为跨非实时 (Non-RT)、近实时 (Near-RT) 和实时 (RT) 控制循环的协调层次结构：(i) 非 RT RIC 中的大型语言模型 (LLM) 代理将操作员意图转换为策略并管理模型生命周期。 (ii)近RT RIC中的小语言模型（SLM）代理执行低延迟优化，并可以激活、调整或禁用现有控制应用程序； (iii) 分布式单元附近的无线物理层基础模型 (WPFM) 代理提供靠近空中接口的快速推理。我们描述了这些代理如何通过标准化 O-RAN 接口和遥测进行协作。使用基于开源模型、软件和数据集构建的概念验证实现，我们在两个代表性场景中演示了所提出的代理方法：非平稳条件下的鲁棒操作和意图驱动的切片资源控制。

</details>

---

## 5. TabTracer: Monte Carlo Tree Search for Complex Table Reasoning with Large Language Models / TabTracer：使用大型语言模型进行复杂表推理的蒙特卡罗树搜索

**Date**: 2026-02-15 | **arXiv**: [2602.14089v1](http://arxiv.org/abs/2602.14089v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14089v1)

**Categories**: cs.DB, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Large language models (LLMs) have emerged as powerful tools for natural language table reasoning, where there are two main categories of methods. Prompt-based approaches rely on language-only inference or one-pass program generation without step-level verification. Agent-based approaches use tools in a closed loop, but verification is often local and backtracking is limited, allowing errors to propagate and increasing cost. Moreover, they rely on chain- or beam-style trajectories that are typically combinatorially redundant, leading to high token costs. In this paper, we propose TabTracer, an agentic framework that coordinates multi-step tool calls over intermediate table states, with explicit state tracking for verification and rollback. First, it enforces step-level verification with typed operations and lightweight numeric and format checks to provide reliable rewards and suppress hallucinations. Second, execution-feedback Monte Carlo Tree Search maintains a search tree of candidate table states and uses backpropagated reflection scores to guide UCB1 selection and rollback via versioned snapshots. Third, it reduces redundancy with budget-aware pruning, deduplication, and state hashing with a monotonicity gate to cut token cost. Comprehensive evaluation on TabFact, WikiTQ, and CRT datasets shows that TabTracer outperforms state-of-the-art baselines by up to 6.7% in accuracy while reducing token consumption by 59--84%.

大型语言模型 (LLM) 已成为自然语言表推理的强大工具，其中有两大类方法。基于提示的方法依赖于纯语言推理或一次性程序生成，无需步骤级验证。基于代理的方法在闭环中使用工具，但验证通常是本地的，并且回溯受到限制，导致错误传播并增加成本。此外，它们依赖于链式或梁式轨迹，这些轨迹通常是组合冗余的，导致代币成本很高。在本文中，我们提出了 TabTracer，这是一个代理框架，可协调中间表状态上的多步骤工具调用，并具有用于验证和回滚的显式状态跟踪。首先，它通过类型化操作和轻量级数字和格式检查强制执行步骤级验证，以提供可靠的奖励并抑制幻觉。其次，执行反馈蒙特卡罗树搜索维护候选表状态的搜索树，并使用反向传播反射分数来通过版本化快照指导 UCB1 选择和回滚。第三，它通过预算感知修剪、重复数据删除和带有单调性门的状态散列来减少冗余，以降低代币成本。对 TabFact、WikiTQ 和 CRT 数据集的综合评估表明，TabTracer 的准确率比最先进的基线高出 6.7%，同时减少了 59--84% 的代币消耗。

</details>

---

## 6. Experiential Reinforcement Learning / 体验式强化学习

**Date**: 2026-02-15 | **arXiv**: [2602.13949v1](http://arxiv.org/abs/2602.13949v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.13949v1)

**Categories**: cs.LG, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Reinforcement learning has become the central approach for language models (LMs) to learn from environmental reward or feedback. In practice, the environmental feedback is usually sparse and delayed. Learning from such signals is challenging, as LMs must implicitly infer how observed failures should translate into behavioral changes for future iterations. We introduce Experiential Reinforcement Learning (ERL), a training paradigm that embeds an explicit experience-reflection-consolidation loop into the reinforcement learning process. Given a task, the model generates an initial attempt, receives environmental feedback, and produces a reflection that guides a refined second attempt, whose success is reinforced and internalized into the base policy. This process converts feedback into structured behavioral revision, improving exploration and stabilizing optimization while preserving gains at deployment without additional inference cost. Across sparse-reward control environments and agentic reasoning benchmarks, ERL consistently improves learning efficiency and final performance over strong reinforcement learning baselines, achieving gains of up to +81% in complex multi-step environments and up to +11% in tool-using reasoning tasks. These results suggest that integrating explicit self-reflection into policy training provides a practical mechanism for transforming feedback into durable behavioral improvement.

强化学习已成为语言模型（LM）从环境奖励或反馈中学习的核心方法。在实践中，环境反馈通常是稀疏且延迟的。从此类信号中学习具有挑战性，因为语言模型必须隐式推断观察到的故障应如何转化为未来迭代的行为变化。我们引入了体验式强化学习（ERL），这是一种将明确的经验-反思-巩固循环嵌入到强化学习过程中的训练范例。给定一项任务，该模型会生成初始尝试，接收环境反馈，并产生指导改进的第二次尝试的反思，第二次尝试的成功得到加强并内化到基本政策中。该过程将反馈转化为结构化行为修正，改进探索并稳定优化，同时保留部署收益，而无需额外的推理成本。在稀疏奖励控制环境和代理推理基准中，ERL 在强大的强化学习基线上持续提高学习效率和最终性能，在复杂的多步骤环境中实现高达 +81% 的增益，在使用工具的推理任务中实现高达 +11% 的增益。这些结果表明，将明确的自我反思纳入政策培训提供了将反馈转化为持久行为改善的实用机制。

</details>

---

## 7. When Benchmarks Lie: Evaluating Malicious Prompt Classifiers Under True Distribution Shift / 当基准撒谎时：评估真实分布变化下的恶意提示分类器

**Date**: 2026-02-15 | **arXiv**: [2602.14161v1](http://arxiv.org/abs/2602.14161v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14161v1)

**Categories**: cs.LG

**Code**: https://github.com/maxf-zn/prompt-mining

<details><summary><b>Abstract / 摘要</b></summary>

Detecting prompt injection and jailbreak attacks is critical for deploying LLM-based agents safely. As agents increasingly process untrusted data from emails, documents, tool outputs, and external APIs, robust attack detection becomes essential. Yet current evaluation practices and production systems have fundamental limitations. We present a comprehensive analysis using a diverse benchmark of 18 datasets spanning harmful requests, jailbreaks, indirect prompt injections, and extraction attacks. We propose Leave-One-Dataset-Out (LODO) evaluation to measure true out-of-distribution generalization, revealing that the standard practice of train-test splits from the same dataset sources severely overestimates performance: aggregate metrics show an 8.4 percentage point AUC inflation, but per-dataset gaps range from 1% to 25% accuracy-exposing heterogeneous failure modes. To understand why classifiers fail to generalize, we analyze Sparse Auto-Encoder (SAE) feature coefficients across LODO folds, finding that 28% of top features are dataset-dependent shortcuts whose class signal depends on specific dataset compositions rather than semantic content. We systematically compare production guardrails (PromptGuard 2, LlamaGuard) and LLM-as-judge approaches on our benchmark, finding all three fail on indirect attacks targeting agents (7-37% detection) and that PromptGuard 2 and LlamaGuard cannot evaluate agentic tool injection due to architectural limitations. Finally, we show that LODO-stable SAE features provide more reliable explanations for classifier decisions by filtering dataset artifacts. We release our evaluation framework at https://github.com/maxf-zn/prompt-mining to establish LODO as the appropriate protocol for prompt attack detection research.

检测即时注入和越狱攻击对于安全部署基于 LLM 的代理至关重要。随着代理越来越多地处理来自电子邮件、文档、工具输出和外部 API 的不可信数据，强大的攻击检测变得至关重要。然而，当前的评估实践和生产系统存在根本性的局限性。我们使用 18 个数据集的不同基准进行了全面分析，涵盖有害请求、越狱、间接提示注入和提取攻击。我们提出了留一数据集排除（LODO）评估来衡量真正的分布外泛化，揭示了来自相同数据集源的训练-测试分割的标准做法严重高估了性能：聚合指标显示 AUC 膨胀 8.4 个百分点，但每个数据集的差距范围从 1% 到 25% 不等，暴露了异构故障模式。为了理解分类器无法泛化的原因，我们分析了 LODO 折叠中的稀疏自动编码器 (SAE) 特征系数，发现 28% 的顶级特征是数据集相关的快捷方式，其类信号取决于特定的数据集组成而不是语义内容。我们在基准测试中系统地比较了生产护栏（PromptGuard 2、LlamaGuard）和 LLM-as-judge 方法，发现所有三种方法都在针对代理的间接攻击（7-37% 检测）上失败，并且由于架构限制，PromptGuard 2 和 LlamaGuard 无法评估代理工具注入。最后，我们表明 LODO 稳定的 SAE 特征通过过滤数据集工件为分类器决策提供更可靠的解释。我们在 https://github.com/maxf-zn/prompt-mining 发布了我们的评估框架，以将 LODO 建立为即时攻击检测研究的适当协议。

</details>

---

## 8. Decentralized Federated Learning With Energy Harvesting Devices / 使用能量收集设备的去中心化联合学习

**Date**: 2026-02-15 | **arXiv**: [2602.14051v1](http://arxiv.org/abs/2602.14051v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14051v1)

**Categories**: cs.LG, eess.SP

<details><summary><b>Abstract / 摘要</b></summary>

Decentralized federated learning (DFL) enables edge devices to collaboratively train models through local training and fully decentralized device-to-device (D2D) model exchanges. However, these energy-intensive operations often rapidly deplete limited device batteries, reducing their operational lifetime and degrading the learning performance. To address this limitation, we apply energy harvesting technique to DFL systems, allowing edge devices to extract ambient energy and operate sustainably. We first derive the convergence bound for wireless DFL with energy harvesting, showing that the convergence is influenced by partial device participation and transmission packet drops, both of which further depend on the available energy supply. To accelerate convergence, we formulate a joint device scheduling and power control problem and model it as a multi-agent Markov decision process (MDP). Traditional MDP algorithms (e.g., value or policy iteration) require a centralized coordinator with access to all device states and exhibit exponential complexity in the number of devices, making them impractical for large-scale decentralized networks. To overcome these challenges, we propose a fully decentralized policy iteration algorithm that leverages only local state information from two-hop neighboring devices, thereby substantially reducing both communication overhead and computational complexity. We further provide a theoretical analysis showing that the proposed decentralized algorithm achieves asymptotic optimality. Finally, comprehensive numerical experiments on real-world datasets are conducted to validate the theoretical results and corroborate the effectiveness of the proposed algorithm.

去中心化联合学习 (DFL) 使边缘设备能够通过本地训练和完全去中心化的设备到设备 (D2D) 模型交换来协作训练模型。然而，这些能源密集型操作通常会迅速耗尽有限的设备电池，从而缩短其使用寿命并降低学习性能。为了解决这一限制，我们将能量收集技术应用于 DFL 系统，使边缘设备能够提取环境能量并可持续运行。我们首先推导出具有能量收集的无线 DFL 的收敛界限，表明收敛受到部分设备参与和传输数据包丢失的影响，这两者都进一步取决于可用的能量供应。为了加速收敛，我们制定了联合设备调度和功率控制问题，并将其建模为多智能体马尔可夫决策过程（MDP）。传统的 MDP 算法（例如，值或策略迭代）需要一个能够访问所有设备状态的集中式协调器，并且在设备数量方面表现出指数复杂性，这使得它们对于大规模去中心化网络来说不切实际。为了克服这些挑战，我们提出了一种完全去中心化的策略迭代算法，该算法仅利用来自两跳相邻设备的本地状态信息，从而大大减少通信开销和计算复杂性。我们进一步提供了理论分析，表明所提出的去中心化算法实现了渐近最优性。最后，对真实数据集进行了全面的数值实验，以验证理论结果并证实所提出算法的有效性。

</details>

---

## 9. S2SServiceBench: A Multimodal Benchmark for Last-Mile S2S Climate Services / S2SServiceBench：最后一英里 S2S 气候服务的多式联运基准

**Date**: 2026-02-15 | **arXiv**: [2602.14017v1](http://arxiv.org/abs/2602.14017v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14017v1)

**Categories**: cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Subseasonal-to-seasonal (S2S) forecasts play an essential role in providing a decision-critical weeks-to-months planning window for climate resilience and sustainability, yet a growing bottleneck is the last-mile gap: translating scientific forecasts into trusted, actionable climate services, requiring reliable multimodal understanding and decision-facing reasoning under uncertainty. Meanwhile, multimodal large language models (MLLMs) and corresponding agentic paradigms have made rapid progress in supporting various workflows, but it remains unclear whether they can reliably generate decision-making deliverables from operational service products (e.g., actionable signal comprehension, decision-making handoff, and decision analysis & planning) under uncertainty. We introduce S2SServiceBench, a multimodal benchmark for last-mile S2S climate services curated from an operational climate-service system to evaluate this capability. S2SServiceBenchcovers 10 service products with about 150+ expert-selected cases in total, spanning six application domains - Agriculture, Disasters, Energy, Finance, Health, and Shipping. Each case is instantiated at three service levels, yielding around 500 tasks and 1,000+ evaluation items across climate resilience and sustainability applications. Using S2SServiceBench, we benchmark state-of-the-art MLLMs and agents, and analyze performance across products and service levels, revealing persistent challenges in S2S service plot understanding and reasoning - namely, actionable signal comprehension, operationalizing uncertainty into executable handoffs, and stable, evidence-grounded analysis and planning for dynamic hazards-while offering actionable guidance for building future climate-service agents.

次季节到季节 (S2S) 预报在为气候复原力和可持续性提供关键的数周至数月规划窗口方面发挥着至关重要的作用，但日益严重的瓶颈是最后一英里的差距：将科学预报转化为可信、可操作的气候服务，需要在不确定性下进行可靠的多模式理解和面向决策的推理。与此同时，多模态大语言模型（MLLM）和相应的代理范式在支持各种工作流程方面取得了快速进展，但仍不清楚它们是否能够在不确定性下从运营服务产品（例如，可操作的信号理解、决策切换以及决策分析和规划）可靠地生成决策交付成果。我们引入了 S2SServiceBench，这是一个针对最后一英里 S2S 气候服务的多式联运基准，它是根据可操作的气候服务系统策划的，以评估这种能力。 S2SServiceBench涵盖10个服务产品，总计约150多个专家精选案例，涵盖农业、灾害、能源、金融、健康和航运六大应用领域。每个案例都在三个服务级别进行实例化，产生大约 500 个任务和 1,000 多个涉及气候适应力和可持续性应用的评估项目。使用 S2SServiceBench，我们对最先进的 MLLM 和代理进行基准测试，并分析产品和服务级别的性能，揭示 S2S 服务图理解和推理中持续存在的挑战，即可操作的信号理解、将不确定性转化为可执行的切换，以及针对动态危险的稳定、基于证据的分析和规划，同时为构建未来的气候服务代理提供可行的指导。

</details>

---

## 10. A Multi-Agent Framework for Code-Guided, Modular, and Verifiable Automated Machine Learning / 用于代码引导、模块化和可验证的自动化机器学习的多代理框架

**Date**: 2026-02-15 | **arXiv**: [2602.13937v1](http://arxiv.org/abs/2602.13937v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.13937v1)

**Categories**: cs.LG, cs.SE

<details><summary><b>Abstract / 摘要</b></summary>

Automated Machine Learning (AutoML) has revolutionized the development of data-driven solutions; however, traditional frameworks often function as "black boxes", lacking the flexibility and transparency required for complex, real-world engineering tasks. Recent Large Language Model (LLM)-based agents have shifted toward code-driven approaches. However, they frequently suffer from hallucinated logic and logic entanglement, where monolithic code generation leads to unrecoverable runtime failures. In this paper, we present iML, a novel multi-agent framework designed to shift AutoML from black-box prompting to a code-guided, modular, and verifiable architectural paradigm. iML introduces three main ideas: (1) Code-Guided Planning, which synthesizes a strategic blueprint grounded in autonomous empirical profiling to eliminate hallucination; (2) Code-Modular Implementation, which decouples preprocessing and modeling into specialized components governed by strict interface contracts; and (3) Code-Verifiable Integration, which enforces physical feasibility through dynamic contract verification and iterative self-correction. We evaluate iML across MLE-BENCH and the newly introduced iML-BENCH, comprising a diverse range of real-world Kaggle competitions. The experimental results show iML's superiority over state-of-the-art agents, achieving a valid submission rate of 85% and a competitive medal rate of 45% on MLE-BENCH, with an average standardized performance score (APS) of 0.77. On iML-BENCH, iML significantly outperforms the other approaches by 38%-163% in APS. Furthermore, iML maintains a robust 70% success rate even under stripped task descriptions, effectively filling information gaps through empirical profiling. These results highlight iML's potential to bridge the gap between stochastic generation and reliable engineering, marking a meaningful step toward truly AutoML.

自动化机器学习 (AutoML) 彻底改变了数据驱动解决方案的开发；然而，传统框架往往充当“黑匣子”，缺乏复杂的现实工程任务所需的灵活性和透明度。最近基于大型语言模型（LLM）的代理已经转向代码驱动的方法。然而，它们经常遭受逻辑幻觉和逻辑纠缠的困扰，其中单一代码生成会导致不可恢复的运行时故障。在本文中，我们提出了 iML，这是一种新颖的多代理框架，旨在将 AutoML 从黑盒提示转变为代码引导、模块化和可验证的架构范例。 iML 引入了三个主要思想：（1）代码引导规划，综合基于自主经验分析的战略蓝图，以消除幻觉； (2) 代码模块化实现，将预处理和建模解耦为受严格接口契约控制的专用组件； (3) 代码可验证集成，通过动态合约验证和迭代自我修正来增强物理可行性。我们通过 MLE-BENCH 和新推出的 iML-BENCH 评估 iML，其中包括各种真实的 Kaggle 竞赛。实验结果表明，iML 优于最先进的智能体，在 MLE-BENCH 上实现了 85% 的有效提交率和 45% 的竞争奖牌率，平均标准化性能得分 (APS) 为 0.77。在 iML-BENCH 上，iML 在 APS 方面明显优于其他方法 38%-163%。此外，即使在任务描述被剥离的情况下，iML 也能保持 70% 的成功率，通过经验分析有效地填补信息空白。这些结果凸显了 iML 在弥合随机生成和可靠工程之间差距的潜力，标志着迈向真正的 AutoML 的有意义的一步。

</details>

---

## 11. It Takes Two to Tango: A Holistic Simulator for Joint Order Scheduling and Multi-Agent Path Finding in Robotic Warehouses / Tango 需要两个人：用于机器人仓库中联合订单调度和多代理路径查找的整体模拟器

**Date**: 2026-02-15 | **arXiv**: [2602.13999v1](http://arxiv.org/abs/2602.13999v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.13999v1)

**Categories**: cs.RO

<details><summary><b>Abstract / 摘要</b></summary>

The prevailing paradigm in Robotic Mobile Fulfillment Systems (RMFS) typically treats order scheduling and multi-agent pathfinding as isolated sub-problems. We argue that this decoupling is a fundamental bottleneck, masking the critical dependencies between high-level dispatching and low-level congestion. Existing simulators fail to bridge this gap, often abstracting away heterogeneous kinematics and stochastic execution failures. We propose WareRover, a holistic simulation platform that enforces a tight coupling between OS and MAPF via a unified, closed-loop optimization interface. Unlike standard benchmarks, WareRover integrates dynamic order streams, physics-aware motion constraints, and non-nominal recovery mechanisms into a single evaluation loop. Experiments reveal that SOTA algorithms often falter under these realistic coupled constraints, demonstrating that WareRover provides a necessary and challenging testbed for robust, next-generation warehouse coordination. The project and video is available at https://hhh-x.github.io/WareRover/.

机器人移动履行系统 (RMFS) 中的主流范例通常将订单调度和多代理寻路视为孤立的子问题。我们认为这种解耦是一个根本瓶颈，掩盖了高层调度和低层拥塞之间的关键依赖关系。现有的模拟器无法弥补这一差距，通常会抽象出异构运动学和随机执行失败。我们提出了 WareRover，一个整体仿真平台，通过统一的闭环优化接口强制操作系统和 MAPF 之间的紧密耦合。与标准基准测试不同，WareRover 将动态指令流、物理感知运动约束和非标称恢复机制集成到单个评估循环中。实验表明，SOTA 算法在这些现实的耦合约束下经常会出现问题，这表明 WareRover 为强大的下一代仓库协调提供了必要且具有挑战性的测试平台。该项目和视频可在 https://hhh-x.github.io/WareRover/ 获取。

</details>

---

## 12. DTBench: A Synthetic Benchmark for Document-to-Table Extraction / DTBench：文档到表提取的综合基准

**Date**: 2026-02-14 | **arXiv**: [2602.13812v1](http://arxiv.org/abs/2602.13812v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.13812v1)

**Categories**: cs.DB, cs.AI, cs.MA

**Code**: https://github.com/ZJU-DAILY/DTBench.

<details><summary><b>Abstract / 摘要</b></summary>

Document-to-table (Doc2Table) extraction derives structured tables from unstructured documents under a target schema, enabling reliable and verifiable SQL-based data analytics. Although large language models (LLMs) have shown promise in flexible information extraction, their ability to produce precisely structured tables remains insufficiently understood, particularly for indirect extraction that requires complex capabilities such as reasoning and conflict resolution. Existing benchmarks neither explicitly distinguish nor comprehensively cover the diverse capabilities required in Doc2Table extraction.We argue that a capability-aware benchmark is essential for systematic evaluation. However, constructing such benchmarks using human-annotated document-table pairs is costly, difficult to scale, and limited in capability coverage. To address this, we adopt a reverse Table2Doc paradigm and design a multi-agent synthesis workflow to generate documents from ground-truth tables. Based on this approach, we present DTBench, a synthetic benchmark that adopts a proposed two-level taxonomy of Doc2Table capabilities, covering 5 major categories and 13 subcategories. We evaluate several mainstream LLMs on DTBench, and demonstrate substantial performance gaps across models, as well as persistent challenges in reasoning, faithfulness, and conflict resolution. DTBench provides a comprehensive testbed for data generation and evaluation, facilitating future research on Doc2Table extraction. The benchmark is publicly available at https://github.com/ZJU-DAILY/DTBench.

文档到表 (Doc2Table) 提取可在目标模式下从非结构化文档派生出结构化表，从而实现可靠且可验证的基于 SQL 的数据分析。尽管大型语言模型 (LLM) 在灵活的信息提取方面显示出了希望，但它们生成精确结构化表格的能力仍然没有得到充分理解，特别是对于需要推理和冲突解决等复杂功能的间接提取。现有的基准既没有明确区分也没有全面涵盖 Doc2Table 提取所需的各种功能。我们认为，能力感知基准对于系统评估至关重要。然而，使用人工注释的文档表对构建此类基准成本高昂、难以扩展且能力覆盖范围有限。为了解决这个问题，我们采用反向 Table2Doc 范例并设计多代理合成工作流程以从真实表生成文档。基于这种方法，我们提出了 DTBench，这是一个综合基准，采用了拟议的 Doc2Table 功能两级分类法，涵盖 5 个主要类别和 13 个子类别。我们在 DTBench 上评估了几个主流的法学硕士，并展示了模型之间巨大的性能差距，以及在推理、忠实性和冲突解决方面持续存在的挑战。 DTBench 为数据生成和评估提供了一个全面的测试平台，促进了未来 Doc2Table 提取的研究。该基准测试可在 https://github.com/ZJU-DAILY/DTBench 上公开获取。

</details>

---

## 13. An end-to-end agentic pipeline for smart contract translation and quality evaluation / 用于智能合约翻译和质量评估的端到端代理管道

**Date**: 2026-02-14 | **arXiv**: [2602.13808v1](http://arxiv.org/abs/2602.13808v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.13808v1)

**Categories**: cs.AI, cs.SE

<details><summary><b>Abstract / 摘要</b></summary>

We present an end-to-end framework for systematic evaluation of LLM-generated smart contracts from natural-language specifications. The system parses contractual text into structured schemas, generates Solidity code, and performs automated quality assessment through compilation and security checks. Using CrewAI-style agent teams with iterative refinement, the pipeline produces structured artifacts with full provenance metadata. Quality is measured across five dimensions, including functional completeness, variable fidelity, state-machine correctness, business-logic fidelity, and code quality aggregated into composite scores. The framework supports paired evaluation against ground-truth implementations, quantifying alignment and identifying systematic error modes such as logic omissions and state transition inconsistencies. This provides a reproducible benchmark for empirical research on smart contract synthesis quality and supports extensions to formal verification and compliance checking.

我们提出了一个端到端框架，用于根据自然语言规范对法学硕士生成的智能合约进行系统评估。该系统将合同文本解析为结构化模式，生成 Solidity 代码，并通过编译和安全检查执行自动质量评估。使用 CrewAI 风格的代理团队进行迭代细化，该管道可生成具有完整来源元数据的结构化工件。质量是通过五个维度来衡量的，包括功能完整性、变量保真度、状态机正确性、业务逻辑保真度以及汇总为综合分数的代码质量。该框架支持针对真实实现的配对评估，量化对齐并识别系统错误模式，例如逻辑遗漏和状态转换不一致。这为智能合约合成质量的实证研究提供了可重复的基准，并支持形式验证和合规性检查的扩展。

</details>

---

## 14. OR-Agent: Bridging Evolutionary Search and Structured Research for Automated Algorithm Discovery / OR-Agent：连接进化搜索和结构化研究以实现自动算法发现

**Date**: 2026-02-14 | **arXiv**: [2602.13769v1](http://arxiv.org/abs/2602.13769v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.13769v1)

**Categories**: cs.AI, cs.CE, cs.NE

**Code**: https://github.com/qiliuchn/OR-Agent.

<details><summary><b>Abstract / 摘要</b></summary>

Automating scientific discovery in complex, experiment-driven domains requires more than iterative mutation of programs; it demands structured hypothesis management, environment interaction, and principled reflection. We present OR-Agent, a configurable multi-agent research framework designed for automated exploration in rich experimental environments. OR-Agent organizes research as a structured tree-based workflow that explicitly models branching hypothesis generation and systematic backtracking, enabling controlled management of research trajectories beyond simple mutation-crossover loops. At its core, we introduce an evolutionary-systematic ideation mechanism that unifies evolutionary selection of research starting points, comprehensive research plan generation, and coordinated exploration within a research tree. We further propose a hierarchical optimization-inspired reflection system: short-term experimental reflection operates as a form of verbal gradient providing immediate corrective signals; long-term reflection accumulates cross-experiment insights as verbal momentum; and memory compression serves as a regularization mechanism analogous to weight decay, preserving essential signals while mitigating drift. Together, these components form a principled architecture governing research dynamics. We conduct extensive experiments across classical combinatorial optimization benchmarks-including traveling salesman, capacitated vehicle routing, bin packing, orienteering, and multiple knapsack problems-as well as simulation-based cooperative driving scenarios. Results demonstrate that OR-Agent outperforms strong evolutionary baselines while providing a general, extensible, and inspectable framework for AI-assisted scientific discovery. OR-Agent source code and experiments data are publicly available at https://github.com/qiliuchn/OR-Agent.

在复杂的、实验驱动的领域中实现科学发现的自动化需要的不仅仅是程序的迭代突变；它需要结构化的假设管理、环境交互和有原则的反思。我们提出了 OR-Agent，这是一个可配置的多代理研究框架，专为在丰富的实验环境中进行自动探索而设计。 OR-Agent 将研究组织为基于树的结构化工作流程，明确模拟分支假设生成和系统回溯，从而实现对研究轨迹的受控管理，超越简单的突变交叉循环。其核心是，我们引入了一种进化系统思想机制，该机制将研究起点的进化选择、综合研究计划的生成以及研究树内的协调探索统一起来。我们进一步提出了一种分层优化启发的反思系统：短期实验反思以言语梯度的形式运行，提供即时的纠正信号；长期反思积累跨实验见解作为口头动力；内存压缩作为一种类似于权重衰减的正则化机制，在减少漂移的同时保留基本信号。这些组件共同构成了一个管理研究动态的原则架构。我们对经典组合优化基准（包括旅行推销员、有能力的车辆路线、装箱、定向运动和多个背包问题）以及基于模拟的协作驾驶场景进行了广泛的实验。结果表明，OR-Agent 的性能优于强大的进化基线，同时为人工智能辅助的科学发现提供了通用的、可扩展的和可检查的框架。 OR-Agent 源代码和实验数据可在 https://github.com/qiliuchn/OR-Agent 上公开获取。

</details>

---

## 15. PrivAct: Internalizing Contextual Privacy Preservation via Multi-Agent Preference Training / PrivAct：通过多代理偏好训练内部化上下文隐私保护

**Date**: 2026-02-14 | **arXiv**: [2602.13840v1](http://arxiv.org/abs/2602.13840v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.13840v1)

**Categories**: cs.CL

**Code**: https://github.com/chengyh23/PrivAct.

<details><summary><b>Abstract / 摘要</b></summary>

Large language model (LLM) agents are increasingly deployed in personalized tasks involving sensitive, context-dependent information, where privacy violations may arise in agents' action due to the implicitness of contextual privacy. Existing approaches rely on external, inference-time interventions which are brittle, scenario-specific, and may expand the privacy attack surface. We propose PrivAct, a contextual privacy-aware multi-agent learning framework that internalizes contextual privacy preservation directly into models' generation behavior for privacy-compliant agentic actions. By embedding privacy preferences into each agent, PrivAct enhances system-wide contextual integrity while achieving a more favorable privacy-helpfulness tradeoff. Experiments across multiple LLM backbones and benchmarks demonstrate consistent improvements in contextual privacy preservation, reducing leakage rates by up to 12.32% while maintaining comparable helpfulness, as well as zero-shot generalization and robustness across diverse multi-agent topologies. Code is available at https://github.com/chengyh23/PrivAct.

大型语言模型（LLM）代理越来越多地部署在涉及敏感、上下文相关信息的个性化任务中，由于上下文隐私的隐含性，代理的行为可能会出现隐私侵犯。现有的方法依赖于外部的推理时间干预，这些干预是脆弱的、特定于场景的，并且可能会扩大隐私攻击面。我们提出了 PrivAct，一种上下文隐私感知的多代理学习框架，它将上下文隐私保护直接内化到模型的生成行为中，以实现符合隐私的代理操作。通过将隐私偏好嵌入到每个代理中，PrivAct 增强了系统范围的上下文完整性，同时实现了更有利的隐私-有用性权衡。跨多个 LLM 主干和基准的实验表明，上下文隐私保护方面得到了一致的改进，将泄漏率降低了 12.32%，同时保持了相当的有用性，以及跨不同多代理拓扑的零样本泛化和鲁棒性。代码可在 https://github.com/ Chengyh23/PrivAct 获取。

</details>

---

## 16. OMGs: A multi-agent system supporting MDT decision-making across the ovarian tumour care continuum / OMG：支持整个卵巢肿瘤护理连续体中 MDT 决策的多代理系统

**Date**: 2026-02-14 | **arXiv**: [2602.13793v1](http://arxiv.org/abs/2602.13793v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.13793v1)

**Categories**: cs.CL

<details><summary><b>Abstract / 摘要</b></summary>

Ovarian tumour management has increasingly relied on multidisciplinary tumour board (MDT) deliberation to address treatment complexity and disease heterogeneity. However, most patients worldwide lack access to timely expert consensus, particularly in resource-constrained centres where MDT resources are scarce or unavailable. Here we present OMGs (Ovarian tumour Multidisciplinary intelligent aGent System), a multi-agent AI framework where domain-specific agents deliberate collaboratively to integrate multidisciplinary evidence and generate MDT-style recommendations with transparent rationales. To systematically evaluate MDT recommendation quality, we developed SPEAR (Safety, Personalization, Evidence, Actionability, Robustness) and validated OMGs across diverse clinical scenarios spanning the care continuum. In multicentre re-evaluation, OMGs achieved performance comparable to expert MDT consensus ($4.45 \pm 0.30$ versus $4.53 \pm 0.23$), with higher Evidence scores (4.57 versus 3.92). In prospective multicentre evaluation (59 patients), OMGs demonstrated high concordance with routine MDT decisions. Critically, in paired human-AI studies, OMGs most substantially enhanced clinicians' recommendations in Evidence and Robustness, the dimensions most compromised when multidisciplinary expertise is unavailable. These findings suggest that multi-agent deliberative systems can achieve performance comparable to expert MDT consensus, with potential to expand access to specialized oncology expertise in resource-limited settings.

卵巢肿瘤的治疗越来越依赖多学科肿瘤委员会（MDT）的审议来解决治疗的复杂性和疾病的异质性。然而，世界各地的大多数患者无法及时获得专家共识，特别是在 MDT 资源稀缺或不可用的资源有限的中心。在这里，我们提出 OMG（卵巢肿瘤多学科智能 aGent 系统），这是一个多智能体人工智能框架，其中特定领域的智能体协作审议以整合多学科证据并生成具有透明原理的 MDT 式建议。为了系统地评估 MDT 推荐质量，我们开发了 SPEAR（安全性、个性化、证据、可操作性、稳健性），并在跨越护理连续体的不同临床场景中验证了 OMG。在多中心重新评估中，OMG 的表现与专家 MDT 共识相当（$4.45 \pm 0.30$ vs $4.53 \pm 0.23$），且证据分数更高（4.57 vs 3.92）。在前瞻性多中心评估（59 名患者）中，OMG 表现出与常规 MDT 决策高度一致。至关重要的是，在人类与人工智能配对研究中，OMG 最大幅度地增强了临床医生在证据和稳健性方面的建议，而当多学科专业知识不可用时，这两个维度受到的影响最大。这些研究结果表明，多主体审议系统可以实现与专家 MDT 共识相当的性能，并有可能在资源有限的环境中扩大获得专门肿瘤学专业知识的机会。

</details>

---

## 17. AnomaMind: Agentic Time Series Anomaly Detection with Tool-Augmented Reasoning / AnomaMind：利用工具增强推理进行代理时间序列异常检测

**Date**: 2026-02-14 | **arXiv**: [2602.13807v1](http://arxiv.org/abs/2602.13807v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.13807v1)

**Categories**: cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Time series anomaly detection is critical in many real-world applications, where effective solutions must localize anomalous regions and support reliable decision-making under complex settings. However, most existing methods frame anomaly detection as a purely discriminative prediction task with fixed feature inputs, rather than an evidence-driven diagnostic process. As a result, they often struggle when anomalies exhibit strong context dependence or diverse patterns. We argue that these limitations stem from the lack of adaptive feature preparation, reasoning-aware detection, and iterative refinement during inference. To address these challenges, we propose AnomaMind, an agentic time series anomaly detection framework that reformulates anomaly detection as a sequential decision-making process. AnomaMind operates through a structured workflow that progressively localizes anomalous intervals in a coarse-to-fine manner, augments detection through multi-turn tool interactions for adaptive feature preparation, and refines anomaly decisions via self-reflection. The workflow is supported by a set of reusable tool engines, enabling context-aware diagnostic analysis. A key design of AnomaMind is an explicitly designed hybrid inference mechanism for tool-augmented anomaly detection. In this mechanism, general-purpose models are responsible for autonomous tool interaction and self-reflective refinement, while core anomaly detection decisions are learned through reinforcement learning under verifiable workflow-level feedback, enabling task-specific optimization within a flexible reasoning framework. Extensive experiments across diverse settings demonstrate that AnomaMind consistently improves anomaly detection performance. The code is available at https://anonymous.4open.science/r/AnomaMind.

时间序列异常检测在许多实际应用中至关重要，有效的解决方案必须定位异常区域并支持复杂环境下的可靠决策。然而，大多数现有方法将异常检测视为具有固定特征输入的纯粹判别性预测任务，而不是证据驱动的诊断过程。因此，当异常表现出强烈的上下文依赖性或不同的模式时，他们常常会陷入困境。我们认为这些限制源于缺乏自适应特征准备、推理感知检测和推理过程中的迭代细化。为了应对这些挑战，我们提出了 AnomaMind，一种代理时间序列异常检测框架，它将异常检测重新表述为顺序决策过程。 AnomaMind 通过结构化工作流程进行操作，以从粗到细的方式逐步定位异常间隔，通过多轮工具交互增强检测以进行自适应特征准备，并通过自我反思完善异常决策。该工作流程由一组可重用工具引擎支持，从而实现上下文感知诊断分析。 AnomaMind 的一个关键设计是明确设计的用于工具增强异常检测的混合推理机制。在这种机制中，通用模型负责自主工具交互和自我反思细化，而核心异常检测决策是通过可验证的工作流级反馈下的强化学习来学习的，从而在灵活的推理框架内实现特定于任务的优化。跨不同设置的大量实验表明 AnomaMind 持续提高异常检测性能。该代码可在 https://anonymous.4open.science/r/AnomaMind 获取。

</details>

---

## 18. Cast-R1: Learning Tool-Augmented Sequential Decision Policies for Time Series Forecasting / Cast-R1：用于时间序列预测的学习工具增强顺序决策策略

**Date**: 2026-02-14 | **arXiv**: [2602.13802v1](http://arxiv.org/abs/2602.13802v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.13802v1)

**Categories**: cs.LG

**Code**: https://github.com/Xiaoyu-Tao/Cast-R1-TS.

<details><summary><b>Abstract / 摘要</b></summary>

Time series forecasting has long been dominated by model-centric approaches that formulate prediction as a single-pass mapping from historical observations to future values. Despite recent progress, such formulations often struggle in complex and evolving settings, largely because most forecasting models lack the ability to autonomously acquire informative evidence, reason about potential future changes, or revise predictions through iterative decision processes. In this work, we propose Cast-R1, a learned time series forecasting framework that reformulates forecasting as a sequential decision-making problem. Cast-R1 introduces a memory-based state management mechanism that maintains decision-relevant information across interaction steps, enabling the accumulation of contextual evidence to support long-horizon reasoning. Building on this formulation, forecasting is carried out through a tool-augmented agentic workflow, in which the agent autonomously interacts with a modular toolkit to extract statistical features, invoke lightweight forecasting models for decision support, perform reasoning-based prediction, and iteratively refine forecasts through self-reflection. To train Cast-R1, we adopt a two-stage learning strategy that combines supervised fine-tuning with multi-turn reinforcement learning, together with a curriculum learning scheme that progressively increases task difficulty to improve policy learning. Extensive experiments on multiple real-world time series datasets demonstrate the effectiveness of Cast-R1. We hope this work provides a practical step towards further exploration of agentic paradigms for time series modeling. Our code is available at https://github.com/Xiaoyu-Tao/Cast-R1-TS.

时间序列预测长期以来一直以模型为中心的方法主导，这些方法将预测制定为从历史观测到未来值的单通道映射。尽管最近取得了进展，但此类公式往往在复杂且不断变化的环境中举步维艰，很大程度上是因为大多数预测模型缺乏自主获取信息证据、推理未来潜在变化或通过迭代决策过程修改预测的能力。在这项工作中，我们提出了 Cast-R1，这是一种学习时间序列预测框架，它将预测重新表述为顺序决策问题。 Cast-R1 引入了基于内存的状态管理机制，可在交互步骤中维护决策相关信息，从而能够积累上下文证据以支持长期推理。在此基础上，预测是通过工具增强的代理工作流程进行的，其中代理自主地与模块化工具包交互以提取统计特征，调用轻量级预测模型以提供决策支持，执行基于推理的预测，并通过自我反思迭代地完善预测。为了训练Cast-R1，我们采用了将监督微调与多轮强化学习相结合的两阶段学习策略，以及逐步增加任务难度以改善政策学习的课程学习方案。对多个真实时间序列数据集的大量实验证明了 Cast-R1 的有效性。我们希望这项工作为进一步探索时间序列建模的代理范式提供了实际的一步。我们的代码可在 https://github.com/Xiaoyu-Tao/Cast-R1-TS 获取。

</details>

---



</details>

<details><summary><b>2026-02-16 (7 papers)</b></summary>

# arXiv Agent Papers - 2026-02-16

**Paper Count**: 7

---

## 1. Asynchronous Verified Semantic Caching for Tiered LLM Architectures / 分层 LLM 架构的异步验证语义缓存

**Date**: 2026-02-13 | **arXiv**: [2602.13165v1](http://arxiv.org/abs/2602.13165v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.13165v1)

**Categories**: cs.IR, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Large language models (LLMs) now sit in the critical path of search, assistance, and agentic workflows, making semantic caching essential for reducing inference cost and latency. Production deployments typically use a tiered static-dynamic design: a static cache of curated, offline vetted responses mined from logs, backed by a dynamic cache populated online. In practice, both tiers are commonly governed by a single embedding similarity threshold, which induces a hard tradeoff: conservative thresholds miss safe reuse opportunities, while aggressive thresholds risk serving semantically incorrect responses. We introduce \textbf{Krites}, an asynchronous, LLM-judged caching policy that expands static coverage without changing serving decisions. On the critical path, Krites behaves exactly like a standard static threshold policy. When the nearest static neighbor of the prompt falls just below the static threshold, Krites asynchronously invokes an LLM judge to verify whether the static response is acceptable for the new prompt. Approved matches are promoted into the dynamic cache, allowing future repeats and paraphrases to reuse curated static answers and expanding static reach over time. In trace-driven simulations on conversational and search workloads, Krites increases the fraction of requests served with curated static answers (direct static hits plus verified promotions) by up to $\textbf{3.9}$ times for conversational traffic and search-style queries relative to tuned baselines, with unchanged critical path latency.

大型语言模型 (LLM) 现在位于搜索、辅助和代理工作流程的关键路径中，使得语义缓存对于降低推理成本和延迟至关重要。生产部署通常使用分层的静态-动态设计：从日志中挖掘的经过策划、离线审查的响应的静态缓存，由在线填充的动态缓存提供支持。在实践中，这两层通常由单个嵌入相似性阈值控制，这会导致一个艰难的权衡：保守的阈值会错过安全重用机会，而激进的阈值则有可能提供语义上不正确的响应。我们引入了 \textbf{Krites}，这是一种异步的、LLM 判断的缓存策略，可以在不改变服务决策的情况下扩展静态覆盖范围。在关键路径上，Krites 的行为与标准静态阈值策略完全相同。当提示的最近静态邻居低于静态阈值时，Krites 异步调用 LLM 判断来验证静态响应是否适合新提示。批准的匹配将提升到动态缓存中，允许将来的重复和释义重用策划的静态答案并随着时间的推移扩大静态范围。在对话和搜索工作负载的跟踪驱动模拟中，相对于调整后的基线，Krites 将对话流量和搜索式查询的通过策划的静态答案（直接静态点击加上经过验证的促销）提供的请求比例提高了高达 $\textbf{3.9}$ 倍，且关键路径延迟不变。

</details>

---

## 2. In-Context Autonomous Network Incident Response: An End-to-End Large Language Model Agent Approach / 上下文自治网络事件响应：一种端到端的大型语言模型代理方法

**Date**: 2026-02-13 | **arXiv**: [2602.13156v1](http://arxiv.org/abs/2602.13156v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.13156v1)

**Categories**: cs.CR, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Rapidly evolving cyberattacks demand incident response systems that can autonomously learn and adapt to changing threats. Prior work has extensively explored the reinforcement learning approach, which involves learning response strategies through extensive simulation of the incident. While this approach can be effective, it requires handcrafted modeling of the simulator and suppresses useful semantics from raw system logs and alerts. To address these limitations, we propose to leverage large language models' (LLM) pre-trained security knowledge and in-context learning to create an end-to-end agentic solution for incident response planning. Specifically, our agent integrates four functionalities, perception, reasoning, planning, and action, into one lightweight LLM (14b model). Through fine-tuning and chain-of-thought reasoning, our LLM agent is capable of processing system logs and inferring the underlying network state (perception), updating its conjecture of attack models (reasoning), simulating consequences under different response strategies (planning), and generating an effective response (action). By comparing LLM-simulated outcomes with actual observations, the LLM agent repeatedly refines its attack conjecture and corresponding response, thereby demonstrating in-context adaptation. Our agentic approach is free of modeling and can run on commodity hardware. When evaluated on incident logs reported in the literature, our agent achieves recovery up to 23% faster than those of frontier LLMs.

快速发展的网络攻击需要事件响应系统能够自主学习并适应不断变化的威胁。先前的工作广泛探索了强化学习方法，其中涉及通过对事件的广泛模拟来学习响应策略。虽然这种方法可能很有效，但它需要对模拟器进行手工建模，并抑制原始系统日志和警报中的有用语义。为了解决这些限制，我们建议利用大型语言模型（LLM）预先训练的安全知识和上下文学习来为事件响应计划创建端到端代理解决方案。具体来说，我们的代理将感知、推理、规划和行动四种功能集成到一个轻量级 LLM（14b 模型）中。通过微调和链式推理，我们的LLM代理能够处理系统日志并推断底层网络状态（感知），更新其对攻击模型的猜想（推理），模拟不同响应策略下的后果（计划），并生成有效的响应（行动）。通过将 LLM 模拟的结果与实际观察结果进行比较，LLM 代理反复完善其攻击猜想和相应的响应，从而展示上下文适应能力。我们的代理方法无需建模，并且可以在商用硬件上运行。当对文献中报告的事件日志进行评估时，我们的代理的恢复速度比前沿法学硕士快 23%。

</details>

---

## 3. TRACE: Temporal Reasoning via Agentic Context Evolution for Streaming Electronic Health Records (EHRs) / TRACE：通过代理上下文演化进行时间推理，用于流式传输电子健康记录 (EHR)

**Date**: 2026-02-13 | **arXiv**: [2602.12833v1](http://arxiv.org/abs/2602.12833v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.12833v1)

**Categories**: cs.LG, cs.AI, cs.MA

<details><summary><b>Abstract / 摘要</b></summary>

Large Language Models (LLMs) encode extensive medical knowledge but struggle to apply it reliably to longitudinal patient trajectories, where evolving clinical states, irregular timing, and heterogeneous events degrade performance over time. Existing adaptation strategies rely on fine-tuning or retrieval-based augmentation, which introduce computational overhead, privacy constraints, or instability under long contexts. We introduce TRACE (Temporal Reasoning via Agentic Context Evolution), a framework that enables temporal clinical reasoning with frozen LLMs by explicitly structuring and maintaining context rather than extending context windows or updating parameters. TRACE operates over a dual-memory architecture consisting of a static Global Protocol encoding institutional clinical rules and a dynamic Individual Protocol tracking patient-specific state. Four agentic components, Router, Reasoner, Auditor, and Steward, coordinate over this structured memory to support temporal inference and state evolution. The framework maintains bounded inference cost via structured state compression and selectively audits safety-critical clinical decisions. Evaluated on longitudinal clinical event streams from MIMIC-IV, TRACE significantly improves next-event prediction accuracy, protocol adherence, and clinical safety over long-context and retrieval-augmented baselines, while producing interpretable and auditable reasoning traces.

大型语言模型 (LLM) 编码广泛的医学知识，但很难将其可靠地应用于纵向患者轨迹，其中不断变化的临床状态、不规则的时间安排和异质事件会随着时间的推移而降低性能。现有的适应策略依赖于微调或基于检索的增强，这会带来计算开销、隐私限制或长上下文下的不稳定。我们引入了 TRACE（通过 Agentic Context Evolution 进行时间推理），这是一个框架，通过显式构建和维护上下文而不是扩展上下文窗口或更新参数，可以使用冻结的 LLM 进行时间临床推理。 TRACE 在双内存架构上运行，该架构由编码机构临床规则的静态全局协议和跟踪患者特定状态的动态个人协议组成。四个代理组件（路由器、推理器、审计器和管家）在此结构化内存上进行协调，以支持时间推理和状态演化。该框架通过结构化状态压缩维持有界推理成本，并有选择地审核安全关键的临床决策。 TRACE 对 MIMIC-IV 的纵向临床事件流进行评估，在长上下文和检索增强基线上显着提高了下一个事件预测的准确性、协议遵守性和临床安全性，同时产生可解释和可审计的推理轨迹。

</details>

---

## 4. TraceBack: Multi-Agent Decomposition for Fine-Grained Table Attribution / TraceBack：细粒度表归因的多代理分解

**Date**: 2026-02-13 | **arXiv**: [2602.13059v1](http://arxiv.org/abs/2602.13059v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.13059v1)

**Categories**: cs.CL

<details><summary><b>Abstract / 摘要</b></summary>

Question answering (QA) over structured tables requires not only accurate answers but also transparency about which cells support them. Existing table QA systems rarely provide fine-grained attribution, so even correct answers often lack verifiable grounding, limiting trust in high-stakes settings. We address this with TraceBack, a modular multi-agent framework for scalable, cell-level attribution in single-table QA. TraceBack prunes tables to relevant rows and columns, decomposes questions into semantically coherent sub-questions, and aligns each answer span with its supporting cells, capturing both explicit and implicit evidence used in intermediate reasoning steps. To enable systematic evaluation, we release CITEBench, a benchmark with phrase-to-cell annotations drawn from ToTTo, FetaQA, and AITQA. We further propose FairScore, a reference-less metric that compares atomic facts derived from predicted cells and answers to estimate attribution precision and recall without human cell labels. Experiments show that TraceBack substantially outperforms strong baselines across datasets and granularities, while FairScore closely tracks human judgments and preserves relative method rankings, supporting interpretable and scalable evaluation of table-based QA.

通过结构化表格进行问答 (QA) 不仅需要准确的答案，还需要透明地了解哪些单元格支持它们。现有的桌面 QA 系统很少提供细粒度的归因，因此即使是正确的答案也往往缺乏可验证的基础，从而限制了对高风险环境的信任。我们使用 TraceBack 来解决这个问题，TraceBack 是一个模块化多代理框架，用于在单表 QA 中进行可扩展的单元级归因。 TraceBack 将表格修剪为相关的行和列，将问题分解为语义上连贯的子问题，并将每个答案范围与其支持单元格对齐，捕获中间推理步骤中使用的显式和隐式证据。为了实现系统评估，我们发布了 CITEBench，这是一个基准，包含来自 ToTTo、FetaQA 和 AITQA 的短语到单元格注释。我们进一步提出 FairScore，这是一种无参考指标，可以比较从预测细胞得出的原子事实和答案，以估计归因精度和召回率，而无需人类细胞标签。实验表明，TraceBack 在数据集和粒度方面大大优于强大的基线，而 FairScore 密切跟踪人类判断并保留相对方法排名，支持基于表的 QA 的可解释和可扩展的评估。

</details>

---

## 5. SciAgentGym: Benchmarking Multi-Step Scientific Tool-use in LLM Agents / SciAgentGym：LLM 代理中多步骤科学工具使用的基准测试

**Date**: 2026-02-13 | **arXiv**: [2602.12984v1](http://arxiv.org/abs/2602.12984v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.12984v1)

**Categories**: cs.CL

<details><summary><b>Abstract / 摘要</b></summary>

Scientific reasoning inherently demands integrating sophisticated toolkits to navigate domain-specific knowledge. Yet, current benchmarks largely overlook agents' ability to orchestrate tools for such rigorous workflows. To bridge this gap, we introduce SciAgentGym, a scalable interactive environment featuring 1,780 domain-specific tools across four natural science disciplines, supported by a robust execution infrastructure. Complementing this, we present SciAgentBench, a tiered evaluation suite designed to stress-test agentic capabilities from elementary actions to long-horizon workflows. Our evaluation identifies a critical bottleneck: state-of-the-art models struggle with complex scientific tool-use. Even for a leading model like GPT-5, success rates drop sharply from 60.6% to 30.9% as interaction horizons extend, primarily due to failures in multi-step workflow execution. To address this, we propose SciForge, a data synthesis method that models the tool action space as a dependency graph to generate logic-aware training trajectories. By fine-tuning on these trajectories, our SciAgent-8B outperforms the significantly larger Qwen3-VL-235B-Instruct while exhibiting positive cross-domain transfer of scientific tool-use capabilities. These results underscore the promising potential of next-generation autonomous scientific agents.

科学推理本质上需要集成复杂的工具包来导航特定领域的知识。然而，当前的基准在很大程度上忽视了代理为如此严格的工作流程编排工具的能力。为了弥补这一差距，我们推出了 SciAgentGym，这是一个可扩展的交互式环境，具有跨四个自然科学学科的 1,780 个特定领域工具，并由强大的执行基础设施提供支持。作为补充，我们推出了 SciAgentBench，这是一个分层评估套件，旨在对从基本操作到长期工作流程的代理功能进行压力测试。我们的评估发现了一个关键瓶颈：最先进的模型难以应对复杂的科学工具的使用。即使对于像 GPT-5 这样的领先模型，随着交互范围的扩展，成功率也会从 60.6% 急剧下降到 30.9%，这主要是由于多步骤工作流执行失败。为了解决这个问题，我们提出了 SciForge，一种数据合成方法，它将工具操作空间建模为依赖图，以生成逻辑感知的训练轨迹。通过对这些轨迹进行微调，我们的 SciAgent-8B 的性能明显优于更大的 Qwen3-VL-235B-Instruct，同时表现出科学工具使用能力的积极跨域转移。这些结果强调了下一代自主科学代理的巨大潜力。

</details>

---

## 6. UniManip: General-Purpose Zero-Shot Robotic Manipulation with Agentic Operational Graph / UniManip：具有代理操作图的通用零射击机器人操作

**Date**: 2026-02-13 | **arXiv**: [2602.13086v1](http://arxiv.org/abs/2602.13086v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.13086v1)

**Categories**: cs.RO

<details><summary><b>Abstract / 摘要</b></summary>

Achieving general-purpose robotic manipulation requires robots to seamlessly bridge high-level semantic intent with low-level physical interaction in unstructured environments. However, existing approaches falter in zero-shot generalization: end-to-end Vision-Language-Action (VLA) models often lack the precision required for long-horizon tasks, while traditional hierarchical planners suffer from semantic rigidity when facing open-world variations. To address this, we present UniManip, a framework grounded in a Bi-level Agentic Operational Graph (AOG) that unifies semantic reasoning and physical grounding. By coupling a high-level Agentic Layer for task orchestration with a low-level Scene Layer for dynamic state representation, the system continuously aligns abstract planning with geometric constraints, enabling robust zero-shot execution. Unlike static pipelines, UniManip operates as a dynamic agentic loop: it actively instantiates object-centric scene graphs from unstructured perception, parameterizes these representations into collision-free trajectories via a safety-aware local planner, and exploits structured memory to autonomously diagnose and recover from execution failures. Extensive experiments validate the system's robust zero-shot capability on unseen objects and tasks, demonstrating a 22.5% and 25.0% higher success rate compared to state-of-the-art VLA and hierarchical baselines, respectively. Notably, the system enables direct zero-shot transfer from fixed-base setups to mobile manipulation without fine-tuning or reconfiguration. Our open-source project page can be found at https://henryhcliu.github.io/unimanip.

实现通用机器人操作需要机器人在非结构化环境中无缝地连接高级语义意图与低级物理交互。然而，现有方法在零样本泛化方面表现不佳：端到端视觉语言动作（VLA）模型通常缺乏长期任务所需的精度，而传统的分层规划器在面对开放世界变化时会受到语义僵化的困扰。为了解决这个问题，我们提出了 UniManip，这是一个基于双层代理操作图（AOG）的框架，它统一了语义推理和物理基础。通过将用于任务编排的高级代理层与用于动态状态表示的低级场景层相结合，系统不断地将抽象规划与几何约束保持一致，从而实现稳健的零样本执行。与静态管道不同，UniManip 作为动态代理循环运行：它根据非结构化感知主动实例化以对象为中心的场景图，通过安全感知本地规划器将这些表示参数化为无碰撞轨迹，并利用结构化内存来自主诊断执行故障并从执行故障中恢复。大量实验验证了该系统对未见过的物体和任务的强大零样本能力，与最先进的 VLA 和分层基线相比，成功率分别高出 22.5% 和 25.0%。值得注意的是，该系统能够从固定基地设置直接零次转移到移动操纵，无需微调或重新配置。我们的开源项目页面可以在 https://henryhcliu.github.io/unimanip 找到。

</details>

---

## 7. Agentic AI for Robot Control: Flexible but still Fragile / 用于机器人控制的代理人工智能：灵活但仍然脆弱

**Date**: 2026-02-13 | **arXiv**: [2602.13081v1](http://arxiv.org/abs/2602.13081v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.13081v1)

**Categories**: cs.RO

<details><summary><b>Abstract / 摘要</b></summary>

Recent work leverages the capabilities and commonsense priors of generative models for robot control. In this paper, we present an agentic control system in which a reasoning-capable language model plans and executes tasks by selecting and invoking robot skills within an iterative planner and executor loop. We deploy the system on two physical robot platforms in two settings: (i) tabletop grasping, placement, and box insertion in indoor mobile manipulation (Mobipick) and (ii) autonomous agricultural navigation and sensing (Valdemar). Both settings involve uncertainty, partial observability, sensor noise, and ambiguous natural-language commands. The system exposes structured introspection of its planning and decision process, reacts to exogenous events via explicit event checks, and supports operator interventions that modify or redirect ongoing execution. Across both platforms, our proof-of-concept experiments reveal substantial fragility, including non-deterministic suboptimal behavior, instruction-following errors, and high sensitivity to prompt specification. At the same time, the architecture is flexible: transfer to a different robot and task domain largely required updating the system prompt (domain model, affordances, and action catalogue) and re-binding the same tool interface to the platform-specific skill API.

最近的工作利用了机器人控制生成模型的功能和常识先验。在本文中，我们提出了一种代理控制系统，其中具有推理能力的语言模型通过在迭代规划器和执行器循环中选择和调用机器人技能来规划和执行任务。我们在两个物理机器人平台上以两种设置部署该系统：(i) 室内移动操纵中的桌面抓取、放置和盒子插入 (Mobipick) 和 (ii) 自主农业导航和传感 (Valdemar)。这两种设置都涉及不确定性、部分可观察性、传感器噪声和模糊的自然语言命令。该系统公开其规划和决策过程的结构化内省，通过显式事件检查对外部事件做出反应，并支持修改或重定向正在进行的执行的操作员干预。在这两个平台上，我们的概念验证实验揭示了巨大的脆弱性，包括非确定性的次优行为、指令遵循错误以及对提示规范的高度敏感性。同时，该架构非常灵活：转移到不同的机器人和任务域很大程度上需要更新系统提示（域模型、可供性和动作目录）并将相同的工具接口重新绑定到特定于平台的技能API。

</details>

---



</details>

<details><summary><b>2026-02-14 (53 papers)</b></summary>

# arXiv Agent Papers - 2026-02-14

**Paper Count**: 53

---

## 1. UniT: Unified Multimodal Chain-of-Thought Test-time Scaling / UnitT：统一多模式思想链测试时间缩放

**Date**: 2026-02-12 | **arXiv**: [2602.12279v1](http://arxiv.org/abs/2602.12279v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.12279v1)

**Categories**: cs.CV, cs.AI, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Unified models can handle both multimodal understanding and generation within a single architecture, yet they typically operate in a single pass without iteratively refining their outputs. Many multimodal tasks, especially those involving complex spatial compositions, multiple interacting objects, or evolving instructions, require decomposing instructions, verifying intermediate results, and making iterative corrections. While test-time scaling (TTS) has demonstrated that allocating additional inference compute for iterative reasoning substantially improves language model performance, extending this paradigm to unified multimodal models remains an open challenge. We introduce UniT, a framework for multimodal chain-of-thought test-time scaling that enables a single unified model to reason, verify, and refine across multiple rounds. UniT combines agentic data synthesis, unified model training, and flexible test-time inference to elicit cognitive behaviors including verification, subgoal decomposition, and content memory. Our key findings are: (1) unified models trained on short reasoning trajectories generalize to longer inference chains at test time; (2) sequential chain-of-thought reasoning provides a more scalable and compute-efficient TTS strategy than parallel sampling; (3) training on generation and editing trajectories improves out-of-distribution visual reasoning. These results establish multimodal test-time scaling as an effective paradigm for advancing both generation and understanding in unified models.

统一模型可以在单个架构中处理多模态理解和生成，但它们通常在单次传递中运行，而无需迭代地细化其输出。许多多模态任务，特别是那些涉及复杂空间组成、多个交互对象或不断发展的指令的任务，需要分解指令、验证中间结果并进行迭代修正。虽然测试时间扩展 (TTS) 已经证明，为迭代推理分配额外的推理计算可以显着提高语言模型的性能，但将此范式扩展到统一的多模态模型仍然是一个开放的挑战。我们引入了 UnitT，这是一个用于多模式思想链测试时间扩展的框架，它使单个统一模型能够在多轮中进行推理、验证和细化。 UnitT 结合了代理数据合成、统一模型训练和灵活的测试时推理，以引发认知行为，包括验证、子目标分解和内容记忆。我们的主要发现是：（1）在短推理轨迹上训练的统一模型在测试时可以推广到更长的推理链； (2) 顺序思想链推理提供了比并行采样更具可扩展性和计算效率的 TTS 策略； （3）生成和编辑轨迹的训练改善了分布外的视觉推理。这些结果将多模式测试时间缩放确立为促进统一模型的生成和理解的有效范例。

</details>

---

## 2. Agentic Test-Time Scaling for WebAgents / WebAgents 的代理测试时间缩放

**Date**: 2026-02-12 | **arXiv**: [2602.12276v1](http://arxiv.org/abs/2602.12276v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.12276v1)

**Categories**: cs.AI, cs.CL

<details><summary><b>Abstract / 摘要</b></summary>

Test-time scaling has become a standard way to improve performance and boost reliability of neural network models. However, its behavior on agentic, multi-step tasks remains less well-understood: small per-step errors can compound over long horizons; and we find that naive policies that uniformly increase sampling show diminishing returns. In this work, we present CATTS, a simple technique for dynamically allocating compute for multi-step agents. We first conduct an empirical study of inference-time scaling for web agents. We find that uniformly increasing per-step compute quickly saturates in long-horizon environments. We then investigate stronger aggregation strategies, including an LLM-based Arbiter that can outperform naive voting, but that can overrule high-consensus decisions. We show that uncertainty statistics derived from the agent's own vote distribution (entropy and top-1/top-2 margin) correlate with downstream success and provide a practical signal for dynamic compute allocation. Based on these findings, we introduce Confidence-Aware Test-Time Scaling (CATTS), which uses vote-derived uncertainty to allocate compute only when decisions are genuinely contentious. CATTS improves performance on WebArena-Lite and GoBrowse by up to 9.1% over React while using up to 2.3x fewer tokens than uniform scaling, providing both efficiency gains and an interpretable decision rule.

测试时间缩放已成为提高神经网络模型性能和可靠性的标准方法。然而，它在代理、多步骤任务上的行为仍然不太被理解：每一步的小错误可能会在长期内复合；我们发现，统一增加抽样的幼稚政策显示出收益递减。在这项工作中，我们提出了 CATTS，这是一种为多步代理动态分配计算的简单技术。我们首先对网络代理的推理时间缩放进行实证研究。我们发现，在长期环境中，均匀增加的每步计算很快就会饱和。然后，我们研究更强大的聚合策略，包括基于 LLM 的仲裁器，它可以超越朴素投票，但可以推翻高度共识的决策。我们表明，从代理自身投票分布（熵和 top-1/top-2 余量）得出的不确定性统计数据与下游成功相关，并为动态计算分配提供了实用信号。基于这些发现，我们引入了置信度感知测试时间缩放（CATTS），它仅在决策真正有争议时才使用投票产生的不确定性来分配计算。与 React 相比，CATTS 在 WebArena-Lite 和 GoBrowse 上的性能提高了高达 9.1%，同时使用的令牌比统一缩放少了 2.3 倍，从而提供了效率提升和可解释的决策规则。

</details>

---

## 3. CM2: Reinforcement Learning with Checklist Rewards for Multi-Turn and Multi-Step Agentic Tool Use / CM2：针对多轮和多步代理工具使用的强化学习和清单奖励

**Date**: 2026-02-12 | **arXiv**: [2602.12268v1](http://arxiv.org/abs/2602.12268v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.12268v1)

**Categories**: cs.AI

**Code**: https://github.com/namezhenzhang/CM2-RLCR-Tool-Agent.

<details><summary><b>Abstract / 摘要</b></summary>

AI agents are increasingly used to solve real-world tasks by reasoning over multi-turn user interactions and invoking external tools. However, applying reinforcement learning to such settings remains difficult: realistic objectives often lack verifiable rewards and instead emphasize open-ended behaviors; moreover, RL for multi-turn, multi-step agentic tool use is still underexplored; and building and maintaining executable tool environments is costly, limiting scale and coverage. We propose CM2, an RL framework that replaces verifiable outcome rewards with checklist rewards. CM2 decomposes each turn's intended behavior into fine-grained binary criteria with explicit evidence grounding and structured metadata, turning open-ended judging into more stable classification-style decisions. To balance stability and informativeness, our method adopts a strategy of sparse reward assignment but dense evaluation criteria. Training is performed in a scalable LLM-simulated tool environment, avoiding heavy engineering for large tool sets. Experiments show that CM2 consistently improves over supervised fine-tuning. Starting from an 8B Base model and training on an 8k-example RL dataset, CM2 improves over the SFT counterpart by 8 points on tau^-Bench, by 10 points on BFCL-V4, and by 12 points on ToolSandbox. The results match or even outperform similarly sized open-source baselines, including the judging model. CM2 thus provides a scalable recipe for optimizing multi-turn, multi-step tool-using agents without relying on verifiable rewards. Code provided by the open-source community: https://github.com/namezhenzhang/CM2-RLCR-Tool-Agent.

人工智能代理越来越多地用于通过多轮用户交互推理和调用外部工具来解决现实世界的任务。然而，将强化学习应用于此类环境仍然很困难：现实的目标往往缺乏可验证的奖励，而是强调开放式行为；此外，用于多轮、多步代理工具使用的强化学习仍处于探索之中；构建和维护可执行工具环境成本高昂，限制了规模和覆盖范围。我们提出了 CM2，一个 RL 框架，用清单奖励取代可验证的结果奖励。 CM2 将每个回合的预期行为分解为具有明确证据基础和结构化元数据的细粒度二元标准，将开放式判断转变为更稳定的分类式决策。为了平衡稳定性和信息量，我们的方法采用稀疏奖励分配但密集评估标准的策略。培训在可扩展的法学硕士模拟工具环境中进行，避免了大型工具集的繁重工程。实验表明，CM2 始终优于监督微调。从 8B Base 模型开始，在 8k-example RL 数据集上进行训练，CM2 在 tau^-Bench 上比 SFT 模型提高了 8 个点，在 BFCL-V4 上提高了 10 个点，在 ToolSandbox 上提高了 12 个点。结果匹配甚至优于类似规模的开源基线，包括评审模型。因此，CM2 提供了一种可扩展的方法，用于优化多回合、多步骤的工具使用代理，而无需依赖可验证的奖励。开源社区提供的代码：https://github.com/namezhenzhang/CM2-RLCR-Tool-Agent。

</details>

---

## 4. Think like a Scientist: Physics-guided LLM Agent for Equation Discovery / 像科学家一样思考：物理引导的法学硕士方程发现代理

**Date**: 2026-02-12 | **arXiv**: [2602.12259v1](http://arxiv.org/abs/2602.12259v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.12259v1)

**Categories**: cs.AI, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Explaining observed phenomena through symbolic, interpretable formulas is a fundamental goal of science. Recently, large language models (LLMs) have emerged as promising tools for symbolic equation discovery, owing to their broad domain knowledge and strong reasoning capabilities. However, most existing LLM-based systems try to guess equations directly from data, without modeling the multi-step reasoning process that scientists often follow: first inferring physical properties such as symmetries, then using these as priors to restrict the space of candidate equations. We introduce KeplerAgent, an agentic framework that explicitly follows this scientific reasoning process. The agent coordinates physics-based tools to extract intermediate structure and uses these results to configure symbolic regression engines such as PySINDy and PySR, including their function libraries and structural constraints. Across a suite of physical equation benchmarks, KeplerAgent achieves substantially higher symbolic accuracy and greater robustness to noisy data than both LLM and traditional baselines.

通过符号、可解释的公式来解释观察到的现象是科学的基本目标。最近，大语言模型（LLM）因其广泛的领域知识和强大的推理能力而成为符号方程发现的有前途的工具。然而，大多数现有的基于法学硕士的系统试图直接从数据中猜测方程，而不是对科学家经常遵循的多步骤推理过程进行建模：首先推断对称性等物理特性，然后使用这些作为先验来限制候选方程的空间。我们引入 KeplerAgent，这是一个明确遵循这一科学推理过程的代理框架。该代理协调基于物理的工具来提取中间结构，并使用这些结果来配置符号回归引擎，例如 PySINDy 和 PySR，包括它们的函数库和结构约束。在一系列物理方程基准中，KeplerAgent 比 LLM 和传统基准实现了更高的符号准确性和对噪声数据的更强鲁棒性。

</details>

---

## 5. On the Adoption of AI Coding Agents in Open-source Android and iOS Development / 关于人工智能编码代理在开源 Android 和 iOS 开发中的采用

**Date**: 2026-02-12 | **arXiv**: [2602.12144v1](http://arxiv.org/abs/2602.12144v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.12144v1)

**Categories**: cs.SE, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

AI coding agents are increasingly contributing to software development, yet their impact on mobile development has received little empirical attention. In this paper, we present the first category-level empirical study of agent-generated code in open-source mobile app projects. We analyzed PR acceptance behaviors across mobile platforms, agents, and task categories using 2,901 AI-authored pull requests (PRs) in 193 verified Android and iOS open-source GitHub repositories in the AIDev dataset. We find that Android projects have received 2x more AI-authored PRs and have achieved higher PR acceptance rate (71%) than iOS (63%), with significant agent-level variation on Android. Across task categories, PRs with routine tasks (feature, fix, and ui) achieve the highest acceptance, while structural changes like refactor and build achieve lower success and longer resolution times. Furthermore, our evolution analysis shows improvement in PR resolution time on Android through mid-2025 before it declined again. Our findings offer the first evidence-based characterization of AI agents effects on OSS mobile projects and establish empirical baselines for evaluating agent-generated contributions to design platform aware agentic systems.

人工智能编码代理对软件开发的贡献越来越大，但它们对移动开发的影响却很少受到实证关注。在本文中，我们首次对开源移动应用项目中代理生成的代码进行了类别级实证研究。我们使用 AIDev 数据集中 193 个经过验证的 Android 和 iOS 开源 GitHub 存储库中的 2,901 个 AI 编写的拉取请求 (PR) 分析了跨移动平台、代理和任务类别的 PR 接受行为。我们发现，Android 项目收到的 AI 创作 PR 数量是 iOS 的 2 倍，并且 PR 接受率 (71%) 高于 iOS (63%)，并且 Android 上的代理级别差异显着。在各个任务类别中，具有常规任务（功能、修复和 ui）的 PR 获得了最高的接受度，而重构和构建等结构性更改的成功率较低，解决时间也较长。此外，我们的演变分析显示，到 2025 年中期，Android 上的 PR 解决时间会有所改善，然后再次下降。我们的研究结果首次基于证据描述了人工智能代理对 OSS 移动项目的影响，并建立了经验基线，用于评估代理对设计平台感知代理系统的贡献。

</details>

---

## 6. STAR : Bridging Statistical and Agentic Reasoning for Large Model Performance Prediction / STAR：桥接统计和代理推理以进行大型模型性能预测

**Date**: 2026-02-12 | **arXiv**: [2602.12143v1](http://arxiv.org/abs/2602.12143v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.12143v1)

**Categories**: cs.AI, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

As comprehensive large model evaluation becomes prohibitively expensive, predicting model performance from limited observations has become essential. However, existing statistical methods struggle with pattern shifts, data sparsity, and lack of explanation, while pure LLM methods remain unreliable. We propose STAR, a framework that bridges data-driven STatistical expectations with knowledge-driven Agentic Reasoning. STAR leverages specialized retrievers to gather external knowledge and embeds semantic features into Constrained Probabilistic Matrix Factorization (CPMF) to generate statistical expectations with uncertainty. A reasoning module guided by Expectation Violation Theory (EVT) then refines predictions through intra-family analysis, cross-model comparison, and credibility-aware aggregation, producing adjustments with traceable explanations. Extensive experiments show that STAR consistently outperforms all baselines on both score-based and rank-based metrics, delivering a 14.46% gain in total score over the strongest statistical method under extreme sparsity, with only 1--2 observed scores per test model.

随着全面的大型模型评估变得异常昂贵，从有限的观测中预测模型性能变得至关重要。然而，现有的统计方法面临着模式转变、数据稀疏和缺乏解释的问题，而纯粹的法学硕士方法仍然不可靠。我们提出了 STAR，一个将数据驱动的统计期望与知识驱动的代理推理联系起来的框架。 STAR 利用专门的检索器收集外部知识，并将语义特征嵌入到约束概率矩阵分解 (CPMF) 中，以生成具有不确定性的统计期望。然后，由期望违背理论 (EVT) 指导的推理模块通过家庭内分析、跨模型比较和可信度感知聚合来完善预测，并通过可追溯的解释进行调整。大量实验表明，STAR 在基于分数和基于排名的指标上始终优于所有基线，在极端稀疏性下，总分比最强的统计方法高出 14.46%，每个测试模型仅观察到 1--2 个分数。

</details>

---

## 7. Choose Your Agent: Tradeoffs in Adopting AI Advisors, Coaches, and Delegates in Multi-Party Negotiation / 选择你的代理人：在多方谈判中采用人工智能顾问、教练和代表的权衡

**Date**: 2026-02-12 | **arXiv**: [2602.12089v1](http://arxiv.org/abs/2602.12089v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.12089v1)

**Categories**: cs.GT, cs.AI, cs.HC

<details><summary><b>Abstract / 摘要</b></summary>

As AI usage becomes more prevalent in social contexts, understanding agent-user interaction is critical to designing systems that improve both individual and group outcomes. We present an online behavioral experiment (N = 243) in which participants play three multi-turn bargaining games in groups of three. Each game, presented in randomized order, grants \textit{access to} a single LLM assistance modality: proactive recommendations from an \textit{Advisor}, reactive feedback from a \textit{Coach}, or autonomous execution by a \textit{Delegate}; all modalities are powered by an underlying LLM that achieves superhuman performance in an all-agent environment. On each turn, participants privately decide whether to act manually or use the AI modality available in that game. Despite preferring the \textit{Advisor} modality, participants achieve the highest mean individual gains with the \textit{Delegate}, demonstrating a preference-performance misalignment. Moreover, delegation generates positive externalities; even non-adopting users in \textit{access-to-delegate} treatment groups benefit by receiving higher-quality offers. Mechanism analysis reveals that the \textit{Delegate} agent acts as a market maker, injecting rational, Pareto-improving proposals that restructure the trading environment. Our research reveals a gap between agent capabilities and realized group welfare. While autonomous agents can exhibit super-human strategic performance, their impact on realized welfare gains can be constrained by interfaces, user perceptions, and adoption barriers. Assistance modalities should be designed as mechanisms with endogenous participation; adoption-compatible interaction rules are a prerequisite to improving human welfare with automated assistance.

随着人工智能在社会环境中的使用变得越来越普遍，了解代理与用户的交互对于设计可改善个人和群体结果的系统至关重要。我们提出了一个在线行为实验（N = 243），其中参与者以三人为一组玩三个多回合讨价还价游戏。每个游戏以随机顺序呈现，授予 \textit{访问}单一 LLM 协助模式：来自 \textit{Advisor} 的主动建议、来自 \textit{Coach} 的反应性反馈，或由 \textit{Delegate} 自主执行；所有模式均由底层法学硕士提供支持，可在全智能体环境中实现超人的性能。在每一轮中，参与者私下决定是否手动操作或使用该游戏中可用的人工智能模式。尽管更喜欢 \textit{Advisor} 模式，但参与者通过 \textit{Delegate} 实现了最高的平均个人收益，这表明偏好与绩效不一致。此外，授权会产生正外部性；即使 \textit{access-to-delegate} 治疗组中的非采用用户也能通过接收更高质量的优惠而受益。机制分析表明， \textit{Delegate} 代理充当做市商，注入理性的、帕累托改进的建议来重组交易环境。我们的研究揭示了代理人能力与实现的群体福利之间的差距。虽然自主代理可以表现出超人的战略绩效，但它们对实现的福利收益的影响可能会受到界面、用户感知和采用障碍的限制。援助方式应设计为具有内在参与的机制；与采用兼容的交互规则是通过自动化援助改善人类福利的先决条件。

</details>

---

## 8. Differentiable Modal Logic for Multi-Agent Diagnosis, Orchestration and Communication / 用于多代理诊断、编排和通信的可微模态逻辑

**Date**: 2026-02-12 | **arXiv**: [2602.12083v1](http://arxiv.org/abs/2602.12083v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.12083v1)

**Categories**: cs.AI, cs.LO

<details><summary><b>Abstract / 摘要</b></summary>

As multi-agent AI systems evolve from simple chatbots to autonomous swarms, debugging semantic failures requires reasoning about knowledge, belief, causality, and obligation, precisely what modal logic was designed to formalize. However, traditional modal logic requires manual specification of relationship structures that are unknown or dynamic in real systems. This tutorial demonstrates differentiable modal logic (DML), implemented via Modal Logical Neural Networks (MLNNs), enabling systems to learn trust networks, causal chains, and regulatory boundaries from behavioral data alone.   We present a unified neurosymbolic debugging framework through four modalities: epistemic (who to trust), temporal (when events cause failures), deontic (what actions are permitted), and doxastic (how to interpret agent confidence). Each modality is demonstrated on concrete multi-agent scenarios, from discovering deceptive alliances in diplomacy games to detecting LLM hallucinations, with complete implementations showing how logical contradictions become learnable optimization objectives. Key contributions for the neurosymbolic community: (1) interpretable learned structures where trust and causality are explicit parameters, not opaque embeddings; (2) knowledge injection via differentiable axioms that guide learning with sparse data (3) compositional multi-modal reasoning that combines epistemic, temporal, and deontic constraints; and (4) practical deployment patterns for monitoring, active control and communication of multi-agent systems. All code provided as executable Jupyter notebooks.

随着多智能体人工智能系统从简单的聊天机器人发展为自治群体，调试语义故障需要对知识、信念、因果关系和义务进行推理，而这正是模态逻辑旨在形式化的目的。然而，传统的模态逻辑需要手动指定实际系统中未知或动态的关系结构。本教程演示了通过模态逻辑神经网络 (MLNN) 实现的可微模态逻辑 (DML)，使系统能够仅从行为数据中学习信任网络、因果链和监管边界。   我们通过四种模式提出了一个统一的神经符号调试框架：认知（信任谁）、时间（当事件导致失败时）、道义（允许哪些行为）和信念（如何解释代理信心）。每种模式都在具体的多智能体场景中进行了演示，从发现外交游戏中的欺骗性联盟到检测 LLM 幻觉，并通过完整的实现展示了逻辑矛盾如何成为可学习的优化目标。对神经符号社区的主要贡献：（1）可解释的学习结构，其中信任和因果关系是显式参数，而不是不透明的嵌入； （2）通过可微公理进行知识注入，指导稀疏数据的学习（3）结合认知、时间和道义约束的组合多模态推理； (4)多智能体系统监控、主动控制和通信的实用部署模式。所有代码均作为可执行 Jupyter 笔记本提供。

</details>

---

## 9. LawThinker: A Deep Research Legal Agent in Dynamic Environments / LawThinker：动态环境中的深度研究法律代理人

**Date**: 2026-02-12 | **arXiv**: [2602.12056v1](http://arxiv.org/abs/2602.12056v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.12056v1)

**Categories**: cs.AI

**Code**: https://github.com/yxy-919/LawThinker-agent

<details><summary><b>Abstract / 摘要</b></summary>

Legal reasoning requires not only correct outcomes but also procedurally compliant reasoning processes. However, existing methods lack mechanisms to verify intermediate reasoning steps, allowing errors such as inapplicable statute citations to propagate undetected through the reasoning chain. To address this, we propose LawThinker, an autonomous legal research agent that adopts an Explore-Verify-Memorize strategy for dynamic judicial environments. The core idea is to enforce verification as an atomic operation after every knowledge exploration step. A DeepVerifier module examines each retrieval result along three dimensions of knowledge accuracy, fact-law relevance, and procedural compliance, with a memory module for cross-round knowledge reuse in long-horizon tasks. Experiments on the dynamic benchmark J1-EVAL show that LawThinker achieves a 24% improvement over direct reasoning and an 11% gain over workflow-based methods, with particularly strong improvements on process-oriented metrics. Evaluations on three static benchmarks further confirm its generalization capability. The code is available at https://github.com/yxy-919/LawThinker-agent .

法律推理不仅需要正确的结果，还需要符合程序的推理过程。然而，现有方法缺乏验证中间推理步骤的机制，从而导致诸如不适用的法规引用之类的错误在推理链中传播而未被检测到。为了解决这个问题，我们提出了 LawThinker，这是一种自主法律研究代理，它针对动态司法环境采用探索-验证-记忆策略。核心思想是在每个知识探索步骤之后将验证作为原子操作来强制执行。 DeepVerifier 模块沿着知识准确性、事实规律相关性和程序合规性三个维度检查每个检索结果，并带有一个内存模块，用于长期任务中的跨轮知识重用。动态基准 J1-EVAL 上的实验表明，LawThinker 比直接推理提高了 24%，比基于工作流的方法提高了 11%，尤其是在面向流程的指标方面的改进尤其显着。对三个静态基准的评估进一步证实了其泛化能力。该代码可在 https://github.com/yxy-919/LawThinker-agent 获取。

</details>

---

## 10. Multi UAVs Preflight Planning in a Shared and Dynamic Airspace / 共享动态空域中的多无人机飞行前规划

**Date**: 2026-02-12 | **arXiv**: [2602.12055v1](http://arxiv.org/abs/2602.12055v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.12055v1)

**Categories**: cs.AI, cs.MA, cs.RO

<details><summary><b>Abstract / 摘要</b></summary>

Preflight planning for large-scale Unmanned Aerial Vehicle (UAV) fleets in dynamic, shared airspace presents significant challenges, including temporal No-Fly Zones (NFZs), heterogeneous vehicle profiles, and strict delivery deadlines. While Multi-Agent Path Finding (MAPF) provides a formal framework, existing methods often lack the scalability and flexibility required for real-world Unmanned Traffic Management (UTM). We propose DTAPP-IICR: a Delivery-Time Aware Prioritized Planning method with Incremental and Iterative Conflict Resolution. Our framework first generates an initial solution by prioritizing missions based on urgency. Secondly, it computes roundtrip trajectories using SFIPP-ST, a novel 4D single-agent planner (Safe Flight Interval Path Planning with Soft and Temporal Constraints). SFIPP-ST handles heterogeneous UAVs, strictly enforces temporal NFZs, and models inter-agent conflicts as soft constraints. Subsequently, an iterative Large Neighborhood Search, guided by a geometric conflict graph, efficiently resolves any residual conflicts. A completeness-preserving directional pruning technique further accelerates the 3D search. On benchmarks with temporal NFZs, DTAPP-IICR achieves near-100% success with fleets of up to 1,000 UAVs and gains up to 50% runtime reduction from pruning, outperforming batch Enhanced Conflict-Based Search in the UTM context. Scaling successfully in realistic city-scale operations where other priority-based methods fail even at moderate deployments, DTAPP-IICR is positioned as a practical and scalable solution for preflight planning in dense, dynamic urban airspace.

在动态共享空域中对大型无人机 (UAV) 机队进行飞行前规划面临着重大挑战，包括临时禁飞区 (NFZ)、异构车辆配置和严格的交付期限。虽然多代理路径查找（MAPF）提供了正式的框架，但现有方法通常缺乏现实世界的无人交通管理（UTM）所需的可扩展性和灵活性。我们提出 DTAPP-IICR：一种具有增量和迭代冲突解决功能的交付时间感知优先规划方法。我们的框架首先根据紧急程度对任务进行优先级排序，生成初始解决方案。其次，它使用 SFIPP-ST 计算往返轨迹，SFIPP-ST 是一种新颖的 4D 单代理规划器（具有软和时间约束的安全飞行间隔路径规划）。 SFIPP-ST 处理异构无人机，严格执行时间 NFZ，并将代理间冲突建模为软约束。随后，由几何冲突图引导的迭代大邻域搜索有效地解决了任何残余冲突。保持完整性的定向修剪技术进一步加速了 3D 搜索。在时间 NFZ 的基准测试中，DTAPP-IICR 在多达 1,000 架无人机的机队中取得了近 100% 的成功，并通过修剪获得了高达 50% 的运行时间减少，优于 UTM 环境中的批量增强型基于冲突的搜索。 DTAPP-IICR 能够在现实的城市规模运营中成功扩展，而其他基于优先级的方法即使在中等部署情况下也会失败，DTAPP-IICR 被定位为在密集、动态的城市空域中进行飞行前规划的实用且可扩展的解决方案。

</details>

---

## 11. Accelerating Robotic Reinforcement Learning with Agent Guidance / 通过代理指导加速机器人强化学习

**Date**: 2026-02-12 | **arXiv**: [2602.11978v1](http://arxiv.org/abs/2602.11978v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11978v1)

**Categories**: cs.RO, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Reinforcement Learning (RL) offers a powerful paradigm for autonomous robots to master generalist manipulation skills through trial-and-error. However, its real-world application is stifled by severe sample inefficiency. Recent Human-in-the-Loop (HIL) methods accelerate training by using human corrections, yet this approach faces a scalability barrier. Reliance on human supervisors imposes a 1:1 supervision ratio that limits fleet expansion, suffers from operator fatigue over extended sessions, and introduces high variance due to inconsistent human proficiency. We present Agent-guided Policy Search (AGPS), a framework that automates the training pipeline by replacing human supervisors with a multimodal agent. Our key insight is that the agent can be viewed as a semantic world model, injecting intrinsic value priors to structure physical exploration. By using executable tools, the agent provides precise guidance via corrective waypoints and spatial constraints for exploration pruning. We validate our approach on two tasks, ranging from precision insertion to deformable object manipulation. Results demonstrate that AGPS outperforms HIL methods in sample efficiency. This automates the supervision pipeline, unlocking the path to labor-free and scalable robot learning. Project website: https://agps-rl.github.io/agps.

强化学习 (RL) 为自主机器人提供了一个强大的范例，让其通过反复试验掌握通用操作技能。然而，其实际应用却因严重的样本效率低下而受到抑制。最近的人在环（HIL）方法通过使用人工修正来加速训练，但这种方法面临可扩展性障碍。对人类监督员的依赖强制实行 1:1 的监督比例，这限制了车队的扩张，操作员在长时间的工作中会感到疲劳，并且由于人员熟练程度不一致而带来很大的差异。我们提出了代理引导策略搜索（AGPS），这是一个通过用多模式代理取代人类监督员来自动化训练流程的框架。我们的主要见解是，代理可以被视为语义世界模型，在结构物理探索之前注入内在价值。通过使用可执行工具，代理通过修正路径点和空间约束提供精确的指导，以进行探索修剪。我们在两项任务上验证了我们的方法，从精确插入到可变形对象操作。结果表明，AGPS 在样本效率方面优于 HIL 方法。这实现了监督管道的自动化，开启了免劳动力且可扩展的机器人学习之路。项目网站：https://agps-rl.github.io/agps。

</details>

---

## 12. AdaptEvolve: Improving Efficiency of Evolutionary AI Agents through Adaptive Model Selection / AdaptEvolve：通过自适应模型选择提高进化人工智能代理的效率

**Date**: 2026-02-12 | **arXiv**: [2602.11931v1](http://arxiv.org/abs/2602.11931v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11931v1)

**Categories**: cs.CL, cs.AI

**Code**: https://github.com/raypretam/adaptive_llm_selection.

<details><summary><b>Abstract / 摘要</b></summary>

Evolutionary agentic systems intensify the trade-off between computational efficiency and reasoning capability by repeatedly invoking large language models (LLMs) during inference. This setting raises a central question: how can an agent dynamically select an LLM that is sufficiently capable for the current generation step while remaining computationally efficient? While model cascades offer a practical mechanism for balancing this trade-off, existing routing strategies typically rely on static heuristics or external controllers and do not explicitly account for model uncertainty. We introduce AdaptEvolve: Adaptive LLM Selection for Multi-LLM Evolutionary Refinement within an evolutionary sequential refinement framework that leverages intrinsic generation confidence to estimate real-time solvability. Empirical results show that confidence-driven selection yields a favourable Pareto frontier, reducing total inference cost by an average of 37.9% across benchmarks while retaining 97.5% of the upper-bound accuracy of static large-model baselines. Our code is available at https://github.com/raypretam/adaptive_llm_selection.

进化代理系统通过在推理过程中重复调用大型语言模型（LLM）来强化计算效率和推理能力之间的权衡。这种设置提出了一个核心问题：代理如何动态选择足以满足当前生成步骤的LLM，同时保持计算效率？虽然模型级联提供了平衡这种权衡的实用机制，但现有的路由策略通常依赖于静态启发式或外部控制器，并且没有明确考虑模型的不确定性。我们引入了 AdaptEvolve：在进化顺序细化框架内进行多 LLM 进化细化的自适应 LLM 选择，该框架利用内在生成置信度来估计实时可解性。实证结果表明，置信驱动的选择产生了有利的帕累托前沿，将基准的总推理成本平均降低了 37.9%，同时保留了静态大型模型基准的 97.5% 的上限精度。我们的代码可在 https://github.com/raypretam/adaptive_llm_selection 获取。

</details>

---

## 13. MEME: Modeling the Evolutionary Modes of Financial Markets / MEME：金融市场演化模式建模

**Date**: 2026-02-12 | **arXiv**: [2602.11918v1](http://arxiv.org/abs/2602.11918v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11918v1)

**Categories**: cs.AI

**Code**: https://github.com/gta0804/MEME.

<details><summary><b>Abstract / 摘要</b></summary>

LLMs have demonstrated significant potential in quantitative finance by processing vast unstructured data to emulate human-like analytical workflows. However, current LLM-based methods primarily follow either an Asset-Centric paradigm focused on individual stock prediction or a Market-Centric approach for portfolio allocation, often remaining agnostic to the underlying reasoning that drives market movements. In this paper, we propose a Logic-Oriented perspective, modeling the financial market as a dynamic, evolutionary ecosystem of competing investment narratives, termed Modes of Thought. To operationalize this view, we introduce MEME (Modeling the Evolutionary Modes of Financial Markets), designed to reconstruct market dynamics through the lens of evolving logics. MEME employs a multi-agent extraction module to transform noisy data into high-fidelity Investment Arguments and utilizes Gaussian Mixture Modeling to uncover latent consensus within a semantic space. To model semantic drift among different market conditions, we also implement a temporal evaluation and alignment mechanism to track the lifecycle and historical profitability of these modes. By prioritizing enduring market wisdom over transient anomalies, MEME ensures that portfolio construction is guided by robust reasoning. Extensive experiments on three heterogeneous Chinese stock pools from 2023 to 2025 demonstrate that MEME consistently outperforms seven SOTA baselines. Further ablation studies, sensitivity analysis, lifecycle case study and cost analysis validate MEME's capacity to identify and adapt to the evolving consensus of financial markets. Our implementation can be found at https://github.com/gta0804/MEME.

法学硕士通过处理大量非结构化数据来模拟类人的分析工作流程，在定量金融领域展现了巨大的潜力。然而，当前基于法学硕士的方法主要遵循侧重于个股预测的以资产为中心的范式或以市场为中心的投资组合配置方法，通常对驱动市场变动的根本原因保持不可知。在本文中，我们提出了一种面向逻辑的视角，将金融市场建模为一个充满竞争性投资叙述的动态的、进化的生态系统，称为思维模式。为了落实这一观点，我们引入了 MEME（金融市场演化模式建模），旨在通过演化逻辑的视角重建市场动态。 MEME 采用多智能体提取模块将噪声数据转换为高保真投资参数，并利用高斯混合模型来揭示语义空间内的潜在共识。为了对不同市场条件之间的语义漂移进行建模，我们还实施了时间评估和对齐机制来跟踪这些模式的生命周期和历史盈利能力。通过优先考虑持久的市场智慧而不是短暂的异常现象，MEME 确保投资组合构建以稳健的推理为指导。 2023 年至 2025 年对三个异质中国股票池进行的广泛实验表明，MEME 的表现始终优于七个 SOTA 基线。进一步的消融研究、敏感性分析、生命周期案例研究和成本分析验证了 MEME 识别和适应不断变化的金融市场共识的能力。我们的实现可以在 https://github.com/gta0804/MEME 找到。

</details>

---

## 14. Agentic AI for Cybersecurity: A Meta-Cognitive Architecture for Governable Autonomy / 用于网络安全的代理人工智能：可治理自治的元认知架构

**Date**: 2026-02-12 | **arXiv**: [2602.11897v1](http://arxiv.org/abs/2602.11897v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11897v1)

**Categories**: cs.CR, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Contemporary AI-driven cybersecurity systems are predominantly architected as model-centric detection and automation pipelines optimized for task-level performance metrics such as accuracy and response latency. While effective for bounded classification tasks, these architectures struggle to support accountable decision-making under adversarial uncertainty, where actions must be justified, governed, and aligned with organizational and regulatory constraints. This paper argues that cybersecurity orchestration should be reconceptualized as an agentic, multi-agent cognitive system, rather than a linear sequence of detection and response components. We introduce a conceptual architectural framework in which heterogeneous AI agents responsible for detection, hypothesis formation, contextual interpretation, explanation, and governance are coordinated through an explicit meta-cognitive judgement function. This function governs decision readiness and dynamically calibrates system autonomy when evidence is incomplete, conflicting, or operationally risky. By synthesizing distributed cognition theory, multi-agent systems research, and responsible AI governance frameworks, we demonstrate that modern security operations already function as distributed cognitive systems, albeit without an explicit organizing principle. Our contribution is to make this cognitive structure architecturally explicit and governable by embedding meta-cognitive judgement as a first-class system function. We discuss implications for security operations centers, accountable autonomy, and the design of next-generation AI-enabled cyber defence architectures. The proposed framework shifts the focus of AI in cybersecurity from optimizing isolated predictions to governing autonomy under uncertainty.

当代人工智能驱动的网络安全系统主要被构建为以模型为中心的检测和自动化管道，针对任务级性能指标（例如准确性和响应延迟）进行了优化。虽然这些架构对于有界分类任务有效，但很难在对抗性不确定性下支持负责任的决策，其中行动必须合理、受管理并与组织和监管约束保持一致。本文认为，网络安全编排应该被重新概念化为一个代理的多代理认知系统，而不是检测和响应组件的线性序列。我们引入了一个概念架构框架，其中负责检测、假设形成、上下文解释、解释和治理的异构人工智能代理通过明确的元认知判断功能进行协调。当证据不完整、相互冲突或存在操作风险时，该功能负责管理决策准备情况并动态校准系统自主性。通过综合分布式认知理论、多主体系统研究和负责任的人工智能治理框架，我们证明现代安全操作已经作为分布式认知系统发挥作用，尽管没有明确的组织原则。我们的贡献是通过将元认知判断嵌入为一流的系统功能，使这种认知结构在架构上变得明确且可管理。我们讨论对安全运营中心、负责任的自治以及下一代人工智能网络防御架构的设计的影响。拟议的框架将人工智能在网络安全中的重点从优化孤立的预测转移到不确定性下的自治管理。

</details>

---

## 15. Towards Fair and Comprehensive Evaluation of Routers in Collaborative LLM Systems / 在协作法学硕士系统中实现路由器的公平和综合评估

**Date**: 2026-02-12 | **arXiv**: [2602.11877v1](http://arxiv.org/abs/2602.11877v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11877v1)

**Categories**: cs.CL, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Large language models (LLMs) have achieved success, but cost and privacy constraints necessitate deploying smaller models locally while offloading complex queries to cloud-based models. Existing router evaluations are unsystematic, overlooking scenario-specific requirements and out-of-distribution robustness. We propose RouterXBench, a principled evaluation framework with three dimensions: router ability, scenario alignment, and cross-domain robustness. Unlike prior work that relies on output probabilities or external embeddings, we utilize internal hidden states that capture model uncertainty before answer generation. We introduce ProbeDirichlet, a lightweight router that aggregates cross-layer hidden states via learnable Dirichlet distributions with probabilistic training. Trained on multi-domain data, it generalizes robustly across in-domain and out-of-distribution scenarios. Our results show ProbeDirichlet achieves 16.68% and 18.86% relative improvements over the best baselines in router ability and high-accuracy scenarios, with consistent performance across model families, model scales, heterogeneous tasks, and agentic workflows.

大型语言模型 (LLM) 已经取得了成功，但成本和隐私限制使得需要在本地部署较小的模型，同时将复杂的查询卸载到基于云的模型。现有的路由器评估不系统，忽视了特定场景的要求和分布外的稳健性。我们提出了RouterXBench，一个原则性的评估框架，具有三个维度：路由器能力、场景对齐和跨域鲁棒性。与依赖输出概率或外部嵌入的先前工作不同，我们利用内部隐藏状态在答案生成之前捕获模型不确定性。我们介绍 ProbeDirichlet，一种轻量级路由器，通过可学习的狄利克雷分布和概率训练来聚合跨层隐藏状态。它经过多域数据的训练，可以在域内和分布外场景中稳健地推广。我们的结果表明，ProbeDirichlet 在路由器能力和高精度场景方面比最佳基线实现了 16.68% 和 18.86% 的相对改进，并且在模型系列、模型规模、异构任务和代理工作流程中具有一致的性能。

</details>

---

## 16. Intelligent AI Delegation / 智能AI委派

**Date**: 2026-02-12 | **arXiv**: [2602.11865v1](http://arxiv.org/abs/2602.11865v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11865v1)

**Categories**: cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

AI agents are able to tackle increasingly complex tasks. To achieve more ambitious goals, AI agents need to be able to meaningfully decompose problems into manageable sub-components, and safely delegate their completion across to other AI agents and humans alike. Yet, existing task decomposition and delegation methods rely on simple heuristics, and are not able to dynamically adapt to environmental changes and robustly handle unexpected failures. Here we propose an adaptive framework for intelligent AI delegation - a sequence of decisions involving task allocation, that also incorporates transfer of authority, responsibility, accountability, clear specifications regarding roles and boundaries, clarity of intent, and mechanisms for establishing trust between the two (or more) parties. The proposed framework is applicable to both human and AI delegators and delegatees in complex delegation networks, aiming to inform the development of protocols in the emerging agentic web.

人工智能代理能够处理日益复杂的任务。为了实现更雄心勃勃的目标，人工智能代理需要能够有意义地将问题分解为可管理的子组件，并安全地将其完成任务委托给其他人工智能代理和人类。然而，现有的任务分解和委托方法依赖于简单的启发式方法，无法动态适应环境变化并稳健地处理意外故障。在这里，我们提出了一个智能人工智能授权的自适应框架——一系列涉及任务分配的决策，其中还包括权力、责任、问责制的转移、有关角色和边界的明确规范、意图的清晰度以及在两方（或多方）之间建立信任的机制。所提出的框架适用于复杂委托网络中的人类和人工智能委托者和受委托者，旨在为新兴代理网络中的协议开发提供信息。

</details>

---

## 17. Zooming without Zooming: Region-to-Image Distillation for Fine-Grained Multimodal Perception / 无需缩放即可缩放：用于细粒度多模态感知的区域到图像蒸馏

**Date**: 2026-02-12 | **arXiv**: [2602.11858v1](http://arxiv.org/abs/2602.11858v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11858v1)

**Categories**: cs.CV, cs.AI, cs.CL, cs.LG

**Code**: https://github.com/inclusionAI/Zooming-without-Zooming.

<details><summary><b>Abstract / 摘要</b></summary>

Multimodal Large Language Models (MLLMs) excel at broad visual understanding but still struggle with fine-grained perception, where decisive evidence is small and easily overwhelmed by global context. Recent "Thinking-with-Images" methods alleviate this by iteratively zooming in and out regions of interest during inference, but incur high latency due to repeated tool calls and visual re-encoding. To address this, we propose Region-to-Image Distillation, which transforms zooming from an inference-time tool into a training-time primitive, thereby internalizing the benefits of agentic zooming into a single forward pass of an MLLM. In particular, we first zoom in to micro-cropped regions to let strong teacher models generate high-quality VQA data, and then distill this region-grounded supervision back to the full image. After training on such data, the smaller student model improves "single-glance" fine-grained perception without tool use. To rigorously evaluate this capability, we further present ZoomBench, a hybrid-annotated benchmark of 845 VQA data spanning six fine-grained perceptual dimensions, together with a dual-view protocol that quantifies the global--regional "zooming gap". Experiments show that our models achieve leading performance across multiple fine-grained perception benchmarks, and also improve general multimodal cognition on benchmarks such as visual reasoning and GUI agents. We further discuss when "Thinking-with-Images" is necessary versus when its gains can be distilled into a single forward pass. Our code is available at https://github.com/inclusionAI/Zooming-without-Zooming.

多模态大型语言模型 (MLLM) 擅长广泛的视觉理解，但仍难以实现细粒度的感知，其中决定性的证据很小，很容易被全球背景所淹没。最近的“用图像思考”方法通过在推理过程中迭代地放大和缩小感兴趣的区域来缓解这一问题，但由于重复的工具调用和视觉重新编码而导致高延迟。为了解决这个问题，我们提出了区域到图像蒸馏，它将缩放从推理时间工具转换为训练时间原语，从而将代理缩放的好处内化到 MLLM 的单个前向传递中。特别是，我们首先放大微裁剪区域，让强大的教师模型生成高质量的 VQA 数据，然后将这种基于区域的监督提炼回完整图像。在对此类数据进行训练后，较小的学生模型无需使用工具即可提高“一眼”的细粒度感知。为了严格评估这种能力，我们进一步提出了 ZoomBench，这是一个跨越六个细粒度感知维度的 845 个 VQA 数据的混合注释基准，以及量化全球区域“缩放差距”的双视图协议。实验表明，我们的模型在多个细粒度感知基准上实现了领先的性能，并且还提高了视觉推理和 GUI 代理等基准上的一般多模态认知。我们进一步讨论何时需要“用图像思考”以及何时可以将其收益提炼为单个前向传递。我们的代码可在 https://github.com/inclusionAI/Zooming-without-Zooming 获取。

</details>

---

## 18. Beyond End-to-End Video Models: An LLM-Based Multi-Agent System for Educational Video Generation / 超越端到端视频模型：用于教育视频生成的基于 LLM 的多代理系统

**Date**: 2026-02-12 | **arXiv**: [2602.11790v1](http://arxiv.org/abs/2602.11790v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11790v1)

**Categories**: cs.AI, cs.CL

<details><summary><b>Abstract / 摘要</b></summary>

Although recent end-to-end video generation models demonstrate impressive performance in visually oriented content creation, they remain limited in scenarios that require strict logical rigor and precise knowledge representation, such as instructional and educational media. To address this problem, we propose LAVES, a hierarchical LLM-based multi-agent system for generating high-quality instructional videos from educational problems. The LAVES formulates educational video generation as a multi-objective task that simultaneously demands correct step-by-step reasoning, pedagogically coherent narration, semantically faithful visual demonstrations, and precise audio--visual alignment. To address the limitations of prior approaches--including low procedural fidelity, high production cost, and limited controllability--LAVES decomposes the generation workflow into specialized agents coordinated by a central Orchestrating Agent with explicit quality gates and iterative critique mechanisms. Specifically, the Orchestrating Agent supervises a Solution Agent for rigorous problem solving, an Illustration Agent that produces executable visualization codes, and a Narration Agent for learner-oriented instructional scripts. In addition, all outputs from the working agents are subject to semantic critique, rule-based constraints, and tool-based compilation checks. Rather than directly synthesizing pixels, the system constructs a structured executable video script that is deterministically compiled into synchronized visuals and narration using template-driven assembly rules, enabling fully automated end-to-end production without manual editing. In large-scale deployments, LAVES achieves a throughput exceeding one million videos per day, delivering over a 95% reduction in cost compared to current industry-standard approaches while maintaining a high acceptance rate.

尽管最近的端到端视频生成模型在面向视觉的内容创建方面表现出了令人印象深刻的性能，但它们在需要严格逻辑严谨性和精确知识表示的场景中仍然受到限制，例如教学和教育媒体。为了解决这个问题，我们提出了 LAVES，一种基于 LLM 的分层多智能体系统，用于根据教育问题生成高质量的教学视频。 LAVES 将教育视频生成制定为一项多目标任务，同时要求正确的逐步推理、教学上连贯的叙述、语义上忠实的视觉演示以及精确的视听对齐。为了解决先前方法的局限性（包括程序保真度低、生产成本高和可控性有限），LAVES 将生成工作流程分解为由具有明确质量门和迭代批评机制的中央编排代理协调的专门代理。具体来说，编排代理监督解决方案代理以严格解决问题，插图代理生成可执行的可视化代码，以及叙述代理以用于面向学习者的教学脚本。此外，工作代理的所有输出都受到语义批评、基于规则的约束和基于工具的编译检查。该系统不是直接合成像素，而是构建一个结构化的可执行视频脚本，该脚本使用模板驱动的组装规则确定性地编译成同步的视觉效果和旁白，从而实现完全自动化的端到端制作，无需手动编辑。在大规模部署中，LAVES 的吞吐量每天超过 100 万个视频，与当前行业标准方法相比，成本降低了 95% 以上，同时保持了较高的接受率。

</details>

---

## 19. FlowMind: Execute-Summarize for Structured Workflow Generation from LLM Reasoning / FlowMind：从 LLM 推理生成结构化工作流的执行总结

**Date**: 2026-02-12 | **arXiv**: [2602.11782v1](http://arxiv.org/abs/2602.11782v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11782v1)

**Categories**: cs.AI, cs.SE

<details><summary><b>Abstract / 摘要</b></summary>

LLMs can solve complex tasks through reasoning and tool use, but accurately translating these solutions into structured workflows remains challenging. We model workflows as sequences of tool use and reformulate the problem as designing a mechanism that can both solve tasks and reliably construct workflows. Prior approaches that build workflows during execution often suffer from inaccuracies due to interference between the two processes. We propose an Execute-Summarize(ES) framework that decouples task execution from workflow construction: the model first completes the task using available tools, then independently reconstructs a structured workflow from execution traces. This separation improves workflow accuracy and robustness. We introduce FlowBench and show through extensive experiments that our approach outperforms existing methods, providing a reliable paradigm for grounding free-form LLM reasoning into structured workflows.

法学硕士可以通过推理和工具使用来解决复杂的任务，但将这些解决方案准确地转化为结构化工作流程仍然具有挑战性。我们将工作流程建模为工具使用的序列，并将问题重新表述为设计一种既可以解决任务又可以可靠地构建工作流程的机制。由于两个流程之间的干扰，先前在执行期间构建工作流的方法常常会出现不准确的情况。我们提出了一个执行总结（ES）框架，将任务执行与工作流构建分离：模型首先使用可用工具完成任务，然后根据执行跟踪独立重建结构化工作流。这种分离提高了工作流程的准确性和稳健性。我们引入 FlowBench 并通过大量实验证明我们的方法优于现有方法，为将自由形式的 LLM 推理融入结构化工作流程提供了可靠的范例。

</details>

---

## 20. Cooperation Breakdown in LLM Agents Under Communication Delays / 沟通延迟导致LLM代理合作中断

**Date**: 2026-02-12 | **arXiv**: [2602.11754v1](http://arxiv.org/abs/2602.11754v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11754v1)

**Categories**: cs.MA, cs.AI, cs.GT

<details><summary><b>Abstract / 摘要</b></summary>

LLM-based multi-agent systems (LLM-MAS), in which autonomous AI agents cooperate to solve tasks, are gaining increasing attention. For such systems to be deployed in society, agents must be able to establish cooperation and coordination under real-world computational and communication constraints. We propose the FLCOA framework (Five Layers for Cooperation/Coordination among Autonomous Agents) to conceptualize how cooperation and coordination emerge in groups of autonomous agents, and highlight that the influence of lower-layer factors - especially computational and communication resources - has been largely overlooked. To examine the effect of communication delay, we introduce a Continuous Prisoner's Dilemma with Communication Delay and conduct simulations with LLM-based agents. As delay increases, agents begin to exploit slower responses even without explicit instructions. Interestingly, excessive delay reduces cycles of exploitation, yielding a U-shaped relationship between delay magnitude and mutual cooperation. These results suggest that fostering cooperation requires attention not only to high-level institutional design but also to lower-layer factors such as communication delay and resource allocation, pointing to new directions for MAS research.

基于LLM的多智能体系统（LLM-MAS），其中自主人工智能智能体合作解决任务，正受到越来越多的关注。为了在社会中部署此类系统，智能体必须能够在现实世界的计算和通信限制下建立合作和协调。我们提出了 FLCOA 框架（自治代理之间合作/协调的五层）来概念化自治代理群体中如何出现合作和协调，并强调较低层因素（尤其是计算和通信资源）的影响在很大程度上被忽视了。为了检查通信延迟的影响，我们引入了具有通信延迟的连续囚徒困境，并使用基于 LLM 的代理进行模拟。随着延迟的增加，即使没有明确的指令，代理也会开始利用较慢的响应。有趣的是，过度延迟会减少利用周期，从而在延迟幅度和相互合作之间产生 U 形关系。这些结果表明，促进合作不仅需要关注高层的制度设计，还需要关注通信延迟和资源分配等较低层的因素，这为MAS研究指明了新的方向。

</details>

---

## 21. AmbiBench: Benchmarking Mobile GUI Agents Beyond One-Shot Instructions in the Wild / AmbiBench：对移动 GUI 代理进行基准测试，超越野外一次性指令

**Date**: 2026-02-12 | **arXiv**: [2602.11750v1](http://arxiv.org/abs/2602.11750v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11750v1)

**Categories**: cs.SE, cs.AI, cs.HC

<details><summary><b>Abstract / 摘要</b></summary>

Benchmarks are paramount for gauging progress in the domain of Mobile GUI Agents. In practical scenarios, users frequently fail to articulate precise directives containing full task details at the onset, and their expressions are typically ambiguous. Consequently, agents are required to converge on the user's true intent via active clarification and interaction during execution. However, existing benchmarks predominantly operate under the idealized assumption that user-issued instructions are complete and unequivocal. This paradigm focuses exclusively on assessing single-turn execution while overlooking the alignment capability of the agent. To address this limitation, we introduce AmbiBench, the first benchmark incorporating a taxonomy of instruction clarity to shift evaluation from unidirectional instruction following to bidirectional intent alignment. Grounded in Cognitive Gap theory, we propose a taxonomy of four clarity levels: Detailed, Standard, Incomplete, and Ambiguous. We construct a rigorous dataset of 240 ecologically valid tasks across 25 applications, subject to strict review protocols. Furthermore, targeting evaluation in dynamic environments, we develop MUSE (Mobile User Satisfaction Evaluator), an automated framework utilizing an MLLM-as-a-judge multi-agent architecture. MUSE performs fine-grained auditing across three dimensions: Outcome Effectiveness, Execution Quality, and Interaction Quality. Empirical results on AmbiBench reveal the performance boundaries of SoTA agents across different clarity levels, quantify the gains derived from active interaction, and validate the strong correlation between MUSE and human judgment. This work redefines evaluation standards, laying the foundation for next-generation agents capable of truly understanding user intent.

基准对于衡量移动 GUI 代理领域的进展至关重要。在实际场景中，用户经常无法在一开始就表达出包含完整任务细节的精确指令，并且他们的表达通常是模糊的。因此，代理需要在执行过程中通过主动澄清和交互来了解用户的真实意图。然而，现有的基准主要是在理想化假设下运行的，即用户发出的指令是完整且明确的。该范例仅专注于评估单轮执行，而忽略了代理的对齐能力。为了解决这个限制，我们引入了 AmbiBench，这是第一个包含指令清晰度分类法的基准测试，将评估从单向指令跟踪转变为双向意图对齐。基于认知差距理论，我们提出了四个清晰度级别的分类：详细、标准、不完整和模糊。我们构建了一个严格的数据集，其中包含 25 个应用程序中 240 个生态有效的任务，并遵守严格的审查协议。此外，针对动态环境中的评估，我们开发了 MUSE（移动用户满意度评估器），这是一个利用 MLLM 作为法官的多代理架构的自动化框架。 MUSE 从三个维度进行细粒度审核：结果有效性、执行质量和交互质量。 AmbiBench 上的实证结果揭示了 SoTA 代理在不同清晰度级别上的性能边界，量化了主动交互带来的收益，并验证了 MUSE 和人类判断之间的强相关性。这项工作重新定义了评估标准，为下一代能够真正理解用户意图的智能体奠定了基础。

</details>

---

## 22. AIR: Improving Agent Safety through Incident Response / AIR：通过事件响应提高代理安全

**Date**: 2026-02-12 | **arXiv**: [2602.11749v1](http://arxiv.org/abs/2602.11749v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11749v1)

**Categories**: cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Large Language Model (LLM) agents are increasingly deployed in practice across a wide range of autonomous applications. Yet current safety mechanisms for LLM agents focus almost exclusively on preventing failures in advance, providing limited capabilities for responding to, containing, or recovering from incidents after they inevitably arise. In this work, we introduce AIR, the first incident response framework for LLM agent systems. AIR defines a domain-specific language for managing the incident response lifecycle autonomously in LLM agent systems, and integrates it into the agent's execution loop to (1) detect incidents via semantic checks grounded in the current environment state and recent context, (2) guide the agent to execute containment and recovery actions via its tools, and (3) synthesize guardrail rules during eradication to block similar incidents in future executions. We evaluate AIR on three representative agent types. Results show that AIR achieves detection, remediation, and eradication success rates all exceeding 90%. Extensive experiments further confirm the necessity of AIR's key design components, show the timeliness and moderate overhead of AIR, and demonstrate that LLM-generated rules can approach the effectiveness of developer-authored rules across domains. These results show that incident response is both feasible and essential as a first-class mechanism for improving agent safety.

大型语言模型（LLM）代理在实践中越来越多地部署在各种自治应用程序中。然而，目前 LLM 代理的安全机制几乎完全专注于提前预防故障，在事件不可避免地发生后提供有限的响应、遏制或恢复能力。在这项工作中，我们介绍了 AIR，这是 LLM 代理系统的第一个事件响应框架。 AIR 定义了一种特定于领域的语言，用于在 LLM 代理系统中自主管理事件响应生命周期，并将其集成到代理的执行循环中，以 (1) 通过基于当前环境状态和最近上下文的语义检查来检测事件，(2) 指导代理通过其工具执行遏制和恢复操作，以及 (3) 在根除期间综合护栏规则，以阻止未来执行中的类似事件。我们根据三种代表性代理类型评估 AIR。结果显示，AIR 的检测、修复和根除成功率均超过 90%。大量的实验进一步证实了AIR关键设计组件的必要性，展示了AIR的及时性和适度的开销，并证明LLM生成的规则可以接近开发人员编写的跨域规则的有效性。这些结果表明，事件响应作为提高代理安全性的一流机制既可行又重要。

</details>

---

## 23. PhyNiKCE: A Neurosymbolic Agentic Framework for Autonomous Computational Fluid Dynamics / PhyNiKCE：自主计算流体动力学的神经符号代理框架

**Date**: 2026-02-12 | **arXiv**: [2602.11666v1](http://arxiv.org/abs/2602.11666v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11666v1)

**Categories**: cs.AI, cs.CL

<details><summary><b>Abstract / 摘要</b></summary>

The deployment of autonomous agents for Computational Fluid Dynamics (CFD), is critically limited by the probabilistic nature of Large Language Models (LLMs), which struggle to enforce the strict conservation laws and numerical stability required for physics-based simulations. Reliance on purely semantic Retrieval Augmented Generation (RAG) often leads to "context poisoning," where agents generate linguistically plausible but physically invalid configurations due to a fundamental Semantic-Physical Disconnect. To bridge this gap, this work introduces PhyNiKCE (Physical and Numerical Knowledgeable Context Engineering), a neurosymbolic agentic framework for trustworthy engineering. Unlike standard black-box agents, PhyNiKCE decouples neural planning from symbolic validation. It employs a Symbolic Knowledge Engine that treats simulation setup as a Constraint Satisfaction Problem, rigidly enforcing physical constraints via a Deterministic RAG Engine with specialized retrieval strategies for solvers, turbulence models, and boundary conditions. Validated through rigorous OpenFOAM experiments on practical, non-tutorial CFD tasks using Gemini-2.5-Pro/Flash, PhyNiKCE demonstrates a 96% relative improvement over state-of-the-art baselines. Furthermore, by replacing trial-and-error with knowledge-driven initialization, the framework reduced autonomous self-correction loops by 59% while simultaneously lowering LLM token consumption by 17%. These results demonstrate that decoupling neural generation from symbolic constraint enforcement significantly enhances robustness and efficiency. While validated on CFD, this architecture offers a scalable, auditable paradigm for Trustworthy Artificial Intelligence in broader industrial automation.

计算流体动力学 (CFD) 自主代理的部署受到大型语言模型 (LLM) 概率性质的严重限制，该模型很难执行基于物理的模拟所需的严格守恒定律和数值稳定性。对纯语义检索增强生成（RAG）的依赖通常会导致“上下文中毒”，即由于基本的语义-物理脱节，代理生成语言上合理但物理上无效的配置。为了弥补这一差距，这项工作引入了 PhyNiKCE（物理和数值知识背景工程），这是一种用于可信工程的神经符号代理框架。与标准黑盒代理不同，PhyNiKCE 将神经规划与符号验证分离。它采用符号知识引擎，将模拟设置视为约束满足问题，通过确定性 RAG 引擎严格执行物理约束，并为求解器、湍流模型和边界条件提供专门的检索策略。使用 Gemini-2.5-Pro/Flash 对实际的非教程 CFD 任务进行严格的 OpenFOAM 实验验证，PhyNiKCE 与最先进的基线相比，相对提高了 96%。此外，通过用知识驱动的初始化代替试错，该框架将自主自我纠正循环减少了 59%，同时将 LLM 代币消耗降低了 17%。这些结果表明，将神经生成与符号约束执行解耦可以显着增强鲁棒性和效率。经过 CFD 验证后，该架构为更广泛的工业自动化中的可信人工智能提供了可扩展、可审计的范例。

</details>

---

## 24. ABot-N0: Technical Report on the VLA Foundation Model for Versatile Embodied Navigation / ABot-N0：多功能嵌入式导航 VLA 基础模型的技术报告

**Date**: 2026-02-12 | **arXiv**: [2602.11598v1](http://arxiv.org/abs/2602.11598v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11598v1)

**Categories**: cs.RO, cs.AI, cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Embodied navigation has long been fragmented by task-specific architectures. We introduce ABot-N0, a unified Vision-Language-Action (VLA) foundation model that achieves a ``Grand Unification'' across 5 core tasks: Point-Goal, Object-Goal, Instruction-Following, POI-Goal, and Person-Following. ABot-N0 utilizes a hierarchical ``Brain-Action'' architecture, pairing an LLM-based Cognitive Brain for semantic reasoning with a Flow Matching-based Action Expert for precise, continuous trajectory generation.   To support large-scale learning, we developed the ABot-N0 Data Engine, curating 16.9M expert trajectories and 5.0M reasoning samples across 7,802 high-fidelity 3D scenes (10.7 $\text{km}^2$). ABot-N0 achieves new SOTA performance across 7 benchmarks, significantly outperforming specialized models. Furthermore, our Agentic Navigation System integrates a planner with hierarchical topological memory, enabling robust, long-horizon missions in dynamic real-world environments.

长期以来，具体化导航一直被特定于任务的架构所分割。我们推出了 ABot-N0，这是一个统一的视觉-语言-动作 (VLA) 基础模型，它实现了 5 个核心任务的“大统一”：点目标、对象目标、指令跟踪、POI 目标和人员跟踪。 ABot-N0 采用分层“大脑动作”架构，将用于语义推理的基于 LLM 的认知大脑与用于精确、连续轨迹生成的基于流匹配的动作专家配对。   为了支持大规模学习，我们开发了 ABot-N0 数据引擎，在 7,802 个高保真 3D 场景 (10.7 $\text{km}^2$) 中整理了 1690 万条专家轨迹和 500 万个推理样本。 ABot-N0 在 7 个基准测试中实现了新的 SOTA 性能，显着优于专用模型。此外，我们的代理导航系统集成了具有分层拓扑内存的规划器，可在动态的现实环境中实现稳健的长视野任务。

</details>

---

## 25. The Five Ws of Multi-Agent Communication: Who Talks to Whom, When, What, and Why -- A Survey from MARL to Emergent Language and LLMs / 多智能体通信的五个 W：谁与谁交谈、何时交谈、交谈内容和原因——从 MARL 到新兴语言和法学硕士的调查

**Date**: 2026-02-12 | **arXiv**: [2602.11583v1](http://arxiv.org/abs/2602.11583v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11583v1)

**Categories**: cs.AI, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Multi-agent sequential decision-making powers many real-world systems, from autonomous vehicles and robotics to collaborative AI assistants. In dynamic, partially observable environments, communication is often what reduces uncertainty and makes collaboration possible. This survey reviews multi-agent communication (MA-Comm) through the Five Ws: who communicates with whom, what is communicated, when communication occurs, and why communication is beneficial. This framing offers a clean way to connect ideas across otherwise separate research threads. We trace how communication approaches have evolved across three major paradigms. In Multi-Agent Reinforcement Learning (MARL), early methods used hand-designed or implicit protocols, followed by end-to-end learned communication optimized for reward and control. While successful, these protocols are frequently task-specific and hard to interpret, motivating work on Emergent Language (EL), where agents can develop more structured or symbolic communication through interaction. EL methods, however, still struggle with grounding, generalization, and scalability, which has fueled recent interest in large language models (LLMs) that bring natural language priors for reasoning, planning, and collaboration in more open-ended settings. Across MARL, EL, and LLM-based systems, we highlight how different choices shape communication design, where the main trade-offs lie, and what remains unsolved. We distill practical design patterns and open challenges to support future hybrid systems that combine learning, language, and control for scalable and interpretable multi-agent collaboration.

多智能体顺序决策为许多现实世界的系统提供动力，从自动驾驶车辆和机器人到协作人工智能助手。在动态的、部分可观察的环境中，沟通通常可以减少不确定性并使协作成为可能。这项调查通过五个 W 来回顾多主体通信 (MA-Comm)：谁与谁通信、通信什么、何时发生通信以及为什么通信是有益的。这个框架提供了一种干净的方式来连接不同研究线索之间的想法。我们追踪沟通方法如何在三个主要范式中演变。在多智能体强化学习（MARL）中，早期方法使用手工设计或隐式协议，然后是针对奖励和控制进行优化的端到端学习通信。虽然成功，但这些协议通常是特定于任务的并且难以解释，从而激发了新兴语言（EL）的工作，其中代理可以通过交互开发更结构化或符号化的通信。然而，EL 方法仍然在基础性、泛化性和可扩展性方面遇到困难，这激发了人们最近对大型语言模型 (LLM) 的兴趣，这些模型为在更开放的环境中进行推理、规划和协作带来了自然语言先验。在基于 MARL、EL 和 LLM 的系统中，我们重点介绍了不同的选择如何塑造沟通设计、主要的权衡所在以及尚未解决的问题。我们提炼实用的设计模式和开放的挑战，以支持未来的混合系统，将学习、语言和控制结合起来，实现可扩展和可解释的多代理协作。

</details>

---

## 26. Learning to Configure Agentic AI Systems / 学习配置代理人工智能系统

**Date**: 2026-02-12 | **arXiv**: [2602.11574v1](http://arxiv.org/abs/2602.11574v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11574v1)

**Categories**: cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Configuring LLM-based agent systems involves choosing workflows, tools, token budgets, and prompts from a large combinatorial design space, and is typically handled today by fixed large templates or hand-tuned heuristics. This leads to brittle behavior and unnecessary compute, since the same cumbersome configuration is often applied to both easy and hard input queries. We formulate agent configuration as a query-wise decision problem and introduce ARC (Agentic Resource & Configuration learner), which learns a light-weight hierarchical policy using reinforcement learning to dynamically tailor these configurations. Across multiple benchmarks spanning reasoning and tool-augmented question answering, the learned policy consistently outperforms strong hand-designed and other baselines, achieving up to 25% higher task accuracy while also reducing token and runtime costs. These results demonstrate that learning per-query agent configurations is a powerful alternative to "one size fits all" designs.

配置基于 LLM 的代理系统涉及从大型组合设计空间中选择工作流程、工具、代币预算和提示，目前通常通过固定的大型模板或手动调整的启发式方法来处理。这会导致脆弱的行为和不必要的计算，因为相同的繁琐配置通常应用于简单输入查询和硬输入查询。我们将代理配置制定为明智的查询决策问题，并引入 ARC（代理资源和配置学习器），它使用强化学习来学习轻量级分层策略，以动态定制这些配置。在涵盖推理和工具增强问答的多个基准中，学习策略始终优于强大的手工设计基准和其他基准，将任务准确性提高了 25%，同时还降低了令牌和运行时成本。这些结果表明，学习每个查询代理配置是“一刀切”设计的强大替代方案。

</details>

---

## 27. Budget-Constrained Agentic Large Language Models: Intention-Based Planning for Costly Tool Use / 预算受限的代理大型语言模型：基于意图的昂贵工具使用规划

**Date**: 2026-02-12 | **arXiv**: [2602.11541v1](http://arxiv.org/abs/2602.11541v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11541v1)

**Categories**: cs.AI, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

We study budget-constrained tool-augmented agents, where a large language model must solve multi-step tasks by invoking external tools under a strict monetary budget. We formalize this setting as sequential decision making in context space with priced and stochastic tool executions, making direct planning intractable due to massive state-action spaces, high variance of outcomes and prohibitive exploration cost. To address these challenges, we propose INTENT, an inference-time planning framework that leverages an intention-aware hierarchical world model to anticipate future tool usage, risk-calibrated cost, and guide decisions online. Across cost-augmented StableToolBench, INTENT strictly enforces hard budget feasibility while substantially improving task success over baselines, and remains robust under dynamic market shifts such as tool price changes and varying budgets.

我们研究预算受限的工具增强代理，其中大型语言模型必须在严格的货币预算下通过调用外部工具来解决多步骤任务。我们将这种设置形式化为上下文空间中具有定价和随机工具执行的顺序决策，由于巨大的状态动作空间、结果的高方差和令人望而却步的探索成本，使得直接规划变得棘手。为了应对这些挑战，我们提出了 INTENT，这是一种推理时间规划框架，它利用意图感知的分层世界模型来预测未来的工具使用、风险校准成本并指导在线决策。在成本增加的 StableToolBench 中，INTENT 严格执行硬预算可行性，同时大幅提高任务成功率（较基准），并在工具价格变化和预算变化等动态市场变化下保持稳健。

</details>

---

## 28. CausalAgent: A Conversational Multi-Agent System for End-to-End Causal Inference / CausalAgent：用于端到端因果推理的会话式多代理系统

**Date**: 2026-02-12 | **arXiv**: [2602.11527v1](http://arxiv.org/abs/2602.11527v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11527v1)

**Categories**: cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Causal inference holds immense value in fields such as healthcare, economics, and social sciences. However, traditional causal analysis workflows impose significant technical barriers, requiring researchers to possess dual backgrounds in statistics and computer science, while manually selecting algorithms, handling data quality issues, and interpreting complex results. To address these challenges, we propose CausalAgent, a conversational multi-agent system for end-to-end causal inference. The system innovatively integrates Multi-Agent Systems (MAS), Retrieval-Augmented Generation (RAG), and the Model Context Protocol (MCP) to achieve automation from data cleaning and causal structure learning to bias correction and report generation through natural language interaction. Users need only upload a dataset and pose questions in natural language to receive a rigorous, interactive analysis report. As a novel user-centered human-AI collaboration paradigm, CausalAgent explicitly models the analysis workflow. By leveraging interactive visualizations, it significantly lowers the barrier to entry for causal analysis while ensuring the rigor and interpretability of the process.

因果推理在医疗保健、经济学和社会科学等领域具有巨大的价值。然而，传统的因果分析工作流程存在很大的技术障碍，要求研究人员拥有统计学和计算机科学的双重背景，同时手动选择算法、处理数据质量问题和解释复杂的结果。为了应对这些挑战，我们提出了 CausalAgent，一种用于端到端因果推理的会话式多智能体系统。该系统创新性地集成了多智能体系统（MAS）、检索增强生成（RAG）和模型上下文协议（MCP），通过自然语言交互实现从数据清理和因果结构学习到偏差校正和报告生成的自动化。用户只需上传数据集并用自然语言提出问题，即可收到严谨的交互式分析报告。作为一种新颖的以用户为中心的人类与人工智能协作范例，CausalAgent 显式地建模了分析工作流程。通过利用交互式可视化，它显着降低了因果分析的进入门槛，同时确保了过程的严谨性和可解释性。

</details>

---

## 29. AgentLeak: A Full-Stack Benchmark for Privacy Leakage in Multi-Agent LLM Systems / AgentLeak：多代理 LLM 系统中隐私泄露的全栈基准

**Date**: 2026-02-12 | **arXiv**: [2602.11510v1](http://arxiv.org/abs/2602.11510v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11510v1)

**Categories**: cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Multi-agent Large Language Model (LLM) systems create privacy risks that current benchmarks cannot measure. When agents coordinate on tasks, sensitive data passes through inter-agent messages, shared memory, and tool arguments; pathways that output-only audits never inspect. We introduce AgentLeak, to the best of our knowledge the first full-stack benchmark for privacy leakage covering internal channels, spanning 1,000 scenarios across healthcare, finance, legal, and corporate domains, paired with a 32-class attack taxonomy and three-tier detection pipeline. Testing GPT-4o, GPT-4o-mini, Claude 3.5 Sonnet, Mistral Large, and Llama 3.3 70B across 4,979 traces reveals that multi-agent configurations reduce per-channel output leakage (C1: 27.2% vs 43.2% in single-agent) but introduce unmonitored internal channels that raise total system exposure to 68.9% (OR-aggregated across C1, C2, C5). Internal channels account for most of this gap: inter-agent messages (C2) leak at 68.8%, compared to 27.2% on C1 (output channel). This means that output-only audits miss 41.7% of violations. Claude 3.5 Sonnet, which emphasizes safety alignment in its design, achieves the lowest leakage rates on both external (3.3%) and internal (28.1%) channels, suggesting that model-level safety training may transfer to internal channel protection. Across all five models and four domains, the pattern C2 > C1 holds consistently, confirming that inter-agent communication is the primary vulnerability. These findings underscore the need for coordination frameworks that incorporate internal-channel privacy protections and enforce privacy controls on inter-agent communication.

多代理大型语言模型 (LLM) 系统会产生当前基准无法衡量的隐私风险。当代理协调任务时，敏感数据会通过代理间消息、共享内存和工具参数传递；仅输出审计从不检查的路径。我们推出 AgentLeak，据我们所知，这是第一个涵盖内部渠道的隐私泄露全栈基准，涵盖医疗保健、金融、法律和企业领域的 1,000 个场景，并配有 32 类攻击分类法和三层检测管道。在 4,979 个跟踪中测试 GPT-4o、GPT-4o-mini、Claude 3.5 Sonnet、Mistral Large 和 Llama 3.3 70B 表明，多代理配置减少了每个通道的输出泄漏（C1：27.2% vs 单代理的 43.2%），但引入了不受监控的内部通道，将系统总体暴露率提高到 68.9%（C1 上的 OR 聚合） C2、C5)。内部渠道占了这一差距的大部分：代理间消息 (C2) 泄漏率为 68.8%，而 C1（输出通道）泄漏率为 27.2%。这意味着仅输出审计会漏掉 41.7% 的违规行为。 Claude 3.5 Sonnet在设计中强调安全一致性，在外部（3.3％）和内部（28.1％）通道上实现了最低泄漏率，这表明模型级安全训练可能会转移到内部通道保护。在所有五个模型和四个域中，模式 C2 > C1 一致，证实代理间通信是主要漏洞。这些发现强调了协调框架的必要性，该框架纳入内部渠道隐私保护并对代理间通信实施隐私控制。

</details>

---

## 30. SIGHT: Reinforcement Learning with Self-Evidence and Information-Gain Diverse Branching for Search Agent / SIGHT：具有不证自明和信息的强化学习 - 为搜索代理获得多样化分支

**Date**: 2026-02-12 | **arXiv**: [2602.11551v1](http://arxiv.org/abs/2602.11551v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11551v1)

**Categories**: cs.CL

<details><summary><b>Abstract / 摘要</b></summary>

Reinforcement Learning (RL) has empowered Large Language Models (LLMs) to master autonomous search for complex question answering. However, particularly within multi-turn search scenarios, this interaction introduces a critical challenge: search results often suffer from high redundancy and low signal-to-noise ratios. Consequently, agents easily fall into "Tunnel Vision," where the forced interpretation of early noisy retrievals leads to irreversible error accumulation. To address these challenges, we propose SIGHT, a framework that enhances search-based reasoning through Self-Evidence Support (SES) and Information-Gain Driven Diverse Branching. SIGHT distills search results into high-fidelity evidence via SES and calculates an Information Gain score to pinpoint pivotal states where observations maximally reduce uncertainty. This score guides Dynamic Prompting Interventions - including de-duplication, reflection, or adaptive branching - to spawn new branches with SES. Finally, by integrating SES and correctness rewards via Group Relative Policy Optimization, SIGHT internalizes robust exploration strategies without external verifiers. Experiments on single-hop and multi-hop QA benchmarks demonstrate that SIGHT significantly outperforms existing approaches, particularly in complex reasoning scenarios, using fewer search steps.

强化学习 (RL) 使大型语言模型 (LLM) 能够掌握复杂问题回答的自主搜索。然而，特别是在多轮搜索场景中，这种交互带来了一个严峻的挑战：搜索结果往往存在高冗余和低信噪比。因此，智能体很容易陷入“隧道视野”，即对早期噪声检索的强制解释导致不可逆的错误累积。为了应对这些挑战，我们提出了 SIGHT，这是一个通过自明支持 (SES) 和信息增益驱动的多样化分支来增强基于搜索的推理的框架。 SIGHT 通过 SES 将搜索结果提炼成高保真证据，并计算信息增益分数，以查明观察结果最大限度地减少不确定性的关键状态。该分数指导动态提示干预（包括重复数据删除、反射或自适应分支）以使用 SES 生成新分支。最后，通过组相对策略优化整合 SES 和正确性奖励，SIGHT 内部化了强大的探索策略，无需外部验证者。单跳和多跳 QA 基准测试表明，SIGHT 显着优于现有方法，特别是在复杂的推理场景中，使用更少的搜索步骤。

</details>

---

## 31. Convex Markov Games and Beyond: New Proof of Existence, Characterization and Learning Algorithms for Nash Equilibria / 凸马尔可夫博弈及其他：纳什均衡的新存在证明、表征和学习算法

**Date**: 2026-02-12 | **arXiv**: [2602.12181v1](http://arxiv.org/abs/2602.12181v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.12181v1)

**Categories**: cs.GT, cs.LG, cs.MA

<details><summary><b>Abstract / 摘要</b></summary>

Convex Markov Games (cMGs) were recently introduced as a broad class of multi-agent learning problems that generalize Markov games to settings where strategic agents optimize general utilities beyond additive rewards. While cMGs expand the modeling frontier, their theoretical foundations, particularly the structure of Nash equilibria (NE) and guarantees for learning algorithms, are not yet well understood. In this work, we address these gaps for an extension of cMGs, which we term General Utility Markov Games (GUMGs), capturing new applications requiring coupling between agents' occupancy measures. We prove that in GUMGs, Nash equilibria coincide with the fixed points of projected pseudo-gradient dynamics (i.e., first-order stationary points), enabled by a novel agent-wise gradient domination property. This insight also yields a simple proof of NE existence using Brouwer's fixed-point theorem. We further show the existence of Markov perfect equilibria. Building on this characterization, we establish a policy gradient theorem for GUMGs and design a model-free policy gradient algorithm. For potential GUMGs, we establish iteration complexity guarantees for computing approximate-NE under exact gradients and provide sample complexity bounds in both the generative model and on-policy settings. Our results extend beyond prior work restricted to zero-sum cMGs, providing the first theoretical analysis of common-interest cMGs.

凸马尔可夫博弈（cMG）最近被引入作为一类广泛的多智能体学习问题，它将马尔可夫博弈推广到战略智能体优化一般效用而不是附加奖励的环境。虽然 cMG 扩展了建模前沿，但其理论基础，特别是纳什均衡 (NE) 的结构和学习算法的保证，尚未得到很好的理解。在这项工作中，我们解决了 cMG 扩展的这些差距，我们将其称为通用效用马尔可夫游戏（GUMG），捕获需要在代理占用度量之间进行耦合的新应用程序。我们证明，在 GUMG 中，纳什均衡与投影伪梯度动力学的不动点（即一阶驻点）一致，这是由新颖的智能体梯度支配属性实现的。这种见解还使用布劳威尔不动点定理产生了 NE 存在的简单证明。我们进一步证明了马尔可夫完美均衡的存在。在此表征的基础上，我们建立了 GUMG 的策略梯度定理并设计了无模型策略梯度算法。对于潜在的 GUMG，我们建立了迭代复杂性保证，用于在精确梯度下计算近似 NE，并在生成模型和在策略设置中提供样本复杂性界限。我们的结果超出了之前仅限于零和 cMG 的工作，提供了对共同利益 cMG 的第一个理论分析。

</details>

---

## 32. PrefillShare: A Shared Prefill Module for KV Reuse in Multi-LLM Disaggregated Serving / PrefillShare：用于多 LLM 分解服务中 KV 重用的共享预填充模块

**Date**: 2026-02-12 | **arXiv**: [2602.12029v1](http://arxiv.org/abs/2602.12029v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.12029v1)

**Categories**: cs.LG, cs.DC

<details><summary><b>Abstract / 摘要</b></summary>

Multi-agent systems increasingly orchestrate multiple specialized language models to solve complex real-world problems, often invoking them over a shared context. This execution pattern repeatedly processes the same prompt prefix across models. Consequently, each model redundantly executes the prefill stage and maintains its own key-value (KV) cache, increasing aggregate prefill load and worsening tail latency by intensifying prefill-decode interference in existing LLM serving stacks. Disaggregated serving reduces such interference by placing prefill and decode on separate GPUs, but disaggregation does not fundamentally eliminate inter-model redundancy in computation and KV storage for the same prompt. To address this issue, we propose PrefillShare, a novel algorithm that enables sharing the prefill stage across multiple models in a disaggregated setting. PrefillShare factorizes the model into prefill and decode modules, freezes the prefill module, and fine-tunes only the decode module. This design allows multiple task-specific models to share a prefill module and the KV cache generated for the same prompt. We further introduce a routing mechanism that enables effective prefill sharing across heterogeneous models in a vLLM-based disaggregated system. PrefillShare not only matches full fine-tuning accuracy on a broad range of tasks and models, but also delivers 4.5x lower p95 latency and 3.9x higher throughput in multi-model agent workloads.

多代理系统越来越多地协调多个专门的语言模型来解决复杂的现实问题，通常在共享上下文中调用它们。此执行模式跨模型重复处理相同的提示前缀。因此，每个模型都会冗余地执行预填充阶段并维护自己的键值 (KV) 缓存，从而通过加剧现有 LLM 服务堆栈中的预填充解码干扰来增加总预填充负载并恶化尾部延迟。分解服务通过将预填充和解码放在单独的 GPU 上来减少此类干扰，但分解并不能从根本上消除同一提示的计算和 KV 存储中的模型间冗余。为了解决这个问题，我们提出了 PrefillShare，这是一种新颖的算法，可以在分类设置中跨多个模型共享预填充阶段。 PrefillShare 将模型分解为预填充和解码模块，冻结预填充模块，仅微调解码模块。这种设计允许多个特定于任务的模型共享预填充模块以及为同一提示生成的 KV 缓存。我们进一步引入了一种路由机制，可以在基于 vLLM 的分解系统中跨异构模型实现有效的预填充共享。 PrefillShare 不仅可以满足各种任务和模型的全面微调精度，而且还可以在多模型代理工作负载中提供 4.5 倍的 p95 延迟降低和 3.9 倍的吞吐量提高。

</details>

---

## 33. Towards Sustainable Investment Policies Informed by Opponent Shaping / 制定以对手塑造为指导的可持续投资政策

**Date**: 2026-02-12 | **arXiv**: [2602.11829v1](http://arxiv.org/abs/2602.11829v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11829v1)

**Categories**: cs.LG, cs.GT

<details><summary><b>Abstract / 摘要</b></summary>

Addressing climate change requires global coordination, yet rational economic actors often prioritize immediate gains over collective welfare, resulting in social dilemmas. InvestESG is a recently proposed multi-agent simulation that captures the dynamic interplay between investors and companies under climate risk. We provide a formal characterization of the conditions under which InvestESG exhibits an intertemporal social dilemma, deriving theoretical thresholds at which individual incentives diverge from collective welfare. Building on this, we apply Advantage Alignment, a scalable opponent shaping algorithm shown to be effective in general-sum games, to influence agent learning in InvestESG. We offer theoretical insights into why Advantage Alignment systematically favors socially beneficial equilibria by biasing learning dynamics toward cooperative outcomes. Our results demonstrate that strategically shaping the learning processes of economic agents can result in better outcomes that could inform policy mechanisms to better align market incentives with long-term sustainability goals.

应对气候变化需要全球协调，但理性的经济行为体往往将眼前利益置于集体福利之上，从而导致社会困境。 InvestESG 是最近提出的多主体模拟，可捕捉气候风险下投资者和公司之间的动态相互作用。我们对 InvestESG 表现出跨期社会困境的条件进行了正式描述，得出了个人激励与集体福利背离的理论阈值。在此基础上，我们应用优势对齐（Advantage Alignment）（一种可扩展的对手塑造算法，在一般和博弈中被证明是有效的）来影响 InvestESG 中的代理学习。我们提供理论见解，解释为什么优势对齐通过将学习动态偏向合作结果来系统地支持社会有益的平衡。我们的结果表明，战略性地塑造经济主体的学习过程可以带来更好的结果，从而为政策机制提供信息，以更好地将市场激励与长期可持续发展目标结合起来。

</details>

---

## 34. Deep Kernel Fusion for Transformers / 变形金刚的深度内核融合

**Date**: 2026-02-12 | **arXiv**: [2602.11808v1](http://arxiv.org/abs/2602.11808v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11808v1)

**Categories**: cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Agentic LLM inference with long contexts is increasingly limited by memory bandwidth rather than compute. In this setting, SwiGLU MLP blocks, whose large weights exceed cache capacity, become a major yet under-optimized bottleneck. We propose DeepFusionKernel, a deeply fused kernel that cuts HBM traffic and boosts cache reuse, delivering up to 13.2% speedup on H100 and 9.7% on A100 over SGLang. Integrated with SGLang and paired with a kernel scheduler, DeepFusionKernel ensures consistent accelerations over generation lengths, while remaining adaptable to diverse models, inference configurations, and hardware platforms.

具有长上下文的代理 LLM 推理越来越受到内存带宽而不是计算的限制。在这种情况下，SwiGLU MLP 块的权重超过了缓存容量，成为主要但未优化的瓶颈。我们提出了 DeepFusionKernel，这是一种深度融合的内核，可减少 HBM 流量并提高缓存重用，与 SGLang 相比，在 H100 上可实现高达 13.2% 的加速，在 A100 上可实现 9.7% 的加速。 DeepFusionKernel 与 SGLang 集成并与内核调度程序配合使用，可确保在一代长度内实现一致的加速，同时保持对不同模型、推理配置和硬件平台的适应性。

</details>

---

## 35. A Generic Framework for Fair Consensus Clustering in Streams / 流中公平共识聚类的通用框架

**Date**: 2026-02-12 | **arXiv**: [2602.11500v1](http://arxiv.org/abs/2602.11500v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11500v1)

**Categories**: cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Consensus clustering seeks to combine multiple clusterings of the same dataset, potentially derived by considering various non-sensitive attributes by different agents in a multi-agent environment, into a single partitioning that best reflects the overall structure of the underlying dataset. Recent work by Chakraborty et al, introduced a fair variant under proportionate fairness and obtained a constant-factor approximation by naively selecting the best closest fair input clustering; however, their offline approach requires storing all input clusterings, which is prohibitively expensive for most large-scale applications.   In this paper, we initiate the study of fair consensus clustering in the streaming model, where input clusterings arrive sequentially and memory is limited. We design the first constant-factor algorithm that processes the stream while storing only a logarithmic number of inputs. En route, we introduce a new generic algorithmic framework that integrates closest fair clustering with cluster fitting, yielding improved approximation guarantees not only in the streaming setting but also when revisited offline. Furthermore, the framework is fairness-agnostic: it applies to any fairness definition for which an approximately close fair clustering can be computed efficiently. Finally, we extend our methods to the more general k-median consensus clustering problem.

共识聚类旨在将同一数据集的多个聚类（可能是通过考虑多智能体环境中不同智能体的各种非敏感属性而得出的）组合成最能反映底层数据集整体结构的单个分区。 Chakraborty 等人最近的工作引入了比例公平下的公平变体，并通过天真地选择最佳最接近的公平输入聚类来获得常数因子近似值；然而，他们的离线方法需要存储所有输入聚类，这对于大多数大型应用程序来说过于昂贵。   在本文中，我们发起了流模型中公平共识聚类的研究，其中输入聚类按顺序到达且内存有限。我们设计了第一个常数因子算法，该算法在处理流的同时仅存储对数数量的输入。在此过程中，我们引入了一种新的通用算法框架，该框架将最接近公平聚类与聚类拟合相结合，不仅在流媒体设置中而且在离线重新访问时都产生改进的近似保证。此外，该框架与公平性无关：它适用于可以有效计算近似接近公平聚类的任何公平性定义。最后，我们将我们的方法扩展到更一般的 k 中值一致性聚类问题。

</details>

---

## 36. RF-Modulated Adaptive Communication Improves Multi-Agent Robotic Exploration / 射频调制自适应通信改善多智能体机器人探索

**Date**: 2026-02-12 | **arXiv**: [2602.12074v1](http://arxiv.org/abs/2602.12074v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.12074v1)

**Categories**: cs.RO

<details><summary><b>Abstract / 摘要</b></summary>

Reliable coordination and efficient communication are critical challenges for multi-agent robotic exploration of environments where communication is limited. This work introduces Adaptive-RF Transmission (ART), a novel communication-aware planning algorithm that dynamically modulates transmission location based on signal strength and data payload size, enabling heterogeneous robot teams to share information efficiently without unnecessary backtracking. We further explore an extension to this approach called ART-SST, which enforces signal strength thresholds for high-fidelity data delivery. Through over 480 simulations across three cave-inspired environments, ART consistently outperforms existing strategies, including full rendezvous and minimum-signal heuristic approaches, achieving up to a 58% reduction in distance traveled and up to 52% faster exploration times compared to baseline methods. These results demonstrate that adaptive, payload-aware communication significantly improves coverage efficiency and mission speed in complex, communication-constrained environments, offering a promising foundation for future planetary exploration and search-and-rescue missions.

可靠的协调和高效的通信是多智能体机器人在通信受限的环境中探索的关键挑战。这项工作引入了自适应射频传输（ART），这是一种新颖的通信感知规划算法，可根据信号强度和数据有效负载大小动态调制传输位置，使异构机器人团队能够有效地共享信息，而无需不必要的回溯。我们进一步探索了这种方法的扩展，称为 ART-SST，它强制执行高保真数据传输的信号强度阈值。通过在三个洞穴环境中进行超过 480 次模拟，ART 始终优于现有策略，包括完全交会和最小信号启发式方法，与基线方法相比，行驶距离减少了 58%，探索时间缩短了 52%。这些结果表明，自适应有效载荷感知通信可显着提高复杂、通信受限环境中的覆盖效率和任务速度，为未来的行星探索和搜索救援任务奠定良好的基础。

</details>

---

## 37. Adaptive-Horizon Conflict-Based Search for Closed-Loop Multi-Agent Path Finding / 用于闭环多智能体路径查找的自适应地平线冲突搜索

**Date**: 2026-02-12 | **arXiv**: [2602.12024v1](http://arxiv.org/abs/2602.12024v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.12024v1)

**Categories**: cs.RO

<details><summary><b>Abstract / 摘要</b></summary>

MAPF is a core coordination problem for large robot fleets in automated warehouses and logistics. Existing approaches are typically either open-loop planners, which generate fixed trajectories and struggle to handle disturbances, or closed-loop heuristics without reliable performance guarantees, limiting their use in safety-critical deployments. This paper presents ACCBS, a closed-loop algorithm built on a finite-horizon variant of CBS with a horizon-changing mechanism inspired by iterative deepening in MPC. ACCBS dynamically adjusts the planning horizon based on the available computational budget, and reuses a single constraint tree to enable seamless transitions between horizons. As a result, it produces high-quality feasible solutions quickly while being asymptotically optimal as the budget increases, exhibiting anytime behavior. Extensive case studies demonstrate that ACCBS combines flexibility to disturbances with strong performance guarantees, effectively bridging the gap between theoretical optimality and practical robustness for large-scale robot deployment.

MAPF 是自动化仓库和物流中大型机器人车队的核心协调问题。现有的方法通常是开环规划器，它生成固定轨迹并难以处理干扰，或者是闭环启发式，没有可靠的性能保证，限制了它们在安全关键型部署中的使用。本文提出了 ACCBS，这是一种基于 CBS 的有限水平变体构建的闭环算法，其水平变化机制受到 MPC 迭代深化的启发。 ACCBS 根据可用的计算预算动态调整规划范围，并重用单个约束树以实现范围之间的无缝转换。因此，它可以快速产生高质量的可行解决方案，同时随着预算的增加而渐近最优，表现出随时行为。大量案例研究表明，ACCBS 将抗干扰灵活性与强大的性能保证相结合，有效缩小了大规模机器人部署的理论最优性和实际鲁棒性之间的差距。

</details>

---

## 38. AC-MASAC: An Attentive Curriculum Learning Framework for Heterogeneous UAV Swarm Coordination / AC-MASAC：异构无人机群协调的细心课程学习框架

**Date**: 2026-02-12 | **arXiv**: [2602.11735v1](http://arxiv.org/abs/2602.11735v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11735v1)

**Categories**: cs.RO

**Code**: https://github.com/Wanhao-Liu/AC-MASAC

<details><summary><b>Abstract / 摘要</b></summary>

Cooperative path planning for heterogeneous UAV swarms poses significant challenges for Multi-Agent Reinforcement Learning (MARL), particularly in handling asymmetric inter-agent dependencies and addressing the risks of sparse rewards and catastrophic forgetting during training. To address these issues, this paper proposes an attentive curriculum learning framework (AC-MASAC). The framework introduces a role-aware heterogeneous attention mechanism to explicitly model asymmetric dependencies. Moreover, a structured curriculum strategy is designed, integrating hierarchical knowledge transfer and stage-proportional experience replay to address the issues of sparse rewards and catastrophic forgetting. The proposed framework is validated on a custom multi-agent simulation platform, and the results show that our method has significant advantages over other advanced methods in terms of Success Rate, Formation Keeping Rate, and Success-weighted Mission Time. The code is available at \textcolor{red}{https://github.com/Wanhao-Liu/AC-MASAC}.

异构无人机群的协作路径规划对多智能体强化学习（MARL）提出了重大挑战，特别是在处理不对称智能体间依赖关系以及解决训练期间稀疏奖励和灾难性遗忘的风险方面。为了解决这些问题，本文提出了一个专注的课程学习框架（AC-MASAC）。该框架引入了角色感知异构注意力机制来显式建模不对称依赖关系。此外，还设计了结构化的课程策略，将分层知识转移和阶段比例经验重播相结合，以解决奖励稀疏和灾难性遗忘的问题。所提出的框架在定制的多智能体仿真平台上进行了验证，结果表明我们的方法在成功率、编队保持率和成功加权任务时间方面比其他先进方法具有显着优势。代码可在 \textcolor{red}{https://github.com/Wanhao-Liu/AC-MASAC} 获取。

</details>

---

## 39. Distributionally Robust Cooperative Multi-Agent Reinforcement Learning via Robust Value Factorization / 通过鲁棒价值分解的分布式鲁棒协作多智能体强化学习

**Date**: 2026-02-11 | **arXiv**: [2602.11437v1](http://arxiv.org/abs/2602.11437v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11437v1)

**Categories**: cs.AI, cs.MA

**Code**: https://github.com/crqu/robust-coMARL.

<details><summary><b>Abstract / 摘要</b></summary>

Cooperative multi-agent reinforcement learning (MARL) commonly adopts centralized training with decentralized execution, where value-factorization methods enforce the individual-global-maximum (IGM) principle so that decentralized greedy actions recover the team-optimal joint action. However, the reliability of this recipe in real-world settings remains unreliable due to environmental uncertainties arising from the sim-to-real gap, model mismatch, and system noise. We address this gap by introducing Distributionally robust IGM (DrIGM), a principle that requires each agent's robust greedy action to align with the robust team-optimal joint action. We show that DrIGM holds for a novel definition of robust individual action values, which is compatible with decentralized greedy execution and yields a provable robustness guarantee for the whole system. Building on this foundation, we derive DrIGM-compliant robust variants of existing value-factorization architectures (e.g., VDN/QMIX/QTRAN) that (i) train on robust Q-targets, (ii) preserve scalability, and (iii) integrate seamlessly with existing codebases without bespoke per-agent reward shaping. Empirically, on high-fidelity SustainGym simulators and a StarCraft game environment, our methods consistently improve out-of-distribution performance. Code and data are available at https://github.com/crqu/robust-coMARL.

协作多智能体强化学习（MARL）通常采用集中训练和分散执行的方式，其中价值分解方法强制执行个体全局最大值（IGM）原则，以便分散的贪婪行动恢复团队最优的联合行动。然而，由于模拟与真实差距、模型不匹配和系统噪声引起的环境不确定性，该方法在现实环境中的可靠性仍然不可靠。我们通过引入分布式鲁棒 IGM (DrIGM) 来解决这一差距，该原则要求每个智能体的鲁棒贪婪行动与鲁棒团队最优联合行动保持一致。我们表明，DrIGM 支持鲁棒个人行动值的新颖定义，它与去中心化贪婪执行兼容，并为整个系统提供了可证明的鲁棒性保证。在此基础上，我们推导出现有价值分解架构（例如 VDN/QMIX/QTRAN）的符合 DrIGM 的稳健变体，这些变体（i）在稳健的 Q 目标上进行训练，（ii）保持可扩展性，以及（iii）与现有代码库无缝集成，无需定制每个代理奖励塑造。根据经验，在高保真 SustainGym 模拟器和星际争霸游戏环境中，我们的方法持续改进分布外性能。代码和数据可在 https://github.com/crqu/robust-coMARL 获取。

</details>

---

## 40. When Visibility Outpaces Verification: Delayed Verification and Narrative Lock-in in Agentic AI Discourse / 当可见性超过验证时：代理人工智能话语中的延迟验证和叙事锁定

**Date**: 2026-02-11 | **arXiv**: [2602.11412v1](http://arxiv.org/abs/2602.11412v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11412v1)

**Categories**: cs.CY, cs.AI, cs.HC

<details><summary><b>Abstract / 摘要</b></summary>

Agentic AI systems-autonomous entities capable of independent planning and execution-reshape the landscape of human-AI trust. Long before direct system exposure, user expectations are mediated through high-stakes public discourse on social platforms. However, platform-mediated engagement signals (e.g., upvotes) may inadvertently function as a ``credibility proxy,'' potentially stifling critical evaluation.   This paper investigates the interplay between social proof and verification timing in online discussions of agentic AI. Analyzing a longitudinal dataset from two distinct Reddit communities with contrasting interaction cultures-r/OpenClaw and r/Moltbook-we operationalize verification cues via reproducible lexical rules and model the ``time-to-first-verification'' using a right-censored survival analysis framework.   Our findings reveal a systemic ``Popularity Paradox'': high-visibility discussions in both subreddits experience significantly delayed or entirely absent verification cues compared to low-visibility threads. This temporal lag creates a critical window for ``Narrative Lock-in,'' where early, unverified claims crystallize into collective cognitive biases before evidence-seeking behaviors emerge. We discuss the implications of this ``credibility-by-visibility'' effect for AI safety and propose ``epistemic friction'' as a design intervention to rebalance engagement-driven platforms.

代理人工智能系统——能够独立规划和执行的自主实体——重塑了人类与人工智能信任的格局。早在系统直接暴露之前，用户的期望就通过社交平台上高风险的公共话语来调节。然而，平台介导的参与信号（例如，赞成票）可能会无意中充当“可信度代理”，从而可能抑制批判性评估。   本文研究了代理人工智能在线讨论中社会证明和验证时间之间的相互作用。通过分析来自两个不同 Reddit 社区（r/OpenClaw 和 r/Moltbook）的纵向数据集，我们通过可重复的词汇规则来操作验证线索，并使用右审查生存分析框架对“首次验证时间”进行建模。   我们的研究结果揭示了一个系统性的“受欢迎度悖论”：与低可见度主题相比，两个 Reddit 子版块中高可见度讨论的验证提示明显延迟或完全缺失。这种时间滞后为“叙事锁定”创造了一个关键窗口，即早期未经证实的主张在寻求证据的行为出现之前就具体化为集体认知偏见。我们讨论了这种“可见性可信度”效应对人工智能安全的影响，并提出“认知摩擦”作为重新平衡参与驱动平台的设计干预措施。

</details>

---

## 41. TRACER: Trajectory Risk Aggregation for Critical Episodes in Agentic Reasoning / TRACER：主体推理中关键事件的轨迹风险聚合

**Date**: 2026-02-11 | **arXiv**: [2602.11409v1](http://arxiv.org/abs/2602.11409v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11409v1)

**Categories**: cs.AI

**Code**: https://github.com/sinatayebati/agent-tracer.

<details><summary><b>Abstract / 摘要</b></summary>

Estimating uncertainty for AI agents in real-world multi-turn tool-using interaction with humans is difficult because failures are often triggered by sparse critical episodes (e.g., looping, incoherent tool use, or user-agent miscoordination) even when local generation appears confident. Existing uncertainty proxies focus on single-shot text generation and therefore miss these trajectory-level breakdown signals. We introduce TRACER, a trajectory-level uncertainty metric for dual-control Tool-Agent-User interaction. TRACER combines content-aware surprisal with situational-awareness signals, semantic and lexical repetition, and tool-grounded coherence gaps, and aggregates them using a tail-focused risk functional with a MAX-composite step risk to surface decisive anomalies. We evaluate TRACER on $τ^2$-bench by predicting task failure and selective task execution. To this end, TRACER improves AUROC by up to 37.1% and AUARC by up to 55% over baselines, enabling earlier and more accurate detection of uncertainty in complex conversational tool-use settings. Our code and benchmark are available at https://github.com/sinatayebati/agent-tracer.

估计人工智能代理在现实世界中使用多轮工具与人类交互的不确定性是很困难的，因为即使本地生成看起来很自信，失败也常常是由稀疏的关键事件（例如循环、不连贯的工具使用或用户代理不协调）触发的。现有的不确定性代理专注于单次文本生成，因此错过了这些轨迹级故障信号。我们引入了 TRACER，一种用于双控制工具-代理-用户交互的轨迹级不确定性度量。 TRACER 将内容感知惊喜与情境感知信号、语义和词汇重复以及基于工具的连贯性差距相结合，并使用以尾部为中心的风险函数和 MAX 复合步骤风险来聚合它们，以显示决定性的异常。我们通过预测任务失败和选择性任务执行来在 $τ^2$-bench 上评估 TRACER。为此，TRACER 与基线相比，AUROC 提高了 37.1%，AUARC 提高了 55%，从而能够更早、更准确地检测复杂的对话工具使用设置中的不确定性。我们的代码和基准可在 https://github.com/sinatayebati/agent-tracer 获取。

</details>

---

## 42. ReplicatorBench: Benchmarking LLM Agents for Replicability in Social and Behavioral Sciences / ReplicatorBench：对法学硕士代理在社会和行为科学中的可复制性进行基准测试

**Date**: 2026-02-11 | **arXiv**: [2602.11354v1](http://arxiv.org/abs/2602.11354v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11354v1)

**Categories**: cs.AI, cs.CL

**Code**: https://github.com/CenterForOpenScience/llm-benchmarking.

<details><summary><b>Abstract / 摘要</b></summary>

The literature has witnessed an emerging interest in AI agents for automated assessment of scientific papers. Existing benchmarks focus primarily on the computational aspect of this task, testing agents' ability to reproduce or replicate research outcomes when having access to the code and data. This setting, while foundational, (1) fails to capture the inconsistent availability of new data for replication as opposed to reproduction, and (2) lacks ground-truth diversity by focusing only on reproducible papers, thereby failing to evaluate an agent's ability to identify non-replicable research. Furthermore, most benchmarks only evaluate outcomes rather than the replication process. In response, we introduce ReplicatorBench, an end-to-end benchmark, including human-verified replicable and non-replicable research claims in social and behavioral sciences for evaluating AI agents in research replication across three stages: (1) extraction and retrieval of replication data; (2) design and execution of computational experiments; and (3) interpretation of results, allowing a test of AI agents' capability to mimic the activities of human replicators in real world. To set a baseline of AI agents' capability, we develop ReplicatorAgent, an agentic framework equipped with necessary tools, like web search and iterative interaction with sandboxed environments, to accomplish tasks in ReplicatorBench. We evaluate ReplicatorAgent across four underlying large language models (LLMs), as well as different design choices of programming language and levels of code access. Our findings reveal that while current LLM agents are capable of effectively designing and executing computational experiments, they struggle with retrieving resources, such as new data, necessary to replicate a claim. All code and data are publicly available at https://github.com/CenterForOpenScience/llm-benchmarking.

文献见证了人们对用于自动评估科学论文的人工智能代理的兴趣。现有的基准主要侧重于该任务的计算方面，测试代理在访问代码和数据时重现或复制研究结果的能力。这种设置虽然是基础性的，但（1）无法捕获用于复制的新数据的可用性与复制相反，（2）由于仅关注可复制的论文而缺乏真实的多样性，从而无法评估代理识别不可复制研究的能力。此外，大多数基准测试仅评估结果而不是复制过程。为此，我们推出了 ReplicatorBench，这是一个端到端基准测试，包括社会和行为科学领域经过人类验证的可复制和不可复制的研究主张，用于跨三个阶段评估研究复制中的人工智能代理：（1）复制数据的提取和检索； (2) 计算实验的设计和执行； （3）结果解释，允许测试人工智能代理在现实世界中模仿人类复制者活动的能力。为了设定人工智能代理能力的基线，我们开发了 ReplicatorAgent，这是一个代理框架，配备了必要的工具，例如网络搜索和与沙盒环境的迭代交互，以完成 ReplicatorBench 中的任务。我们通过四种底层大语言模型 (LLM) 以及不同的编程语言设计选择和代码访问级别来评估 ReplicatorAgent。我们的研究结果表明，虽然当前的法学硕士代理人能够有效地设计和执行计算实验，但他们在检索复制主张所需的资源（例如新数据）方面遇到了困难。所有代码和数据均可在 https://github.com/CenterForOpenScience/llm-benchmarking 上公开获取。

</details>

---

## 43. Pushing Forward Pareto Frontiers of Proactive Agents with Behavioral Agentic Optimization / 通过行为代理优化推进主动代理的帕累托前沿

**Date**: 2026-02-11 | **arXiv**: [2602.11351v1](http://arxiv.org/abs/2602.11351v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11351v1)

**Categories**: cs.AI, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Proactive large language model (LLM) agents aim to actively plan, query, and interact over multiple turns, enabling efficient task completion beyond passive instruction following and making them essential for real-world, user-centric applications. Agentic reinforcement learning (RL) has recently emerged as a promising solution for training such agents in multi-turn settings, allowing interaction strategies to be learned from feedback. However, existing pipelines face a critical challenge in balancing task performance with user engagement, as passive agents can not efficiently adapt to users' intentions while overuse of human feedback reduces their satisfaction. To address this trade-off, we propose BAO, an agentic RL framework that combines behavior enhancement to enrich proactive reasoning and information-gathering capabilities with behavior regularization to suppress inefficient or redundant interactions and align agent behavior with user expectations. We evaluate BAO on multiple tasks from the UserRL benchmark suite, and demonstrate that it substantially outperforms proactive agentic RL baselines while achieving comparable or even superior performance to commercial LLM agents, highlighting its effectiveness for training proactive, user-aligned LLM agents in complex multi-turn scenarios. Our website: https://proactive-agentic-rl.github.io/.

主动式大语言模型 (LLM) 代理旨在主动规划、查询和多轮交互，从而实现超越被动指令跟随的高效任务完成，并使它们对于现实世界、以用户为中心的应用程序至关重要。代理强化学习（RL）最近成为在多回合设置中训练此类代理的有前景的解决方案，允许从反馈中学习交互策略。然而，现有的管道在平衡任务性能和用户参与度方面面临着严峻的挑战，因为被动代理无法有效地适应用户的意图，而过度使用人类反馈会降低他们的满意度。为了解决这种权衡问题，我们提出了 BAO，这是一种代理 RL 框架，它将行为增强与行为正则化相结合，以丰富主动推理和信息收集能力，以抑制低效或冗余的交互，并使代理行为与用户期望保持一致。我们在 UserRL 基准套件中的多项任务上评估了 BAO，并证明它的性能大大优于主动代理 RL 基线，同时实现了与商业 LLM 代理相当甚至更优越的性能，突出了其在复杂的多回合场景中训练主动、用户一致的 LLM 代理的有效性。我们的网站：https://proactive-agentic-rl.github.io/。

</details>

---

## 44. AgentNoiseBench: Benchmarking Robustness of Tool-Using LLM Agents Under Noisy Condition / AgentNoiseBench：噪声条件下使用工具的 LLM 代理的鲁棒性基准测试

**Date**: 2026-02-11 | **arXiv**: [2602.11348v1](http://arxiv.org/abs/2602.11348v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11348v1)

**Categories**: cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Recent advances in large language models have enabled LLM-based agents to achieve strong performance on a variety of benchmarks. However, their performance in real-world deployments often that observed on benchmark settings, especially in complex and imperfect environments. This discrepancy largely arises because prevailing training and evaluation paradigms are typically built on idealized assumptions, overlooking the inherent stochasticity and noise present in real-world interactions. To bridge this gap, we introduce AgentNoiseBench, a framework for systematically evaluating the robustness of agentic models under noisy environments. We first conduct an in-depth analysis of biases and uncertainties in real-world scenarios and categorize environmental noise into two primary types: user-noise and tool-noise. Building on this analysis, we develop an automated pipeline that injects controllable noise into existing agent-centric benchmarks while preserving task solvability. Leveraging this pipeline, we perform extensive evaluations across a wide range of models with diverse architectures and parameter scales. Our results reveal consistent performance variations under different noise conditions, highlighting the sensitivity of current agentic models to realistic environmental perturbations.

大型语言模型的最新进展使基于 LLM 的代理能够在各种基准测试中实现强大的性能。然而，它们在实际部署中的性能通常是在基准设置上观察到的，特别是在复杂和不完美的环境中。这种差异很大程度上是因为流行的训练和评估范式通常建立在理想化的假设之上，忽视了现实世界交互中存在的固有随机性和噪声。为了弥补这一差距，我们引入了 AgentNoiseBench，这是一个用于系统评估代理模型在噪声环境下的鲁棒性的框架。我们首先对现实场景中的偏差和不确定性进行深入分析，并将环境噪声分为两种主要类型：用户噪声和工具噪声。在此分析的基础上，我们开发了一个自动化管道，将可控噪声注入现有的以代理为中心的基准测试中，同时保持任务的可解决性。利用这个管道，我们对具有不同架构和参数规模的各种模型进行了广泛的评估。我们的结果揭示了不同噪声条件下一致的性能变化，突出了当前代理模型对现实环境扰动的敏感性。

</details>

---

## 45. Security Threat Modeling for Emerging AI-Agent Protocols: A Comparative Analysis of MCP, A2A, Agora, and ANP / 新兴 AI-Agent 协议的安全威胁建模：MCP、A2A、Agora 和 ANP 的比较分析

**Date**: 2026-02-11 | **arXiv**: [2602.11327v1](http://arxiv.org/abs/2602.11327v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11327v1)

**Categories**: cs.CR, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

The rapid development of the AI agent communication protocols, including the Model Context Protocol (MCP), Agent2Agent (A2A), Agora, and Agent Network Protocol (ANP), is reshaping how AI agents communicate with tools, services, and each other. While these protocols support scalable multi-agent interaction and cross-organizational interoperability, their security principles remain understudied, and standardized threat modeling is limited; no protocol-centric risk assessment framework has been established yet. This paper presents a systematic security analysis of four emerging AI agent communication protocols. First, we develop a structured threat modeling analysis that examines protocol architectures, trust assumptions, interaction patterns, and lifecycle behaviors to identify protocol-specific and cross-protocol risk surfaces. Second, we introduce a qualitative risk assessment framework that identifies twelve protocol-level risks and evaluates security posture across the creation, operation, and update phases through systematic assessment of likelihood, impact, and overall protocol risk, with implications for secure deployment and future standardization. Third, we provide a measurement-driven case study on MCP that formalizes the risk of missing mandatory validation/attestation for executable components as a falsifiable security claim by quantifying wrong-provider tool execution under multi-server composition across representative resolver policies. Collectively, our results highlight key design-induced risk surfaces and provide actionable guidance for secure deployment and future standardization of agent communication ecosystems.

AI代理通信协议的快速发展，包括模型上下文协议（MCP）、Agent2Agent（A2A）、Agora和代理网络协议（ANP），正在重塑AI代理与工具、服务以及彼此之间的通信方式。虽然这些协议支持可扩展的多代理交互和跨组织互操作性，但它们的安全原理仍未得到充分研究，标准化威胁建模也很有限；尚未建立以方案为中心的风险评估框架。本文对四种新兴人工智能代理通信协议进行了系统的安全分析。首先，我们开发结构化威胁建模分析，检查协议架构、信任假设、交互模式和生命周期行为，以识别特定于协议和跨协议的风险面。其次，我们引入了一个定性风险评估框架，该框架可识别 12 个协议级风险，并通过对可能性、影响和整体协议风险的系统评估来评估创建、操作和更新阶段的安全态势，并对安全部署和未来标准化产生影响。第三，我们提供了一个关于 MCP 的测量驱动案例研究，通过量化跨代表性解析器策略的多服务器组合下的错误提供程序工具执行，将可执行组件缺少强制验证/证明的风险形式化为可证伪的安全声明。总的来说，我们的结果突出了设计引起的关键风险面，并为代理通信生态系统的安全部署和未来标准化提供了可行的指导。

</details>

---

## 46. CryptoAnalystBench: Failures in Multi-Tool Long-Form LLM Analysis / CryptoAnalystBench：多工具长格式 LLM 分析中的失败

**Date**: 2026-02-11 | **arXiv**: [2602.11304v1](http://arxiv.org/abs/2602.11304v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11304v1)

**Categories**: cs.CR, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Modern analyst agents must reason over complex, high token inputs, including dozens of retrieved documents, tool outputs, and time sensitive data. While prior work has produced tool calling benchmarks and examined factuality in knowledge augmented systems, relatively little work studies their intersection: settings where LLMs must integrate large volumes of dynamic, structured and unstructured multi tool outputs. We investigate LLM failure modes in this regime using crypto as a representative high data density domain. We introduce (1) CryptoAnalystBench, an analyst aligned benchmark of 198 production crypto and DeFi queries spanning 11 categories; (2) an agentic harness equipped with relevant crypto and DeFi tools to generate responses across multiple frontier LLMs; and (3) an evaluation pipeline with citation verification and an LLM as a judge rubric spanning four user defined success dimensions: relevance, temporal relevance, depth, and data consistency. Using human annotation, we develop a taxonomy of seven higher order error types that are not reliably captured by factuality checks or LLM based quality scoring. We find that these failures persist even in state of the art systems and can compromise high stakes decisions. Based on this taxonomy, we refine the judge rubric to better capture these errors. While the judge does not align with human annotators on precise scoring across rubric iterations, it reliably identifies critical failure modes, enabling scalable feedback for developers and researchers studying analyst style agents. We release CryptoAnalystBench with annotated queries, the evaluation pipeline, judge rubrics, and the error taxonomy, and outline mitigation strategies and open challenges in evaluating long form, multi tool augmented systems.

现代分析代理必须对复杂的高令牌输入进行推理，包括数十个检索到的文档、工具输出和时间敏感数据。虽然之前的工作已经制定了工具调用基准并检查了知识增强系统的真实性，但相对较少的工作研究它们的交叉点：法学硕士必须集成大量动态、结构化和非结构化多工具输出的设置。我们使用加密作为代表性的高数据密度域来研究这种情况下的 LLM 故障模式。我们介绍 (1) CryptoAnalystBench，这是一个分析师一致的基准，涵盖 11 个类别的 198 个生产加密货币和 DeFi 查询； (2) 配备相关加密和 DeFi 工具的代理工具，可在多个前沿 LLM 中生成响应； (3) 具有引文验证和法学硕士作为评判标准的评估流程，涵盖四个用户定义的成功维度：相关性、时间相关性、深度和数据一致性。使用人工注释，我们开发了七种高阶错误类型的分类法，这些错误类型不能通过事实检查或基于法学硕士的质量评分可靠地捕获。我们发现，即使在最先进的系统中，这些失败仍然存在，并且可能会损害高风险的决策。基于这种分类法，我们改进了评判标准，以更好地捕捉这些错误。虽然法官在跨评分标准迭代的精确评分方面与人类注释者不一致，但它可靠地识别关键故障模式，为研究分析师风格代理的开发人员和研究人员提供可扩展的反馈。我们发布了 CryptoAnalystBench，其中包含带注释的查询、评估管道、判断标准和错误分类法，并概述了评估长格式、多工具增强系统时的缓解策略和开放挑战。

</details>

---

## 47. The PBSAI Governance Ecosystem: A Multi-Agent AI Reference Architecture for Securing Enterprise AI Estates / PBSAI 治理生态系统：用于保护企业 AI 资产的多代理 AI 参考架构

**Date**: 2026-02-11 | **arXiv**: [2602.11301v1](http://arxiv.org/abs/2602.11301v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11301v1)

**Categories**: cs.AI, cs.CR

<details><summary><b>Abstract / 摘要</b></summary>

Enterprises are rapidly deploying large language models, retrieval augmented generation pipelines, and tool using agents into production, often on shared high performance computing clusters and cloud accelerator platforms that also support defensive analytics. These systems increasingly function not as isolated models but as AI estates: socio technical systems spanning models, agents, data pipelines, security tooling, human workflows, and hyperscale infrastructure. Existing governance and security frameworks, including the NIST AI Risk Management Framework and systems security engineering guidance, articulate principles and risk functions but do not provide implementable architectures for multi agent, AI enabled cyber defense.   This paper introduces the Practitioners Blueprint for Secure AI (PBSAI) Governance Ecosystem, a multi agent reference architecture for securing enterprise and hyperscale AI estates. PBSAI organizes responsibilities into a twelve domain taxonomy and defines bounded agent families that mediate between tools and policy through shared context envelopes and structured output contracts. The architecture assumes baseline enterprise security capabilities and encodes key systems security techniques, including analytic monitoring, coordinated defense, and adaptive response. A lightweight formal model of agents, context envelopes, and ecosystem level invariants clarifies the traceability, provenance, and human in the loop guarantees enforced across domains. We demonstrate alignment with NIST AI RMF functions and illustrate application in enterprise SOC and hyperscale defensive environments. PBSAI is proposed as a structured, evidence centric foundation for open ecosystem development and future empirical validation.

企业正在快速部署大型语言模型、检索增强生成管道以及使用代理的工具到生产中，通常是在共享的高性能计算集群和云加速器平台上，这些平台也支持防御性分析。这些系统越来越多地不再作为孤立的模型，而是作为人工智能资产：涵盖模型、代理、数据管道、安全工具、人类工作流程和超大规模基础设施的社会技术系统。现有的治理和安全框架，包括 NIST 人工智能风险管理框架和系统安全工程指南，阐明了原则和风险功能，但没有为多代理、人工智能支持的网络防御提供可实施的架构。   本文介绍了安全人工智能 (PBSAI) 治理生态系统的从业者蓝图，这是一种用于保护企业和超大规模人工智能资产的多代理参考架构。 PBSAI 将职责组织成 12 个领域分类法，并定义有界代理系列，这些代理系列通过共享上下文信封和结构化输出契约在工具和策略之间进行调解。该架构假定基线企业安全能力，并对关键系统安全技术进行编码，包括分析监控、协调防御和自适应响应。代理、上下文信封和生态系统级不变量的轻量级正式模型阐明了跨领域强制执行的可追溯性、来源和人在循环保证。我们展示了与 NIST AI RMF 功能的一致性，并说明了在企业 SOC 和超大规模防御环境中的应用。 PBSAI 被提议作为开放生态系统开发和未来实证验证的结构化、以证据为中心的基础。

</details>

---

## 48. FormalJudge: A Neuro-Symbolic Paradigm for Agentic Oversight / FormalJudge：代理监督的神经符号范式

**Date**: 2026-02-11 | **arXiv**: [2602.11136v2](http://arxiv.org/abs/2602.11136v2) | **PDF**: [Link](http://arxiv.org/pdf/2602.11136v2)

**Categories**: cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

As LLM-based agents increasingly operate in high-stakes domains with real-world consequences, ensuring their behavioral safety becomes paramount. The dominant oversight paradigm, LLM-as-a-Judge, faces a fundamental dilemma: how can probabilistic systems reliably supervise other probabilistic systems without inheriting their failure modes? We argue that formal verification offers a principled escape from this dilemma, yet its adoption has been hindered by a critical bottleneck: the translation from natural language requirements to formal specifications. This paper bridges this gap by proposing , a neuro-symbolic framework that employs a bidirectional Formal-of-Thought architecture: LLMs serve as specification compilers that top-down decompose high-level human intent into atomic, verifiable constraints, then bottom-up prove compliance using Dafny specifications and Z3 Satisfiability modulo theories solving, which produces mathematical guarantees rather than probabilistic scores. We validate across three benchmarks spanning behavioral safety, multi-domain constraint adherence, and agentic upward deception detection. Experiments on 7 agent models demonstrate that achieves an average improvement of 16.6% over LLM-as-a-Judge baselines, enables weak-to-strong generalization where a 7B judge achieves over 90% accuracy detecting deception from 72B agents, and provides near-linear safety improvement through iterative refinement.

随着基于法学硕士的代理人越来越多地在具有现实世界后果的高风险领域运作，确保他们的行为安全变得至关重要。占主导地位的监督范式“LLM-as-a-Judge”面临着一个根本性的困境：概率系统如何可靠地监督其他概率系统而不继承它们的故障模式？我们认为形式验证提供了摆脱这种困境的原则性途径，但它的采用却受到了一个关键瓶颈的阻碍：从自然语言要求到形式规范的翻译。本文通过提出一种采用双向思维形式架构的神经符号框架来弥补这一差距：法学硕士作为规范编译器，自上而下地将高级人类意图分解为原子的、可验证的约束，然后使用 Dafny 规范和 Z3 可满足性模理论求解自下而上证明合规性，从而产生数学保证而不是概率分数。我们验证了三个基准，涵盖行为安全、多域约束遵守和代理向上欺骗检测。对 7 个代理模型的实验表明，与 LLM 作为法官的基线相比，平均提高了 16.6%，实现了从弱到强的泛化，其中 7B 法官在检测 72B 代理的欺骗时达到了 90% 以上的准确率，并通过迭代细化提供了近线性的安全性改进。

</details>

---

## 49. Learning to Compose for Cross-domain Agentic Workflow Generation / 学习构建跨域代理工作流生成

**Date**: 2026-02-11 | **arXiv**: [2602.11114v1](http://arxiv.org/abs/2602.11114v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11114v1)

**Categories**: cs.MA, cs.AI, cs.LG, cs.SE

<details><summary><b>Abstract / 摘要</b></summary>

Automatically generating agentic workflows -- executable operator graphs or codes that orchestrate reasoning, verification, and repair -- has become a practical way to solve complex tasks beyond what single-pass LLM generation can reliably handle. Yet what constitutes a good workflow depends heavily on the task distribution and the available operators. Under domain shift, current systems typically rely on iterative workflow refinement to discover a feasible workflow from a large workflow space, incurring high iteration costs and yielding unstable, domain-specific behavior. In response, we internalize a decompose-recompose-decide mechanism into an open-source LLM for cross-domain workflow generation. To decompose, we learn a compact set of reusable workflow capabilities across diverse domains. To recompose, we map each input task to a sparse composition over these bases to generate a task-specific workflow in a single pass. To decide, we attribute the success or failure of workflow generation to counterfactual contributions from learned capabilities, thereby capturing which capabilities actually drive success by their marginal effects. Across stringent multi-domain, cross-domain, and unseen-domain evaluations, our 1-pass generator surpasses SOTA refinement baselines that consume 20 iterations, while substantially reducing generation latency and cost.

自动生成代理工作流程（编排推理、验证和修复的可执行操作图或代码）已成为解决单通道 LLM 生成无法可靠处理的复杂任务的实用方法。然而，良好的工作流程的构成在很大程度上取决于任务分配和可用的操作员。在域转移下，当前系统通常依赖于迭代工作流细化来从大型工作流空间中发现可行的工作流，从而产生高迭代成本并产生不稳定的特定于域的行为。作为回应，我们将分解-重组-决定机制内化到开源 LLM 中，以生成跨域工作流程。为了进行分解，我们学习了一组跨不同领域的紧凑的可重用工作流功能。为了重构，我们将每个输入任务映射到这些基础上的稀疏组合，以在一次传递中生成特定于任务的工作流程。为了做出决定，我们将工作流生成的成功或失败归因于学习能力的反事实贡献，从而捕获哪些能力通过其边际效应真正推动成功。在严格的多域、跨域和未见域评估中，我们的 1-pass 生成器超越了消耗 20 次迭代的 SOTA 细化基线，同时大幅降低了生成延迟和成本。

</details>

---

## 50. GameDevBench: Evaluating Agentic Capabilities Through Game Development / GameDevBench：通过游戏开发评估代理能力

**Date**: 2026-02-11 | **arXiv**: [2602.11103v1](http://arxiv.org/abs/2602.11103v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11103v1)

**Categories**: cs.AI, cs.CL, cs.SE

<details><summary><b>Abstract / 摘要</b></summary>

Despite rapid progress on coding agents, progress on their multimodal counterparts has lagged behind. A key challenge is the scarcity of evaluation testbeds that combine the complexity of software development with the need for deep multimodal understanding. Game development provides such a testbed as agents must navigate large, dense codebases while manipulating intrinsically multimodal assets such as shaders, sprites, and animations within a visual game scene. We present GameDevBench, the first benchmark for evaluating agents on game development tasks. GameDevBench consists of 132 tasks derived from web and video tutorials. Tasks require significant multimodal understanding and are complex -- the average solution requires over three times the amount of lines of code and file changes compared to prior software development benchmarks. Agents still struggle with game development, with the best agent solving only 54.5% of tasks. We find a strong correlation between perceived task difficulty and multimodal complexity, with success rates dropping from 46.9% on gameplay-oriented tasks to 31.6% on 2D graphics tasks. To improve multimodal capability, we introduce two simple image and video-based feedback mechanisms for agents. Despite their simplicity, these methods consistently improve performance, with the largest change being an increase in Claude Sonnet 4.5's performance from 33.3% to 47.7%. We release GameDevBench publicly to support further research into agentic game development.

尽管编码剂取得了快速进展，但其多模式对应物的进展却滞后。一个关键的挑战是缺乏将软件开发的复杂性与深入的多模式理解的需求结合起来的评估测试平台。游戏开发提供了这样一个测试平台，因为代理必须导航大型、密集的代码库，同时在视觉游戏场景中操纵本质上的多模式资产，例如着色器、精灵和动画。我们推出了 GameDevBench，这是第一个评估代理游戏开发任务的基准。 GameDevBench 包含源自网络和视频教程的 132 个任务。任务需要大量的多模式理解，而且很复杂——与之前的软件开发基准相比，平均解决方案需要的代码行数和文件更改量是三倍多。智能体在游戏开发方面仍然举步维艰，最好的智能体只能解决 54.5% 的任务。我们发现感知任务难度和多模态复杂性之间存在很强的相关性，游戏导向任务的成功率从 46.9% 下降到 2D 图形任务的 31.6%。为了提高多模态能力，我们为代理引入了两种简单的基于图像和视频的反馈机制。尽管这些方法很简单，但它们不断提高性能，其中最大的变化是 Claude Sonnet 4.5 的性能从 33.3% 提高到 47.7%。我们公开发布 GameDevBench 以支持对代理游戏开发的进一步研究。

</details>

---

## 51. SurveyLens: A Research Discipline-Aware Benchmark for Automatic Survey Generation / SurveyLens：自动调查生成的研究学科感知基准

**Date**: 2026-02-11 | **arXiv**: [2602.11238v1](http://arxiv.org/abs/2602.11238v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11238v1)

**Categories**: cs.CL

<details><summary><b>Abstract / 摘要</b></summary>

The exponential growth of scientific literature has driven the evolution of Automatic Survey Generation (ASG) from simple pipelines to multi-agent frameworks and commercial Deep Research agents. However, current ASG evaluation methods rely on generic metrics and are heavily biased toward Computer Science (CS), failing to assess whether ASG methods adhere to the distinct standards of various academic disciplines. Consequently, researchers, especially those outside CS, lack clear guidance on using ASG systems to yield high-quality surveys compliant with specific discipline standards. To bridge this gap, we introduce SurveyLens, the first discipline-aware benchmark evaluating ASG methods across diverse research disciplines. We construct SurveyLens-1k, a curated dataset of 1,000 high-quality human-written surveys spanning 10 disciplines. Subsequently, we propose a dual-lens evaluation framework: (1) Discipline-Aware Rubric Evaluation, which utilizes LLMs with human preference-aligned weights to assess adherence to domain-specific writing standards; and (2) Canonical Alignment Evaluation to rigorously measure content coverage and synthesis quality against human-written survey papers. We conduct extensive experiments by evaluating 11 state-of-the-art ASG methods on SurveyLens, including Vanilla LLMs, ASG systems, and Deep Research agents. Our analysis reveals the distinct strengths and weaknesses of each paradigm across fields, providing essential guidance for selecting tools tailored to specific disciplinary requirements.

科学文献的指数级增长推动了自动调查生成（ASG）从简单管道发展到多智能体框架和商业深度研究智能体。然而，当前的ASG评估方法依赖于通用指标，并且严重偏向计算机科学（CS），无法评估ASG方法是否遵循各个学科的独特标准。因此，研究人员，尤其是计算机科学以外的研究人员，缺乏关于使用 ASG 系统进行符合特定学科标准的高质量调查的明确指导。为了弥补这一差距，我们推出了 SurveyLens，这是第一个跨不同研究学科评估 ASG 方法的学科意识基准。我们构建了 SurveyLens-1k，这是一个由 1000 份高质量人工撰写的调查组成的精选数据集，涵盖 10 个学科。随后，我们提出了一个双镜头评估框架：（1）学科意识评估，它利用法学硕士与人类偏好一致的权重来评估对特定领域写作标准的遵守情况； (2) 规范对齐评估，根据人工撰写的调查论文严格衡量内容覆盖范围和综合质量。我们通过在 SurveyLens 上评估 11 种最先进的 ASG 方法来进行广泛的实验，包括 Vanilla LLM、ASG 系统和 Deep Research 代理。我们的分析揭示了跨领域每种范式的独特优势和劣势，为选择适合特定学科要求的工具提供了重要指导。

</details>

---

## 52. Agent-Diff: Benchmarking LLM Agents on Enterprise API Tasks via Code Execution with State-Diff-Based Evaluation / Agent-Diff：通过基于状态差异评估的代码执行，对企业 API 任务上的 LLM 代理进行基准测试

**Date**: 2026-02-11 | **arXiv**: [2602.11224v1](http://arxiv.org/abs/2602.11224v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11224v1)

**Categories**: cs.SE, cs.CL

**Code**: https://github.com/agent-diff-bench/agent-diff.

<details><summary><b>Abstract / 摘要</b></summary>

We present Agent-Diff, a novel benchmarking framework for evaluating agentic Large Language Models (LLMs) on real-world tasks that execute code via external APIs. Agentic LLM performance varies due to differences in models, external tool access, prompt structures, and agentic frameworks. Benchmarks must make fundamental trade-offs between a sandboxed approach that controls for variation in software environments and more ecologically valid approaches employing real services. Agent-Diff attempts to capture the desirable features of both of these approaches by including access to the real API interfaces for software services while sandboxing the environment in which calls are made, processed, and evaluated. This approach relies on two key innovations. The first is a novel state-diff contract, which separates process from outcome - rather than fuzzy trace or parameter matching, we define task success as whether the expected change in environment state was achieved. The second is a novel sandbox that provides a standardized scripting layer that all models use to execute code against external APIs (Slack, Box, Linear, Google Calendar). Thus, we can evaluate different agentic LLMs against a standardized set of contracts using a unified sandbox while still evaluating their performance on real-world service interfaces. Using the Agent-Diff framework, we provide benchmarks for nine LLMs across 224 tasks utilizing enterprise software workflows. In addition, we evaluate the robustness of the framework with ablation experiments to assess the contribution of access to API documentation on benchmark performance. Code and data: https://github.com/agent-diff-bench/agent-diff.

我们提出了 Agent-Diff，这是一种新颖的基准测试框架，用于在通过外部 API 执行代码的现实任务上评估代理大型语言模型 (LLM)。由于模型、外部工具访问、提示结构和代理框架的差异，代理 LLM 的性能有所不同。基准必须在控制软件环境变化的沙盒方法和采用实际服务的更生态有效的方法之间进行基本权衡。 Agent-Diff 尝试通过包括对软件服务的真实 API 接口的访问，同时对进行调用、处理和评估的环境进行沙箱处理，来捕获这两种方法的所需功能。这种方法依赖于两项关键创新。第一个是新颖的状态差异契约，它将过程与结果分开——而不是模糊跟踪或参数匹配，我们将任务成功定义为是否实现了环境状态的预期变化。第二个是一个新颖的沙箱，它提供了一个标准化的脚本层，所有模型都使用该层来针对外部 API（Slack、Box、Linear、Google Calendar）执行代码。因此，我们可以使用统一的沙箱根据一组标准化合同来评估不同的代理法学硕士，同时仍然评估它们在现实世界服务接口上的性能。使用 Agent-Diff 框架，我们利用企业软件工作流程为 9 个法学硕士跨 224 项任务提供基准。此外，我们通过消融实验评估了框架的稳健性，以评估访问 API 文档对基准性能的贡献。代码和数据：https://github.com/agent-diff-bench/agent-diff。

</details>

---

## 53. From Natural Language to Materials Discovery:The Materials Knowledge Navigation Agent / 从自然语言到材料发现：材料知识导航代理

**Date**: 2026-02-11 | **arXiv**: [2602.11123v1](http://arxiv.org/abs/2602.11123v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11123v1)

**Categories**: cs.LG, cond-mat.mtrl-sci

<details><summary><b>Abstract / 摘要</b></summary>

Accelerating the discovery of high-performance materials remains a central challenge across energy, electronics, and aerospace technologies, where traditional workflows depend heavily on expert intuition and computationally expensive simulations. Here we introduce the Materials Knowledge Navigation Agent (MKNA), a language-driven system that translates natural-language scientific intent into executable actions for database retrieval, property prediction, structure generation, and stability evaluation. Beyond automating tool invocation, MKNA autonomously extracts quantitative thresholds and chemically meaningful design motifs from literature and database evidence, enabling data-grounded hypothesis formation. Applied to the search for high-Debye-temperature ceramics, the agent identifies a literature-supported screening criterion (Theta_D > 800 K), rediscovers canonical ultra-stiff materials such as diamond, SiC, SiN, and BeO, and proposes thermodynamically stable, previously unreported Be-C-rich compounds that populate the sparsely explored 1500-1700 K regime. These results demonstrate that MKNA not only finds stable candidates but also reconstructs interpretable design heuristics, establishing a generalizable platform for autonomous, language-guided materials exploration.

加速高性能材料的发现仍然是能源、电子和航空航天技术领域的核心挑战，这些技术的传统工作流程在很大程度上依赖于专家的直觉和计算成本高昂的模拟。在这里，我们介绍材料知识导航代理（MKNA），这是一种语言驱动的系统，可将自然语言的科学意图转化为可执行的操作，用于数据库检索、属性预测、结构生成和稳定性评估。除了自动化工具调用之外，MKNA 还可以从文献和数据库证据中自主提取定量阈值和具有化学意义的设计主题，从而能够形成基于数据的假设。应用于寻找高德拜温度陶瓷时，该代理确定了文献支持的筛选标准（Theta_D > 800 K），重新发现了典型的超硬材料，例如金刚石、SiC、SiN 和 BeO，并提出了热力学稳定、先前未报道的富含 Be-C 的化合物，这些化合物填充了很少探索的 1500-1700 K 范围。这些结果表明，MKNA 不仅找到了稳定的候选者，而且还重建了可解释的设计启发式，为自主的、语言引导的材料探索建立了一个通用平台。

</details>

---



</details>

<!-- PAPERS_CONTENT_END -->
