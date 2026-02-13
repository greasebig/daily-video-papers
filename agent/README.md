# Agent Papers

Daily updates of agent-related arXiv papers.

## Papers Index

<!-- PAPERS_INDEX_START -->
- [2026-02-13](papers/2026-02-13.md) - 59 papers
<!-- PAPERS_INDEX_END -->

## Daily Papers

<!-- PAPERS_CONTENT_START -->
<details><summary><b>2026-02-13 (59 papers)</b></summary>

# arXiv Agent Papers - 2026-02-13

**Paper Count**: 59

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

应对气候变化需要全球协调，但理性的经济行为体往往将眼前利益置于集体福利之上，从而导致社会困境。 InvestESG 是最近提出的多主体模拟，可以捕捉气候风险下投资者和公司之间的动态相互作用。我们对 InvestESG 表现出跨期社会困境的条件进行了正式描述，得出了个人激励与集体福利背离的理论阈值。在此基础上，我们应用优势对齐（Advantage Alignment）（一种可扩展的对手塑造算法，在一般和博弈中被证明是有效的）来影响 InvestESG 中的代理学习。我们提供理论见解，解释为什么优势对齐通过将学习动态偏向合作结果来系统地支持社会有益的平衡。我们的结果表明，战略性地塑造经济主体的学习过程可以带来更好的结果，从而为政策机制提供信息，以更好地将市场激励与长期可持续发展目标结合起来。

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

## 53. Step 3.5 Flash: Open Frontier-Level Intelligence with 11B Active Parameters / 步骤3.5 Flash：利用11B主动参数开放前沿级智能

**Date**: 2026-02-11 | **arXiv**: [2602.10604v1](http://arxiv.org/abs/2602.10604v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.10604v1)

**Categories**: cs.CL, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

We introduce Step 3.5 Flash, a sparse Mixture-of-Experts (MoE) model that bridges frontier-level agentic intelligence and computational efficiency. We focus on what matters most when building agents: sharp reasoning and fast, reliable execution. Step 3.5 Flash pairs a 196B-parameter foundation with 11B active parameters for efficient inference. It is optimized with interleaved 3:1 sliding-window/full attention and Multi-Token Prediction (MTP-3) to reduce the latency and cost of multi-round agentic interactions. To reach frontier-level intelligence, we design a scalable reinforcement learning framework that combines verifiable signals with preference feedback, while remaining stable under large-scale off-policy training, enabling consistent self-improvement across mathematics, code, and tool use. Step 3.5 Flash demonstrates strong performance across agent, coding, and math tasks, achieving 85.4% on IMO-AnswerBench, 86.4% on LiveCodeBench-v6 (2024.08-2025.05), 88.2% on tau2-Bench, 69.0% on BrowseComp (with context management), and 51.0% on Terminal-Bench 2.0, comparable to frontier models such as GPT-5.2 xHigh and Gemini 3.0 Pro. By redefining the efficiency frontier, Step 3.5 Flash provides a high-density foundation for deploying sophisticated agents in real-world industrial environments.

我们引入了 Step 3.5 Flash，这是一种稀疏专家混合 (MoE) 模型，可连接前沿级代理智能和计算效率。在构建代理时，我们关注最重要的事情：敏锐的推理和快速、可靠的执行。步骤 3.5 Flash 将 196B 参数基础与 11B 活动参数配对，以实现高效推理。它通过交错 3:1 滑动窗口/全注意力和多令牌预测 (MTP-3) 进行优化，以减少多轮代理交互的延迟和成本。为了达到前沿水平的智能，我们设计了一个可扩展的强化学习框架，该框架将可验证的信号与偏好反馈相结合，同时在大规模离策略训练下保持稳定，从而实现数学、代码和工具使用方面的一致自我改进。 Step 3.5 Flash 在智能体、编码和数学任务上表现出了强大的性能，在 IMO-AnswerBench 上实现了 85.4%，在 LiveCodeBench-v6 (2024.08-2025.05) 上实现了 86.4%，在 tau2-Bench 上实现了 88.2%，在 BrowseComp（具有上下文管理）上实现了 69.0%，在 Terminal-Bench 2.0 上实现了 51.0%，与前沿型号，如 GPT-5.2 xHigh 和 Gemini 3.0 Pro。通过重新定义效率边界，Step 3.5 Flash 为在现实工业环境中部署复杂的代理提供了高密度基础。

</details>

---

## 54. LHAW: Controllable Underspecification for Long-Horizon Tasks / LHAW：长期任务的可控不足

**Date**: 2026-02-11 | **arXiv**: [2602.10525v1](http://arxiv.org/abs/2602.10525v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.10525v1)

**Categories**: cs.CL, cs.AI, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Long-horizon workflow agents that operate effectively over extended periods are essential for truly autonomous systems. Their reliable execution critically depends on the ability to reason through ambiguous situations in which clarification seeking is necessary to ensure correct task execution. However, progress is limited by the lack of scalable, task-agnostic frameworks for systematically curating and measuring the impact of ambiguity across custom workflows. We address this gap by introducing LHAW (Long-Horizon Augmented Workflows), a modular, dataset-agnostic synthetic pipeline that transforms any well-specified task into controllable underspecified variants by systematically removing information across four dimensions - Goals, Constraints, Inputs, and Context - at configurable severity levels. Unlike approaches that rely on LLM predictions of ambiguity, LHAW validates variants through empirical agent trials, classifying them as outcome-critical, divergent, or benign based on observed terminal state divergence. We release 285 task variants from TheAgentCompany, SWE-Bench Pro and MCP-Atlas according to our taxonomy alongside formal analysis measuring how current agents detect, reason about, and resolve underspecification across ambiguous settings. LHAW provides the first systematic framework for cost-sensitive evaluation of agent clarification behavior in long-horizon settings, enabling development of reliable autonomous systems.

长期有效运行的长视野工作流代理对于真正的自治系统至关重要。它们的可靠执行关键取决于在模糊情况下进行推理的能力，在这种情况下，需要寻求澄清以确保正确的任务执行。然而，由于缺乏可扩展的、与任务无关的框架来系统地管理和衡量自定义工作流程中模糊性的影响，进展受到限制。我们通过引入 LHAW（长视野增强工作流）来解决这一差距，这是一种模块化的、与数据集无关的合成管道，通过在可配置的严重性级别系统地删除四个维度（目标、约束、输入和上下文）的信息，将任何明确指定的任务转换为可控的未指定变体。与依赖 LLM 模糊性预测的方法不同，LHAW 通过经验代理试验来验证变体，并根据观察到的最终状态差异将它们分类为结果关键型、发散型或良性型。我们根据我们的分类法，从 TheAgentCompany、SWE-Bench Pro 和 MCP-Atlas 发布了 285 个任务变体，同时进行正式分析，衡量当前代理如何在不明确的设置中检测、推理和解决规范不足的问题。 LHAW 提供了第一个系统框架，用于对长视野环境中的代理澄清行为进行成本敏感的评估，从而能够开发可靠的自主系统。

</details>

---

## 55. TestExplora: Benchmarking LLMs for Proactive Bug Discovery via Repository-Level Test Generation / TestExplora：通过存储库级测试生成对 LLM 进行主动 Bug 发现基准测试

**Date**: 2026-02-11 | **arXiv**: [2602.10471v1](http://arxiv.org/abs/2602.10471v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.10471v1)

**Categories**: cs.SE, cs.CL

<details><summary><b>Abstract / 摘要</b></summary>

Given that Large Language Models (LLMs) are increasingly applied to automate software development, comprehensive software assurance spans three distinct goals: regression prevention, reactive reproduction, and proactive discovery. Current evaluations systematically overlook the third goal. Specifically, they either treat existing code as ground truth (a compliance trap) for regression prevention, or depend on post-failure artifacts (e.g., issue reports) for bug reproduction-so they rarely surface defects before failures. To bridge this gap, we present TestExplora, a benchmark designed to evaluate LLMs as proactive testers within full-scale, realistic repository environments. TestExplora contains 2,389 tasks from 482 repositories and hides all defect-related signals. Models must proactively find bugs by comparing implementations against documentation-derived intent, using documentation as the oracle. Furthermore, to keep evaluation sustainable and reduce leakage, we propose continuous, time-aware data collection. Our evaluation reveals a significant capability gap: state-of-the-art models achieve a maximum Fail-to-Pass (F2P) rate of only 16.06%. Further analysis indicates that navigating complex cross-module interactions and leveraging agentic exploration are critical to advancing LLMs toward autonomous software quality assurance. Consistent with this, SWEAgent instantiated with GPT-5-mini achieves an F2P of 17.27% and an F2P@5 of 29.7%, highlighting the effectiveness and promise of agentic exploration in proactive bug discovery tasks.

鉴于大型语言模型 (LLM) 越来越多地应用于自动化软件开发，全面的软件保障涵盖三个不同的目标：回归预防、反应性再现和主动发现。目前的评估系统地忽视了第三个目标。具体来说，他们要么将现有代码视为用于回归预防的基本事实（合规性陷阱），要么依赖于故障后工件（例如问题报告）来进行错误再现 - 因此他们很少在故障之前暴露缺陷。为了弥补这一差距，我们推出了 TestExplora，这是一个基准测试，旨在评估法学硕士在全面、真实的存储库环境中作为主动测试人员的能力。 TestExplora 包含来自 482 个存储库的 2,389 个任务，并隐藏所有与缺陷相关的信号。模型必须使用文档作为预言机，通过将实现与文档派生的意图进行比较来主动发现错误。此外，为了保持评估的可持续性并减少泄漏，我们建议进行连续的、具有时间意识的数据收集。我们的评估揭示了巨大的能力差距：最先进的模型的最大失败率 (F2P) 仅达到 16.06%。进一步的分析表明，导航复杂的跨模块交互和利用代理探索对于推进法学硕士走向自主软件质量保证至关重要。与此一致，使用 GPT-5-mini 实例化的 SWEAgent 实现了 17.27% 的 F2P 和 29.7% 的 F2P@5，凸显了代理探索在主动 bug 发现任务中的有效性和前景。

</details>

---

## 56. The Landscape of Prompt Injection Threats in LLM Agents: From Taxonomy to Analysis / LLM 代理中的即时注入威胁概况：从分类到分析

**Date**: 2026-02-11 | **arXiv**: [2602.10453v1](http://arxiv.org/abs/2602.10453v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.10453v1)

**Categories**: cs.CR, cs.CL

<details><summary><b>Abstract / 摘要</b></summary>

The evolution of Large Language Models (LLMs) has resulted in a paradigm shift towards autonomous agents, necessitating robust security against Prompt Injection (PI) vulnerabilities where untrusted inputs hijack agent behaviors. This SoK presents a comprehensive overview of the PI landscape, covering attacks, defenses, and their evaluation practices. Through a systematic literature review and quantitative analysis, we establish taxonomies that categorize PI attacks by payload generation strategies (heuristic vs. optimization) and defenses by intervention stages (text, model, and execution levels). Our analysis reveals a key limitation shared by many existing defenses and benchmarks: they largely overlook context-dependent tasks, in which agents are authorized to rely on runtime environmental observations to determine actions. To address this gap, we introduce AgentPI, a new benchmark designed to systematically evaluate agent behavior under context-dependent interaction settings. Using AgentPI, we empirically evaluate representative defenses and show that no single approach can simultaneously achieve high trustworthiness, high utility, and low latency. Moreover, we show that many defenses appear effective under existing benchmarks by suppressing contextual inputs, yet fail to generalize to realistic agent settings where context-dependent reasoning is essential. This SoK distills key takeaways and open research problems, offering structured guidance for future research and practical deployment of secure LLM agents.

大型语言模型 (LLM) 的发展导致了向自主代理的范式转变，因此需要针对即时注入 (PI) 漏洞（即不受信任的输入劫持代理行为）提供强大的安全性。该 SoK 全面概述了 PI 领域，涵盖攻击、防御及其评估实践。通过系统的文献综述和定量分析，我们建立了分类法，根据有效负载生成策略（启发式与优化）对 PI 攻击进行分类，并根据干预阶段（文本、模型和执行级别）对防御进行分类。我们的分析揭示了许多现有防御和基准所共有的一个关键限制：它们在很大程度上忽略了上下文相关的任务，在这些任务中，代理被授权依赖运行时环境观察来确定行动。为了解决这一差距，我们引入了 AgentPI，这是一个新的基准，旨在系统地评估上下文相关交互设置下的代理行为。使用 AgentPI，我们根据经验评估了代表性防御，并表明没有任何一种方法可以同时实现高可信度、高实用性和低延迟。此外，我们表明，许多防御措施在现有基准下通过抑制上下文输入而显得有效，但无法推广到上下文相关推理至关重要的现实代理设置。该 SoK 提炼了关键要点和开放研究问题，为安全 LLM 代理的未来研究和实际部署提供结构化指导。

</details>

---

## 57. From Natural Language to Materials Discovery:The Materials Knowledge Navigation Agent / 从自然语言到材料发现：材料知识导航代理

**Date**: 2026-02-11 | **arXiv**: [2602.11123v1](http://arxiv.org/abs/2602.11123v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11123v1)

**Categories**: cs.LG, cond-mat.mtrl-sci

<details><summary><b>Abstract / 摘要</b></summary>

Accelerating the discovery of high-performance materials remains a central challenge across energy, electronics, and aerospace technologies, where traditional workflows depend heavily on expert intuition and computationally expensive simulations. Here we introduce the Materials Knowledge Navigation Agent (MKNA), a language-driven system that translates natural-language scientific intent into executable actions for database retrieval, property prediction, structure generation, and stability evaluation. Beyond automating tool invocation, MKNA autonomously extracts quantitative thresholds and chemically meaningful design motifs from literature and database evidence, enabling data-grounded hypothesis formation. Applied to the search for high-Debye-temperature ceramics, the agent identifies a literature-supported screening criterion (Theta_D > 800 K), rediscovers canonical ultra-stiff materials such as diamond, SiC, SiN, and BeO, and proposes thermodynamically stable, previously unreported Be-C-rich compounds that populate the sparsely explored 1500-1700 K regime. These results demonstrate that MKNA not only finds stable candidates but also reconstructs interpretable design heuristics, establishing a generalizable platform for autonomous, language-guided materials exploration.

加速高性能材料的发现仍然是能源、电子和航空航天技术领域的核心挑战，这些技术的传统工作流程在很大程度上依赖于专家的直觉和计算成本高昂的模拟。在这里，我们介绍材料知识导航代理（MKNA），这是一种语言驱动的系统，可将自然语言的科学意图转化为可执行的操作，用于数据库检索、属性预测、结构生成和稳定性评估。除了自动化工具调用之外，MKNA 还可以从文献和数据库证据中自主提取定量阈值和具有化学意义的设计主题，从而能够形成基于数据的假设。应用于寻找高德拜温度陶瓷时，该代理确定了文献支持的筛选标准（Theta_D > 800 K），重新发现了典型的超硬材料，例如金刚石、SiC、SiN 和 BeO，并提出了热力学稳定、先前未报道的富含 Be-C 的化合物，这些化合物填充了很少探索的 1500-1700 K 范围。这些结果表明，MKNA 不仅找到了稳定的候选者，而且还重建了可解释的设计启发式，为自主的、语言引导的材料探索建立了一个通用平台。

</details>

---

## 58. Co-jump: Cooperative Jumping with Quadrupedal Robots via Multi-Agent Reinforcement Learning / 协同跳跃：通过多智能体强化学习与四足机器人协同跳跃

**Date**: 2026-02-11 | **arXiv**: [2602.10514v1](http://arxiv.org/abs/2602.10514v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.10514v1)

**Categories**: cs.RO, cs.AI, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

While single-agent legged locomotion has witnessed remarkable progress, individual robots remain fundamentally constrained by physical actuation limits. To transcend these boundaries, we introduce Co-jump, a cooperative task where two quadrupedal robots synchronize to execute jumps far beyond their solo capabilities. We tackle the high-impulse contact dynamics of this task under a decentralized setting, achieving synchronization without explicit communication or pre-specified motion primitives. Our framework leverages Multi-Agent Proximal Policy Optimization (MAPPO) enhanced by a progressive curriculum strategy, which effectively overcomes the sparse-reward exploration challenges inherent in mechanically coupled systems. We demonstrate robust performance in simulation and successful transfer to physical hardware, executing multi-directional jumps onto platforms up to 1.5 m in height. Specifically, one of the robots achieves a foot-end elevation of 1.1 m, which represents a 144% improvement over the 0.45 m jump height of a standalone quadrupedal robot, demonstrating superior vertical performance. Notably, this precise coordination is achieved solely through proprioceptive feedback, establishing a foundation for communication-free collaborative locomotion in constrained environments.

虽然单代理腿运动已经取得了显着的进步，但单个机器人仍然从根本上受到物理驱动限制的限制。为了超越这些界限，我们引入了协同跳跃，这是一种合作任务，其中两个四足机器人同步执行远远超出其单独能力的跳跃。我们在分散的设置下处理该任务的高脉冲接触动力学，无需显式通信或预先指定的运动基元即可实现同步。我们的框架利用渐进式课程策略增强的多智能体近端策略优化（MAPPO），有效克服了机械耦合系统固有的稀疏奖励探索挑战。我们在模拟中展示了强大的性能，并成功转移到物理硬件，在高达 1.5 m 的平台上执行多向跳跃。具体来说，其中一台机器人的足端高度达到了 1.1 m，这比独立四足机器人的 0.45 m 跳跃高度提高了 144%，展示了卓越的垂直性能。值得注意的是，这种精确的协调仅通过本体感觉反馈来实现，为受限环境中的无通信协作运动奠定了基础。

</details>

---

## 59. Autonomous Continual Learning of Computer-Use Agents for Environment Adaptation / 计算机使用代理的自主持续学习以适应环境

**Date**: 2026-02-10 | **arXiv**: [2602.10356v1](http://arxiv.org/abs/2602.10356v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.10356v1)

**Categories**: cs.CL

**Code**: https://github.com/OSU-NLP-Group/ACuRL.

<details><summary><b>Abstract / 摘要</b></summary>

Real-world digital environments are highly diverse and dynamic. These characteristics cause agents to frequently encounter unseen scenarios and distribution shifts, making continual learning in specific environments essential for computer-use agents (CUAs). However, a key challenge lies in obtaining high-quality and environment-grounded agent data without relying on costly human annotation. In this work, we introduce ACuRL, an Autonomous Curriculum Reinforcement Learning framework that continually adapts agents to specific environments with zero human data. The agent first explores target environments to acquire initial experiences. During subsequent iterative training, a curriculum task generator leverages these experiences together with feedback from the previous iteration to synthesize new tasks tailored for the agent's current capabilities. To provide reliable reward signals, we introduce CUAJudge, a robust automatic evaluator for CUAs that achieves 93% agreement with human judgments. Empirically, our method effectively enables both intra-environment and cross-environment continual learning, yielding 4-22% performance gains without catastrophic forgetting on existing environments. Further analyses show highly sparse updates (e.g., 20% parameters), which helps explain the effective and robust adaptation. Our data and code are available at https://github.com/OSU-NLP-Group/ACuRL.

现实世界的数字环境是高度多样化和动态的。这些特征导致代理经常遇到看不见的场景和分布变化，使得在特定环境中持续学习对于计算机使用代理（CUA）至关重要。然而，一个关键的挑战在于如何在不依赖昂贵的人工注释的情况下获得高质量且基于环境的代理数据。在这项工作中，我们引入了 ACuRL，这是一种自主课程强化学习框架，它能够在零人类数据的情况下不断使代理适应特定环境。代理首先探索目标环境以获得初始经验。在随后的迭代训练中，课程任务生成器利用这些经验以及先前迭代的反馈来合成适合代理当前能力的新任务。为了提供可靠的奖励信号，我们引入了 CUAJudge，这是一种强大的 CUA 自动评估器，与人类判断的一致性达到 93%。根据经验，我们的方法有效地实现了环境内和跨环境的持续学习，获得了 4-22% 的性能提升，并且不会对现有环境造成灾难性的遗忘。进一步的分析显示高度稀疏的更新（例如 20% 的参数），这有助于解释有效且稳健的适应。我们的数据和代码可在 https://github.com/OSU-NLP-Group/ACuRL 获取。

</details>

---



</details>

<!-- PAPERS_CONTENT_END -->
