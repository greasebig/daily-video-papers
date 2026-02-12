# World Model Papers

Daily updates of world model related arXiv papers.

## Papers Index

<!-- PAPERS_INDEX_START -->
- [2026-02-12](papers/2026-02-12.md) - 15 papers
<!-- PAPERS_INDEX_END -->

## Daily Papers

<!-- PAPERS_CONTENT_START -->
<details><summary><b>2026-02-12 (15 papers)</b></summary>

# arXiv World Model Papers - 2026-02-12

**Paper Count**: 15

---

## 1. ContactGaussian-WM: Learning Physics-Grounded World Model from Videos / ContactGaussian-WM：从视频中学习基于物理的世界模型

**Date**: 2026-02-11 | **arXiv**: [2602.11021v1](http://arxiv.org/abs/2602.11021v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11021v1)

**Categories**: cs.RO, cs.AI, cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Developing world models that understand complex physical interactions is essential for advancing robotic planning and simulation.However, existing methods often struggle to accurately model the environment under conditions of data scarcity and complex contact-rich dynamic motion.To address these challenges, we propose ContactGaussian-WM, a differentiable physics-grounded rigid-body world model capable of learning intricate physical laws directly from sparse and contact-rich video sequences.Our framework consists of two core components: (1) a unified Gaussian representation for both visual appearance and collision geometry, and (2) an end-to-end differentiable learning framework that differentiates through a closed-form physics engine to infer physical properties from sparse visual observations.Extensive simulations and real-world evaluations demonstrate that ContactGaussian-WM outperforms state-of-the-art methods in learning complex scenarios, exhibiting robust generalization capabilities.Furthermore, we showcase the practical utility of our framework in downstream applications, including data synthesis and real-time MPC.

开发理解复杂物理交互的世界模型对于推进机器人规划和仿真至关重要。然而，现有方法通常很难在数据稀缺和复杂的接触丰富的动态运动条件下准确地对环境进行建模。为了应对这些挑战，我们提出了 ContactGaussian-WM，这是一种基于物理的可微刚体世界模型，能够直接从稀疏和接触丰富的视频序列中学习复杂的物理定律。我们的框架由两个核心组件组成：（1）视觉外观和碰撞的统一高斯表示（2）端到端可微学习框架，通过封闭式物理引擎进行区分，从稀疏的视觉观察中推断物理属性。广泛的模拟和现实世界评估表明，ContactGaussian-WM 在学习复杂场景方面优于最先进的方法，展现出强大的泛化能力。此外，我们展示了我们的框架在下游应用中的实际效用，包括数据合成和实时 MPC。

</details>

---

## 2. Affordances Enable Partial World Modeling with LLMs / 可供性使法学硕士能够进行部分世界建模

**Date**: 2026-02-11 | **arXiv**: [2602.10390v1](http://arxiv.org/abs/2602.10390v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.10390v1)

**Categories**: cs.LG, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Full models of the world require complex knowledge of immense detail. While pre-trained large models have been hypothesized to contain similar knowledge due to extensive pre-training on vast amounts of internet scale data, using them directly in a search procedure is inefficient and inaccurate. Conversely, partial models focus on making high quality predictions for a subset of state and actions: those linked through affordances that achieve user intents~\citep{khetarpal2020can}. Can we posit large models as partial world models? We provide a formal answer to this question, proving that agents achieving task-agnostic, language-conditioned intents necessarily possess predictive partial-world models informed by affordances. In the multi-task setting, we introduce distribution-robust affordances and show that partial models can be extracted to significantly improve search efficiency. Empirical evaluations in tabletop robotics tasks demonstrate that our affordance-aware partial models reduce the search branching factor and achieve higher rewards compared to full world models.

完整的世界模型需要大量细节的复杂知识。虽然由于对大量互联网规模数据进行了广泛的预训练，预训练的大型模型被假设包含类似的知识，但直接在搜索过程中使用它们是低效且不准确的。相反，部分模型专注于对状态和操作的子集进行高质量预测：通过实现用户意图的可供性链接的状态和操作~\citep{khetarpal2020can}。我们可以将大型模型视为部分世界模型吗？我们对这个问题提供了一个正式的答案，证明实现与任务无关、语言条件化意图的智能体必然拥有由可供性告知的预测部分世界模型。在多任务设置中，我们引入了分布鲁棒性可供性，并表明可以提取部分模型以显着提高搜索效率。桌面机器人任务的实证评估表明，与完整世界模型相比，我们的可供性感知部分模型减少了搜索分支因子并获得了更高的奖励。

</details>

---

## 3. Neuro-Symbolic Synergy for Interactive World Modeling / 交互式世界建模的神经符号协同作用

**Date**: 2026-02-11 | **arXiv**: [2602.10480v1](http://arxiv.org/abs/2602.10480v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.10480v1)

**Categories**: cs.CL

<details><summary><b>Abstract / 摘要</b></summary>

Large language models (LLMs) exhibit strong general-purpose reasoning capabilities, yet they frequently hallucinate when used as world models (WMs), where strict compliance with deterministic transition rules--particularly in corner cases--is essential. In contrast, Symbolic WMs provide logical consistency but lack semantic expressivity. To bridge this gap, we propose Neuro-Symbolic Synergy (NeSyS), a framework that integrates the probabilistic semantic priors of LLMs with executable symbolic rules to achieve both expressivity and robustness. NeSyS alternates training between the two models using trajectories inadequately explained by the other. Unlike rule-based prompting, the symbolic WM directly constrains the LLM by modifying its output probability distribution. The neural WM is fine-tuned only on trajectories not covered by symbolic rules, reducing training data by 50% without loss of accuracy. Extensive experiments on three distinct interactive environments, i.e., ScienceWorld, Webshop, and Plancraft, demonstrate NeSyS's consistent advantages over baselines in both WM prediction accuracy and data efficiency.

大型语言模型 (LLM) 表现出强大的通用推理能力，但在用作世界模型 (WM) 时，它们经常产生幻觉，在这种情况下，严格遵守确定性转换规则（尤其是在极端情况下）至关重要。相比之下，符号 WM 提供逻辑一致性，但缺乏语义表达能力。为了弥补这一差距，我们提出了神经符号协同（NeSyS），这是一个将法学硕士的概率语义先验与可执行符号规则相结合的框架，以实现表达性和鲁棒性。 NeSyS 使用另一个模型无法充分解释的轨迹在两个模型之间交替训练。与基于规则的提示不同，符号WM通过修改其输出概率分布来直接约束LLM。神经 WM 仅对符号规则未涵盖的轨迹进行微调，从而在不损失准确性的情况下减少 50% 的训练数据。在三个不同的交互环境（即 ScienceWorld、Webshop 和 Plancraft）上进行的大量实验证明了 NeSyS 在 WM 预测准确性和数据效率方面相对于基线具有一致的优势。

</details>

---

## 4. RISE: Self-Improving Robot Policy with Compositional World Model / RISE：利用组合世界模型自我改进机器人政策

**Date**: 2026-02-11 | **arXiv**: [2602.11075v1](http://arxiv.org/abs/2602.11075v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11075v1)

**Categories**: cs.RO

<details><summary><b>Abstract / 摘要</b></summary>

Despite the sustained scaling on model capacity and data acquisition, Vision-Language-Action (VLA) models remain brittle in contact-rich and dynamic manipulation tasks, where minor execution deviations can compound into failures. While reinforcement learning (RL) offers a principled path to robustness, on-policy RL in the physical world is constrained by safety risk, hardware cost, and environment reset. To bridge this gap, we present RISE, a scalable framework of robotic reinforcement learning via imagination. At its core is a Compositional World Model that (i) predicts multi-view future via a controllable dynamics model, and (ii) evaluates imagined outcomes with a progress value model, producing informative advantages for the policy improvement. Such compositional design allows state and value to be tailored by best-suited yet distinct architectures and objectives. These components are integrated into a closed-loop self-improving pipeline that continuously generates imaginary rollouts, estimates advantages, and updates the policy in imaginary space without costly physical interaction. Across three challenging real-world tasks, RISE yields significant improvement over prior art, with more than +35% absolute performance increase in dynamic brick sorting, +45% for backpack packing, and +35% for box closing, respectively.

尽管模型容量和数据采集不断扩展，但视觉-语言-动作 (VLA) 模型在接触丰富的动态操作任务中仍然很脆弱，在这些任务中，微小的执行偏差可能会导致失败。虽然强化学习 (RL) 提供了实现稳健性的原则性途径，但物理世界中的策略 RL 受到安全风险、硬件成本和环境重置的限制。为了弥补这一差距，我们推出了 RISE，这是一个通过想象力进行机器人强化学习的可扩展框架。其核心是组合世界模型，（i）通过可控动态模型预测多视角未来，（ii）通过进步价值模型评估想象的结果，为政策改进提供信息优势。这种组合设计允许通过最适合但独特的架构和目标来定制状态和价值。这些组件被集成到一个闭环自我改进管道中，该管道不断生成想象中的部署、估计优势并更新想象空间中的策略，而无需昂贵的物理交互。在三个具有挑战性的现实世界任务中，RISE 比现有技术取得了显着改进，动态砖块分类的绝对性能提高了 35% 以上，背包包装的绝对性能提高了 45%，盒子关闭的性能提高了 35%。

</details>

---

## 5. Scaling World Model for Hierarchical Manipulation Policies / 分级操纵策略的扩展世界模型

**Date**: 2026-02-11 | **arXiv**: [2602.10983v1](http://arxiv.org/abs/2602.10983v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.10983v1)

**Categories**: cs.RO

<details><summary><b>Abstract / 摘要</b></summary>

Vision-Language-Action (VLA) models are promising for generalist robot manipulation but remain brittle in out-of-distribution (OOD) settings, especially with limited real-robot data. To resolve the generalization bottleneck, we introduce a hierarchical Vision-Language-Action framework \our{} that leverages the generalization of large-scale pre-trained world model for robust and generalizable VIsual Subgoal TAsk decomposition VISTA. Our hierarchical framework \our{} consists of a world model as the high-level planner and a VLA as the low-level executor. The high-level world model first divides manipulation tasks into subtask sequences with goal images, and the low-level policy follows the textual and visual guidance to generate action sequences. Compared to raw textual goal specification, these synthesized goal images provide visually and physically grounded details for low-level policies, making it feasible to generalize across unseen objects and novel scenarios. We validate both visual goal synthesis and our hierarchical VLA policies in massive out-of-distribution scenarios, and the performance of the same-structured VLA in novel scenarios could boost from 14% to 69% with the guidance generated by the world model. Results demonstrate that our method outperforms previous baselines with a clear margin, particularly in out-of-distribution scenarios. Project page: \href{https://vista-wm.github.io/}{https://vista-wm.github.io}

视觉-语言-动作（VLA）模型对于通用机器人操作很有希望，但在分布外（OOD）设置中仍然很脆弱，尤其是在真实机器人数据有限的情况下。为了解决泛化瓶颈，我们引入了分层视觉-语言-动作框架 \our{}，该框架利用大规模预训练世界模型的泛化来实现稳健且可泛化的视觉子目标任务分解 VISTA。我们的分层框架 \our{} 由作为高级规划器的世界模型和作为低级执行器的 VLA 组成。高层世界模型首先将操作任务划分为具有目标图像的子任务序列，低层策略遵循文本和视觉指导来生成动作序列。与原始文本目标规范相比，这些合成的目标图像为低级策略提供了视觉和物理基础的细节，使得在未见过的物体和新场景中进行泛化成为可能。我们在大规模分布外场景中验证了视觉目标合成和分层 VLA 策略，并且在世界模型生成的指导下，相同结构的 VLA 在新颖场景中的性能可以从 14% 提高到 69%。结果表明，我们的方法明显优于以前的基线，特别是在分布外的情况下。项目页面：\href{https://vista-wm.github.io/}{https://vista-wm.github.io}

</details>

---

## 6. Say, Dream, and Act: Learning Video World Models for Instruction-Driven Robot Manipulation / 说、梦想和行动：学习指令驱动机器人操作的视频世界模型

**Date**: 2026-02-11 | **arXiv**: [2602.10717v1](http://arxiv.org/abs/2602.10717v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.10717v1)

**Categories**: cs.RO

<details><summary><b>Abstract / 摘要</b></summary>

Robotic manipulation requires anticipating how the environment evolves in response to actions, yet most existing systems lack this predictive capability, often resulting in errors and inefficiency. While Vision-Language Models (VLMs) provide high-level guidance, they cannot explicitly forecast future states, and existing world models either predict only short horizons or produce spatially inconsistent frames. To address these challenges, we propose a framework for fast and predictive video-conditioned action. Our approach first selects and adapts a robust video generation model to ensure reliable future predictions, then applies adversarial distillation for fast, few-step video generation, and finally trains an action model that leverages both generated videos and real observations to correct spatial errors. Extensive experiments show that our method produces temporally coherent, spatially accurate video predictions that directly support precise manipulation, achieving significant improvements in embodiment consistency, spatial referring ability, and task completion over existing baselines. Codes & Models will be released.

机器人操纵需要预测环境如何响应行动而演变，但大多数现有系统缺乏这种预测能力，常常导致错误和低效率。虽然视觉语言模型（VLM）提供高级指导，但它们无法明确预测未来状态，并且现有的世界模型要么仅预测短期情况，要么产生空间不一致的框架。为了应对这些挑战，我们提出了一个快速、预测性视频条件动作框架。我们的方法首先选择并调整一个强大的视频生成模型，以确保可靠的未来预测，然后应用对抗性蒸馏来快速、几步视频生成，最后训练一个动作模型，利用生成的视频和真实观察来纠正空间错误。大量的实验表明，我们的方法产生时间上一致、空间上准确的视频预测，直接支持精确操作，在现有基线的基础上实现了实施例一致性、空间参考能力和任务完成度的显着改进。代码和型号将被发布。

</details>

---

## 7. Olaf-World: Orienting Latent Actions for Video World Modeling / Olaf-World：定向视频世界建模的潜在动作

**Date**: 2026-02-10 | **arXiv**: [2602.10104v1](http://arxiv.org/abs/2602.10104v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.10104v1)

**Categories**: cs.CV, cs.AI, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Scaling action-controllable world models is limited by the scarcity of action labels. While latent action learning promises to extract control interfaces from unlabeled video, learned latents often fail to transfer across contexts: they entangle scene-specific cues and lack a shared coordinate system. This occurs because standard objectives operate only within each clip, providing no mechanism to align action semantics across contexts. Our key insight is that although actions are unobserved, their semantic effects are observable and can serve as a shared reference. We introduce Seq$Δ$-REPA, a sequence-level control-effect alignment objective that anchors integrated latent action to temporal feature differences from a frozen, self-supervised video encoder. Building on this, we present Olaf-World, a pipeline that pretrains action-conditioned video world models from large-scale passive video. Extensive experiments demonstrate that our method learns a more structured latent action space, leading to stronger zero-shot action transfer and more data-efficient adaptation to new control interfaces than state-of-the-art baselines.

动作可控世界模型的扩展受到动作标签稀缺的限制。虽然潜在动作学习有望从未标记的视频中提取控制界面，但学习到的潜在动作通常无法跨上下文迁移：它们纠缠了特定于场景的线索并且缺乏共享的坐标系。发生这种情况是因为标准目标仅在每个剪辑内运行，没有提供跨上下文对齐动作语义的机制。我们的主要见解是，虽然动作是不可观察的，但它们的语义效果是可观察的并且可以作为共享参考。我们引入了 Seq$Δ$-REPA，这是一种序列级控制效果对齐目标，它将集成的潜在动作锚定到来自冻结的自监督视频编码器的时间特征差异。在此基础上，我们提出了 Olaf-World，这是一个从大规模被动视频中预训练动作条件视频世界模型的管道。大量的实验表明，我们的方法学习了一个更加结构化的潜在动作空间，与最先进的基线相比，可以实现更强的零样本动作转移和更高效的数据适应新的控制界面。

</details>

---

## 8. Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning / 代理世界模型：代理强化学习的无限合成环境

**Date**: 2026-02-10 | **arXiv**: [2602.10090v2](http://arxiv.org/abs/2602.10090v2) | **PDF**: [Link](http://arxiv.org/pdf/2602.10090v2)

**Categories**: cs.AI, cs.CL, cs.LG

**Code**: https://github.com/Snowflake-Labs/agent-world-model.

<details><summary><b>Abstract / 摘要</b></summary>

Recent advances in large language model (LLM) have empowered autonomous agents to perform complex tasks that require multi-turn interactions with tools and environments. However, scaling such agent training is limited by the lack of diverse and reliable environments. In this paper, we propose Agent World Model (AWM), a fully synthetic environment generation pipeline. Using this pipeline, we scale to 1,000 environments covering everyday scenarios, in which agents can interact with rich toolsets (35 tools per environment on average) and obtain high-quality observations. Notably, these environments are code-driven and backed by databases, providing more reliable and consistent state transitions than environments simulated by LLMs. Moreover, they enable more efficient agent interaction compared with collecting trajectories from realistic environments. To demonstrate the effectiveness of this resource, we perform large-scale reinforcement learning for multi-turn tool-use agents. Thanks to the fully executable environments and accessible database states, we can also design reliable reward functions. Experiments on three benchmarks show that training exclusively in synthetic environments, rather than benchmark-specific ones, yields strong out-of-distribution generalization. The code is available at https://github.com/Snowflake-Labs/agent-world-model.

大语言模型 (LLM) 的最新进展使自主代理能够执行需要与工具和环境进行多轮交互的复杂任务。然而，由于缺乏多样化和可靠的环境，扩展此类代理训练受到限制。在本文中，我们提出了代理世界模型（AWM），一个完全合成的环境生成管道。使用此管道，我们可以扩展到涵盖日常场景的 1,000 个环境，其中代理可以与丰富的工具集（平均每个环境 35 个工具）进行交互并获得高质量的观察结果。值得注意的是，这些环境是代码驱动的，并由数据库支持，提供比法学硕士模拟的环境更可靠、更一致的状态转换。此外，与从现实环境中收集轨迹相比，它们可以实现更有效的代理交互。为了证明该资源的有效性，我们对多轮工具使用代理进行大规模强化学习。得益于完全可执行的环境和可访问的数据库状态，我们还可以设计可靠的奖励函数。对三个基准的实验表明，仅在合成环境中进行训练，而不是在特定于基准的环境中进行训练，可以产生强大的分布外泛化能力。该代码可在 https://github.com/Snowflake-Labs/agent-world-model 获取。

</details>

---

## 9. Optimistic World Models: Efficient Exploration in Model-Based Deep Reinforcement Learning / 乐观世界模型：基于模型的深度强化学习的有效探索

**Date**: 2026-02-10 | **arXiv**: [2602.10044v1](http://arxiv.org/abs/2602.10044v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.10044v1)

**Categories**: cs.LG, cs.AI, eess.SY

<details><summary><b>Abstract / 摘要</b></summary>

Efficient exploration remains a central challenge in reinforcement learning (RL), particularly in sparse-reward environments. We introduce Optimistic World Models (OWMs), a principled and scalable framework for optimistic exploration that brings classical reward-biased maximum likelihood estimation (RBMLE) from adaptive control into deep RL. In contrast to upper confidence bound (UCB)-style exploration methods, OWMs incorporate optimism directly into model learning by augmentation with an optimistic dynamics loss that biases imagined transitions toward higher-reward outcomes. This fully gradient-based loss requires neither uncertainty estimates nor constrained optimization. Our approach is plug-and-play with existing world model frameworks, preserving scalability while requiring only minimal modifications to standard training procedures. We instantiate OWMs within two state-of-the-art world model architectures, leading to Optimistic DreamerV3 and Optimistic STORM, which demonstrate significant improvements in sample efficiency and cumulative return compared to their baseline counterparts.

高效探索仍然是强化学习（RL）的核心挑战，特别是在稀疏奖励环境中。我们引入了乐观世界模型 (OWM)，这是一种用于乐观探索的有原则且可扩展的框架，它将经典的奖励偏差最大似然估计 (RBMLE) 从自适应控制引入深度强化学习。与上置信界 (UCB) 式的探索方法相比，OWM 通过乐观动态损失的增强，将乐观主义直接融入到模型学习中，这种损失使想象的转变偏向于更高回报的结果。这种完全基于梯度的损失既不需要不确定性估计，也不需要约束优化。我们的方法是与现有的世界模型框架即插即用，保持可扩展性，同时只需要对标准训练程序进行最少的修改。我们在两个最先进的世界模型架构中实例化了 OWM，从而产生了 Optimistic DreamerV3 和 Optimistic STORM，与基准模型相比，它们在样本效率和累积回报方面表现出了显着的改进。

</details>

---

## 10. Code2World: A GUI World Model via Renderable Code Generation / Code2World：通过可渲染代码生成的 GUI 世界模型

**Date**: 2026-02-10 | **arXiv**: [2602.09856v1](http://arxiv.org/abs/2602.09856v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.09856v1)

**Categories**: cs.CV, cs.AI, cs.CL, cs.HC

**Code**: https://github.com/AMAP-ML/Code2World.

<details><summary><b>Abstract / 摘要</b></summary>

Autonomous GUI agents interact with environments by perceiving interfaces and executing actions. As a virtual sandbox, the GUI World model empowers agents with human-like foresight by enabling action-conditioned prediction. However, existing text- and pixel-based approaches struggle to simultaneously achieve high visual fidelity and fine-grained structural controllability. To this end, we propose Code2World, a vision-language coder that simulates the next visual state via renderable code generation. Specifically, to address the data scarcity problem, we construct AndroidCode by translating GUI trajectories into high-fidelity HTML and refining synthesized code through a visual-feedback revision mechanism, yielding a corpus of over 80K high-quality screen-action pairs. To adapt existing VLMs into code prediction, we first perform SFT as a cold start for format layout following, then further apply Render-Aware Reinforcement Learning which uses rendered outcome as the reward signal by enforcing visual semantic fidelity and action consistency. Extensive experiments demonstrate that Code2World-8B achieves the top-performing next UI prediction, rivaling the competitive GPT-5 and Gemini-3-Pro-Image. Notably, Code2World significantly enhances downstream navigation success rates in a flexible manner, boosting Gemini-2.5-Flash by +9.5% on AndroidWorld navigation. The code is available at https://github.com/AMAP-ML/Code2World.

自主 GUI 代理通过感知界面并执行操作与环境进行交互。作为一个虚拟沙箱，GUI World 模型通过启用动作条件预测，使代理具有类似人类的远见。然而，现有的基于文本和像素的方法很难同时实现高视觉保真度和细粒度的结构可控性。为此，我们提出了 Code2World，一种视觉语言编码器，可通过可渲染代码生成来模拟下一个视觉状态。具体来说，为了解决数据稀缺问题，我们通过将 GUI 轨迹转换为高保真 HTML 并通过视觉反馈修订机制完善合成代码来构建 AndroidCode，从而生成超过 80K 高质量屏幕操作对的语料库。为了使现有的 VLM 适应代码预测，我们首先执行 SFT 作为格式布局遵循的冷启动，然后进一步应用渲染感知强化学习，通过强制视觉语义保真度和动作一致性，使用渲染结果作为奖励信号。大量实验表明，Code2World-8B 实现了性能最佳的下一个 UI 预测，可与竞争性的 GPT-5 和 Gemini-3-Pro-Image 相媲美。值得注意的是，Code2World 以灵活的方式显着提高了下游导航的成功率，使 Gemini-2.5-Flash 在 AndroidWorld 导航上提高了 9.5%。该代码可从 https://github.com/AMAP-ML/Code2World 获取。

</details>

---

## 11. On Emergent Social World Models -- Evidence for Functional Integration of Theory of Mind and Pragmatic Reasoning in Language Models / 论新兴的社会世界模型——心灵理论和语用推理在语言模型中功能整合的证据

**Date**: 2026-02-10 | **arXiv**: [2602.10298v1](http://arxiv.org/abs/2602.10298v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.10298v1)

**Categories**: cs.CL

<details><summary><b>Abstract / 摘要</b></summary>

This paper investigates whether LMs recruit shared computational mechanisms for general Theory of Mind (ToM) and language-specific pragmatic reasoning in order to contribute to the general question of whether LMs may be said to have emergent "social world models", i.e., representations of mental states that are repurposed across tasks (the functional integration hypothesis). Using behavioral evaluations and causal-mechanistic experiments via functional localization methods inspired by cognitive neuroscience, we analyze LMs' performance across seven subcategories of ToM abilities (Beaudoin et al., 2020) on a substantially larger localizer dataset than used in prior like-minded work. Results from stringent hypothesis-driven statistical testing offer suggestive evidence for the functional integration hypothesis, indicating that LMs may develop interconnected "social world models" rather than isolated competencies. This work contributes novel ToM localizer data, methodological refinements to functional localization techniques, and empirical insights into the emergence of social cognition in artificial systems.

本文研究了 LM 是否为一般心智理论 (ToM) 和特定于语言的语用推理引入共享计算机制，以便回答 LM 是否可以说具有新兴的“社会世界模型”这一普遍问题，即跨任务重新调整用途的心理状态的表示（功能整合假设）。通过受认知神经科学启发的功能定位方法进行行为评估和因果机制实验，我们在比之前志同道合的工作中使用的定位器数据集大得多的定位器数据集上分析了 LM 在 ToM 能力的七个子类别中的表现（Beaudoin 等人，2020）。严格的假设驱动的统计测试的结果为功能整合假设提供了暗示性证据，表明 LM 可能会开发相互关联的“社会世界模型”，而不是孤立的能力。这项工作贡献了新颖的 ToM 定位器数据、功能定位技术的方法改进以及对人工系统中社会认知的出现的实证见解。

</details>

---

## 12. VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model / VLA-JEPA：利用潜在世界模型增强视觉-语言-动作模型

**Date**: 2026-02-10 | **arXiv**: [2602.10098v1](http://arxiv.org/abs/2602.10098v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.10098v1)

**Categories**: cs.RO, cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Pretraining Vision-Language-Action (VLA) policies on internet-scale video is appealing, yet current latent-action objectives often learn the wrong thing: they remain anchored to pixel variation rather than action-relevant state transitions, making them vulnerable to appearance bias, nuisance motion, and information leakage. We introduce VLA-JEPA, a JEPA-style pretraining framework that sidesteps these pitfalls by design. The key idea is \emph{leakage-free state prediction}: a target encoder produces latent representations from future frames, while the student pathway sees only the current observation -- future information is used solely as supervision targets, never as input. By predicting in latent space rather than pixel space, VLA-JEPA learns dynamics abstractions that are robust to camera motion and irrelevant background changes. This yields a simple two-stage recipe -- JEPA pretraining followed by action-head fine-tuning -- without the multi-stage complexity of prior latent-action pipelines. Experiments on LIBERO, LIBERO-Plus, SimplerEnv and real-world manipulation tasks show that VLA-JEPA achieves consistent gains in generalization and robustness over existing methods.

在互联网规模的视频上预训练视觉-语言-动作（VLA）策略很有吸引力，但当前的潜在动作目标经常学到错误的东西：它们仍然锚定于像素变化而不是与动作相关的状态转换，这使得它们容易受到外观偏差、令人讨厌的运动和信息泄漏的影响。我们引入了 VLA-JEPA，这是一种 JEPA 风格的预训练框架，它通过设计避开了这些陷阱。关键思想是 \emph{无泄漏状态预测}：目标编码器从未来帧生成潜在表示，而学生路径只能看到当前的观察结果 - 未来信息仅用作监督目标，从不用作输入。通过在潜在空间而不是像素空间中进行预测，VLA-JEPA 学习了对相机运动和不相关背景变化具有鲁棒性的动态抽象。这产生了一个简单的两阶段配方——JEPA 预训练，然后是动作头微调——没有先前潜在动作管道的多阶段复杂性。对 LIBERO、LIBERO-Plus、SimplerEnv 和现实世界操作任务的实验表明，VLA-JEPA 在泛化性和鲁棒性方面比现有方法取得了一致的进步。

</details>

---

## 13. NavDreamer: Video Models as Zero-Shot 3D Navigators / NavDreamer：作为零镜头 3D 导航器的视频模型

**Date**: 2026-02-10 | **arXiv**: [2602.09765v1](http://arxiv.org/abs/2602.09765v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.09765v1)

**Categories**: cs.RO

<details><summary><b>Abstract / 摘要</b></summary>

Previous Vision-Language-Action models face critical limitations in navigation: scarce, diverse data from labor-intensive collection and static representations that fail to capture temporal dynamics and physical laws. We propose NavDreamer, a video-based framework for 3D navigation that leverages generative video models as a universal interface between language instructions and navigation trajectories. Our main hypothesis is that video's ability to encode spatiotemporal information and physical dynamics, combined with internet-scale availability, enables strong zero-shot generalization in navigation. To mitigate the stochasticity of generative predictions, we introduce a sampling-based optimization method that utilizes a VLM for trajectory scoring and selection. An inverse dynamics model is employed to decode executable waypoints from generated video plans for navigation. To systematically evaluate this paradigm in several video model backbones, we introduce a comprehensive benchmark covering object navigation, precise navigation, spatial grounding, language control, and scene reasoning. Extensive experiments demonstrate robust generalization across novel objects and unseen environments, with ablation studies revealing that navigation's high-level decision-making nature makes it particularly suited for video-based planning.

以前的视觉-语言-动作模型在导航方面面临着严重的局限性：来自劳动密集型收集的稀缺且多样化的数据以及无法捕捉时间动态和物理定律的静态表示。我们提出了 NavDreamer，一种基于视频的 3D 导航框架，利用生成视频模型作为语言指令和导航轨迹之间的通用接口。我们的主要假设是，视频编码时空信息和物理动力学的能力，与互联网规模的可用性相结合，可以在导航中实现强大的零样本泛化。为了减轻生成预测的随机性，我们引入了一种基于采样的优化方法，该方法利用 VLM 进行轨迹评分和选择。采用逆动态模型从生成的导航视频计划中解码可执行航路点。为了系统地评估几个视频模型主干中的这种范式，我们引入了一个涵盖对象导航、精确导航、空间基础、语言控制和场景推理的综合基准。大量的实验证明了对新物体和看不见的环境的强大泛化能力，消融研究表明导航的高级决策性质使其特别适合基于视频的规划。

</details>

---

## 14. WorldArena: A Unified Benchmark for Evaluating Perception and Functional Utility of Embodied World Models / WorldArena：评估具体世界模型的感知和功能效用的统一基准

**Date**: 2026-02-09 | **arXiv**: [2602.08971v2](http://arxiv.org/abs/2602.08971v2) | **PDF**: [Link](http://arxiv.org/pdf/2602.08971v2)

**Categories**: cs.CV, cs.RO

<details><summary><b>Abstract / 摘要</b></summary>

While world models have emerged as a cornerstone of embodied intelligence by enabling agents to reason about environmental dynamics through action-conditioned prediction, their evaluation remains fragmented. Current evaluation of embodied world models has largely focused on perceptual fidelity (e.g., video generation quality), overlooking the functional utility of these models in downstream decision-making tasks. In this work, we introduce WorldArena, a unified benchmark designed to systematically evaluate embodied world models across both perceptual and functional dimensions. WorldArena assesses models through three dimensions: video perception quality, measured with 16 metrics across six sub-dimensions; embodied task functionality, which evaluates world models as data engines, policy evaluators, and action planners integrating with subjective human evaluation. Furthermore, we propose EWMScore, a holistic metric integrating multi-dimensional performance into a single interpretable index. Through extensive experiments on 14 representative models, we reveal a significant perception-functionality gap, showing that high visual quality does not necessarily translate into strong embodied task capability. WorldArena benchmark with the public leaderboard is released at https://world-arena.ai, providing a framework for tracking progress toward truly functional world models in embodied AI.

虽然世界模型已成为体现智能的基石，使智能体能够通过行动条件预测来推理环境动态，但它们的评估仍然支离破碎。目前对具体世界模型的评估主要集中在感知保真度（例如视频生成质量），而忽视了这些模型在下游决策任务中的功能效用。在这项工作中，我们介绍了 WorldArena，这是一个统一的基准，旨在跨感知和功能维度系统地评估具体世界模型。 WorldArena 通过三个维度评估模型：视频感知质量，通过 6 个子维度的 16 个指标进行衡量；体现任务功能，将世界模型评估为数据引擎、政策评估者和与主观人类评估相结合的行动规划者。此外，我们提出了 EWMScore，这是一种将多维性能集成到单个可解释指数中的整体指标。通过对 14 个代表性模型的广泛实验，我们揭示了显着的感知功能差距，表明高视觉质量并不一定转化为强大的具体任务能力。 WorldArena 基准测试和公共排行榜在 https://world-arena.ai 上发布，提供了一个框架，用于跟踪具体人工智能中真正功能性世界模型的进展。

</details>

---

## 15. Reduced-order Control and Geometric Structure of Learned Lagrangian Latent Dynamics / 学习拉格朗日潜在动力学的降阶控制和几何结构

**Date**: 2026-02-09 | **arXiv**: [2602.08963v1](http://arxiv.org/abs/2602.08963v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.08963v1)

**Categories**: cs.RO, math.OC

<details><summary><b>Abstract / 摘要</b></summary>

Model-based controllers can offer strong guarantees on stability and convergence by relying on physically accurate dynamic models. However, these are rarely available for high-dimensional mechanical systems such as deformable objects or soft robots. While neural architectures can learn to approximate complex dynamics, they are either limited to low-dimensional systems or provide only limited formal control guarantees due to a lack of embedded physical structure. This paper introduces a latent control framework based on learned structure-preserving reduced-order dynamics for high-dimensional Lagrangian systems. We derive a reduced tracking law for fully actuated systems and adopt a Riemannian perspective on projection-based model-order reduction to study the resulting latent and projected closed-loop dynamics. By quantifying the sources of modeling error, we derive interpretable conditions for stability and convergence. We extend the proposed controller and analysis to underactuated systems by introducing learned actuation patterns. Experimental results on simulated and real-world systems validate our theoretical investigation and the accuracy of our controllers.

基于模型的控制器可以依靠物理上精确的动态模型来为稳定性和收敛性提供强有力的保证。然而，这些很少适用于高维机械系统，例如可变形物体或软机器人。虽然神经架构可以学习近似复杂的动力学，但它们要么仅限于低维系统，要么由于缺乏嵌入式物理结构而仅提供有限的形式控制保证。本文介绍了一种基于学习结构保持降阶动力学的高维拉格朗日系统的潜在控制框架。我们推导了全驱动系统的简化跟踪定律，并采用基于投影的模型降阶的黎曼观点来研究由此产生的潜在和投影闭环动力学。通过量化建模误差的来源，我们得出了稳定性和收敛性的可解释条件。我们通过引入学习的驱动模式将所提出的控制器和分析扩展到欠驱动系统。模拟和现实系统的实验结果验证了我们的理论研究和控制器的准确性。

</details>

---



</details>

<!-- PAPERS_CONTENT_END -->
