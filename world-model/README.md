# World Model Papers

Daily updates of world model related arXiv papers.

## Papers Index

<!-- PAPERS_INDEX_START -->
- [2026-02-18](papers/2026-02-18.md) - 6 papers
- [2026-02-17](papers/2026-02-17.md) - 1 papers
- [2026-02-16](papers/2026-02-16.md) - 1 papers
- [2026-02-14](papers/2026-02-14.md) - 12 papers
<!-- PAPERS_INDEX_END -->

## Daily Papers

<!-- PAPERS_CONTENT_START -->
<details><summary><b>2026-02-18 (6 papers)</b></summary>

# arXiv World Model Papers - 2026-02-18

**Paper Count**: 6

---

## 1. VLM-DEWM: Dynamic External World Model for Verifiable and Resilient Vision-Language Planning in Manufacturing / VLM-DEWM：用于制造中可验证和弹性视觉语言规划的动态外部世界模型

**Date**: 2026-02-17 | **arXiv**: [2602.15549v1](http://arxiv.org/abs/2602.15549v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.15549v1)

**Categories**: cs.RO, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Vision-language model (VLM) shows promise for high-level planning in smart manufacturing, yet their deployment in dynamic workcells faces two critical challenges: (1) stateless operation, they cannot persistently track out-of-view states, causing world-state drift; and (2) opaque reasoning, failures are difficult to diagnose, leading to costly blind retries. This paper presents VLM-DEWM, a cognitive architecture that decouples VLM reasoning from world-state management through a persistent, queryable Dynamic External World Model (DEWM). Each VLM decision is structured into an Externalizable Reasoning Trace (ERT), comprising action proposal, world belief, and causal assumption, which is validated against DEWM before execution. When failures occur, discrepancy analysis between predicted and observed states enables targeted recovery instead of global replanning. We evaluate VLM-DEWM on multi-station assembly, large-scale facility exploration, and real-robot recovery under induced failures. Compared to baseline memory-augmented VLM systems, VLM DEWM improves state-tracking accuracy from 56% to 93%, increases recovery success rate from below 5% to 95%, and significantly reduces computational overhead through structured memory. These results establish VLM-DEWM as a verifiable and resilient solution for long-horizon robotic operations in dynamic manufacturing environments.

视觉语言模型（VLM）显示了智能制造中高层规划的前景，但它们在动态工作单元中的部署面临着两个关键挑战：（1）无状态操作，它们无法持续跟踪视野外的状态，导致世界状态漂移； (2) 推理不透明，故障难以诊断，导致盲目重试成本高昂。本文提出了 VLM-DEWM，这是一种认知架构，通过持久的、可查询的动态外部世界模型 (DEWM) 将 VLM 推理与世界状态管理解耦。每个 VLM 决策都被构建为外部化推理跟踪 (ERT)，其中包括行动建议、世界信念和因果假设，并在执行前针对 DEWM 进行验证。当发生故障时，预测状态和观察状态之间的差异分析可以实现有针对性的恢复，而不是全局重新规划。我们评估了 VLM-DEWM 的多站装配、大规模设施探索以及诱发故障下的真实机器人恢复。与基线内存增强 VLM 系统相比，VLM DEWM 将状态跟踪精度从 56% 提高到 93%，将恢复成功率从 5% 以下提高到 95%，并通过结构化内存显着降低计算开销。这些结果使 VLM-DEWM 成为动态制造环境中长期机器人操作的可验证且有弹性的解决方案。

</details>

---

## 2. World-Model-Augmented Web Agents with Action Correction / 具有动作校正功能的世界模型增强网络代理

**Date**: 2026-02-17 | **arXiv**: [2602.15384v1](http://arxiv.org/abs/2602.15384v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.15384v1)

**Categories**: cs.AI, cs.CL

<details><summary><b>Abstract / 摘要</b></summary>

Web agents based on large language models have demonstrated promising capability in automating web tasks. However, current web agents struggle to reason out sensible actions due to the limitations of predicting environment changes, and might not possess comprehensive awareness of execution risks, prematurely performing risky actions that cause losses and lead to task failure. To address these challenges, we propose WAC, a web agent that integrates model collaboration, consequence simulation, and feedback-driven action refinement. To overcome the cognitive isolation of individual models, we introduce a multi-agent collaboration process that enables an action model to consult a world model as a web-environment expert for strategic guidance; the action model then grounds these suggestions into executable actions, leveraging prior knowledge of environmental state transition dynamics to enhance candidate action proposal. To achieve risk-aware resilient task execution, we introduce a two-stage deduction chain. A world model, specialized in environmental state transitions, simulates action outcomes, which a judge model then scrutinizes to trigger action corrective feedback when necessary. Experiments show that WAC achieves absolute gains of 1.8% on VisualWebArena and 1.3% on Online-Mind2Web.

基于大型语言模型的 Web 代理在自动化 Web 任务方面表现出了良好的能力。然而，当前的网络代理由于预测环境变化的局限性，难以推理出合理的行动，并且可能不具备全面的执行风险意识，过早地执行风险行动，造成损失并导致任务失败。为了应对这些挑战，我们提出了 WAC，这是一种集成了模型协作、结果模拟和反馈驱动的动作细化的网络代理。为了克服各个模型的认知隔离，我们引入了多智能体协作流程，使行动模型能够作为网络环境专家咨询世界模型以获取战略指导；然后，行动模型将这些建议转化为可执行的行动，利用环境状态转换动态的先验知识来增强候选行动建议。为了实现风险感知的弹性任务执行，我们引入了两阶段推论链。专门研究环境状态转换的世界模型会模拟行动结果，然后法官模型会对其进行仔细检查，以在必要时触发行动纠正反馈。实验表明，WAC 在 VisualWebArena 上获得了 1.8% 的绝对收益，在 Online-Mind2Web 上获得了 1.3% 的绝对收益。

</details>

---

## 3. Feasibility-aware Imitation Learning from Observation with Multimodal Feedback / 通过多模态反馈进行观察的可行性感知模仿学习

**Date**: 2026-02-17 | **arXiv**: [2602.15351v1](http://arxiv.org/abs/2602.15351v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.15351v1)

**Categories**: cs.RO

<details><summary><b>Abstract / 摘要</b></summary>

Imitation learning frameworks that learn robot control policies from demonstrators' motions via hand-mounted demonstration interfaces have attracted increasing attention. However, due to differences in physical characteristics between demonstrators and robots, this approach faces two limitations: i) the demonstration data do not include robot actions, and ii) the demonstrated motions may be infeasible for robots. These limitations make policy learning difficult. To address them, we propose Feasibility-Aware Behavior Cloning from Observation (FABCO). FABCO integrates behavior cloning from observation, which complements robot actions using robot dynamics models, with feasibility estimation. In feasibility estimation, the demonstrated motions are evaluated using a robot-dynamics model, learned from the robot's execution data, to assess reproducibility under the robot's dynamics. The estimated feasibility is used for multimodal feedback and feasibility-aware policy learning to improve the demonstrator's motions and learn robust policies. Multimodal feedback provides feasibility through the demonstrator's visual and haptic senses to promote feasible demonstrated motions. Feasibility-aware policy learning reduces the influence of demonstrated motions that are infeasible for robots, enabling the learning of policies that robots can execute stably. We conducted experiments with 15 participants on two tasks and confirmed that FABCO improves imitation learning performance by more than 3.2 times compared to the case without feasibility feedback.

通过手持式演示界面从演示者的动作中学习机器人控制策略的模仿学习框架引起了越来越多的关注。然而，由于演示者和机器人之间的物理特征差异，这种方法面临两个限制：i）演示数据不包括机器人动作，ii）演示的运动对于机器人来说可能不可行。这些限制使得政策学习变得困难。为了解决这些问题，我们提出了基于观察的可行性感知行为克隆（FABCO）。 FABCO 将观察行为克隆与可行性评估相结合，利用机器人动力学模型补充机器人动作。在可行性评估中，使用从机器人的执行数据中学习的机器人动力学模型来评估演示的运动，以评估机器人动力学下的再现性。估计的可行性用于多模式反馈和可行性感知政策学习，以改进演示者的动作并学习稳健的政策。多模态反馈通过演示者的视觉和触觉提供可行性，以促进可行的演示动作。可行性感知策略学习减少了机器人无法执行的演示动作的影响，从而能够学习机器人可以稳定执行的策略。我们对 15 名参与者进行了两项任务的实验，证实 FABCO 与没有可行性反馈的情况相比，模仿学习性能提高了 3.2 倍以上。

</details>

---

## 4. Cold-Start Personalization via Training-Free Priors from Structured World Models / 通过结构化世界模型的免训练先验进行冷启动个性化

**Date**: 2026-02-16 | **arXiv**: [2602.15012v1](http://arxiv.org/abs/2602.15012v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.15012v1)

**Categories**: cs.CL, cs.AI, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

Cold-start personalization requires inferring user preferences through interaction when no user-specific historical data is available. The core challenge is a routing problem: each task admits dozens of preference dimensions, yet individual users care about only a few, and which ones matter depends on who is asking. With a limited question budget, asking without structure will miss the dimensions that matter. Reinforcement learning is the natural formulation, but in multi-turn settings its terminal reward fails to exploit the factored, per-criterion structure of preference data, and in practice learned policies collapse to static question sequences that ignore user responses. We propose decomposing cold-start elicitation into offline structure learning and online Bayesian inference. Pep (Preference Elicitation with Priors) learns a structured world model of preference correlations offline from complete profiles, then performs training-free Bayesian inference online to select informative questions and predict complete preference profiles, including dimensions never asked about. The framework is modular across downstream solvers and requires only simple belief models. Across medical, mathematical, social, and commonsense reasoning, Pep achieves 80.8% alignment between generated responses and users' stated preferences versus 68.5% for RL, with 3-5x fewer interactions. When two users give different answers to the same question, Pep changes its follow-up 39-62% of the time versus 0-28% for RL. It does so with ~10K parameters versus 8B for RL, showing that the bottleneck in cold-start elicitation is the capability to exploit the factored structure of preference data.

冷启动个性化需要在没有特定于用户的历史数据可用时通过交互推断用户偏好。核心挑战是路由问题：每个任务都有数十个偏好维度，但个人用户只关心其中几个，而哪些维度重要取决于提出请求的人。由于问题预算有限，没有结构的提问会错过重要的维度。强化学习是自然的公式，但在多轮设置中，其最终奖励无法利用偏好数据的分解的、按标准的结构，并且在实践中学习的策略崩溃为忽略用户响应的静态问题序列。我们建议将冷启动启发分解为离线结构学习和在线贝叶斯推理。 Pep（先验偏好启发）从完整的配置文件中离线学习偏好相关性的结构化世界模型，然后在线执行免训练贝叶斯推理以选择信息性问题并预测完整的偏好配置文件，包括从未询问过的维度。该框架在下游求解器中是模块化的，并且只需要简单的信念模型。在医学、数学、社会和常识推理方面，Pep 在生成的响应和用户陈述的偏好之间实现了 80.8% 的一致性，而 RL 的一致性为 68.5%，交互次数减少了 3-5 倍。当两个用户对同一问题给出不同答案时，Pep 在 39-62% 的情况下会更改其后续内容，而 RL 的这一比例为 0-28%。与 RL 的 8B 参数相比，它使用约 10K 参数来实现这一点，这表明冷启动启发的瓶颈是利用偏好数据的分解结构的能力。

</details>

---

## 5. World Models for Policy Refinement in StarCraft II / 星际争霸 II 中政策细化的世界模型

**Date**: 2026-02-16 | **arXiv**: [2602.14857v1](http://arxiv.org/abs/2602.14857v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14857v1)

**Categories**: cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Large Language Models (LLMs) have recently shown strong reasoning and generalization capabilities, motivating their use as decision-making policies in complex environments. StarCraft II (SC2), with its massive state-action space and partial observability, is a challenging testbed. However, existing LLM-based SC2 agents primarily focus on improving the policy itself and overlook integrating a learnable, action-conditioned transition model into the decision loop. To bridge this gap, we propose StarWM, the first world model for SC2 that predicts future observations under partial observability. To facilitate learning SC2's hybrid dynamics, we introduce a structured textual representation that factorizes observations into five semantic modules, and construct SC2-Dynamics-50k, the first instruction-tuning dataset for SC2 dynamics prediction. We further develop a multi-dimensional offline evaluation framework for predicted structured observations. Offline results show StarWM's substantial gains over zero-shot baselines, including nearly 60% improvements in resource prediction accuracy and self-side macro-situation consistency. Finally, we propose StarWM-Agent, a world-model-augmented decision system that integrates StarWM into a Generate--Simulate--Refine decision loop for foresight-driven policy refinement. Online evaluation against SC2's built-in AI demonstrates consistent improvements, yielding win-rate gains of 30%, 15%, and 30% against Hard (LV5), Harder (LV6), and VeryHard (LV7), respectively, alongside improved macro-management stability and tactical risk assessment.

大型语言模型（LLM）最近表现出强大的推理和泛化能力，促使它们在复杂环境中用作决策策略。星际争霸 II (SC2) 具有巨大的状态动作空间和部分可观测性，是一个具有挑战性的测试平台。然而，现有的基于 LLM 的 SC2 智能体主要侧重于改进策略本身，而忽略了将可学习的、以行动为条件的转换模型集成到决策循环中。为了弥补这一差距，我们提出了 StarWM，这是 SC2 的第一个世界模型，可以在部分可观测性下预测未来的观测结果。为了促进学习 SC2 的混合动力学，我们引入了一种结构化文本表示，将观察结果分解为五个语义模块，并构建了 SC2-Dynamics-50k，这是第一个用于 SC2 动力学预测的指令调整数据集。我们进一步开发了一个用于预测结构化观察的多维离线评估框架。离线结果显示，StarWM 较零样本基线有大幅提升，包括资源预测精度和自身宏观情况一致性方面提高了近 60%。最后，我们提出了 StarWM-Agent，这是一个世界模型增强决策系统，它将 StarWM 集成到生成-模拟-细化决策循环中，以实现前瞻驱动的政策细化。针对 SC2 内置 AI 的在线评估显示出持续的改进，相对于 Hard (LV5)、Harder (LV6) 和 VeryHard (LV7) 胜率分别提高了 30%、15% 和 30%，同时改善了宏观管理稳定性和战术风险评估。

</details>

---

## 6. WebWorld: A Large-Scale World Model for Web Agent Training / WebWorld：用于 Web 代理训练的大规模世界模型

**Date**: 2026-02-16 | **arXiv**: [2602.14721v1](http://arxiv.org/abs/2602.14721v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.14721v1)

**Categories**: cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Web agents require massive trajectories to generalize, yet real-world training is constrained by network latency, rate limits, and safety risks. We introduce \textbf{WebWorld} series, the first open-web simulator trained at scale. While existing simulators are restricted to closed environments with thousands of trajectories, WebWorld leverages a scalable data pipeline to train on 1M+ open-web interactions, supporting reasoning, multi-format data, and long-horizon simulations of 30+ steps. For intrinsic evaluation, we introduce WebWorld-Bench with dual metrics spanning nine dimensions, where WebWorld achieves simulation performance comparable to Gemini-3-Pro. For extrinsic evaluation, Qwen3-14B trained on WebWorld-synthesized trajectories improves by +9.2\% on WebArena, reaching performance comparable to GPT-4o. WebWorld enables effective inference-time search, outperforming GPT-5 as a world model. Beyond web simulation, WebWorld exhibits cross-domain generalization to code, GUI, and game environments, providing a replicable recipe for world model construction.

网络代理需要大量的轨迹来进行概括，但现实世界的训练受到网络延迟、速率限制和安全风险的限制。我们推出 \textbf{WebWorld} 系列，这是第一个大规模训练的开放网络模拟器。虽然现有模拟器仅限于具有数千条轨迹的封闭环境，但 WebWorld 利用可扩展的数据管道来训练超过 100 万次开放网络交互，支持推理、多格式数据和 30 多个步骤的长期模拟。对于内在评估，我们引入了具有跨越九个维度的双指标的WebWorld-Bench，其中WebWorld实现了与Gemini-3-Pro相当的模拟性能。对于外部评估，在 WebWorld 合成轨迹上训练的 Qwen3-14B 在 WebArena 上提高了 +9.2%，达到了与 GPT-4o 相当的性能。 WebWorld 可实现有效的推理时间搜索，作为世界模型，其性能优于 GPT-5。除了网络模拟之外，WebWorld 还展示了对代码、GUI 和游戏环境的跨域泛化，为世界模型构建提供了可复制的方法。

</details>

---



</details>

<details><summary><b>2026-02-17 (1 papers)</b></summary>

# arXiv World Model Papers - 2026-02-17

**Paper Count**: 1

---

## 1. WoVR: World Models as Reliable Simulators for Post-Training VLA Policies with RL / WoVR：世界模型作为 RL 训练后 VLA 策略的可靠模拟器

**Date**: 2026-02-15 | **arXiv**: [2602.13977v1](http://arxiv.org/abs/2602.13977v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.13977v1)

**Categories**: cs.RO, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Reinforcement learning (RL) promises to unlock capabilities beyond imitation learning for Vision-Language-Action (VLA) models, but its requirement for massive real-world interaction prevents direct deployment on physical robots. Recent work attempts to use learned world models as simulators for policy optimization, yet closed-loop imagined rollouts inevitably suffer from hallucination and long-horizon error accumulation. Such errors do not merely degrade visual fidelity; they corrupt the optimization signal, encouraging policies to exploit model inaccuracies rather than genuine task progress. We propose WoVR, a reliable world-model-based reinforcement learning framework for post-training VLA policies. Instead of assuming a faithful world model, WoVR explicitly regulates how RL interacts with imperfect imagined dynamics. It improves rollout stability through a controllable action-conditioned video world model, reshapes imagined interaction to reduce effective error depth via Keyframe-Initialized Rollouts, and maintains policy-simulator alignment through World Model-Policy co-evolution. Extensive experiments on LIBERO benchmarks and real-world robotic manipulation demonstrate that WoVR enables stable long-horizon imagined rollouts and effective policy optimization, improving average LIBERO success from 39.95% to 69.2% (+29.3 points) and real-robot success from 61.7% to 91.7% (+30.0 points). These results show that learned world models can serve as practical simulators for reinforcement learning when hallucination is explicitly controlled.

强化学习 (RL) 有望为视觉-语言-动作 (VLA) 模型解锁超越模仿学习的功能，但其对大规模现实世界交互的要求阻碍了直接部署在物理机器人上。最近的工作尝试使用学习的世界模型作为政策优化的模拟器，但闭环想象的推出不可避免地会遭受幻觉和长期错误累积的影响。此类错误不仅会降低视觉保真度，还会降低视觉保真度。它们破坏了优化信号，鼓励政策利用模型的不准确性，而不是真正的任务进展。我们提出了 WoVR，一种可靠的基于世界模型的强化学习框架，用于训练后 VLA 策略。 WoVR 没有假设一个忠实的世界模型，而是明确规范强化学习如何与不完美的想象动态相互作用。它通过可控的动作条件视频世界模型提高了推出稳定性，通过关键帧初始化的推出重塑了想象的交互以减少有效错误深度，并通过世界模型-策略共同进化保持策略-模拟器的一致性。对 LIBERO 基准和现实世界机器人操作的大量实验表明，WoVR 能够实现稳定的长期想象部署和有效的策略优化，将 LIBERO 平均成功率从 39.95% 提高到 69.2%（+29.3 分），将真实机器人成功率从 61.7% 提高到 91.7%（+30.0 分）。这些结果表明，当幻觉受到明确控制时，学习的世界模型可以作为强化学习的实用模拟器。

</details>

---



</details>

<details><summary><b>2026-02-16 (1 papers)</b></summary>

# arXiv World Model Papers - 2026-02-16

**Paper Count**: 1

---

## 1. Information-theoretic analysis of world models in optimal reward maximizers / 最优奖励最大化世界模型的信息论分析

**Date**: 2026-02-13 | **arXiv**: [2602.12963v1](http://arxiv.org/abs/2602.12963v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.12963v1)

**Categories**: cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

An important question in the field of AI is the extent to which successful behaviour requires an internal representation of the world. In this work, we quantify the amount of information an optimal policy provides about the underlying environment. We consider a Controlled Markov Process (CMP) with $n$ states and $m$ actions, assuming a uniform prior over the space of possible transition dynamics. We prove that observing a deterministic policy that is optimal for any non-constant reward function then conveys exactly $n \log m$ bits of information about the environment. Specifically, we show that the mutual information between the environment and the optimal policy is $n \log m$ bits. This bound holds across a broad class of objectives, including finite-horizon, infinite-horizon discounted, and time-averaged reward maximization. These findings provide a precise information-theoretic lower bound on the "implicit world model'' necessary for optimality.

人工智能领域的一个重要问题是成功的行为在多大程度上需要对世界的内部表征。在这项工作中，我们量化了最佳策略提供的有关底层环境的信息量。我们考虑一个具有 $n$ 状态和 $m$ 动作的受控马尔可夫过程 (CMP)，假设在可能的过渡动态空间上有一个统一的先验。我们证明，观察对于任何非恒定奖励函数来说都是最佳的确定性策略可以准确地传达 $n \log m$ 位有关环境的信息。具体来说，我们表明环境和最优策略之间的互信息为 $n \log m$ 位。这一界限适用于广泛的目标，包括有限范围、无限范围贴现和时间平均奖励最大化。这些发现为最优性所需的“隐式世界模型”提供了精确的信息论下界。

</details>

---



</details>

<details><summary><b>2026-02-14 (12 papers)</b></summary>

# arXiv World Model Papers - 2026-02-14

**Paper Count**: 12

---

## 1. The Observer Effect in World Models: Invasive Adaptation Corrupts Latent Physics / 世界模型中的观察者效应：侵入性适应破坏了潜在的物理学

**Date**: 2026-02-12 | **arXiv**: [2602.12218v1](http://arxiv.org/abs/2602.12218v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.12218v1)

**Categories**: cs.LG, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Determining whether neural models internalize physical laws as world models, rather than exploiting statistical shortcuts, remains challenging, especially under out-of-distribution (OOD) shifts. Standard evaluations often test latent capability via downstream adaptation (e.g., fine-tuning or high-capacity probes), but such interventions can change the representations being measured and thus confound what was learned during self-supervised learning (SSL). We propose a non-invasive evaluation protocol, PhyIP. We test whether physical quantities are linearly decodable from frozen representations, motivated by the linear representation hypothesis. Across fluid dynamics and orbital mechanics, we find that when SSL achieves low error, latent structure becomes linearly accessible. PhyIP recovers internal energy and Newtonian inverse-square scaling on OOD tests (e.g., $ρ> 0.90$). In contrast, adaptation-based evaluations can collapse this structure ($ρ\approx 0.05$). These findings suggest that adaptation-based evaluation can obscure latent structures and that low-capacity probes offer a more accurate evaluation of physical world models.

确定神经模型是否将物理定律内化为世界模型，而不是利用统计捷径，仍然具有挑战性，特别是在分布外（OOD）变化的情况下。标准评估通常通过下游适应（例如微调或高容量探测）来测试潜在能力，但此类干预可能会改变正在测量的表示，从而混淆自监督学习（SSL）期间学到的内容。我们提出了一种非侵入性评估协议 PhyIP。在线性表示假设的推动下，我们测试物理量是否可以从冻结表示中线性解码。在流体动力学和轨道力学中，我们发现当 SSL 实现低误差时，潜在结构变得可线性访问。 PhyIP 在 OOD 测试中恢复内能和牛顿平方反比缩放（例如 $ρ> 0.90$）。相反，基于适应的评估可以破坏这种结构（$ρ\approx 0.05$）。这些发现表明，基于适应的评估可以掩盖潜在结构，并且低容量探针可以对物理世界模型提供更准确的评估。

</details>

---

## 2. Accelerating Robotic Reinforcement Learning with Agent Guidance / 通过代理指导加速机器人强化学习

**Date**: 2026-02-12 | **arXiv**: [2602.11978v1](http://arxiv.org/abs/2602.11978v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11978v1)

**Categories**: cs.RO, cs.AI

<details><summary><b>Abstract / 摘要</b></summary>

Reinforcement Learning (RL) offers a powerful paradigm for autonomous robots to master generalist manipulation skills through trial-and-error. However, its real-world application is stifled by severe sample inefficiency. Recent Human-in-the-Loop (HIL) methods accelerate training by using human corrections, yet this approach faces a scalability barrier. Reliance on human supervisors imposes a 1:1 supervision ratio that limits fleet expansion, suffers from operator fatigue over extended sessions, and introduces high variance due to inconsistent human proficiency. We present Agent-guided Policy Search (AGPS), a framework that automates the training pipeline by replacing human supervisors with a multimodal agent. Our key insight is that the agent can be viewed as a semantic world model, injecting intrinsic value priors to structure physical exploration. By using executable tools, the agent provides precise guidance via corrective waypoints and spatial constraints for exploration pruning. We validate our approach on two tasks, ranging from precision insertion to deformable object manipulation. Results demonstrate that AGPS outperforms HIL methods in sample efficiency. This automates the supervision pipeline, unlocking the path to labor-free and scalable robot learning. Project website: https://agps-rl.github.io/agps.

强化学习 (RL) 为自主机器人提供了一个强大的范例，让其通过反复试验掌握通用操作技能。然而，其实际应用却因严重的样本效率低下而受到抑制。最近的人在环（HIL）方法通过使用人工修正来加速训练，但这种方法面临可扩展性障碍。对人类监督员的依赖强制实行 1:1 的监督比例，这限制了车队的扩张，操作员在长时间的工作中会感到疲劳，并且由于人员熟练程度不一致而带来很大的差异。我们提出了代理引导策略搜索（AGPS），这是一个通过用多模式代理取代人类监督员来自动化训练流程的框架。我们的主要见解是，代理可以被视为语义世界模型，在结构物理探索之前注入内在价值。通过使用可执行工具，代理通过修正路径点和空间约束提供精确的指导，以进行探索修剪。我们在两项任务上验证了我们的方法，从精确插入到可变形对象操作。结果表明，AGPS 在样本效率方面优于 HIL 方法。这实现了监督管道的自动化，开启了免劳动力且可扩展的机器人学习之路。项目网站：https://agps-rl.github.io/agps。

</details>

---

## 3. Where Bits Matter in World Model Planning: A Paired Mixed-Bit Study for Efficient Spatial Reasoning / 比特在世界模型规划中的作用：高效空间推理的配对混合比特研究

**Date**: 2026-02-12 | **arXiv**: [2602.11882v1](http://arxiv.org/abs/2602.11882v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11882v1)

**Categories**: cs.LG, cs.AI, cs.CV, cs.RO

**Code**: https://github.com/suraj-ranganath/DINO-MBQuant.

<details><summary><b>Abstract / 摘要</b></summary>

Efficient spatial reasoning requires world models that remain reliable under tight precision budgets. We study whether low-bit planning behavior is determined mostly by total bitwidth or by where bits are allocated across modules. Using DINO-WM on the Wall planning task, we run a paired-goal mixed-bit evaluation across uniform, mixed, asymmetric, and layerwise variants under two planner budgets. We observe a consistent three-regime pattern: 8-bit and 6-bit settings remain close to FP16, 3-bit settings collapse, and 4-bit settings are allocation-sensitive. In that transition region, preserving encoder precision improves planning relative to uniform quantization, and near-size asymmetric variants show the same encoder-side direction. In a later strict 22-cell replication with smaller per-cell episode count, the mixed-versus-uniform INT4 sign becomes budget-conditioned, which further highlights the sensitivity of this transition regime. These findings motivate module-aware, budget-aware quantization policies as a broader research direction for efficient spatial reasoning. Code and run artifacts are available at https://github.com/suraj-ranganath/DINO-MBQuant.

高效的空间推理需要世界模型在严格的精度预算下保持可靠。我们研究低位规划行为是否主要由总位宽或位在模块之间分配的位置决定。在 Wall 规划任务上使用 DINO-WM，我们在两个规划器预算下对均匀、混合、不对称和分层变体进行配对目标混合位评估。我们观察到一致的三机制模式：8 位和 6 位设置保持接近 FP16，3 位设置崩溃，4 位设置对分配敏感。在该过渡区域中，保持编码器精度相对于均匀量化改进了规划，并且接近尺寸的不对称变体显示了相同的编码器侧方向。在后来的严格 22 细胞复制中，每个细胞的情节数更小，混合与均匀的 INT4 符号变得受预算限制，这进一步凸显了这种过渡机制的敏感性。这些发现激发了模块感知、预算感知的量化策略作为高效空间推理的更广泛的研究方向。代码和运行工件可在 https://github.com/suraj-ranganath/DINO-MBQuant 获取。

</details>

---

## 4. Budget-Constrained Agentic Large Language Models: Intention-Based Planning for Costly Tool Use / 预算受限的代理大型语言模型：基于意图的昂贵工具使用规划

**Date**: 2026-02-12 | **arXiv**: [2602.11541v1](http://arxiv.org/abs/2602.11541v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11541v1)

**Categories**: cs.AI, cs.LG

<details><summary><b>Abstract / 摘要</b></summary>

We study budget-constrained tool-augmented agents, where a large language model must solve multi-step tasks by invoking external tools under a strict monetary budget. We formalize this setting as sequential decision making in context space with priced and stochastic tool executions, making direct planning intractable due to massive state-action spaces, high variance of outcomes and prohibitive exploration cost. To address these challenges, we propose INTENT, an inference-time planning framework that leverages an intention-aware hierarchical world model to anticipate future tool usage, risk-calibrated cost, and guide decisions online. Across cost-augmented StableToolBench, INTENT strictly enforces hard budget feasibility while substantially improving task success over baselines, and remains robust under dynamic market shifts such as tool price changes and varying budgets.

我们研究预算受限的工具增强代理，其中大型语言模型必须在严格的货币预算下通过调用外部工具来解决多步骤任务。我们将这种设置形式化为上下文空间中具有定价和随机工具执行的顺序决策，由于巨大的状态动作空间、结果的高方差和令人望而却步的探索成本，使得直接规划变得棘手。为了应对这些挑战，我们提出了 INTENT，这是一种推理时间规划框架，它利用意图感知的分层世界模型来预测未来的工具使用、风险校准成本并指导在线决策。在成本增加的 StableToolBench 中，INTENT 严格执行硬预算可行性，同时大幅提高任务成功率（较基准），并在工具价格变化和预算变化等动态市场变化下保持稳健。

</details>

---

## 5. LDA-1B: Scaling Latent Dynamics Action Model via Universal Embodied Data Ingestion / LDA-1B：通过通用嵌入数据摄取扩展潜在动态动作模型

**Date**: 2026-02-12 | **arXiv**: [2602.12215v1](http://arxiv.org/abs/2602.12215v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.12215v1)

**Categories**: cs.RO

<details><summary><b>Abstract / 摘要</b></summary>

Recent robot foundation models largely rely on large-scale behavior cloning, which imitates expert actions but discards transferable dynamics knowledge embedded in heterogeneous embodied data. While the Unified World Model (UWM) formulation has the potential to leverage such diverse data, existing instantiations struggle to scale to foundation-level due to coarse data usage and fragmented datasets. We introduce LDA-1B, a robot foundation model that scales through universal embodied data ingestion by jointly learning dynamics, policy, and visual forecasting, assigning distinct roles to data of varying quality. To support this regime at scale, we assemble and standardize EI-30k, an embodied interaction dataset comprising over 30k hours of human and robot trajectories in a unified format. Scalable dynamics learning over such heterogeneous data is enabled by prediction in a structured DINO latent space, which avoids redundant pixel-space appearance modeling. Complementing this representation, LDA-1B employs a multi-modal diffusion transformer to handle asynchronous vision and action streams, enabling stable training at the 1B-parameter scale. Experiments in simulation and the real world show LDA-1B outperforms prior methods (e.g., $π_{0.5}$) by up to 21\%, 48\%, and 23\% on contact-rich, dexterous, and long-horizon tasks, respectively. Notably, LDA-1B enables data-efficient fine-tuning, gaining 10\% by leveraging 30\% low-quality trajectories typically harmful and discarded.

最近的机器人基础模型在很大程度上依赖于大规模行为克隆，它模仿专家的行为，但丢弃了嵌入异构数据中的可转移的动力学知识。虽然统一世界模型 (UWM) 公式有潜力利用如此多样化的数据，但由于粗略的数据使用和分散的数据集，现有的实例很难扩展到基础级别。我们引入了 LDA-1B，这是一种机器人基础模型，通过联合学习动态、策略和视觉预测，为不同质量的数据分配不同的角色，通过通用的具体数据摄取进行扩展。为了大规模支持这种制度，我们组装并标准化了 EI-30k，这是一个具体的交互数据集，以统一的格式包含超过 30,000 小时的人类和机器人轨迹。通过在结构化 DINO 潜在空间中进行预测，可以实现对此类异构数据的可扩展动态学习，从而避免冗余的像素空间外观建模。作为对这种表示的补充，LDA-1B 采用多模态扩散变压器来处理异步视觉和动作流，从而实现 1B 参数规模的稳定训练。模拟和现实世界的实验表明，在接触丰富、灵巧和长视野任务上，LDA-1B 的性能分别比先前方法（例如 $π_{0.5}$）高出 21\%、48\% 和 23\%。值得注意的是，LDA-1B 能够实现数据高效的微调，通过利用 30% 的通常有害和丢弃的低质量轨迹，获得 10% 的增益。

</details>

---

## 6. VLAW: Iterative Co-Improvement of Vision-Language-Action Policy and World Model / VLAW：愿景-语言-行动政策和世界模型的迭代共同改进

**Date**: 2026-02-12 | **arXiv**: [2602.12063v1](http://arxiv.org/abs/2602.12063v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.12063v1)

**Categories**: cs.RO

**Project**: https://sites.google.com/view/vla-w  <details><summary><b>Abstract / 摘要</b></summary>

The goal of this paper is to improve the performance and reliability of vision-language-action (VLA) models through iterative online interaction. Since collecting policy rollouts in the real world is expensive, we investigate whether a learned simulator-specifically, an action-conditioned video generation model-can be used to generate additional rollout data. Unfortunately, existing world models lack the physical fidelity necessary for policy improvement: they are predominantly trained on demonstration datasets that lack coverage of many different physical interactions (particularly failure cases) and struggle to accurately model small yet critical physical details in contact-rich object manipulation. We propose a simple iterative improvement algorithm that uses real-world roll-out data to improve the fidelity of the world model, which can then, in turn, be used to generate supplemental synthetic data for improving the VLA model. In our experiments on a real robot, we use this approach to improve the performance of a state-of-the-art VLA model on multiple downstream tasks. We achieve a 39.2% absolute success rate improvement over the base policy and 11.6% improvement from training with the generated synthetic rollouts. Videos can be found at this anonymous website: https://sites.google.com/view/vla-w

本文的目标是通过迭代在线交互来提高视觉-语言-动作（VLA）模型的性能和可靠性。由于在现实世界中收集策略推出的成本很高，因此我们研究了是否可以使用学习的模拟器（具体而言，动作条件视频生成模型）来生成额外的推出数据。不幸的是，现有的世界模型缺乏政策改进所需的物理保真度：它们主要是在演示数据集上进行训练的，这些数据集缺乏对许多不同物理交互（特别是失败案例）的覆盖，并且很难在接触丰富的对象操作中准确地模拟微小但关键的物理细节。我们提出了一种简单的迭代改进算法，该算法使用现实世界的转出数据来提高世界模型的保真度，然后可以使用该算法生成补充合成数据以改进 VLA 模型。在我们对真实机器人的实验中，我们使用这种方法来提高最先进的 VLA 模型在多个下游任务上的性能。与基本策略相比，我们的绝对成功率提高了 39.2%，通过生成的综合部署进行训练，绝对成功率提高了 11.6%。视频可以在这个匿名网站上找到：https://sites.google.com/view/vla-w

</details>

---

## 7. HAIC: Humanoid Agile Object Interaction Control via Dynamics-Aware World Model / HAIC：通过动态感知世界模型进行人形敏捷对象交互控制

**Date**: 2026-02-12 | **arXiv**: [2602.11758v1](http://arxiv.org/abs/2602.11758v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11758v1)

**Categories**: cs.RO

<details><summary><b>Abstract / 摘要</b></summary>

Humanoid robots show promise for complex whole-body tasks in unstructured environments. Although Human-Object Interaction (HOI) has advanced, most methods focus on fully actuated objects rigidly coupled to the robot, ignoring underactuated objects with independent dynamics and non-holonomic constraints. These introduce control challenges from coupling forces and occlusions. We present HAIC, a unified framework for robust interaction across diverse object dynamics without external state estimation. Our key contribution is a dynamics predictor that estimates high-order object states (velocity, acceleration) solely from proprioceptive history. These predictions are projected onto static geometric priors to form a spatially grounded dynamic occupancy map, enabling the policy to infer collision boundaries and contact affordances in blind spots. We use asymmetric fine-tuning, where a world model continuously adapts to the student policy's exploration, ensuring robust state estimation under distribution shifts. Experiments on a humanoid robot show HAIC achieves high success rates in agile tasks (skateboarding, cart pushing/pulling under various loads) by proactively compensating for inertial perturbations, and also masters multi-object long-horizon tasks like carrying a box across varied terrain by predicting the dynamics of multiple objects.

人形机器人有望在非结构化环境中执行复杂的全身任务。尽管人机交互（HOI）已经取得了进步，但大多数方法都专注于与机器人刚性耦合的完全驱动物体，而忽略了具有独立动力学和非完整约束的欠驱动物体。这些引入了来自耦合力和遮挡的控制挑战。我们提出了 HAIC，这是一个无需外部状态估计即可跨不同对象动态进行鲁棒交互的统一框架。我们的主要贡献是一个动力学预测器，它仅根据本体感受历史来估计高阶物体状态（速度、加速度）。这些预测被投影到静态几何先验上，形成基于空间的动态占用图，使策略能够推断碰撞边界和盲点中的接触可供性。我们使用不对称微调，其中世界模型不断适应学生政策的探索，确保分布变化下的稳健状态估计。在人形机器人上的实验表明，HAIC 通过主动补偿惯性扰动，在敏捷任务（滑板、各种负载下推/拉车）中取得了很高的成功率，并且还通过预测多个物体的动态来掌握多物体长视距任务，例如在不同地形上搬运箱子。

</details>

---

## 8. Causal-JEPA: Learning World Models through Object-Level Latent Interventions / 因果-JEPA：通过对象级潜在干预学习世界模型

**Date**: 2026-02-11 | **arXiv**: [2602.11389v1](http://arxiv.org/abs/2602.11389v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11389v1)

**Categories**: cs.AI

**Code**: https://github.com/galilai-group/cjepa.

<details><summary><b>Abstract / 摘要</b></summary>

World models require robust relational understanding to support prediction, reasoning, and control. While object-centric representations provide a useful abstraction, they are not sufficient to capture interaction-dependent dynamics. We therefore propose C-JEPA, a simple and flexible object-centric world model that extends masked joint embedding prediction from image patches to object-centric representations. By applying object-level masking that requires an object's state to be inferred from other objects, C-JEPA induces latent interventions with counterfactual-like effects and prevents shortcut solutions, making interaction reasoning essential. Empirically, C-JEPA leads to consistent gains in visual question answering, with an absolute improvement of about 20\% in counterfactual reasoning compared to the same architecture without object-level masking. On agent control tasks, C-JEPA enables substantially more efficient planning by using only 1\% of the total latent input features required by patch-based world models, while achieving comparable performance. Finally, we provide a formal analysis demonstrating that object-level masking induces a causal inductive bias via latent interventions. Our code is available at https://github.com/galilai-group/cjepa.

世界模型需要强大的关系理解来支持预测、推理和控制。虽然以对象为中心的表示提供了有用的抽象，但它们不足以捕获依赖于交互的动态。因此，我们提出了 C-JEPA，这是一种简单而灵活的以对象为中心的世界模型，它将蒙版联合嵌入预测从图像块扩展到以对象为中心的表示。通过应用需要从其他对象推断对象状态的对象级屏蔽，C-JEPA 会引发具有类似反事实效果的潜在干预，并阻止捷径解决方案，从而使交互推理变得至关重要。根据经验，C-JEPA 在视觉问答方面带来了持续的收益，与没有对象级屏蔽的相同架构相比，反事实推理绝对提高了约 20%。在代理控制任务上，C-JEPA 仅使用基于补丁的世界模型所需的总潜在输入特征的 1%，从而实现了更高效的规划，同时实现了可比的性能。最后，我们提供了正式的分析，证明对象级掩蔽通过潜在干预诱发了因果归纳偏差。我们的代码可在 https://github.com/galilai-group/cjepa 获取。

</details>

---

## 9. H-WM: Robotic Task and Motion Planning Guided by Hierarchical World Model / H-WM：分层世界模型引导的机器人任务和运动规划

**Date**: 2026-02-11 | **arXiv**: [2602.11291v1](http://arxiv.org/abs/2602.11291v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11291v1)

**Categories**: cs.RO

<details><summary><b>Abstract / 摘要</b></summary>

World models are becoming central to robotic planning and control, as they enable prediction of future state transitions. Existing approaches often emphasize video generation or natural language prediction, which are difficult to directly ground in robot actions and suffer from compounding errors over long horizons. Traditional task and motion planning relies on symbolic logic world models, such as planning domains, that are robot-executable and robust for long-horizon reasoning. However, these methods typically operate independently of visual perception, preventing synchronized symbolic and perceptual state prediction. We propose a Hierarchical World Model (H-WM) that jointly predicts logical and visual state transitions within a unified bilevel framework. H-WM combines a high-level logical world model with a low-level visual world model, integrating the robot-executable, long-horizon robustness of symbolic reasoning with perceptual grounding from visual observations. The hierarchical outputs provide stable and consistent intermediate guidance for long-horizon tasks, mitigating error accumulation and enabling robust execution across extended task sequences. To train H-WM, we introduce a robotic dataset that aligns robot motion with symbolic states, actions, and visual observations. Experiments across vision-language-action (VLA) control policies demonstrate the effectiveness and generality of the approach.

世界模型正在成为机器人规划和控制的核心，因为它们能够预测未来的状态转换。现有的方法通常强调视频生成或自然语言预测，这些方法很难直接反映机器人的动作，并且在长期范围内会出现复合错误。传统的任务和运动规划依赖于符号逻辑世界模型，例如规划域，这些模型是机器人可执行的并且对于长视野推理来说是鲁棒的。然而，这些方法通常独立于视觉感知进行操作，从而阻止了同步的符号和感知状态预测。我们提出了一种分层世界模型（H-WM），它在统一的双层框架内联合预测逻辑和视觉状态转换。 H-WM 将高级逻辑世界模型与低级视觉世界模型相结合，将机器人可执行的符号推理的长期鲁棒性与视觉观察的感知基础相结合。分层输出为长期任务提供稳定一致的中间指导，减少错误累积并实现跨扩展任务序列的稳健执行。为了训练 H-WM，我们引入了一个机器人数据集，该数据集将机器人运动与符号状态、动作和视觉观察对齐。视觉-语言-动作（VLA）控制策略的实验证明了该方法的有效性和通用性。

</details>

---

## 10. RISE: Self-Improving Robot Policy with Compositional World Model / RISE：利用组合世界模型自我改进机器人政策

**Date**: 2026-02-11 | **arXiv**: [2602.11075v1](http://arxiv.org/abs/2602.11075v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11075v1)

**Categories**: cs.RO

<details><summary><b>Abstract / 摘要</b></summary>

Despite the sustained scaling on model capacity and data acquisition, Vision-Language-Action (VLA) models remain brittle in contact-rich and dynamic manipulation tasks, where minor execution deviations can compound into failures. While reinforcement learning (RL) offers a principled path to robustness, on-policy RL in the physical world is constrained by safety risk, hardware cost, and environment reset. To bridge this gap, we present RISE, a scalable framework of robotic reinforcement learning via imagination. At its core is a Compositional World Model that (i) predicts multi-view future via a controllable dynamics model, and (ii) evaluates imagined outcomes with a progress value model, producing informative advantages for the policy improvement. Such compositional design allows state and value to be tailored by best-suited yet distinct architectures and objectives. These components are integrated into a closed-loop self-improving pipeline that continuously generates imaginary rollouts, estimates advantages, and updates the policy in imaginary space without costly physical interaction. Across three challenging real-world tasks, RISE yields significant improvement over prior art, with more than +35% absolute performance increase in dynamic brick sorting, +45% for backpack packing, and +35% for box closing, respectively.

尽管模型容量和数据采集不断扩展，但视觉-语言-动作 (VLA) 模型在接触丰富的动态操作任务中仍然很脆弱，在这些任务中，微小的执行偏差可能会导致失败。虽然强化学习 (RL) 提供了实现稳健性的原则性途径，但物理世界中的策略 RL 受到安全风险、硬件成本和环境重置的限制。为了弥补这一差距，我们推出了 RISE，这是一个通过想象力进行机器人强化学习的可扩展框架。其核心是组合世界模型，（i）通过可控动态模型预测多视角未来，（ii）通过进步价值模型评估想象的结果，为政策改进提供信息优势。这种组合设计允许通过最适合但独特的架构和目标来定制状态和价值。这些组件被集成到一个闭环自我改进管道中，该管道不断生成想象中的部署、估计优势并更新想象空间中的策略，而无需昂贵的物理交互。在三个具有挑战性的现实世界任务中，RISE 比现有技术取得了显着改进，动态砖块分类的绝对性能提高了 35% 以上，背包包装的绝对性能提高了 45%，盒子关闭的性能提高了 35%。

</details>

---

## 11. ContactGaussian-WM: Learning Physics-Grounded World Model from Videos / ContactGaussian-WM：从视频中学习基于物理的世界模型

**Date**: 2026-02-11 | **arXiv**: [2602.11021v1](http://arxiv.org/abs/2602.11021v1) | **PDF**: [Link](http://arxiv.org/pdf/2602.11021v1)

**Categories**: cs.RO, cs.AI, cs.CV

<details><summary><b>Abstract / 摘要</b></summary>

Developing world models that understand complex physical interactions is essential for advancing robotic planning and simulation.However, existing methods often struggle to accurately model the environment under conditions of data scarcity and complex contact-rich dynamic motion.To address these challenges, we propose ContactGaussian-WM, a differentiable physics-grounded rigid-body world model capable of learning intricate physical laws directly from sparse and contact-rich video sequences.Our framework consists of two core components: (1) a unified Gaussian representation for both visual appearance and collision geometry, and (2) an end-to-end differentiable learning framework that differentiates through a closed-form physics engine to infer physical properties from sparse visual observations.Extensive simulations and real-world evaluations demonstrate that ContactGaussian-WM outperforms state-of-the-art methods in learning complex scenarios, exhibiting robust generalization capabilities.Furthermore, we showcase the practical utility of our framework in downstream applications, including data synthesis and real-time MPC.

开发理解复杂物理交互的世界模型对于推进机器人规划和仿真至关重要。然而，现有方法通常很难在数据稀缺和复杂的接触丰富的动态运动条件下准确地对环境进行建模。为了应对这些挑战，我们提出了 ContactGaussian-WM，这是一种基于物理的可微刚体世界模型，能够直接从稀疏和接触丰富的视频序列中学习复杂的物理定律。我们的框架由两个核心组件组成：（1）视觉外观和碰撞的统一高斯表示（2）端到端可微学习框架，通过封闭式物理引擎进行区分，从稀疏的视觉观察中推断物理属性。广泛的模拟和现实世界评估表明，ContactGaussian-WM 在学习复杂场景方面优于最先进的方法，展现出强大的泛化能力。此外，我们展示了我们的框架在下游应用中的实际效用，包括数据合成和实时 MPC。

</details>

---

## 12. Scaling World Model for Hierarchical Manipulation Policies / 分级操纵策略的扩展世界模型

**Date**: 2026-02-11 | **arXiv**: [2602.10983v2](http://arxiv.org/abs/2602.10983v2) | **PDF**: [Link](http://arxiv.org/pdf/2602.10983v2)

**Categories**: cs.RO

<details><summary><b>Abstract / 摘要</b></summary>

Vision-Language-Action (VLA) models are promising for generalist robot manipulation but remain brittle in out-of-distribution (OOD) settings, especially with limited real-robot data. To resolve the generalization bottleneck, we introduce a hierarchical Vision-Language-Action framework \our{} that leverages the generalization of large-scale pre-trained world model for robust and generalizable VIsual Subgoal TAsk decomposition VISTA. Our hierarchical framework \our{} consists of a world model as the high-level planner and a VLA as the low-level executor. The high-level world model first divides manipulation tasks into subtask sequences with goal images, and the low-level policy follows the textual and visual guidance to generate action sequences. Compared to raw textual goal specification, these synthesized goal images provide visually and physically grounded details for low-level policies, making it feasible to generalize across unseen objects and novel scenarios. We validate both visual goal synthesis and our hierarchical VLA policies in massive out-of-distribution scenarios, and the performance of the same-structured VLA in novel scenarios could boost from 14% to 69% with the guidance generated by the world model. Results demonstrate that our method outperforms previous baselines with a clear margin, particularly in out-of-distribution scenarios. Project page: \href{https://vista-wm.github.io/}{https://vista-wm.github.io}

视觉-语言-动作（VLA）模型对于通用机器人操作很有希望，但在分布外（OOD）设置中仍然很脆弱，尤其是在真实机器人数据有限的情况下。为了解决泛化瓶颈，我们引入了分层视觉-语言-动作框架 \our{}，该框架利用大规模预训练世界模型的泛化来实现稳健且可泛化的视觉子目标任务分解 VISTA。我们的分层框架 \our{} 由作为高级规划器的世界模型和作为低级执行器的 VLA 组成。高层世界模型首先将操作任务划分为具有目标图像的子任务序列，低层策略遵循文本和视觉指导来生成动作序列。与原始文本目标规范相比，这些合成的目标图像为低级策略提供了视觉和物理基础的细节，使得在未见过的物体和新场景中进行泛化成为可能。我们在大规模分布外场景中验证了视觉目标合成和分层 VLA 策略，并且在世界模型生成的指导下，相同结构的 VLA 在新颖场景中的性能可以从 14% 提高到 69%。结果表明，我们的方法明显优于以前的基线，特别是在分布外的情况下。项目页面：\href{https://vista-wm.github.io/}{https://vista-wm.github.io}

</details>

---



</details>

<!-- PAPERS_CONTENT_END -->
