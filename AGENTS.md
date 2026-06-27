# 项目协作记录

## 固定协作规则

- 回答与项目文档尽量使用中文。
- 在当前工作目录内工作，不参考其他目录的项目代码。
- 新项目需要维护 `AGENTS.md`，每次任务结束同步更新当前状态。
- 如当前目录是 GitHub 仓库，任务结束后按用户要求同步上传；当前目录已初始化为 Git 仓库，远程目标为 GitHub 公开仓库 `coinrainy/Auto_Research`。
- 每轮任务结束时给出建议的后续命令。

## 当前研究目标

- 目标：在 2026 年投稿/发表一篇图对比学习方向的顶会或顶刊论文。
- 主要任务：节点分类。
- 当前状态：尚无稳定研究方向，也没有明确具体 idea。
- 当前工作流：`academic-research-suite` -> `deep-research` -> `socratic` mode。

## 本轮任务记录

- 日期：2026-06-27（UTC）。
- 阶段：Socratic Layer 1 已完成第一轮方向收敛，进入候选 research question 初筛。
- 目标：围绕 Graph Contrastive Learning / Graph Self-Supervised Learning 的节点分类算法论文，聚焦有限算力下可实验验证的贡献。
- 用户约束：单卡 RTX 3060 12GB；优先 Cora/Citeseer/PubMed 与小中型 heterophily 数据集；不做大规模 OGB 主线；重点关注语义保持增强、可靠正负样本构造、假负/假正/难负样本不均衡。
- 当前输出：形成 3 个候选方向：
  1. 语义保持的自适应图增强 / 视图构造；
  2. reliability-aware 正负样本对构造；
  3. 图增强与样本对可靠性的联合校准 / curriculum 框架。
- 初步判断：方向 1 和方向 3 更适合作为 2026 年投稿目标；方向 2 文献拥挤度更高，需要更明确地区分 ProGCL、PMGCL 等正负样本挖掘工作。
- 用户选择：RQ3，即“图增强与样本对可靠性的联合校准 / curriculum 框架”作为主方向。
- 当前 Socratic 进展：[INSIGHT] 用户倾向将语义保持增强与正负样本可靠性放入同一个闭环算法框架，而不是只单独改 augmentation 或 mining。
- Layer 2 方法假设更新：
  - [INSIGHT] view semantic consistency 不依赖单一信号，而采用特征相似度、结构/多跳一致性、teacher-student embedding stability、局部 homophily/heterophily 等组合信号。
  - [INSIGHT] 联合校准机制采用交替更新闭环：保守增强 -> 估计 positive/negative pair reliability -> 加权/过滤 contrastive loss -> reliability 反向调节下一轮 augmentation 强度。
  - [INSIGHT] 方法倾向保留负样本，因此需要显式处理 false negative 与 hard negative imbalance。
  - [INSIGHT] homophilic 与 heterophilic 局部区域不使用完全统一的增强策略，而是根据局部可靠性自动调节 edge dropping、feature masking、hard negative 强度。
  - [INSIGHT] reliability score 的 3 个主信号暂定为 teacher-student embedding stability、cross-view prediction consistency、high-pass/low-pass response difference；feature similarity 与 structural proximity 暂作辅助分析。
  - [INSIGHT] reliability score 的主作用是加权 contrastive loss，同时可辅助低可靠样本过滤与 augmentation 强度调节。
  - [INSIGHT] 用户主动识别的主要风险包括 heuristic 质疑、额外超参数导致的伪提升，以及交替更新造成错误累积。
  - [INSIGHT] 若方法主要在 heterophily 数据集提升、在 Cora/Citeseer/PubMed 上保持接近 baseline，用户仍认为方向成立。
- Devil's Advocate Checkpoint 1 初步压力测试：主要 Major issue 是 reliability score 的三个主信号均可能依赖模型自身预测，存在 self-confirming loop；需通过 warm-up、stop-gradient/EMA、更新频率控制、ablation 与错误累积诊断实验解决。
- Layer 3 证据设计更新：
  - [INSIGHT] 核心证据不应只是 heterophily accuracy 提升，而应同时满足性能提升与机制诊断成立。
  - [INSIGHT] 关键机制证据包括 heterophily 节点分类性能提升、weighted false negative mass 下降、high-reliability positive pair 的跨视图一致性更高；embedding clustering 稳定性仅作为辅助证据。
  - [INSIGHT] 明确失败判据包括 shuffled reliability 也有效、closed-loop 不如 one-shot、augmentation control only 或 loss weighting only 接近完整方法、去掉 high/low-pass gate 不影响 heterophily、homophily 数据集明显退化。
  - [INSIGHT] 最关键 ablation 是 random/shuffled reliability score：保留分布、loss 形式、augmentation schedule 与模型结构，仅打乱 score 和节点/pair 对应关系。
  - [INSIGHT] high-pass/low-pass response difference 不直接并入 pair reliability，而作为局部图类型 context gate，用于解释 stability/consistency 信号并调节 augmentation strength。
- 下一步：进入 Socratic Layer 4，围绕 self-confirming loop、复杂度/超参数、homophily 退化风险、理论叙事边界与 reviewer 攻击面进行 critical self-examination。
- Layer 4 批判性自审更新：
  - [INSIGHT] 若 reviewer 质疑模块过多，用户愿意优先砍掉或降级 curriculum 与 augmentation closed-loop，而不是 pair reliability 本身。
  - [INSIGHT] 最小不可砍贡献单元是基于 embedding stability 与 cross-view prediction consistency 的 pair reliability weighted contrastive loss。
  - [INSIGHT] high/low-pass gate、augmentation strength control、curriculum schedule 与 hard filtering 可作为扩展模块、辅助门控或附录实验，而非主 reliability 定义。
  - [INSIGHT] 用户不再强行坚持交替闭环必然必要，而是将 two-stage pipeline 设为强 baseline，再用实验判断 closed-loop 是否带来额外收益。
  - [INSIGHT] 保守 contribution statement 收缩为：提出轻量 reliability-aware contrastive calibration 框架，通过机制诊断证明其在异配图上减少错误对比信号，同时在同配图上保持接近现有 GCL baseline 的表现。
- [INSIGHT] closed-loop 必要性将用 2x2 因子消融检验：fixed reliability vs EMA slow update，fixed augmentation vs reliability-guided augmentation，对应 two-stage fixed reliability、dynamic reliability only、augmentation feedback only、full closed-loop 四个版本。
- [INSIGHT] 若 closed-loop 不优于 two-stage，用户愿意将主方法收缩为 two-stage reliability-weighted GCL，并把 closed-loop 作为 optional refinement、negative result 或 discussion。
- [INSIGHT] 主方法不依赖 high/low-pass gate 也成立；该 gate 应作为辅助解释和可选扩展，标题与摘要应避免把它包装成核心贡献。
- [INSIGHT] 若仅 3/7 个 heterophily 数据集提升，但 shuffled reliability 与 false negative mass 诊断支持机制，贡献应定位为机制性贡献 + 条件性有效，而非全面 SOTA。
- 下一步：进入 Socratic Layer 5，明确论文的“so what”、目标读者、贡献表述、投稿定位与最小可发表单位。
- Layer 5 贡献定位更新：
  - [INSIGHT] 一句话贡献：在 GCL 中并非所有增强视图和正负样本对都同等可靠；pair reliability 可在无标签条件下由跨视图 embedding stability 与 prediction consistency 估计，并用于加权 contrastive loss，从而诊断并削弱错误正样本对齐与疑似 false negative。
  - [INSIGHT] 论文定位为“机制分析 + 方法论文”，而不是单纯 SOTA 方法论文或完全 heterophily 专项论文。
  - [INSIGHT] 论文主线结构建议：Problem diagnosis -> Method -> Mechanism evidence -> Performance evidence -> Optional extension。
  - [INSIGHT] 投稿策略倾向稳妥路线：若结果显著强于预期再冲更高层级；默认面向图学习/机器学习应用友好的会议或期刊。
  - [INSIGHT] 最小可发表实验单元包括 node classification、shuffled reliability、weighted false negative mass、view consistency、reliability ablation、homophily non-degradation。
  - [INSIGHT] 2x2 closed-loop ablation、high/low-pass gate 分析、超参数敏感性、聚类可视化、runtime/复杂度分析可根据结果进入附录或主文补充。
- 当前推荐主 RQ：在无标签节点分类的 Graph Contrastive Learning 中，能否通过跨视图 embedding stability 与 prediction consistency 估计 pair reliability，并用 reliability-weighted contrastive loss 削弱不可靠正样本与疑似 false negative 的影响，从而在部分 heterophily graphs 上提升稳健性，同时不显著损害 homophily graphs 上的性能？
- 下一步执行建议：进入 experiment-agent 或手动实验规划阶段，先实现 two-stage reliability-weighted GCL 最小原型，再做 shuffled reliability 与 false negative mass 诊断。
- GitHub 上传记录：已将当前目录初始化为 Git 仓库，创建并绑定 GitHub 仓库 `coinrainy/Auto_Research`，已推送 `main` 分支；2026-06-27 已按用户要求将仓库可见性改为公开。
- 2026-06-27 后续路线更新：
  - 当前应切换到 `academic-research-suite` -> `experiment-agent` -> `plan` 阶段，而不是继续扩大选题或直接写论文。
  - 第 1 步：搭建可复现实验仓库骨架，固定数据集加载、训练入口、配置文件、日志与结果表格式。
  - 第 2 步：先复现 2-3 个轻量 baseline（建议 GRACE、BGRL、CCA-SSG 或 DGI），确认 Cora/CiteSeer/PubMed 与至少 3 个 heterophily 数据集能稳定跑通。
  - 第 3 步：实现 two-stage reliability-weighted GCL 最小原型，只包含 warm-up、embedding stability、prediction consistency、reliability-weighted InfoNCE。
  - 第 4 步：优先实现 shuffled reliability、weighted false negative mass、view consistency 三个机制诊断，而不是先堆更多模型模块。
  - 第 5 步：若最小原型在机制诊断上成立，再决定是否加入 high/low-pass context gate 与 closed-loop augmentation。
  - 当前不建议事项：暂不写完整论文大纲；暂不冲大规模 OGB；暂不把 closed-loop、curriculum、high/low-pass gate 全部放进主方法。
