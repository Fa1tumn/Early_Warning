# 研究执行 TODO List

> 配合 `论文草案.md` 使用，按顺序推进。每项标注是否需要导师确认。

---

## 阶段 0：方向确认（优先做，避免返工）

- [ ] **找导师确认选题定位**：按之前想好的话术，确认"self-improving agent 只是叙事框架挂靠，核心还是循环预警"这个理解是否正确，还是他想要更大幅度地转向 self-evolving/权重级方向
- [ ] 确认导师承诺的 "ai-native code engineering" repo 是否还有其他要发的，还是就是那个 HuggingFace 排行榜
- [ ] 确认实验室的算力/API 预算量级，这决定第 3 步任务集规模

---

## 阶段 1：技术可行性摸底（1-3 天，零成本高优先级）

- [x] **验证 LangGraph/AutoGen 能否拿到 token-level logprobs**
  - 结论：本机（8GB显存）用 HF transformers 本地跑 Qwen2.5-1.5B-Instruct，`model.generate(output_scores=True)` + LangGraph 单节点 state graph 可以完整拿到每个生成 token 的 logprob 和熵，非阉割版、非 top-N 近似。代码：`code/stage1_logprobs/probe_logprobs.py`
  - 换到4090机器只需替换 MODEL_NAME 为更大模型（如 Qwen2.5-14B-Instruct），其余代码不变
- [x] **下载并跑通 MAST-Data（已确定为主数据集）**（HuggingFace `mcemri/MAD`，文件 `MAD_full_dataset.json`）
  - 已探明数据结构：1242 条记录，`mast_annotation` 字段是 MAST 14类失败模式的 0/1 标注，编号 "1.3" = Step Repetition（已用 MAST 论文 arXiv:2503.13657 核实）
  - 筛出 451 条 Step Repetition 轨迹（占 36.3%），已存前20条样例到 `code/stage1_mast_data/step_repetition_samples.json`
  - 另有 286 条无任何失败标注的"干净"轨迹，可作对照参考
- [x] **下载 TRAIL benchmark（已确定为辅助验证集）**，读几条 span 级错误标注样例，确认字段是否包含你需要的时间/位置信息
  - 真实仓库 `PatronusAI/TRAIL`（gated，已申请通过），GAIA 117条 + SWE-Bench 31条轨迹，各自带 span 级 processed_annotations
  - 原始轨迹是完整 OpenTelemetry span 结构（`timestamp`、`duration`、`span_id`/`parent_span_id`/`child_spans` 层级），错误标注通过 `location`（span_id）精确指回具体 span —— **时间/位置信息齐全，满足精细时序分析需求**
  - 116 份标注（1份JSON格式损坏已跳过）累计 580 条错误，前3类：Formatting Errors(124)、Instruction Non-compliance(64)、Goal Deviation(61)；TRAIL 的分类体系与 MAST 不同，不含"Step Repetition"这个具体标签，交叉验证时需要按语义（如 Goal Deviation / Task Orchestration）映射到循环行为，而非直接按标签筛选
  - 代码：`code/stage1_trail/download_and_inspect.py`
- [ ] Who&When / TraceElephant / AgentPex 暂不下载，已列入"暂不使用"后备注记录，不重复投入时间

---

## 阶段 2：核心假设验证（存在性证明，不涉及对照实验）

- [ ] 在主数据集 MAST-Data 中筛选"循环/重复"+"memory 相关错误"的轨迹（优先），再用 TRAIL 作交叉验证
- [ ] 人工标注：循环真正开始的那一步（ground truth 起点）
- [ ] 分析循环起点前 N 步，是否存在可识别的异常模式（可以先用简单统计量，比如动作重复率、embedding 相似度趋势）
- [ ] **产出判断**：如果现有数据集里就能看出"循环前有征兆"，说明假设成立，可以继续；如果看不出来，需要重新设计自建轨迹采集方案

---

## 阶段 3：自建轨迹采集（现有数据集验证通过后再做）

- [ ] 用 LangGraph 或 AutoGen 选定 1-2 个框架，跑通一个最小 Agent（不追求复杂，先打通流程）
- [ ] 在 GAIA 或 ALFWorld 中选 10-20 个已知容易触发循环的任务作为种子任务
- [ ] 采集每一步的：工具调用序列、输出 embedding、token 级熵
- [ ] 人工标注哪些轨迹真的发生了循环，标出起始点

---

## 阶段 4：信号分析与关联性验证

- [ ] 检验熵/语义漂移信号是否在循环起始点前出现异常（核心实验）
- [ ] 计算"提前量"（领先多少步/token）
- [ ] 尝试把预警信号定位到具体的 Memory 条目，验证关联性是否成立
- [ ] 人工抽样核验 Memory 定位是否准确

---

## 阶段 5：干预实验（探索性，非对照）

- [ ] 设计一个最简单的干预动作（如临时屏蔽某工具、插入重新聚焦提示）
- [ ] 触发预警后执行干预，观察是否规避了循环
- [ ] 用 witcheer/agentic-score-leaderboard 的 "Reality Anchor" 思路：拿一批人工标注的真实循环/非循环轨迹做交叉校验，避免自建指标"测不出真实差异"

---

## 阶段 6：写作与整理

- [ ] 把阶段 2-5 的结果补充进 `论文草案.md` 对应章节
- [ ] 整理图表（信号轨迹图、提前量分布图等）
- [ ] 精简 Related Work，按最终实验结果调整贡献点表述
- [ ] 再次找导师过一遍完整草稿

---

## 当前所在位置

👉 阶段 1 三项技术摸底（logprobs 可行性、MAST-Data、TRAIL）已全部跑通，详见上方各条目。
**下一步：阶段 0 的导师确认**（选题定位 + repo确认 + 算力预算，其中算力已初步确定为本地4090部署）仍需完成；确认后即可进入阶段 2 核心假设验证。
