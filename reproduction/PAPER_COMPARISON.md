# 复现结果与原论文对比（Reproduction vs. Paper）

- 论文：*LLMs Get Lost in Evolving User Intent*（arXiv 2607.20734，microsoft/evolving-intent）
- 复现模型：Kimi K2.6（构建/评测/judge/naturalizer 全程锁定，reasoning=medium）
- 本文档日期：2026-08-19；结论口径基于当时的本地证据文件
- 证据：[BIRD/SWE 汇总](remaining_experiments_report.json)、[GSM8K summary](runs/plan-a-kimi-k2.6-n10-20260814/summary.json)、[BrowseComp+ 停止状态](browsecomp_stopped_progress_2026-08-18.json)

## 一句话定位

现象级部分复现：论文最核心的主张（Single → Evolve 的 "getting lost" 退化）在 BIRD-SQL 上强复现且幅度超出论文全部模型，在 SWE-bench Verified 上方向一致但幅度远小于论文同模型数值；GSM8K 因 n=10 无判定力；BrowseComp+ 无数据。论文的全部机制性分析（场景分解、recap 缓解、逐轮 intent tracking）零覆盖。

## 1. 主结果逐项对照

论文 Table 1 报告了 9 个模型的四域成绩。下表把复现的 Kimi K2.6 结果对到论文同一行，并给出论文 9 个模型的相对退化区间作参照（括号为相对变化）：

| 域 | 论文 K2.6 Single→Evolve | 论文 9 模型退化区间 | 复现 Single→Evolve | 判定 |
|---|---|---|---|---|
| GSM8K（论文 n=200 / 复现 n=10） | 96.5 → 79.5（-17.6%） | -16.2% ~ -23.0% | 90.0 → 90.0（0.0%） | 无法判定（样本不足） |
| BIRD-SQL（双方 n=100） | 75.0 → 68.0（-9.3%） | -4.0% ~ -30.3% | 69.0 → 45.0（**-34.8%**） | **方向复现，幅度越过论文最差值** |
| BrowseComp+（论文 n=100） | 55.0 → 52.0（-5.5%） | -5.5% ~ -70.6% | NOT_RUN | 无数据 |
| SWE-Bench Verified（双方 n=50） | 86.0 → 72.0（-16.3%） | +0.0% ~ -100% | 78.0 → 76.0（-2.6%） | 方向复现，幅度落在论文跨模型离散范围内 |

## 2. 相同点（论文语言 + 复现证据）

1. **核心现象在两个正式域都成立。** 论文摘要的中心命题是 "strong static-setting performance does not transfer to the evolving-intent setting, with substantial drops across model families"。BIRD-SQL（-24pp / -34.8% 相对）是这句话的强证据；SWE 的 -2pp 同向。论文称之为 "LLMs get lost"——在 BIRD evolve 中，模型在 6 次 intent transition 后丢失了超过三分之一的可解任务。
2. **协议要素严格对齐。** 论文 Table 1 定义的正式设置——"Evolve applies each transition type twice, yielding six intent transitions per dialogue (seven turns including the initial turn)"——与复现一致（T=7, g=2, p=2）。其余对齐项：medium reasoning（论文："All models are run at default reasoning for each turn (medium, where applicable)"）；native verifier（BIRD 用 SQL 执行集合比较、SWE 用官方 harness 的 FAIL_TO_PASS+PASS_TO_PASS，无 LLM judge）；SWE 的 mini-SWE-agent v2 脚手架；SWE 每轮工具预算 200（论文专门注明 "We raised the per-turn budget to 200 for Kimi K2.6 and DeepSeek V3.2"）；BrowseComp+ 每轮 50 次搜索上限。
3. **SWE 结果落在论文的跨模型离散范围内。** 论文 SWE 列是全表方差最大的域（DeepSeek V3.2 +0.0%、Gemini 3.1 Pro -2.3%，而 GPT 5.1 / Grok 4.20 / Mistral Large 3 直接 -100% 归零）。论文将其归为 "an extreme degradation regime"——部分 agent "easily exhaust the tool-call budget, lingering in extended thinking and timing out"。复现的 -2.6% 与 Gemini/DeepSeek 同档：本复现的 K2.6 agent 未落入预算耗尽模式；与论文 K2.6 的点估计（-16.3%）不符，但未超出论文自身的模型间分布。

## 3. 不同点

1. **域间退化模式与论文倒挂。** 论文的模式是 "degradation is particularly large in Search and SWE"，BIRD 相对温和（9 个模型中 7 个在 -4%~-11%）。复现恰好相反：BIRD 最重（-34.8%，大于论文最差的 DeepSeek V3.2 -30.3%），SWE 最轻。因此当前数据不支持论文 "跨域一致的 substantial drops" 这一表述——两个完成的域一强一弱。
2. **BIRD 绝对值全面偏低，且 Evolve 侧偏离更大。** Single 低 6pp（69 vs 75），Evolve 低 23pp（45 vs 68）。最大的混杂变量是**构建模型不同**：论文明确构建（intent 抽取、counterfactual/predecessor 生成）"all run with GPT 5.1"，本复现按既定要求全程用 kimi-k2.6 构建。构建模型既影响 single 侧难度（partial gold 的自然度），也影响 evolve 侧的可跟踪性，因此 -34.8% 不能直接读作"比论文更严重"——它部分是数据集构造差异。次要已知偏差：700 个 turn 中 3 个构造 gold 无法在 SQLite 执行（int64 溢出/超时），按仓库比较语义判 incorrect 并落 `gold_error` 审计字段，最多贡献 3/700 的偏负（逐条见 experiments/*/bird_sql_n100/*.compact.json）。
3. **GSM8K 与论文所有模型不一致，但无判定力。** 论文 9 个模型全部 -16%~-23%；复现 Full Evolve 为 +0.0%。n=10 的 Wilson 95% 区间（90%: 59.6–98.2；80%: 49.0–94.3）彼此重叠，论文使用 n=200。结论只能是"未能确认"，不是"证伪"。
4. **论文的机制层结论全部未检验。** 对照论文的研究问题：RQ1（evolving vs single-turn，Table 1）交了 2/4 个域的部分答案；RQ2（transition 的类型/数量/组合/顺序——Figure 4 "Scaling intent transitions monotonically degrades accuracy"、Table 2 "function switch consistently emerges as the most challenging transition type"、Figure 5 "LLMs gradually shift away from earlier context after function switches"）零覆盖——场景表只跑了两端（Single 与 Reveal+Revise+Switch），单独的 reveal/revise/switch 及两两组合均无数据；RQ3（Figure 6 的 prompt recap / oracle recap 缓解，以及 "even with oracle recap, all scenarios still fall short of single-turn accuracy"）零覆盖；Table 3 的逐轮 intent tracking（reveal 99→98、switch 89→82 的塌陷梯度）与 Figure 7 的难度复合效应同样零覆盖。论文对 SWE 的机制解释——"the accumulated interaction context and tool traces can themselves become distractors"（预算耗在 grep/sed 等探索而非执行）——轨迹数据在手但尚未做该分析。

## 4. 复现定位（最终口径）

- **成立**：论文中心现象（fully-specified → evolving-intent 的显著退化）在 BIRD-SQL 上以超论文幅度复现，在 SWE 上方向性复现；实验协议（场景定义、预算、verifier、reasoning 档位）逐项对齐。
- **不成立/未定**：跨域一致性（域间模式倒挂）、GSM8K 退化（样本不足）、BrowseComp+（未跑）；论文全部机制性归因（function switch 主导、组合放大、recap 部分恢复、intent tracking 塌陷）无证据。
- **归因注意**：唯一记录在案的方法差异是构建模型（GPT 5.1 → kimi-k2.6），它是 BIRD 幅度偏大与域间模式倒挂的第一嫌疑变量，报告时应作为 caveat 写明，而非归结为模型行为差异。

按当前证据强度，准确描述是：**论文 Table 1 在 2/4 域、论文机制分析 0/5 项的部分复现**——比"四 benchmark 正式复现"少，但比"初步验证"多：BIRD 的 -34.8% 本身已是一条可独立成立的强结果。
