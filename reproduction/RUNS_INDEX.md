# 实验运行索引（Runs Index）

> 检索入口：每一条实验运行的**时间、内容、规模、状态、证据路径、费用**。
> 生成方式：人工核对磁盘与 git 索引，数据截至 **2026-08-19**。
> 配套阅读：[README 首页速览](../README.md#复现结果速览2026-08-19) · [复现 vs 论文对比](PAPER_COMPARISON.md) · [运行手册](reproduction_runbook_2026-08-15.html)

## 命名规范

| 位置 | 约定 | 示例 |
|---|---|---|
| `reproduction/runs/` | `<campaign>-<model>[-n<规模>][-<变体>][-<yyyymmdd>]` | `bird-sql-kimi-k2.6`、`plan-a-kimi-k2.6-n10-20260814` |
| `evaluation/experiments/` | `<scenario>/<dataset>/<model>[_t<g>_p<p>].json` | `combined_independent/bird_sql_n100/kimi-k2.6_t7_g2_p2.json` |
| `evaluation/swe_runs/` | `<model>/`（manifest / results / usage.jsonl） | `kimi-k2.6/` |
| 报告类 | `reproduction/` 根下带日期文件名 | `reproduction_runbook_2026-08-15.html` |

时间戳统一 `yyyymmdd`；本地专属（不入 git）的目录在状态列标 **本地**。

## 正式实验（按时间排序）

| # | 运行 | 日期 | 内容 | 规模 / 设置 | 状态 | 关键证据（✦ = 已入 git） | 费用 (USD) |
|---|---|---|---|---|---|---|---:|
| 1 | `runs/plan-a-kimi-k2.6-n10-20260814/` | 08-14 | GSM8K Plan A：构建 + 4 设置配对评测 | n=10，T1 / T4g1p1 / repeat-T7 / T7g2p2 | ✅ 完成（结论：无判定力） | ✦ `manifest.json`、`summary.{json,html,csv}`、`results/`、`cost_ledger.jsonl` | 见 cost_ledger |
| 2 | `runs/swe-bench-verified-kimi-k2.6/` | 08-15 | SWE-bench Verified 构建（stage1–5 + 选择） | 50 published IDs | ✅ 构建完成 | 本地 `artifacts/stage{1..5}_*.json`、`checkpoints/`、`selection.json`；产物 `final_dataset/swe_bench_verified_final.json`（本地） | 9.85 |
| 3 | `runs/bird-sql-kimi-k2.6/` | 08-15→16 | BIRD-SQL 构建（35 库按需下载） | 100 published IDs | ✅ 构建完成 | 本地 `construction.log`、`bird_data/`、`selected_published_n100.json`；产物 `final_dataset/bird_sql_final.json`（本地） | 22.04 |
| 4 | `evaluation/swe_runs/kimi-k2.6/` | 08-15→17 | SWE 配对评测（Modal sandbox + 官方 verifier） | 50 paired，T1 vs T7g2p2 | ✅ 完成：78% vs 76% | ✦ `manifest.json`、`results/{single,evolve}.json`、`usage.jsonl`；本地 `trajectories/`、`logs/`、逐任务 JSON | 252.58 |
| 5 | `evaluation/experiments/{fully_specified,combined_independent}/bird_sql_n100/` | 08-16→17 | BIRD-SQL 配对评测（本机 SQLite 原生 verifier） | 100 paired，T1 vs T7g2p2 | ✅ 完成：69% vs 45% | ✦ `kimi-k2.6.json`（single 全量）、两份 `*.compact.json`（逐样本精简版）；本地全量 evolve 聚合（241 MB，超 GitHub 上限）与 checkpoint | 8.68 |
| 6 | `runs/browsecomp-plan-a-n100/` | 08-15→18 | BrowseComp+ 构建（Modal detached，三阶段） | 100 IDs：stage1 ✅ / stage2 ✅ / stage3 10/100 | ⏸ 按要求停止，checkpoint 可恢复 | 本地 `stage*.json` 与 checkpoints；明文 query/gold 仅在私有 Volume | 159.95 |

费用合计（已确认，不含未结算 reservation）：约 **453 USD**；逐事件账本均在上述 `usage*.jsonl` ✦。

## 辅助与验证运行（本地保留）

| 目录 | 日期 | 内容 | 说明 |
|---|---|---|---|
| `runs/canary/` | 08-15 | SWE seaborn-3069 金丝雀（v6→v8）+ verifier 通道验证 | 被 `evaluation/swe_bench/modal_app.py` 引用，勿移动 |
| `runs/_archive/plan-a-dry-run-{n20,final}/` | 08-14 | Plan A 干跑（20 样本 / 终版） | 已被正式运行取代 |
| `runs/_archive/review-paired-safety/` | 08-14 | 配对安全性复核 | 一次性检查 |
| `runs/_archive/paper-remaining-kimi-k2.6-20260815/` | 08-15 | 正式运行前 preflight | 一次性检查 |

## 评测基础设施（本地日志，gitignored）

| 位置 | 内容 |
|---|---|
| `evaluation/logs/{bird,browsecomp}/` | 各 benchmark 评测的 tee 日志（`<bench>__<model>__<scenario>.log`） |
| `tmp/` | 会话级运维文件（watcher 日志、哨兵标记、下载暂存） |
| `final_dataset/` | 三个 benchmark 的最终数据集（含 BrowseComp+ 明文，永不入库） |

## 报告与文档（全部 ✦ 入 git）

| 文件 | 内容 |
|---|---|
| [PAPER_COMPARISON.md](PAPER_COMPARISON.md) | 复现 vs 论文逐表逐图对比 |
| [remaining_experiments_report.json](remaining_experiments_report.json) / [.html](remaining_experiments_report.html) | finalizer 汇总：覆盖率 / 准确率 / 费用 / 模型审计 |
| [reproduction_runbook_2026-08-15.html](reproduction_runbook_2026-08-15.html) | 运行手册：不可变条件、验收顺序、故障与修正 |
| [plan_2026-08-14.html](plan_2026-08-14.html) | 最初规划文档 |
| [browsecomp_stopped_progress_2026-08-18.json](browsecomp_stopped_progress_2026-08-18.json) | BrowseComp+ 停止时点的进度与费用快照 |
| [../REPRODUCTION.md](../REPRODUCTION.md) | Plan A GSM8K 协议与结果 |
| [../TRANSPARENCY.md](../TRANSPARENCY.md) | 上游透明度声明 |
