# 三 Benchmark 正式复现流程（5090-only）

本文记录《LLMs Get Lost in Evolving User Intent》中 GSM8K 之外三个 benchmark 的正式复现流程、已发生故障及强制预防措施。它是运行说明，不包含 BrowseComp+ 明文 query、gold documents 或任何密钥。

## 1. 实验范围与当前证据

| Benchmark | 规模 | Single | Evolve | Verifier | 当前状态 |
| --- | ---: | --- | --- | --- | --- |
| BIRD-SQL | 100 paired | T=1, g=0, p=0 | T=7, g=2, p=2 | SQL execution | 完成：69/100 与 45/100 |
| BrowseComp+ | 100 paired | T=1, g=0, p=0 | T=7, g=2, p=2 | gold answer judge | 用户要求提前结束：Stage 1=100、Stage 2=100、Stage 3=10；评测 NOT_RUN |
| SWE-bench Verified | 50 paired | T=1, g=0, p=0 | T=7, g=2, p=2 | 官方 SWE-bench tests | 完成：39/50 与 38/50 |

论文主实验的 Evolve 对话固定包含 2 次 argument reveal、2 次 argument revision 和 2 次 function switch，共 7 轮。BrowseComp+ 每轮最多 50 次搜索；Kimi K2.6 的 SWE 每轮最多 200 次工具调用。构建、agent、judge 和 naturalizer 均锁定 `kimi-k2.6`，`reasoning=medium`。

BIRD-SQL 与 SWE-bench Verified 已完成。BrowseComp+ 唯一 owner 已于 2026-08-18 按用户要求停止，停止后进程数为 0；最后 checkpoint 已核验为 100/100/10，聚合与原子 checkpoint 一致，`safe_to_resume=true`。本次任务不再恢复，paired 评测与 finalizer 均为 `NOT_RUN`。

停止前已完成并通过测试的并发加速实现为一个 coordinator 管理 10 个确定性互斥 shard、每 shard 2 workers；它没有部署到远端，也没有产生模型调用。该实现只作为安全代码进度保留。

5090-1 检索 preflight 已于 2026-08-17 通过：固定 revision 的 4 个 FAISS 分片、Qwen3-Embedding-8B 和 Qwen3-0.6B snippet tokenizer 均已缓存；实际加载得到 100,195 个 4096 维向量和 100,195 篇 corpus 文档。仅用公开测试问题进行 localhost smoke，返回固定的 5 条结果且 retriever revision 匹配；smoke 服务随后已停止，不包含 BrowseComp+ 私密 query。

## 2. 不可变条件

1. 不使用 Modal 计算或 Modal 检索服务；所有后续运行只在已核验的 `fanruibo-5090-1`（hostname `5090-1`）执行。正式启动器拒绝 `5090-2` 及其 SSH 别名。
2. 模型请求只能解析到 `kimi-k2.6`，所有 usage 事件必须同时满足 `requested_model == resolved_model == kimi-k2.6`。
3. 请求中不得发送 `max_tokens`、`max_completion_tokens` 或 `max_output_tokens`。
4. 费用只记录实际 usage，不设置费用 hard cap，不把 reservation 当成已确认费用。
5. 密钥只从本机只读 CC Switch 数据库进入进程内存；不得写入文件、命令参数、日志或 Git。
6. BrowseComp+ 明文 query、gold documents、完整私密构建产物及密钥不得进入 Git。
7. Checkpoint 只按真实 task ID 恢复，每个成功 task 原子写入；不得按数组下标或日志行数推断进度。

## 3. 5090-only 执行顺序

### 3.1 启动前

1. 确认具体节点、hostname、GPU 空闲显存、磁盘和唯一操作者。
2. 同时检查本机进程、SSH 目标进程、`pipeline_state.json` 和 owner PID。只要原 owner 或其 construction worker 仍存活，就停止启动流程；不得为了增加并发再开第二个 pipeline，也不得靠删除 `pipeline.lock` 绕过单 owner 门禁。
3. 确认没有 `repro_lane_keeper`、BrowseComp watcher、旧 evaluation gate 或 Modal task。仓库保留的 `browsecomp_*_modal.py` 是旧入口和共享代码来源，不得执行其中的 `modal run`、`modal deploy`、`.remote()` 或迁移入口；5090 专用环境不得安装 Modal 依赖。
4. 首次迁移时核验 Stage 1=100、Stage 2=100，并记录当时真实的 Stage 3 已提交数；最初安全基线是 Stage 3=0，后续恢复必须接受并验证当前的非零已提交数，绝不能重置回 0。迁移前后记录文件数和 SHA-256 manifest。
5. 在 5090 上安装精确版本依赖，下载固定 revision 的 Qwen3-Embedding-8B、BrowseComp+ corpus 和 FAISS index。
   若节点到 `huggingface.co` 的连接被重置，只使用固定的 `https://hf-mirror.com` 端点；revision 校验保持不变。
   Stage 3 与检索服务共享同一 HF cache，避免重复下载 corpus；FAISS index 使用单独的固定 revision 目录。
6. 运行 preflight：在 5090 上验证模型可用性，并检查模型锁、输出字段静态扫描、manifest 100 IDs、checkpoint policy、GPU retriever 加载和单次公开检索 smoke。本机不发起模型或模型目录请求。
7. 所有 Python 入口都使用仓库 `.venv/bin/python -m <module>`。在付费调用前分别运行 construction、pipeline 和 evaluation module 的 `--help`，并运行 `bash -n evaluation/scripts/run_browsecomp.sh`；不得使用系统裸 `python` 或 `python path/to/runner.py`。

### 3.2 BrowseComp+ Stage 3

1. 只运行 Stage 3 缺失 task ID；Stage 1/2 checkpoint 必须只读复用。
2. 已停止的正式 owner 使用 16 workers 与 1800 秒 read timeout。仓库另保留未部署的 10 shard × 2 workers 实现；各 shard 按固定 manifest 位置取模分工，使用 construction mode、shard 和 task 三层文件锁，只有 coordinator 能在全部 shard 退出后发布有序聚合。
3. 对 `LLMIncompleteResponse` 或未通过完整 verification/independence gate 的候选做最多 8 次完整样本尝试（首次加最多 7 次重试）；只有完全通过 gate 的结果才能写 checkpoint。accounting 错误、模型不一致、checkpoint policy 不一致及其他系统异常立即失败。
4. 每个成功 task 立即原子提交。停止或网络中断后，只从已提交 task ID 集合恢复。
5. 完成后运行 construction audit，必须得到 100/100/100、固定 ID 顺序和 predecessor independence verification 通过。

不要只用“Stage 3 checkpoint 暂未增长”判断停滞。按当前固定协议，一个完全无重试的样本也至少包含 3 次 predecessor 生成、6 次 similarity judge、4 次 cross-turn judge 和 6 次 functional-independence answer，共 19 次 Kimi 调用；答案不直接相等时还会增加 independence judge，gate rejection 还会触发整链重生成，总计最多 8 次完整样本尝试。16 个 worker 同时推进时，首批链提交前常有约 `16 x 19 = 304` 次调用仍在途中。健康判断必须同时看 owner/worker 存活、usage 时间戳继续更新和活跃连接；只有这些也停止后才按故障处理。

### 3.2.1 Stage 3 安全恢复协议

1. 先记录 owner PID、construction PID、状态文件、checkpoint 数和 usage 文件末次更新时间。进程仍活跃或连接仍有流量时只观察，不强杀、不覆盖状态、不启动第二批。
2. 只有 owner 与 worker 均已退出后，才确认 `pipeline_state.json` 为 `failed` 或存在可解释的非零退出状态；等待 usage 文件停止写入，再开始只读检查。
3. 用 5090 construction 的 `--inspect-only` 检查固定 manifest、Stage 1/2 完整覆盖、当前 Stage 3 task ID 集合、每个 checkpoint 的 `input_sha256`、完整 policy/source hash、结果 gate 和聚合顺序。文件数量本身不是恢复依据。
4. 检查失败时不得删除 checkpoint 或手改 policy。只有输入 hash 不变、差异仅为经过审计的 source bundle hash 时，才允许使用专门迁移；其他差异视为新实验，不能混入当前正式 run。
5. 检查通过后只手动启动一个 owner。恢复入口必须从已提交 task ID 集合补缺；不得配置 watcher、keeper、cron 或自动 relaunch。锁文件可保留，是否有活 owner 由非阻塞文件锁判断，不由文件是否存在判断。
6. `None`、`verification_passed != true` 或 `independence_passed != true` 都是候选 rejection：每次都从原 Stage 2 输入重新生成完整 predecessor chain，总计最多 8 次完整样本尝试，不复用被拒绝的半条 chain。重试耗尽后让当前 pipeline 明确失败并保留其他已提交 checkpoint；accounting、模型锁、policy 和其他系统异常不进入此重试。

### 3.3 BrowseComp+ paired 评测

1. 同一 5090 进程内加载固定 revision 的 Qwen3-Embedding-8B 与 FAISS index，不使用 Modal HTTP endpoint。
2. 先跑 Single 100，再跑 Evolve 100；每个 task 结果原子 checkpoint，重启只补缺失 task。
3. Agent、naturalizer 和 judge 均使用 `kimi-k2.6`、`reasoning=medium`；每轮搜索上限 50。
4. 两组均须覆盖固定 100 IDs；任何失败 task、空 prediction、缺 response 或模型审计异常都不计完成。
5. Pipeline 必须把 retriever URL 固定为同机 `127.0.0.1`，并在启动 retriever 子进程前清除模型凭据。不得直接调用通用 evaluation runner 指向外部或旧 Modal endpoint。

### 3.4 收尾

1. 正常完成时，远端 `pipeline_state.json` 必须先为 `complete`，然后才允许拉回。本次因用户提前结束，状态是可解释的 `failed`（SIGTERM），所以没有拉回私密构建产物，只导出脱敏的 100/100/10、usage 和 checkpoint 审计汇总。
2. 运行 `reproduction.finalize_remaining_experiments`，要求状态为 `complete`，覆盖 BIRD 100/100、BrowseComp+ 100/100、SWE 50/50。
3. 运行全量 pytest、静态输出上限扫描、模型审计和覆盖待推送历史的 secret scan。只在精确暂存候选安全产物后，用同一环境运行 `.venv/bin/python -m reproduction.audit_safe_export`。该审计只读取当时的 Git index，不读取未暂存工作区或历史；任何 `git add`、报告重生成或 amend 之后都必须重跑。最终一次必须证明 BrowseComp+ 明文 query、gold document 片段、实际 CC Switch key 和原始 BrowseComp 结果路径均未进入提交。
4. 更新 `remaining_experiments_report.json` 与 HTML，明确实际准确率、退化、usage、耗时，以及 BrowseComp+ 的 `stopped_incomplete` / `NOT_RUN` 边界。
5. 仅提交代码、测试、固定配置、安全 manifest、聚合 JSON/HTML 和允许公开的 compact 证据。
6. 推送 private `main` 到 `RRiiiccckkk/evolving-intent-reproduction` 后，停止复现心跳。

## 4. 已发生故障与防复发规则

| 故障 | 根因 | 防复发规则 |
| --- | --- | --- |
| 输出为空或 JSON 截断 | 客户端输出 token 上限 | 禁止三个 `max_*tokens` 字段；静态测试验证实际 payload |
| 中断后重复计费或跳样本 | 批次末保存、按数组下标恢复 | 每 task 原子 checkpoint；按 task ID 集合恢复 |
| Stage 3 长调用频繁 600 秒超时 | predecessor chain 是长生成，不是全局 API 拥堵 | read timeout 1800 秒；单 owner 内 worker=16；incomplete response 有限重试 |
| 单个 Stage 3 候选未通过 independence 后取消整批 | 核心 executor 把候选 rejection 当作 stage 系统失败 | wrapper 总计最多 8 次完整样本尝试（首次加最多 7 次重试）；不降低 gate，不提交失败候选；系统异常仍立即失败 |
| Pipeline 状态显示 16 workers、wrapper 实际仍跑 12 | launcher 与 Stage 3 wrapper 各自维护 worker 常量 | construction CLI、pipeline 和 wrapper 共用同一个 `STAGE3_WORKERS=16`；测试断言命令与运行常量一致，正式启动后核对日志中的实际 workers |
| 评测入口找不到 `intent_construction` | 用文件路径执行 runner 时，Python 只把 `evaluation/runners` 放进模块搜索路径 | 固定使用仓库虚拟环境并以 `python -m evaluation.runners.run_browsecomp_experiment` 启动；远端预检必须先通过 `--help` |
| Owner 仍活跃时从旧 checkpoint 再开一批 | 误把 checkpoint 暂停增长或状态文件滞后当成 owner 已死 | 同时核验 owner/worker PID、usage 更新时间、连接和非阻塞锁；活跃时只观察，退出后 inspect-only，通过后只手动恢复一个 owner |
| Content filter 中断 chain | 生成上下文被判高风险 | 当前样本重新生成 chain，不提交半成品 checkpoint |
| Checkpoint policy 不匹配 | 源码变化导致 source bundle 指纹变化 | 只允许审计明确的 source hash 迁移；输入 hash 或其他 policy 变化立即拒绝 |
| Modal app 被停止后又重启 | 多个 watcher、keeper 和会话 cron 同时拥有重启权 | 5090-only 模式不设自动 redeploy；唯一 owner + PID/lock；接管先停旧 owner |
| 截止后仍产生 Kimi 调用 | 旧 keeper 在另一会话重新拉起 app | 启动前扫描本机进程和远端任务；检测到第二 owner 立即停止，不自动恢复 |
| 非交互 SSH 使用系统 Python | 评测脚本调用裸 `python`，Stage 3 完成后才暴露缺失依赖 | 固定使用仓库 `.venv/bin/python`，启动前验证可执行，并用回归测试禁止退回裸 `python` |
| SWE 基础设施失败仍继续调用模型 | transport error 被当成工具观察 | `returncode=-1` 或执行异常立即终止 task，不继续喂给 agent |
| BIRD gold SQL 无法执行 | 构造 SQL 的 int64 溢出或超时 | 按仓库比较语义判 incorrect，并记录 `gold_error`，不得把 verifier 异常伪装成正确 |
| 拉回后的 usage 副本被重复计费 | canonical 账本与 `usage/` 原账本内容相同但 inode 不同 | finalizer 按完整内容 SHA-256 去重；复制账本回归测试必须保持调用数与费用不变 |
| 安全审计在暂存前运行 | 审计的是旧 Git index，新产物随后仍可能泄露 | 精确暂存后审计；每次修改 index 都重跑，并另做待推送历史 secret scan |
| 费用统计错误 | reservation 与 confirmed usage 混算 | confirmed、excluded、reservation 分列；不设费用停止门槛 |

## 5. 验收证据

实验完成的最低证据同时包括：固定样本覆盖、原生 verifier 结果、逐 task checkpoint、唯一模型 usage 审计、无输出上限 payload 测试、secret scan、私密数据排除、全量测试、最终报告以及 GitHub private `main` 的远端 commit。缺任何一项都不得称为“完成复现”。
