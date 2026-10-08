# 第一阶段：校园网 H100 执行手册

这份手册对应 `codex/case-relevance-phase1` 分支的正式查询冻结、候选收集、人工标注与评测流程。
在校园网服务器终端执行 Git 拉取和 GPU 命令；本地开发端不需要连接服务器 SSH。
已有环境、历史数据、索引和两套权重直接复用。以下命令不下载模型、不创建环境、不重编码历史向量。

所有查询、审核、标签、完整预检、运行记录和诊断放在 Git 忽略的 `data/` 下。
不要把真实投诉或地址写入命令行；用私有 JSON/JSONL/TSV 文件传入。
每次运行使用新编号，保留失败目录和冻结版本。本手册按同一个 Bash 会话分步执行。

## 1. 拉取开发分支并确认实际环境

进入服务器已有仓库后执行：

```bash
git status --short
git fetch origin codex/case-relevance-phase1
git switch codex/case-relevance-phase1
git pull --ff-only origin codex/case-relevance-phase1
git rev-parse HEAD
conda env list
umask 077
test -e deploy/.env.retrieval || cp deploy/.env.retrieval.example deploy/.env.retrieval
chmod 600 deploy/.env.retrieval
```

若本地尚无开发分支，`git switch` 会自动尝试跟踪同名远端分支；有未提交修改或分叉时先处理，
不要用 `reset --hard` 覆盖服务器已有工作。应当在正式冻结前拉取最终代码；正式冻结要求相关源码已提交且与记录的 HEAD 一致，冻结后相关源码改变会阻止收集。

用服务器编辑器检查私有 `deploy/.env.retrieval`。仓库默认环境名是 `civic-rag-retrieval`；
如果 `conda env list` 的实际名称是 `civi-rag-retrieval`，就把 `CONDA_RETRIEVAL_ENV` 改为该实际名称。
不根据口头简称改环境名，不另建一套环境。脚本会 `source deploy/.env.retrieval`，因此文件中已设置的值
会覆盖调用脚本前同名的 shell 环境变量；需要覆盖时统一修改这份私有文件。

核对 `RETRIEVAL_DATASET`、`RETRIEVAL_LEXICAL_INDEX`、`RETRIEVAL_DENSE_INDEX`、
`RETRIEVAL_MODEL_PATH` 和 `RERANKER_MODEL_PATH` 指向现有完整资源。
设备使用分配到的 H100，`RETRIEVAL_DEVICE=cuda`；dtype、embedding max_length 必须与已有 dense manifest 相容，
不要为绕过检查随意改 dtype 或删除 manifest。默认检索参数为 hybrid、case_k=50、max_terms=32。

本阶段建议将 `RETRIEVAL_ADDRESS_INDEX` 设置为新的 `data/case-relevance-phase1-v1/address-index-v1`。
当前地址解析源码与旧索引指纹不同的时候，只重建地址索引；旧目录保留。

```bash
source deploy/.env.retrieval
export CONDA_RETRIEVAL_ENV
PHASE1_ROOT=data/case-relevance-phase1-v1
mkdir -p "$PHASE1_ROOT"
bash deploy/inspect-retrieval-env.sh
conda run --no-capture-output -n "$CONDA_RETRIEVAL_ENV" python -c 'import sys; print(sys.executable); print(sys.version)'
bash deploy/check-reranker-env.sh
conda run --no-capture-output -n "$CONDA_RETRIEVAL_ENV" python -m retrieval_baseline.prepare_model \
  --output "$RETRIEVAL_MODEL_PATH" --verify-only
conda run --no-capture-output -n "$CONDA_RETRIEVAL_ENV" python -m retrieval_baseline.prepare_reranker \
  --output "$RERANKER_MODEL_PATH" --verify-only
bash deploy/run-case-search.sh prepare-address
```

`prepare-address` 需要输出目录尚不存在。如果已有该版本的完整索引，不重复运行；预检将核对其内容与源码。
`inspect-retrieval-env.sh` 会生成只读诊断文本到私有 `data/` 目录，列出指定环境的包版本和关键导入结果。
如果环境检查失败，先根据诊断定位，不要直接升级或卸载现有包。单独模型文件检查不等于模型能够联合推理。

## 2. 静态预检和真实联合推理

```bash
static_status=0
bash deploy/run-phase1.sh preflight --output "$PHASE1_ROOT/preflight-static-001.json" || static_status=$?
printf 'preflight_exit_code=%s\n' "$static_status"
```

预检退出码：`0` 表示完整联合预检通过；`1` 表示文件、环境或演练失败；`2` 表示静态检查通过、推理尚未运行。
不带 `--smoke-queries` 永远不会形成正式放行证据。缺文件或依赖时 `status=failed`，`inference.status=not_run`。
完整私有 JSON 中包含实际解释器、包版本、CUDA/设备、内存/磁盘、资源哈希、实际参数及联合推理源码指纹；终端只输出摘要。

人工准备 `$PHASE1_ROOT/smoke-queries.json`：JSON 数组，至少覆盖以下三行结构。
这里列的是字段要求，不是可以直接通过的业务查询；使用已经隔离的调试/演练样本并登记其来源排除记录。

| mode | query | address |
|---|---|---|
| `problem` | 私有问题文本 | 不提供 |
| `address` | 私有地址文本 | 不提供 |
| `combined` | 与问题模式一致的问题文本 | 与地址模式一致的地址文本 |

不要读旧 `queries.test` 或 `qrels.test` 来构造演练。问题与综合模式都需要产生非空 dense 候选并完成 reranker 打分；
只加载两个模型、只有地址匹配或返回空候选，均不算联合通过。

```bash
bash deploy/run-phase1.sh preflight \
  --smoke-queries "$PHASE1_ROOT/smoke-queries.json" \
  --output "$PHASE1_ROOT/preflight-001.json"
```

只有 `status=passed`、`inference.status=passed` 和 `joint_hybrid_reranker_passed=true` 才继续正式冻结。
更换模型、索引、解释器、依赖、推理参数或联合推理源码后，重新做联合预检并使用新输出文件。
文档和独立报告模块不属于联合推理源码绑定范围。

## 3. 来源排除、顺序审核和正式查询

先由人员创建 `$PHASE1_ROOT/source-exclusions.json`，字段约定如下。空白姓名/时间不是审核通过。

| 字段 | 内容 |
|---|---|
| `schema_version` | 固定为 `phase1-source-exclusions-v1` |
| `version` | 本次来源排除表版本 |
| `excluded_source_ids` | 调试、演练或已看结果的来源 ID 数组 |
| `excluded_group_ids` | 需要整体隔离的关联组 ID 数组 |
| `reviewed_by`、`reviewed_at` | 实际审核人及含时区时间 |
| `history_complete` | 调试/演练来源历史是否完整，布尔值 |
| `limitations` | 来源历史限制的字符串数组；历史不全时必须明确填写 |

程序另行排除旧 dev 查询对应的全部关联组；不把来源历史不全伪装成完整隔离。
来源排除表在准备候选之前核定，之后修改应重新准备和审核。

```bash
bash deploy/run-phase1.sh prepare-queries \
  --exclusions "$PHASE1_ROOT/source-exclusions.json" \
  --limit 80 --output "$PHASE1_ROOT/sampling-001"
cp -n "$PHASE1_ROOT/sampling-001/candidates.jsonl" \
  "$PHASE1_ROOT/sampling-001/reviewed-candidates.jsonl"
cp -n "$PHASE1_ROOT/sampling-001/protocol.template.json" \
  "$PHASE1_ROOT/sampling-001/protocol.json"
```

人工只填写 `reviewed-candidates.jsonl` 每行的 `review` 对象，保留来源、原文、候选顺序和组代表不变。
按 `review-schema.json` 从前往后审核，到第 10 个合格且互相独立的场景为止；不能先看检索结果再挑样本。
审核覆盖范围、来源、意图、信息时点、地址可解析性、近重复/关联事件、选择理由和改写依据。
`query_as_of` 无可靠证据时填写 `unknown`；已知后续处理信息需剔除并记录；只有导出时文本证据时，
使用 `export_text_only`，且协议须显式允许这一不确定性并写入限制。

`protocol.json` 需要填写协议版本 `v1.1`、实验标识、场景范围、审核人/时间，
确认 `selection_without_retrieval_results=true`，并明确 `allow_temporal_uncertainty`。
助手不代填人工结论。重复问题、重复地址查询或关联事件会排除，继续取后续候选；不能复用为独立计权样本。
80 个候选仍不足时，使用新目录和更大 `--limit`，保留原审核并在同一确定顺序上继续。

```bash
bash deploy/run-phase1.sh finalize-queries \
  --exclusions "$PHASE1_ROOT/source-exclusions.json" \
  --worksheet "$PHASE1_ROOT/sampling-001/reviewed-candidates.jsonl" \
  --sampling-plan "$PHASE1_ROOT/sampling-001/sampling-plan.json" \
  --protocol "$PHASE1_ROOT/sampling-001/protocol.json" \
  --output "$PHASE1_ROOT/query-freeze-001"
```

产物为 `queries.json`、`query-provenance.jsonl`、`sampling-audit.jsonl`、`query-id-map.json`。
共有 10 场景、30 查询；固定 6 场景校准、4 场景留出评价。随机编号不暴露来源、模式或分区信息。

## 4. 冻结实际配置并正式收集

从已通过的联合预检机械提取 `runtime_config`，不手工猜测默认值。以下操作只复制机器配置，不代替人工审核。

```bash
conda run --no-capture-output -n "$CONDA_RETRIEVAL_ENV" python -c '
import json, sys
from pathlib import Path
from retrieval_baseline.search import _write_private_json
report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report["status"] != "passed":
    raise SystemExit("Joint preflight is not passed")
_write_private_json(Path(sys.argv[2]), report["runtime_config"])
' "$PHASE1_ROOT/preflight-001.json" "$PHASE1_ROOT/config-001.json"

bash deploy/run-phase1.sh freeze \
  --queries "$PHASE1_ROOT/query-freeze-001/queries.json" \
  --provenance "$PHASE1_ROOT/query-freeze-001/query-provenance.jsonl" \
  --protocol "$PHASE1_ROOT/sampling-001/protocol.json" \
  --config "$PHASE1_ROOT/config-001.json" \
  --preflight "$PHASE1_ROOT/preflight-001.json" \
  --exclusions "$PHASE1_ROOT/source-exclusions.json" \
  --sampling-audit "$PHASE1_ROOT/query-freeze-001/sampling-audit.jsonl" \
  --id-map "$PHASE1_ROOT/query-freeze-001/query-id-map.json" \
  --output "$PHASE1_ROOT/freeze-manifest-001.json"

bash deploy/run-case-evaluation.sh collect \
  --queries "$PHASE1_ROOT/query-freeze-001/queries.json" \
  --freeze-manifest "$PHASE1_ROOT/freeze-manifest-001.json" \
  --retriever hybrid --case-k 50 \
  --output "$PHASE1_ROOT/run-001"
```

正式 `collect` 在加载模型前重新核对审核文件、配置、相关源码、运行环境和全部资源身份。
输出包含 `collection-status.json`、冻结输入快照、查询和逐查询检索结果。
只有全部 30 查询成功落盘且完整 manifest 有效，状态才为 `completed`。
正常零候选计为成功查询并单独登记；错误、强制中断残留 `running`、缺 manifest 都不能进入正式评价。

## 5. 导出、人标、复核和标签冻结

```bash
bash deploy/run-case-evaluation.sh export \
  --run "$PHASE1_ROOT/run-001" --depth 10 \
  --output "$PHASE1_ROOT/labels-v1"
```

人工编辑 `labels-v1/judgments.tsv` 的评分/事件组/备注列，保留查询、候选正文及编号。
评分为整数 0/1/2：无关、部分相关、直接相关。问题模式填 `problem_grade`，地址模式填 `address_grade`，
综合模式两项都填；综合有效评分取两项较小值。不要按 reranker 分数反推人工标签。
使用能保留制表符、引用换行和原始正文的编辑器；导入时会检查 TSV 往返和原文字段。

在 `labels-v1/review-log.jsonl` 中人工登记至少 20% 候选对的复核，覆盖所有非空模式。
每行包含 `annotation_id`、`reviewer`、带时区的 `reviewed_at`、`method`、`resolution`，
以及 `initial_grades`、`review_grades`、`final_grades` 三个分项字典。
分项字典按模式包含必填的 `problem_grade`/`address_grade`，值为整数 0/1/2；最终分项须与当前 TSV 一致。
`method=independent` 必须由原标注人以外的人员复核；`delayed_self` 使用原标注人，
还需有带时区的 `annotated_at`，且自复核至少晚 24 小时。`resolution` 写真实一致结论或分歧裁决依据。

```bash
read -r -p '实际标注人员代号: ' ANNOTATOR_NAME
bash deploy/run-case-evaluation.sh freeze-labels \
  --run "$PHASE1_ROOT/run-001" --labels "$PHASE1_ROOT/labels-v1" \
  --annotator "$ANNOTATOR_NAME"
```

标签冻结会绑定 run、共同池、TSV、复核日志和校准/留出两分区的有效标签哈希。
不能给空白复核日志补造记录，也不能在冻结后原地修改标签。

## 6. 校准、未过滤评价、过滤评价与报告

先只用校准分区完成阈值尝试并冻结策略，或记录无法发布策略的失败结论。
在这一步结束前，不运行留出集的 `evaluate` 或 `report`，避免提前看到留出指标。

```bash
cal_status=0
bash deploy/run-case-evaluation.sh calibrate \
  --run "$PHASE1_ROOT/run-001" --labels "$PHASE1_ROOT/labels-v1" \
  --k 5 --target-precision 0.9 --min-results 5 --min-queries 3 \
  --output "$PHASE1_ROOT/calibration-001" || cal_status=$?
printf 'calibration_exit_code=%s\n' "$cal_status"
```

`calibrate --output` 是新的尝试目录，必须与 run 目录同级，包含 `status.json` 和 `threshold-search.json`。
问题与综合两模式均成功时才发布 `policy.json`；失败原因/支持量保留在 `status.json`。
0.9 是校准集经验目标，不是总体精度保证。达不到时不放宽冻结目标或借用留出标签反复调阈值。

确认 run、标签和输入有效后，无论校准成功，还是因支持量不足、无合格阈值等原因失败，
都可以保存未过滤的留出评价。若状态为 `input_error`、`execution_error`、`interrupted` 或仍为 `running`，
先排查并完成有效校准尝试，不把异常当成已完成的科学结论。

```bash
bash deploy/run-case-evaluation.sh evaluate \
  --run "$PHASE1_ROOT/run-001" --labels "$PHASE1_ROOT/labels-v1" \
  --partition evaluation --k 5 --ndcg-k 5 \
  --output "$PHASE1_ROOT/evaluation-unfiltered-001.json"
```

只有校准成功且 `policy.json` 已发布，才继续过滤评价与报告：

```bash
if (( cal_status == 0 )); then
  bash deploy/run-case-evaluation.sh evaluate \
    --run "$PHASE1_ROOT/run-001" --labels "$PHASE1_ROOT/labels-v1" \
    --partition evaluation --k 5 --ndcg-k 5 \
    --policy "$PHASE1_ROOT/calibration-001/policy.json" \
    --output "$PHASE1_ROOT/evaluation-filtered-001.json"

  bash deploy/run-phase1.sh report \
    --run "$PHASE1_ROOT/run-001" --labels "$PHASE1_ROOT/labels-v1" \
    --evaluation "$PHASE1_ROOT/evaluation-filtered-001.json" \
    --policy "$PHASE1_ROOT/calibration-001/policy.json" \
    --calibration-status "$PHASE1_ROOT/calibration-001/status.json" \
    --output "$PHASE1_ROOT/report-filtered-001"
fi
```

校准失败仍可生成未过滤报告：

```bash
bash deploy/run-phase1.sh report \
  --run "$PHASE1_ROOT/run-001" --labels "$PHASE1_ROOT/labels-v1" \
  --evaluation "$PHASE1_ROOT/evaluation-unfiltered-001.json" \
  --calibration-status "$PHASE1_ROOT/calibration-001/status.json" \
  --output "$PHASE1_ROOT/report-unfiltered-001"
```

报告目录中 `public/report.json` 和 `public/report.md` 是聚合结果；
`private/diagnostics.json` 和 `private/diagnostics.md` 含查询与案例正文，按私有业务资料保存。
错误归因只有人工明确填写的 `[problem_mismatch]`、`[address_conflict]`、`[insufficient_evidence]` 才计入对应类别，
标签低分不会自动推断成模型错误或地址冲突。共同池 nDCG 不是全库 Recall，10 场景结果只用于探索性验收。

## 7. 标签修订、失败重试和本地离线评价

需要纠正标签时先创建新版本：

```bash
read -r -p '本次标签修订依据: ' LABEL_REVISION_REASON
bash deploy/run-case-evaluation.sh revise-labels \
  --run "$PHASE1_ROOT/run-001" --labels "$PHASE1_ROOT/labels-v1" \
  --output "$PHASE1_ROOT/labels-v2" --reason "$LABEL_REVISION_REASON"
```

人工在新 TSV 中纠错并更新复核记录。任何必填评分分项发生变化的候选对都需新的复核时间，晚于上一版冻结；
`initial_grades` 等于旧版分项，`final_grades` 等于新版，`resolution` 写明纠错证据。
只有备注变更时可以沿用评分复核，但仍记录本次操作者和修订原因。之后对 `labels-v2` 再执行 `freeze-labels`。
新版 `revision-record.json` 是冻结绑定的权威修订记录，全局 `label-revisions.jsonl` 只是追加索引。
校准分区的有效标签变更必须新建 `calibration-002` 重新校准；只有留出分区有效标签变更时保留原阈值并重新评价。

收集失败保留 `run-001`，排查后用 `run-002`，不能删掉失败查询或手写完成状态。
预检/输入/配置变化需要重新冻结，输出使用新编号。校准重试使用新的校准目录；
查看留出指标后不能依据这些指标反复调阈值或放宽目标，已保存的未过滤评价仍可报告。
`--drill` 仅用于隔离的演练目录，正式 export/calibrate/evaluate/report 默认拒绝演练产物；
`init-queries` 生成的是旧示例，不能作为正式来源审核和冻结的替代。

H100 完成候选收集后，可以通过已有的可信文件传输方式复制完整 `run-001`、所有相关 `labels-v*`、
`calibration-*` 目录及修订记录回本地，保持这些目录彼此同级且原目录名不变。
策略按校准目录名称核对状态与原始策略，修订标签按同级历史标签目录核对旧版冻结记录。
保留 run 中的冻结输入快照，不能只拷 `runs.jsonl`；
保存过的 evaluation/report 也可一起复制。不要人工修改冻结文件中记录的服务器资源路径。
本地使用同一代码版本的 Python 3.11/3.12，以普通 `python3 -m` 调用离线入口，无须下载两套模型或复制向量索引：

```bash
python3 -m retrieval_baseline.case_eval evaluate \
  --run "$PHASE1_ROOT/run-001" --labels "$PHASE1_ROOT/labels-v1" \
  --partition evaluation --k 5 --ndcg-k 5 \
  --policy "$PHASE1_ROOT/calibration-001/policy.json" \
  --output "$PHASE1_ROOT/evaluation-local-001.json"
python3 -m retrieval_baseline.case_report \
  --run "$PHASE1_ROOT/run-001" --labels "$PHASE1_ROOT/labels-v1" \
  --evaluation "$PHASE1_ROOT/evaluation-local-001.json" \
  --policy "$PHASE1_ROOT/calibration-001/policy.json" \
  --calibration-status "$PHASE1_ROOT/calibration-001/status.json" \
  --output "$PHASE1_ROOT/report-local-001"
```

离线检查验证已完成 run、冻结快照、标签和策略的关联；新的正式收集仍须在目标环境重新通过活资源门。
文件可复制不代表可以将数据提交 GitHub；公开交付只包含代码、操作说明和经过核对的聚合报告。
