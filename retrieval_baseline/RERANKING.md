# 案例重排与人工相关性评测

这一阶段比较相同候选集合上的原文混合检索与 BGE reranker，评估返回的投诉案例
是否符合用户的问题及地址要求。历史知识引用 qrels 和已有冻结测试集不参与这次标注。

检索仍只使用 `case_content`。BM25/BGE-M3 先召回候选，reranker 再同时阅读
“查询 + 每条候选原文”打分；综合查询的候选始终限定在地址匹配集合内。
默认对案例 RRF 的前 50 条重排，不只重排已经显示的前 5 条。纯地址查询保持名称匹配。

重排分数是原始 logit，不是相关概率。首次试跑只改变排序，不自动过滤；人工标注完成后，
可以校准门槛并允许返回少于 K 条，甚至返回空列表。不能根据两条示例自行认定有效门槛。

## 1. 独立检查环境、下载模型

已有 `civic-rag-retrieval` 环境可以直接复用，无需新增 Python 包，也无需重新编码历史向量。
环境检查脚本不升级现有依赖，以免改变旧向量索引的编码配置。新机器按
[检索环境配置](../deploy/RETRIEVAL_SETUP.md) 创建环境后再检查。

```bash
cd /seu_share/home/huangkai/220243809/12345/excel_rag
git pull --ff-only origin main
bash deploy/check-reranker-env.sh
bash deploy/download-reranker-model.sh
```

下载源是 [ModelScope BAAI/bge-reranker-v2-m3](https://modelscope.cn/models/BAAI/bge-reranker-v2-m3)，
权重约 2.27 GB，默认保存到 `models/bge-reranker-v2-m3`，与 BGE-M3 embedding 模型分开。
下载不需要 GPU。首次加载会记录本地模型全部配置、分词器及权重的 SHA256；推理仅访问本地文件。
可显式设置 `RERANKER_MODEL_REVISION` 固定 ModelScope 版本，默认 `master`。
已有文件只检查时运行 `bash deploy/download-reranker-model.sh --verify-only`。

下载和环境检查使用显式环境变量，不读取实验 `.env.retrieval`。自定义位置时，例如：

```bash
RERANKER_MODEL_PATH=models/bge-reranker-v2-m3 bash deploy/download-reranker-model.sh
```

运行阶段默认参数见 `deploy/.env.retrieval.example`；可在私有 `deploy/.env.retrieval` 中设置
`RERANKER_MODEL_PATH`、`RERANKER_MAX_LENGTH`（默认 1024）、`RERANKER_BATCH_SIZE`（默认 4）。
设备和浮点精度沿用检索配置。改变重排 max_length/dtype/模型后需要重新校准门槛。

## 2. H100 上试跑一条

在平台已经分配的 H100 任务里执行，运行入口不调用 Git：

```bash
cd /seu_share/home/huangkai/220243809/12345/excel_rag
bash deploy/run-case-search.sh search --mode combined --retriever hybrid --address '花语馨苑' --query '楼道长期没人打扫' --rerank --top-k 5 --output data/case-relevance-v1/reranked-example-001.json
```

结果的 `matching.reranker` 保存重排分数和原召回排名，`reranking.stats` 保存截断及 OOM 重试数量。
不指定 `--relevance-policy` 时，`relevance_filter.enabled=false`；仍可能出现不相关候选。
`timing_seconds` 分别记录 embedding 模型/向量索引加载校验、查询编码、向量搜索与元数据读取、
reranker 加载和打分。`retrieval_total` 包含前三项，不应再与它们相加。
主构造器的数据/词法索引初始校验不在逐查询计时内，因此这也不是完整进程启动耗时。

同一进程连续试查询可复用模型和索引，输入一行查询后回车，Ctrl-D 结束：

```bash
bash deploy/run-case-search.sh search --mode problem --retriever hybrid --rerank --interactive --top-k 5
```

不要用单次冷启动耗时当作稳定查询延迟。真实业务文本可用 `--query-file`，避免进入 shell 历史。

## 3. 准备并冻结查询草稿

```bash
bash deploy/run-case-evaluation.sh init-queries --output data/case-relevance-v1/queries.json
```

该文件是可编辑的 **30 条人工编写示例**：10 个场景，每个场景有问题、地址、综合三种查询。
地点只是示例，不保证库存充分。这些查询不是从原测试集抽取的，也不是代表性业务抽样。
请在收集候选前检查意图、地点和语句，优先用真实新投诉中的用户表达替换示例。

字段是 `query_id`、`scenario_id`、`partition`、`mode`、`query`，综合查询另有 `address`。
默认前 6 个场景共 18 条属于 `calibration`，后 4 个场景共 12 条属于 `evaluation`。
同一事件、场景的改写和不同模式应使用相同 `scenario_id`，不能跨分区；不要把相同案例正文
既作为查询又留在候选库中，用自匹配证明检索效果。修改查询后必须重新收集到新目录。

## 4. H100 收集共同候选，CPU 导出标注表

```bash
bash deploy/run-case-evaluation.sh collect --queries data/case-relevance-v1/queries.json --output data/case-relevance-v1/run-001
bash deploy/run-case-evaluation.sh export --run data/case-relevance-v1/run-001 --output data/case-relevance-v1/labels-001 --depth 10
```

每条查询只检索一次，保存原始与重排后的完整候选排名。随后从两种排名各取前 10 条，
去重后导出共同标注池。因此是 30 条查询、最多 600 对“查询—案例”，并非只填 30 个格子。
纯地址查询两种排名相同，不重复导出。默认模型/索引在整个批次内复用。

标注文件是 **`data/case-relevance-v1/labels-001/judgments.tsv`**，可在 Excel 中作为 UTF-8、
制表符分隔文件打开。表中隐藏路线、分数、排名以及分区；以 `pair_...` 标识每一对，
避免 Excel 把 19 位工单 ID 四舍五入。长文、换行和 Unicode 分隔符保留在同一个单元格内。
保存时仍使用 UTF-8 制表符文本，保持列名及查询、原文字段不变，不要删除行。

只填写以下标签与可选备注：

| 查询模式 | 必填列 | 可留空列 |
| --- | --- | --- |
| `problem` | `problem_grade` | `address_grade` |
| `address` | `address_grade` | `problem_grade` |
| `combined` | `problem_grade`、`address_grade` | 无 |

统一使用 0/1/2：

- `2`：核心问题直接相关，明确提出的约束有依据；地址标签要求目标地点及明确限定不冲突。
- `1`：部分相关、只有宽泛主题相同，或关键条件缺失、地址身份有歧义。
- `0`：问题不同、明确否定了所需事实、地点冲突等不相关情形。

例如“楼道无人打扫”对应“消防通道被车堵住”是问题 0，即使是同一小区；
“小区卫生差”没有楼道/保洁依据时可记问题 1；“楼道一周无人打扫”明确涉及相同核心问题，
在没有明确持续时长门槛时可记 2。若查询明确要求“持续一个月”，则必须核对这一约束。
在标注前固定这种口径，不根据模型分数改变标准。详细地址已被隐藏时，不推断楼栋一致。

`related_event_group` 和 `notes` 可选。内容相似或同日反映不自动视为同一事件，当前指标不会
据此合并或删除工单。留空标签表示未判断，评分/校准遇到必填空格会停止，不把它当作 0。
对于空候选查询，标签清单的 `manifest.json/empty_pool_queries` 会单独列出；这不证明库中无相关案例。

## 5. 校准门槛，冻结后评价

默认门槛目标是开发标注上的 `returned_precision >= 0.9`，在满足它的阈值中尽量多保留结果。
每种模式至少要求 5 个已返回候选、覆盖 3 个查询；纯地址模式不校准语义阈值。
样本不足或达不到目标会报错，不会用“全部返回空”伪装成功。这只是开发集经验目标，不是性能保证。

```bash
bash deploy/run-case-evaluation.sh calibrate --run data/case-relevance-v1/run-001 --labels data/case-relevance-v1/labels-001 --output data/case-relevance-v1/policy-001.json --k 5 --target-precision 0.9
bash deploy/run-case-evaluation.sh evaluate --run data/case-relevance-v1/run-001 --labels data/case-relevance-v1/labels-001 --policy data/case-relevance-v1/policy-001.json --output data/case-relevance-v1/evaluation-001.json
```

校准只读取 `calibration` 的标签。`evaluate` 默认只评 `evaluation`，同时报告原始排名、重排、
重排加筛选三组结果，并按查询模式分组。不要根据 evaluation 的结果反复调门槛后仍称其为独立测试；
更大实验需要新的未见查询。基线和模型改变时，应扩充共同标注池再比较。

| 指标 | 本项目定义 |
| --- | --- |
| `precision@5` | 每条查询前 5 条中 grade=2 的数量 / 5，再取查询平均；返回 3 条全相关仍为 0.6 |
| `returned_precision` | 所有查询实际返回前 5 条中的 grade=2 总数 / 实际返回总数；全不返回为 null |
| `pooled_ndcg@5` | 用 0/1/2 的增益 `2^grade-1` 计算；理想排序仅限共同人工标注池，三组都比较前 5 条 |
| `query_coverage` | 至少返回一条结果的查询比例；不表示这些结果一定相关 |
| `mean_returned` | 每条查询平均实际返回条数（最多 5） |

综合查询的最终 grade 取 `min(problem_grade, address_grade)`；只有两项都直接相关才算正例。
标注池无非零增益的查询，nDCG 为 null 并单独报告有效分母。未标注的相关案例不可视为负例，
本阶段不报告全库 Recall，30 条示例也不足以证明总体提升。
含筛选策略时，nDCG 截止位置不能超过实际返回预算 K，避免把重排前 10 条与筛选后最多 5 条
混为同一预算比较。仅比较未筛选两组排名时，可不传 `--policy`，另设 `--ndcg-k 10`。

## 6. 应用已经验证的门槛

```bash
bash deploy/run-case-search.sh search --mode combined --retriever hybrid --address '花语馨苑' --query '楼道长期没人打扫' --rerank --relevance-policy data/case-relevance-v1/policy-001.json --top-k 5 --output data/case-relevance-v1/filtered-example-001.json
```

策略绑定模型、索引、查询模式、候选预算、打分参数和实现指纹；不匹配时明确拒绝应用。
按 K=5 校准后不能用同一策略请求 K=10。空候选不加载 reranker；所有分数低于门槛时，
`result_status=below_relevance_threshold`，不从其他地址补齐、不保底返回不相关案例。

所有实验目录和结果文件拒绝覆盖；中断的收集目录没有完成 manifest 时不能用于评测，
保留它并换新目录重跑。数据、标注、策略与结果留在 Git 忽略的 `data/` 下，
新结果文件在 POSIX 上为 0600。GPU 上的模型兼容性、真实质量及吞吐需要服务器实测。
