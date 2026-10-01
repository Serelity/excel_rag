# 原文检索基线：先建立可复现的对照组

本阶段直接使用 `case_content` 建立历史案例检索和知识条目推荐基线。
不调用 Qwen，不使用抽取结果，不等待 20 条抽取标注完成。
后续在相同语料、查询、标签和评价口径下增加 embedding 和 LLM 抽取实验。

已完成的全量构建和首轮开发集指标见 [RESULTS.md](RESULTS.md)。
下一阶段的原文 BGE-M3 语义检索、检查点和配对比较见 [DENSE.md](DENSE.md)。

现有数据包含历史案例、知识引用 ID 和标题，没有可验证的知识正文。
因此当前产物是“相似案例候选 + 知识条目推荐”，不能据此生成权威政策答案。
当前自动指标衡量历史知识引用的找回程度；案例相似性需要另外的人审相关性标签。

## 服务器运行

使用已有的 `civic-rag-extract` Conda 环境，Python 3.11/3.12 和自带的 SQLite FTS5 即可。
本模块只用标准库，不需要新增 pip 依赖、Docker、GPU、模型下载或 vLLM。
虽然不用 GPU，全量建索引仍需要 CPU、内存和磁盘；请通过平台的计算任务运行，
不要在受限的登录节点上长时间建索引。

先按已有方式把仓库代码更新到服务器，然后在计算任务中执行：

```bash
cd /seu_share/home/huangkai/220243809/12345/excel_rag
bash deploy/run-retrieval-baseline.sh
```

脚本默认环境名已写明，不依赖 `$CONDA_EXTRACT_ENV` 或 `deploy/.env.semantic`，
运行过程中不调用 Git。若环境名不同，用 `CONDA_BASELINE_ENV` 指定。
目录已存在时拒绝覆盖；完整重跑时使用新目录：

```bash
BASELINE_OUTPUT="$PWD/data/retrieval-baseline-v1-rerun" bash deploy/run-retrieval-baseline.sh
```

也可以分步运行，某阶段中断后使用该阶段的新输出目录继续，不必重做已完成阶段：

```bash
conda activate civic-rag-extract
python -m retrieval_baseline build --input data/raw/t_order_master.sanitized.v1_9.tsv --output data/retrieval-baseline-v1/dataset --dev-start 2026-01-01 --test-start 2026-02-01 --queries-per-split 200 --seed 42
python -m retrieval_baseline index --dataset data/retrieval-baseline-v1/dataset --output data/retrieval-baseline-v1/index
python -m retrieval_baseline evaluate --dataset data/retrieval-baseline-v1/dataset --index data/retrieval-baseline-v1/index --output data/retrieval-baseline-v1/dev-bm25 --split dev --case-k 50 --max-terms 32
```

读取汇总结果（不输出工单正文）：

```bash
python -m json.tool data/retrieval-baseline-v1/dev-bm25/report.json
```

## 数据协议

输入固定为脱敏 TSV `data/raw/t_order_master.sanitized.v1_9.tsv`。
所有原始数据、数据库、查询集、标签和检索结果都在 Git 忽略的 `data/` 下。

| 字段 | 用途 |
| --- | --- |
| `case_content` | 唯一检索输入；保存原文，仅分词和重复判定时做规范化 |
| `id`、源逻辑行号 | 唯一标识和回溯；重复或空 ID 会阻止构建 |
| `order_id` | 隔离关联组，不直接认定为重复工单，也不作为检索文本 |
| `call_time` | 时间切分；不是原文内容的版本时间 |
| `knowledge_quote` | 语料期构造知识目录和案例引用边；开发/测试期仅作观察标签 |
| `delete_flag`、`order_invalid_type` | 样本资格过滤 |
| 一级业务类别 | 保存供后续分层检查，不输入检索、不据此选择查询 |

处理顺序和规则：

1. 排除非 `delete_flag=0`、有无效原因、空正文及正文含制表符的记录。
   统计按顺序互斥计数。制表符规则是针对已发现表格嵌入问题的保守过滤，
   不代表识别了所有污染文本。
2. 对合格记录，按相同非空 `order_id` 或相同规范化正文建立传递关联组。
   正文指纹使用 NFKC 后删除空白；原文保留。相同 `order_id` 的不同叙述仍可
   在历史索引中保留，只要求整个关联组不跨切分。
3. 语料期为 `call_time < 2026-01-01`，开发期为 2026 年 1 月，
   测试期为 `call_time >= 2026-02-01`。2025 年秋季记录稀疏，故不拿它单独作验证期。
   横跨多个时期的关联组整体排除；包含缺失/非法时间的关联组也不进入有效切分。
   审计中的 `records_missing_time` 表示全组时间均不可用，`records_cross_period`
   包含真正跨时期组以及“有效时间 + 不可用时间”混合组。
4. 历史语料按规范化相同正文合并为一个检索文档，引用取这些历史重复文本记录的并集。
   每个知识 ID 使用 `type:value`，保留历史语料中的全部标题别名，不使用未来标题。
   `type` 的业务含义未获确认，作为命名空间保留，不自行赋义。
5. 开发/测试各取最多 200 个关联组：先以最小源行号代表该组，再按固定种子和 ID 的
   SHA256 排序抽样。不要求有引用，也不要求引用在历史目录中。
   查询标签只取该代表记录的引用，不把关联组其他记录当作完整标签。
6. 显式检查 `order_id`、规范化正文和关联组的切分交集均为零。
   manifest 记录输入、数据库和各 JSONL 文件 SHA256；索引和评估会验证关联文件。

引用损坏时保留正文，但标签记为未知并记录异常数。无引用也记为未知。
没有做近重复语义去重；相同正文之外的转述仍可能跨切分。
资格字段、正文和引用是导出时的状态，可能包含后续处理信息，因此本实验是
**导出文本条件下的时间隔离离线评估，不是实际线上时点回放**。

## 三条固定基线

采用成熟的 BM25 和 RRF 方法，通过 SQLite FTS5 实现本地、可复现的轻量对照。
中文使用相邻双字 token，英文和数字保留整词；这不是经过调优的中文分词系统。
每个查询在对应历史索引词表中取最多 32 个词，按
`min(查询词频, 3) / log2(文档频率 + 2)` 排序，OR 匹配后由 FTS5 BM25 排名。
这是带固定查询词筛选的 BM25 变体，后续比较应保持该配置一致。

| 方法 | 路径 |
| --- | --- |
| `title_bm25` | 正文 → 历史知识标题/别名 BM25 → Top 10 条目 |
| `case_bm25_vote` | 正文 → Top 50 历史案例 BM25 → 按案例引用投票 → Top 10 条目 |
| `title_case_rrf` | 对以上两路各 Top 10 结果进行 RRF 融合，常数 60 |

案例投票权重为 `1 / ((60 + 案例排名) × 该案例不同引用数)`。
同一关联组对同一知识 ID 只取最大一次贡献，减轻重复提交的影响。
融合仅在两路已截断的候选上进行；若后续增大融合候选池，应作为配置变更报告。
`rankings.jsonl` 保留各路条目 ID、Top 50 案例来源，以及案例投票的支持记录。

方法参考：[SQLite FTS5 BM25](https://www.sqlite.org/fts5.html#the_bm25_function)、
[RRF 原论文 DOI](https://doi.org/10.1145/1571941.1572114)。
本阶段没有接入完整开源 RAG 平台；先冻结数据和评价接口，之后可用
Elasticsearch/OpenSearch 等后端替换检索实现并作同协议比较。

## 案例级混合检索

完成同一开发集的 BM25 和 BGE-M3 评估后，复用两路已经保存的历史案例候选：

1. 每路取已有 Top 50 案例，按案例 `source_id` 去重，最多得到 100 个候选。
2. 用等权 RRF 计算 `1 / (60 + BM25 排名) + 1 / (60 + dense 排名)`；
   某路没有该案例时，该路贡献为零。同分按 `source_id` 固定排序。
3. 截取统一的 Top 50 案例，再使用与单路基线相同的引用投票，推荐 Top 10 知识条目。

因此混合方案不是把 100 个案例直接送入投票，也不是融合已有知识 Top 10。
下游案例预算仍为 50，但上游需要两路检索，成本不能视为与单路相同。
整个过程不使用查询标签选候选，不增加 LLM、reranker 或新的 embedding。
输入报告、候选与数据集必须来自同一数据协议，程序会校验来源和查询一致性。
当前脚本只运行 `dev`，不评估保留测试集。

服务器已完成两路评估后执行：

```bash
cd /seu_share/home/huangkai/220243809/12345/excel_rag
git pull --ff-only origin main
bash deploy/run-hybrid-baseline.sh
conda run --no-capture-output -n civic-rag-retrieval python -m json.tool data/retrieval-hybrid-v1/dev-hybrid/report.json
```

脚本使用已有 `civic-rag-retrieval` 环境，不依赖 `$CONDA_EXTRACT_ENV`，也不调用 Git。
更新代码的 `git pull` 是单独的终端操作。无需重新下载模型、编码 400,120 条正文或启动 GPU；
只需要现成的数据集、BM25 元数据索引和两路报告/候选文件。它可在 CPU 计算任务中运行。
`deploy/.env.retrieval` 可覆盖默认路径，新增的 `RETRIEVAL_HYBRID_REPORT` 指定输出目录。
目录已存在时拒绝覆盖；重跑可在环境配置中指定新的输出目录。

默认读取以下目录：

| 目录 | 用途 |
| --- | --- |
| `data/retrieval-baseline-v1/dataset` | 冻结的语料、开发查询与观察引用 |
| `data/retrieval-baseline-v1/index` | 历史案例元数据与引用投票索引 |
| `data/retrieval-baseline-v1/dev-bm25` | BM25 报告与 Top 50 案例 |
| `data/retrieval-bge-m3-v1/dev-dense` | BGE-M3 报告与 Top 50 案例 |

输出保存在 `data/retrieval-hybrid-v1/dev-hybrid/`：

| 文件 | 用途 |
| --- | --- |
| `report.json` | 三种方法的指标、混合减单路的配对区间及输入指纹 |
| `rankings.jsonl` | 混合后的案例、原两路排名、RRF 分数和知识推荐 |
| `diagnosis.jsonl` | 逐查询检查候选并集、最终 Top 50 和引用投票的漏召回位置 |

先看两路候选并集是否覆盖目标，再看是否因融合 Top 50 截断丢失，最后检查引用投票。
并集覆盖高不等于最终 Top 10 召回一定提高；需要实际运行后的配对结果。
新报告中的耗时仅代表读取缓存候选后的融合与投票，不包含在线 BM25、query embedding
和 dense 搜索，不能与单路在线延迟直接比较，也不是生产 SLA。

## 指标及解释

对有观察引用的查询，设引用集合为 G、历史目录为 C、前 K 推荐为 R：

- `observed_recall@K`：逐查询计算 `|G ∩ R| / |G|` 后平均，K=1/5/10。
  目录外目标保留在分母，不能偷偷过滤。
- `observed_hit@K`：有至少一个观察引用被命中的查询比例。
- `target_catalog_coverage`：所有观察目标中落在历史目录里的比例。
- `catalog_recall_ceiling`：逐查询 `|G ∩ C| / |G|` 的平均，是当前目录下的召回上限。
- `known_target_recall@K`：只对 `G ∩ C` 非空的查询评价目录内目标，必须和总体召回一起看。
- 未标注查询参与检索和耗时统计，不进入监督指标；另报数量。

这不是完整相关性金标：历史坐席可能漏引、错引，系统也可能找到历史未引用的合理条目。
因此不能把未引用条目直接算作错误，也不报告“准确率/Precision”作为结论。
本轮不会评估测试集；200 条开发查询仅作初始比较，不支持最终效果或统计显著性结论。
记录的延迟是本机三路串行查询总耗时，不包含离线建库和文件哈希检查，不是生产 SLA。

## 产物和后续

| 文件 | 用途 |
| --- | --- |
| `dataset/manifest.json` | 过滤、切分、覆盖率和文件校验审计 |
| `dataset/dataset.sqlite3` | 合格记录、关联组、历史语料、引用和目录 |
| `dataset/queries.dev.jsonl`、`qrels.dev.jsonl` | 开发查询与观察引用，分开保存 |
| `dataset/queries.test.jsonl`、`qrels.test.jsonl` | 保留测试集，方案冻结前不评价 |
| `index/index.sqlite3`、`manifest.json` | 两类 FTS5 索引及数据来源 |
| `dev-bm25/report.json`、`rankings.jsonl` | 开发指标及可回溯候选结果 |

原文 BM25、dense 与案例级 hybrid 实现已具备；先运行混合评估，再审查开发集漏召回和目录外目标。
随后比较“原文”“仅 LLM 抽取”“原文 + 抽取”三种表示，固定 embedding、候选数和重排器，
才有条件把差异归因于抽取。已有 20 条抽取样本继续用作开发/回归，不能直接充当独立测试集。
最终需要从多个检索器的候选池抽取案例/条目做盲审标注，分别评价案例相关性与条目相关性，
并报告配对置信区间、延迟和抽取成本，再决定是否扩到全量 LLM 抽取。
