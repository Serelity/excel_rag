# 相似投诉案例检索

输入新的投诉正文、地址或投诉问题，直接返回相关历史投诉原文。
需要候选重排、人工案例标注和相关性门槛时，继续阅读 [重排与评测流程](RERANKING.md)。
主流程不再通过历史知识引用投票，不把知识条目召回率当作案例相关性。
正文检索的两端仍然只使用 `case_content`，时间、分类等字段只用于结果展示。

## 三种查询方式

| 模式 | 输入 | 匹配目标 |
| --- | --- | --- |
| `problem` | 投诉问题或新的投诉正文 | BM25 关键词、BGE-M3 语义，或案例级 RRF 混合排名 |
| `address` | 道路、小区、楼栋、机构等名称 | 从历史正文提取的地址名称，保留原文匹配依据 |
| `combined` | 问题描述及 `--address` | 在地址匹配的历史案例集合中检索问题 |

`problem` 模式不会自动排除正文里的地址。它是当前原文基线，不等价于已经完成
LLM 的问题抽取。`combined` 用明确的地址约束，避免把全文相似度误认为地点一致性。
约束可以作用于所有地址匹配的历史案例，不只是全库 Top 50 的后过滤。

地址模式是名称匹配，不是地理距离检索。名称相近不一定是同一个地方；当前没有可靠的
地址别名库、行政区划地址库或坐标解析服务，不能据此声称找到附近案例。
当前是“命名地点/道路 + 楼栋或门牌”的字面匹配基线，楼栋等信息可能缺失或写法不同，
结果需要查看原文证据。暂不支持单元、房间、楼层、期数以及 `A区`、北区等精细条件。
含这些条件的地址查询会报错，要求改成较粗的地点/道路查询；在 `combined` 模式下，
可以通过 `--address '幸福小区'` 明确指定支持的地址约束，不会静默忽略条件并宣称精确命中。
历史正文仍可以含这些精细信息，按其中可识别的较粗地点名称查询并不会排除这些案例。
默认不主动扩大到更宽地点；`--allow-broader` 是显式允许扩大地点匹配的选项，
仅在没有结果时放宽受支持的楼栋/门牌条件，不绕过不支持的精细条件或行政区约束。
不能把扩大后的结果当作精确地址命中。

## 服务器运行

已有 `civic-rag-retrieval` 环境、本地 BGE-M3 模型、冻结历史数据和两种索引即可。
不需要重新创建 Conda 环境、重新下载模型或重新编码 400,120 条正文。
运行入口不调用 Git；Git 只用于在终端更新代码。

### 1. 更新代码，单独准备地址索引

地址索引仅依赖 Python 标准库，可以在 CPU 计算任务中构建，不需要 H100。

```bash
cd /seu_share/home/huangkai/220243809/12345/excel_rag
git pull --ff-only origin main
bash deploy/run-case-search.sh prepare-address
```

默认地址索引输出到 `data/case-search-v1/address-index/`，已有目录会拒绝覆盖。
POSIX 系统上新建索引目录权限为 `0700`，不改变已有目录权限；索引中的地址证据也属于私有数据。
更换数据快照或地址提取版本时，应使用新的索引目录，不要删除旧索引来混用版本。
私有配置可在 `deploy/.env.retrieval` 中设置
`RETRIEVAL_ADDRESS_INDEX=data/case-search-v1/address-index`；该文件不存在时直接使用默认值。
其余路径和设备参数复用 [环境配置示例](../deploy/.env.retrieval.example)。

### 2. 地址查询

以下名称只是演示，不保证历史库里存在对应案例。请替换为数据中的地点。

```bash
cd /seu_share/home/huangkai/220243809/12345/excel_rag
bash deploy/run-case-search.sh search --mode address --query '幸福小区3号楼' --top-k 10
```

地址查询不加载 BGE-M3 或 FAISS，不需要 GPU。若精细地址没有结果，可先缩短为已知
小区名称查询；需要程序允许更宽匹配时，明确追加 `--allow-broader` 并核对匹配依据。

### 3. 问题查询

CPU 关键词基线：

```bash
bash deploy/run-case-search.sh search --mode problem --retriever bm25 --query '楼道长期没人打扫' --top-k 10
```

在已有 H100 计算任务中运行语义或混合检索：

```bash
bash deploy/run-case-search.sh search --mode problem --retriever dense --query '楼道长期没人打扫' --top-k 10
bash deploy/run-case-search.sh search --mode problem --retriever hybrid --query '楼道长期没人打扫' --top-k 10
```

默认 `--retriever hybrid`，两路各取 `--case-k 50`，按案例 ID 去重后 RRF 排序，
直接返回案例；没有知识引用投票。地址和 `bm25` 单路不加载 embedding 模型。
语义查询会复用已有向量索引，仅编码这次查询。
现有 dense 索引使用 `float16` 表征配置，运行会严格校验。不要通过切换
`--device cpu --dtype float32` 复用它；CPU 编码要求 `float32`，与该索引配置不一致会被拒绝。
需要 CPU 查询时先使用上述 BM25 或地址入口。

### 4. 地址约束的问题查询

```bash
bash deploy/run-case-search.sh search --mode combined --retriever hybrid --address '幸福小区' --query '楼道长期没人打扫' --top-k 10
```

这里的 `--address` 是明确的名称约束。查看结果时分别确认地点匹配和问题相关性，
不能只因为同小区就判定问题相似。地址找不到时不要静默改为全库问题查询。

### 5. 连续查询或从私有文件读取

连续查询可复用已加载的模型和索引，不必每次重新启动；输入一条查询后回车，EOF 退出。

```bash
bash deploy/run-case-search.sh search --mode problem --retriever hybrid --interactive --top-k 10
```

`--query` 会进入 shell 历史和进程参数。真实新投诉建议放在仅自己可读的 UTF-8 文本文件，
使用 `--query-file`，不要把姓名、电话、身份证或详细住址粘贴到共享终端和聊天中。
假设已经在服务器准备好 `data/private-query.txt`：

```bash
bash deploy/run-case-search.sh search --mode problem --retriever hybrid --query-file data/private-query.txt --top-k 10
```

单次查询可选择写入一个新 JSON 文件，不覆盖旧文件；父目录须可写。
POSIX 系统上结果文件以 `0600` 独占创建，新建的直接父目录使用 `0700`；已有父目录
权限不变。Windows 上不能据此认定 ACL 已验证，仍需按所在系统检查访问权限。

```bash
bash deploy/run-case-search.sh search --mode problem --retriever hybrid --query-file data/private-query.txt --top-k 10 --output data/case-search-v1/private-result-001.json
```

`--query`、`--query-file`、`--interactive` 三选一。返回的原文可能包含个人信息；
屏幕输出和结果文件都是私有数据，不应直接作为公开演示或提交 Git。
结果没有指定 `--output` 时直接打印，交互模式不写单次结果文件。

## 结果与下一阶段评价

结果直接展示历史案例 ID、原文、时间以及检索或地址匹配依据。
每个候选的检索分数是排序信号，不是“相似概率”或准确率；查看匹配依据后再判断案例是否有用。

已有 400,120 条案例来自冻结、时间隔离、正文去重的历史语料，不是全量实时生产库。
同正文的多次提交可能只保留一个代表案例，因此结果不能直接用来统计投诉次数、频率或完整提交时间线。
当前不提供新增投诉的增量入库、在线服务权限、脱敏展示或可靠的生产地址归一化。
首次加载会校验索引和模型完整性，哈希读取及模型加载可能耗时，不能与逐查询延迟混在一起
作为线上 SLA。真实资源峰值和响应时间仍需在服务器测量。
地址索引 `manifest.json` 中的 `cases_with_addresses`、`indexed_entities` 只是规则识别数量，
不是人工标注后的地址准确率或召回率。

此前开发集上的混合 `observed_recall@10=83.62%` 是历史知识引用恢复结果，
不能用于证明这一案例检索入口的 Recall。下一阶段建立独立的案例相关性标签：

- 地址查询：判定同地点、较宽地点和无关地点，保留楼栋及同名歧义。
- 问题查询：判定问题核心是否相同、仅主题相关还是无关，允许跨地点相关。
- 综合查询：分别标注地点和问题，再依据查询意图判定综合相关性。

先报告 `Precision@5/10`、`nDCG@10` 及具体错误类型；没有足够完整的相关案例标签时，
不宣称相似案例 Recall。保持既有测试集冻结，再用同一批查询和人工判断比较原文检索、
LLM 地址/问题抽取检索和后续 reranker，避免用知识引用或模型自己生成的判断代替金标。
