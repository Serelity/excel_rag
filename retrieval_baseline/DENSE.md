# 原文语义检索：与 BM25 保持同一数据协议

本阶段的两个检索端都只用 `case_content`：历史正文编码后建库，留出查询正文编码后检索。
使用 BGE-M3 的 dense 向量；不调用 Qwen3，不使用其稀疏/ColBERT 输出，不把分类、
知识引用、处理字段或 LLM 抽取结果拼进模型输入。

运行路线：正文 → BGE-M3 → Top 50 历史案例 → 与 BM25 共用的引用投票 → Top 10 知识条目。
报告评价的是历史知识引用召回；目前没有独立案例相关性金标。

## 第一步：单独创建 Conda 环境

环境与模型准备已拆为独立流程，完整说明见 [RETRIEVAL_SETUP.md](../deploy/RETRIEVAL_SETUP.md)。
先完成这两步，到文件检查通过即可；后续 GPU 检查和实验另行运行。

服务器已有的 `civic-rag-extract` 环境继续用于 Qwen3 抽取。
新增 `civic-rag-retrieval`，避免检索依赖影响 vLLM。

```bash
cd /seu_share/home/huangkai/220243809/12345/excel_rag
git pull --ff-only origin main
bash deploy/create-retrieval-env.sh
```

这一步需要能访问 Conda/PyPI/PyTorch 软件源，但不需要访问 Hugging Face。
PyTorch 默认从 cu124 源安装；如平台有等价镜像，可设置 `PYTORCH_CUDA_INDEX_URL`。
创建环境不需要占用 H100。环境内固定 Python 3.11、PyTorch 2.6.0+cu124、
Transformers 4.51.3、Sentence Transformers 3.4.1、NumPy 1.26.4、FAISS CPU 1.10.0。

## 第二步：从 ModelScope 准备 embedding 模型

这是新的 embedding 模型，目录与已有 `models/Qwen3-30B-A3B` 分开。
已核对 ModelScope 的 `Xorbits/bge-m3` 仓库含权重、tokenizer、modules 和 CLS pooling 配置。

```bash
bash deploy/download-retrieval-model.sh
```

完整下载可能包含两种权重格式，需预留数 GB 磁盘。运行只读取本地模型，
不自动下载、不允许远程模型代码。下载目录已有完整模型时可跳过下载。
每次运行记录模型配置、tokenizer 和权重的内容 SHA256，下载源的浮动版本不作为唯一身份。
下载脚本成功时出现 `"model_files_check": "passed"`；此处只检查文件和配置，不加载模型。
已经准备过模型时，使用 `bash deploy/download-retrieval-model.sh --verify-only` 离线检查。

## 第三步：检查历史库，然后在 GPU 节点做小检查

必须先有上一阶段的三个产物：

```text
data/retrieval-baseline-v1/dataset/manifest.json
data/retrieval-baseline-v1/index/manifest.json
data/retrieval-baseline-v1/dev-bm25/report.json
```

若服务器还没有构建它们，先在 CPU 计算任务中运行一次：

```bash
CONDA_BASELINE_ENV=civic-rag-retrieval bash deploy/run-retrieval-baseline.sh
```

已有完整产物就直接复用；该脚本拒绝覆盖已有目录。不要将另一个数据版本的本地报告
与服务器新建的索引混用，比较程序会检查版本指纹。

下面的命令开始需要 GPU 计算节点，使用已有平台的 H100 任务启动方式：

```bash
cd /seu_share/home/huangkai/220243809/12345/excel_rag
bash deploy/run-dense-baseline.sh check
```

成功时出现 `"check": "passed"` 和 `"dimension": 1024`。
检查只编码两条人工示例，用于验证本地模型、GPU 和输出形状，不能证明检索质量。

## 第四步：完整构建、开发集评估和对比

```bash
bash deploy/run-dense-baseline.sh all
```

默认每批 8 条、每 512 条保存一个检查点、最长 8192 tokens，只评估开发集。
编码使用 H100；FAISS 使用 CPU 精确内积检索，归一化后等价于余弦排序。
先避免近似索引的召回损失，后续再独立比较 ANN 性能。
建议任务分配至少 4 个 CPU 线程、16 GB 主内存并预留额外模型/索引磁盘空间；
实际峰值和耗时需在服务器测量，尚未给出吞吐承诺。
约 40 万条、1024 维 float32 向量本体约 1.64 GB；同时保留分片和 FAISS 索引，
加上数据及模型文件，不能只按这一份向量大小准备磁盘。

分步运行也可以：

```bash
bash deploy/run-dense-baseline.sh build
bash deploy/run-dense-baseline.sh evaluate
bash deploy/run-dense-baseline.sh compare
```

建库中断后，重新运行 `build` 会校验并复用已提交分片。
`evaluate` 和 `compare` 的结果目录必须是新目录，避免把不同运行的报告混在一起；
它们当前不提供逐查询续跑。可以只重跑尚未完成的阶段。

修改配置时，复制 `deploy/.env.retrieval.example` 为 `deploy/.env.retrieval` 后编辑。
脚本也接受同名环境变量；若存在该文件，以文件中的设置为准。
没有这个文件也能使用默认值，运行脚本不依赖 Git。

例如，在未创建配置文件的情况下重新评估已有索引：

```bash
RETRIEVAL_DENSE_REPORT=data/retrieval-bge-m3-v1/dev-dense-rerun bash deploy/run-dense-baseline.sh evaluate
RETRIEVAL_DENSE_REPORT=data/retrieval-bge-m3-v1/dev-dense-rerun RETRIEVAL_COMPARISON=data/retrieval-bge-m3-v1/comparison-rerun bash deploy/run-dense-baseline.sh compare
```

## 这版实现的工程能力

| 能力 | 具体行为 |
| --- | --- |
| 数据一致性 | 只遍历冻结历史库的去重正文，按 rid 固定顺序编码 |
| 检查点 | 向量和 ID 先完整写入，再原子提交检查点；中断最多重算未提交分片 |
| 续跑校验 | 验证分片 SHA256、维度、有限值、归一化及与语料一致的 ID 顺序 |
| 防止混用 | 数据、模型内容、编码参数、关键包版本、实现文件指纹改变时拒绝续跑 |
| 并发保护 | 同一目录只允许一个建库进程；操作系统文件锁随退出释放 |
| 显存不足 | 批量大小逐次减半重试；单条也失败时停止，已提交分片可复用 |
| 结果追溯 | 保存索引清单、逐查询候选、支持案例、查询向量和阶段耗时 |
| 公平比较 | 两路使用同一数据、查询、标签、候选数和历史引用关系，调用同一投票函数 |

默认不降低模型的最长输入限制来掩盖 OOM。若单条也 OOM，需要明确选择更短的
`RETRIEVAL_MAX_LENGTH` 并使用全新的索引和结果目录；这是实验条件变化。
超出设定 token 数的正文会按 tokenizer 的右截断规则处理，语料和查询都记录截断数。
查询还按是否截断分别报告指标。BM25 使用完整原文，因此截断存在时不能把差异
完全归因于匹配方式，后续可研究长文分块，而不是忽略信息丢失。

当前“续跑”针对同一个冻结快照；新增数据的增量索引、线上切换和回滚尚未实现。
GPU 计算及 FAISS 截止位的同分候选可能存在微小差异，不承诺跨硬件逐位一致。
修改批量大小、设备或依赖版本也会被视为新配置；保持默认配置可直接续跑。

## 输出怎么看

```text
data/retrieval-bge-m3-v1/
  index/run.json             数据、模型、运行配置身份
  index/checkpoint.json      已提交分片及编码统计
  index/shard-*.npy          正文向量及对应的历史文档 ID
  index/index.faiss          全量精确检索索引
  index/manifest.json        完成标记、索引校验及编码统计
  dev-dense/report.json      语义检索开发指标、查询截断和耗时
  dev-dense/rankings.jsonl   每条查询的案例候选及知识排名
  dev-dense/query-vectors.npy 与冻结查询顺序一致的向量
  comparison/report.json    BM25 与 dense 的配对比较
  comparison/diagnosis.jsonl 每条有引用查询的漏召回定位
```

重点查看：

1. 两路 `observed_recall@1/5/10`、目录覆盖率与 `known_target_recall@K`。
2. `observed_reference_candidate_coverage`：Top 50 案例的引用并集能覆盖多少观察目标。
   它仍是知识引用覆盖率，不能称为“相似案例 Recall”。
3. `diagnosis.jsonl`：区分目录外目标、案例候选引用中没有目标、投票后目标掉出 Top 10。
4. `dense_minus_bm25`：每条有标签查询的配对 Recall 差值、胜/负/平数量，
   以及固定种子的 2000 次配对 bootstrap 区间。开发集结论仍是探索性结论，
   区间不会纠正弱标签缺失或样本选择偏差。
5. 编码截断数、OOM 重试数、查询编码/精确检索耗时。旧 BM25 报告的延迟包含三路，
   dense 报告为单路，不能直接拿二者比值宣称某个检索器加速多少。

**本地验证范围**：使用可控测试向量验证真实 FAISS 检索、续跑、故障恢复、版本拒绝、
共用投票和配对评价，并检查 BGE 适配器的离线参数与 OOM 行为。
本机未准备真实 BGE-M3 权重，也未运行 H100 全量编码；真实模型兼容性、资源峰值和
召回结果需完成服务器 `check` 和后续全量运行后确认，当前没有新的语义检索成绩。

参考模型仓库：[ModelScope Xorbits/bge-m3](https://modelscope.cn/models/Xorbits/bge-m3)。
