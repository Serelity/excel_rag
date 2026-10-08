# 抽取规范 v1：H100 试抽执行手册

2026-10-08 用户提供服务器诊断：`civic-rag-retrieval`中Python3.11.16、Torch2.6.0+cu124及其余受检包版本匹配，但没有vLLM。用户随后要求重新安装一个环境，现采用独立的 **`civic-rag-extract-v1`** 运行本轮抽取，检索继续使用原环境；GPU仍用服务器H100。服务器新环境安装、GPU预检及模型结果尚待实际执行。

## 执行分工

| 工作 | 执行位置 |
| --- | --- |
| 大模型字段抽取、小批量试抽及后续批处理 | 服务器分配的H100 |
| BGE-M3向量编码、需要模型推理的dense/hybrid检索 | 服务器H100，复用现有环境、权重及索引 |
| BGE reranker重排 | 服务器H100 |
| 规范与代码开发、输入准备、逐字证据校验、人工审阅、离线统计 | 本地CPU；需要时也可在服务器CPU环境执行 |

沿用服务器终端Git拉取代码、提交H100任务的流程。本次按用户新指令创建独立抽取环境，复用已有数据和模型权重。本地无需具备GPU或模型权重，模型试抽不以本地GPU检查通过为前提。

## 已有入口与本轮接入

- 服务器私有`deploy/.env.semantic`继续提供已有Qwen3-30B-A3B权重路径、指纹及服务参数。本轮入口默认查找`civic-rag-extract-v1`，以其绝对环境路径运行服务和客户端；旧配置中的`CONDA_EXTRACT_ENV`不改变本轮选择。需要明确选择其他已有环境时传`--conda-env NAME`，不会自动回退到其他环境。
- [run-qwen3-vllm.sh](run-qwen3-vllm.sh)已有H100模型服务启动逻辑：复用本地权重、单张可见GPU、服务器回环地址服务及环境检查。
- [run-qwen3-pilot.sh](run-qwen3-pilot.sh)仍调用旧`semantic_extraction.run`及v4提示词/schema。它不能直接作为新规范的80条试抽命令。
- 新增 [run-case-contract.sh](run-case-contract.sh) 提供`inspect / smoke / all`三个阶段；[批量适配器](../semantic_extraction/case_contract/runner.py)使用v1提示词和输出结构，保存请求/回复与逐字校验结果。
- [环境检查](inspect-case-contract-env.py)只检查指定名称，缺失或依赖不兼容时保存诊断后退出；不自动改名、切换环境、安装依赖或升级现有检索环境。
- [独立环境安装](create-case-contract-env.sh)是唯一新增的安装入口：只新建`civic-rag-extract-v1`，同名环境已存在则停止；不读取旧私有配置来决定安装目标。

## 1. Git同步代码，在服务器重建固定80条输入

代码沿用既有Git流程：本地完成提交并推送到`origin/codex/case-relevance-phase1`，服务器再拉取。服务器在已有仓库根目录执行：

```bash
git status --short
git fetch origin codex/case-relevance-phase1
git switch codex/case-relevance-phase1
git pull --ff-only origin codex/case-relevance-phase1
git rev-parse --short HEAD
```

若服务器有未提交修改或分支分叉，先保留并处理，不用强制重置覆盖。核对服务器HEAD与本次交付提交一致，并确认新入口及整个`semantic_extraction/case_contract/`目录已到位，再执行环境检查。未推送的本地文件不能通过服务器`git pull`获得。

### 新建独立抽取环境

在允许联网安装软件的服务器终端、仓库根目录中执行。安装不需要申请GPU，不要求当前shell已激活新环境：

```bash
bash deploy/create-case-contract-env.sh
```

安装脚本新建Python3.11环境，使用[固定核心依赖](requirements-case-contract.txt)安装vLLM0.8.5、Torch2.6.0+cu124和对应的torchvision/torchaudio，以及本轮抽取依赖。安装完成后执行`pip check`、包导入、CUDA运行库版本和既有环境检查；成功时输出`environment_created=civic-rag-extract-v1`。这不代表模型加载或GPU推理已通过。

安装清单、pip解析报告、依赖快照和诊断保存在自动生成的`data/case-relevance-phase1-v1/contract-install-*/`内。同名环境已存在时不更新、不删除；若安装中途失败，保留环境和记录，依据具体错误继续处理。

### 准备输入

如果此前`extraction-contract-v1-002`已经准备成功，可直接复用，跳过重新准备。它与新环境独立；运行时传`--input-dir data/case-relevance-phase1-v1/extraction-contract-v1-002`。

服务器已有之前实验的原始数据和检索数据。拉取代码后，在仓库根目录用固定环境执行CPU准备脚本，无需重新上传原始数据：

```bash
conda run --no-capture-output -n civic-rag-extract-v1 \
  python deploy/prepare-case-contract-inputs.py
```

脚本在仓库根目录依次寻找已有的`data/case-relevance-phase1-v1/sampling-001/candidates.jsonl`、`data/retrieval-baseline-v1/dataset/dataset.sqlite3`、`data/raw/t_order_master.sanitized.v1_9.tsv`，使用第一个存在的来源。它按[固定80条指纹表](../research/specs/case-contract-run-v1/development-selection.json)匹配原文，不重新抽样，也不执行空白或字符规范化。该表随Git同步，只含样例编号和哈希，不含真实正文或业务ID。

如果数据放在其他位置，明确指定已有原始TSV或检索SQLite文件：

```bash
conda run --no-capture-output -n civic-rag-extract-v1 \
  python deploy/prepare-case-contract-inputs.py \
  --source /已有数据的绝对路径/t_order_master.sanitized.v1_9.tsv
```

也支持进程环境中已设置的`RAG_INPUT_PATH`；优先级为`--source`、`RAG_INPUT_PATH`、上述默认路径。准备脚本只用Python标准库，不加载GPU包，不读取或修改私有`.env`文件。它全程扫描所选来源；若有缺失，只报告未匹配的B编号并停止，不换一批样例，也不自动改用其他来源。

默认生成`data/case-relevance-phase1-v1/extraction-contract-v1-001/`，可直接接下面的试抽命令。该目录必须不存在；已有目录需要保留时，用`--output NEW_DIR`并在后续试抽中传相同的`--input-dir NEW_DIR`。成功输出必须包含`sample_count=80`、`matches_reviewed_inputs=true`，且`inputs_sha256`为`3148be30f85834343fa656c1869ddf0c159a425f68e59a82b666d4236fc4ac79`。

生成包包含逐字节一致的`inputs.jsonl`及新的`manifest.json`、`summary.json`、`source-provenance.json`、`selection.json`。新清单记录本次数据来源、完整正文流指纹和脚本指纹；不是对旧准备清单的复制。人工复核意见和`acceptance-matrix.jsonl`保留在本地，模型运行不需要上传它们，服务器也不从原始表伪造这些人工材料。模型结果回传后再对照本地标准验收。

## 2. 在服务器做只读环境检查

在服务器仓库根目录执行，检查不需要分配GPU：

```bash
conda env list
bash deploy/run-case-contract.sh inspect \
  --output data/case-relevance-phase1-v1/contract-env-extract-v1-001
```

脚本从Conda实际清单查找`civic-rag-extract-v1`，用对应绝对路径运行检查。同名环境存在于多个位置时会停止，避免选错。`environment.json`记录实际解释器、版本和选中路径；不兼容时报告具体组件的实际值和所需值。

现有Qwen服务部署要求Python3.11、openai1.75.0、pydantic2.11.4、torch2.6.0、transformers4.51.3、vllm0.8.5。检查沿用这些已定义的版本约定。`ready_for_gpu_preflight`只表示包记录及Pydantic导入适合继续；不表示CUDA、权重加载或新schema推理已通过。`pytest`是可选测试包，未安装不影响此次抽取。

## 3. 申请H100后先跑10条

复用已有`deploy/.env.semantic`中的真实权重路径。新入口不覆盖该文件；只在本次运行目录写入有效的非密钥配置。密钥从进程环境传递，不写入请求文件。

```bash
bash deploy/run-case-contract.sh smoke \
  --input-dir data/case-relevance-phase1-v1/extraction-contract-v1-002 \
  --output data/case-relevance-phase1-v1/contract-smoke-extract-v1-001
```

上述命令复用本轮此前准备的`002`输入。若在默认`001`目录准备，则将`--input-dir`改为该目录；省略时仍默认`extraction-contract-v1-001`。私有配置不在默认位置时用`--env-file`。输出目录必须不存在；重试使用新编号。

首批固定选取B001、B015、B032、B040、B013、B016、B019、B026、B053、B069，按原输入顺序执行。这是有意覆盖规则边界的开发集合，不能当随机准确率样本。请求串行，temperature=0、seed=42、关闭thinking；默认最大输出8192 tokens，可通过私有配置中的`CASE_CONTRACT_MAX_TOKENS`明确改变。

脚本核对输入和代码指纹，检查H100及权重，启动已有vLLM服务，核对模型别名，试抽并校验，最后回收本次启动的进程。已占用端口会停止任务，不终止已有服务。实际可见GPU沿用平台分配的`CUDA_VISIBLE_DEVICES`。

## 4. 检查结果，再运行80条

在`contract-smoke-extract-v1-001/`内：

| 路径 | 内容 |
| --- | --- |
| environment.json | 环境名称、绝对路径、包检查结果 |
| runtime.env、vllm.log、job.status | 非密钥有效配置、服务启动/运行日志、任务退出状态 |
| job-manifest.json | 任务回收服务后绑定完整环境、配置、日志和抽取产物；同时记录退出码 |
| extraction/plan.json、config.json、inputs.json | 实际输入、参数和代码指纹 |
| extraction/attempts/ | 每次实际请求、原始响应、HTTP/校验状态、耗时；属于私有业务材料 |
| extraction/records/、results.jsonl | 逐样例校验结果，包含拒收记录 |
| extraction/summary.json、manifest.json | 聚合状态、各文件哈希；完整完成后才生成 |

`status=completed`仅表示所有选中输出通过结构及逐字校验；`semantic_review_status`仍为not_run。`completed_with_errors`会以非零退出码结束，保留拒收及错误类型。没有有效manifest、任务failed/interrupted或尚在running时不能视为完成。

回传整个运行目录，包含外层job清单与内层extraction清单。只有退出码为0且两份清单所绑定文件齐全一致，才说明服务器试抽完整结束；它仍不代替语义验收。环境检查或输入准备阶段失败可能只有诊断文件，不会伪造完整任务清单。

仅超时/连接异常及HTTP408、429、500、502、503、504重试一次，所有尝试都留档。JSON/字段/证据错误或截断回复直接拒收；不复用旧版压缩恢复提示词，不静默删字段。根据失败原因另开新运行，不能覆盖旧目录或改写完成状态。

核对10条字段语义与明确反馈后，可以继续完整80条开发抽取：

```bash
bash deploy/run-case-contract.sh all \
  --input-dir data/case-relevance-phase1-v1/extraction-contract-v1-002 \
  --output data/case-relevance-phase1-v1/contract-full-extract-v1-001
```

这一步仍需对新模型输出做语义验收；原80条人工审阅通过不能直接转写为模型输出通过。不同模型/参数/提示词需另存运行，不与前轮合并当同一实验。

## 输入与结果交接

本轮私有开发输入位于`data/case-relevance-phase1-v1/extraction-contract-v1-001/inputs.jsonl`。sample_id只作传输匹配，模型用户消息仅取每行`input.case_content`。准备清单及原文指纹与输入一起保留。

代码和无正文的固定指纹表通过既有Git流程交接；服务器从已有数据重建输入，人工复核资料保留本地。服务器结果通过既有可信文件传输方式回到本地后，可以运行v1证据校验和人工审阅。真实投诉、复核资料及模型输出继续保存在Git忽略的数据目录，无需复制GPU权重。

先前5项旧版异步测试的阻塞发生在本地Windows事件循环初始化，尚未进入模拟客户端；这些是CPU测试，不是GPU推理失败。可在服务器正常的Python测试环境补跑，结果另行记录，不将“安排在H100服务器运行”写成“已经运行通过”。

当前输入准备方式见[服务器输入重建记录](../research/topics/case-content-extraction/15-server-input-reconstruction.md)，适配器交付见[H100试抽入口开发记录](../research/topics/case-content-extraction/14-h100-adapter-delivery.md)，规范本身的历史记录见[规范交付记录](../research/topics/case-content-extraction/13-extraction-spec-v1-delivery.md)。
