# H100 试抽适配器开发记录

日期：2026-10-08。用户已明确环境名称为 **`civic-rag-retrieval`**。GPU推理在服务器H100运行；本轮在本地完成接入开发及无模型测试，未访问服务器或启动真实模型。

## 本轮交付

- `case_contract.runner prepare/run`：从有指纹的开发包选择输入，绑定原文、参数、规范和源码，再向服务器本机vLLM串行请求。保存每次请求/响应、失败记录、逐字校验结果、汇总和产物哈希。
- `case_contract.transport`：同步HTTP客户端，仅访问`http://127.0.0.1:<port>/v1`，关闭代理及重定向；不依赖Windows异步事件循环。
- `deploy/inspect-case-contract-env.py`：通过Conda实际清单检查指定环境及现有服务所需版本。服务和客户端固定使用同一个`civic-rag-retrieval`环境；没有自动安装、升级或切换。
- `deploy/run-case-contract.sh`：只读检查、10条试抽、80条开发抽取三个入口。复用已有权重和vLLM启动器，独立记录有效配置和日志，结束时回收本次服务。
- 旧`run-qwen3-vllm.sh`增加可选的绝对Conda环境路径参数，供新入口精确使用已查到的环境；未提供该参数时保留旧版按名称启动的行为。

首批10条有意覆盖地点、对象/现象、公司主体、咨询意图、冲突、事项分组、纯流程和内容不足。这是开发回归选择，不产生独立准确率。

## 校验与边界

新增28项测试通过；抽取模块同步回归共93项通过，其中包含前轮65项，5项旧版异步测试仍未计入。测试使用构造输入与模拟HTTP回复，没有调用真实模型。覆盖完整落盘、传输重试上限、输出截断、模型别名不符、schema/证据错误、运行中断、输入及代码指纹变动、敏感输入隔离、拒绝外部URL/重定向，以及固定环境检查。

新增Python代码通过Ruff检查；两个Bash脚本通过实际Bash的`-n`语法检查。该检查只解析脚本，没有执行服务器流程。服务器Conda清单、依赖导入、CUDA/H100、模型加载、vLLM schema兼容性和抽取质量均待实测。

旧规范包与已复核80条快照保持不变。新增适配器自有运行版本`case-contract-run-v1`，不会把旧v4缓存或输出混入新规范。结构通过不等于语义通过；所有新输出都保留`semantic_review_status=not_run`。

真实运行步骤、输出说明和服务器入口命令见[H100执行手册](../../../deploy/CASE_EXTRACTION_V1_H100.md)。代码先通过`codex/case-relevance-phase1`分支同步到服务器。本记录最初采用完整传输私有开发包；用户随后确认服务器已有原始数据，现改为[按固定指纹重建输入](15-server-input-reconstruction.md)，复核标准留在本地。上面的测试记录产生于提交前；实际交付提交与远端同步状态以Git为准，本机模拟测试不登记为H100结果。

已用真实开发包在本地完成10条输入的`prepare`检查，目录为`data/case-relevance-phase1-v1/contract-adapter-preparation-001/`，状态仍为prepared、model_run=false。它只验证输入和计划生成；服务器入口会在服务器另行准备并绑定当地代码、路径与实际环境。

机器可读检查记录见[适配器validation.json](../../specs/case-contract-run-v1/validation.json)，与前一轮规范校验快照分别保存。
