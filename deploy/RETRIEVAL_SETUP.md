# 本阶段只准备 Conda 环境和 BGE-M3 模型

两步独立运行：环境脚本安装软件，下载脚本从 ModelScope 下载模型并检查必需文件。
它们都不读取业务数据、不启动 vLLM、不编码工单、不建索引、不运行评估。
可以在平台允许联网安装的测试环境/计算任务中完成，无需申请 H100 做推理。

先更新服务器代码：

```bash
cd /seu_share/home/huangkai/220243809/12345/excel_rag
git pull --ff-only origin main
```

## 1. 配置 Conda 环境

```bash
bash deploy/create-retrieval-env.sh
```

环境名称固定默认为 `civic-rag-retrieval`，与现有 `civic-rag-extract` 分开。
脚本创建或更新环境，安装检索依赖及 ModelScope 下载工具，执行 `pip check`、
版本检查、核心模块导入和 SQLite FTS5 检查。

| 软件 | 版本 |
| --- | --- |
| Python | 3.11 |
| PyTorch | 2.6.0+cu124 |
| Transformers | 4.51.3 |
| Sentence Transformers | 3.4.1 |
| NumPy / FAISS CPU | 1.26.4 / 1.10.0 |
| SentencePiece / FileLock | 0.2.0 / 3.18.0 |
| ModelScope | 1.25.0 |

成功标志：`environment_check=passed`。
无 GPU 的节点也可以通过该检查，它只确认软件环境；GPU 驱动兼容性留待后续 H100 检查。
软件安装需要 Conda/PyPI/PyTorch 源，但不需要连接 Hugging Face。
PyTorch 下载镜像可通过 `PYTORCH_CUDA_INDEX_URL` 指定，需提供 cu124 对应 wheel。

## 2. 下载 BGE-M3 模型

第一步成功后执行，不需要手动激活环境：

```bash
bash deploy/download-retrieval-model.sh
```

来源：ModelScope 的 `Xorbits/bge-m3`。默认保存位置：

```text
/seu_share/home/huangkai/220243809/12345/excel_rag/models/bge-m3
```

下载完整快照可能同时包含两种权重格式，建议预留至少 10 GB 可用磁盘。
ModelScope SDK 负责下载和缓存；中断后可以重跑同一个命令，复用 SDK 判定可用的文件，
不承诺所有中断文件都能逐字节续传。默认 revision 为 `master`，如需固定下载版本，
用 `RETRIEVAL_MODEL_REVISION` 指定 ModelScope 的可用 revision。

成功标志：`"model_files_check": "passed"`。
这里检查必需 JSON、配置类型、CLS pooling 和非空权重文件，尚未加载 GPU 模型。
真实权重加载和编码能力要在下一阶段确认；文件存在不能替代推理测试。

已经下载过，想只检查本地文件，不访问网络：

```bash
bash deploy/download-retrieval-model.sh --verify-only
```

自定义目录可以直接指定：

```bash
RETRIEVAL_MODEL_PATH=/your/persistent/models/bge-m3 bash deploy/download-retrieval-model.sh
```

两个准备脚本不读取实验用的 `deploy/.env.retrieval`；环境名通过
`CONDA_RETRIEVAL_ENV` 指定，下载路径通过 `RETRIEVAL_MODEL_PATH` 指定。
如果使用自定义值，后续实验的 `.env.retrieval` 也需要设置同样的环境名和路径。

本阶段到这里结束。保留两个成功标志，之后再按
[语义检索运行说明](../retrieval_baseline/DENSE.md)申请 H100 验证模型并运行实验。
