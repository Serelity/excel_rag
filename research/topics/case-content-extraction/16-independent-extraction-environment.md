# 独立抽取环境：服务器诊断后的部署调整

日期：2026-10-08。用户回传`contract-smoke-002/environment.json`：已找到`civic-rag-retrieval`，客户端检查通过，服务检查失败。Python3.11.16、pydantic2.11.4、openai1.75.0、torch2.6.0+cu124、transformers4.51.3均满足当前元数据检查；vLLM未安装，模型未启动。pytest未安装不影响此轮推理。

用户随后明确要求“重新安装一个环境”。这条新指令替代此前抽取必须使用检索环境的约定。本次独立环境命名为`civic-rag-extract-v1`，与服务器已有的`civic-rag-extract`和`civic-rag-retrieval`区分，原环境和已有输入、权重均保留。

## 代码交付

- `deploy/create-case-contract-env.sh`：显式新建固定名称的环境，不读取旧`.env.semantic`来决定安装目标；已有同名环境即停止，不更新或删除。记录安装前Conda清单、pip解析、依赖快照、检查结果及安装退出码。
- `deploy/requirements-case-contract.txt`：固定本轮核心依赖，统一Torch/torchvision/torchaudio的CUDA12.4 wheel系列。安装后做依赖检查及包导入，实际GPU和模型加载由后续H100任务验证。
- 环境检查和`run-case-contract.sh`默认使用新环境；增加显式`--conda-env NAME`，从Conda清单解析准确路径。客户端和服务共用选定环境，不自动寻找其他可用环境。
- 诊断增加`client_mismatches`和`serve_mismatches`，终端摘要列出不匹配组件、实际值和期望值，避免只返回泛化的不兼容原因。

原抽取规则、schema、提示词和80条输入未改变。此前`002`失败目录保留，新环境首轮输出使用`contract-smoke-extract-v1-001`。已有输入可用`--input-dir .../extraction-contract-v1-002`直接复用。

## 验证与下一步

本地抽取模块同步回归116项通过，5项旧版异步测试未运行。新增覆盖显式环境选择、新默认值、禁止自动回退、非法名称及vLLM缺失的准确诊断。新增安装脚本和修改后的运行入口通过Bash语法检查，Python修改通过Ruff。

用假Conda进行Bash流程演练：新建路径通过，同名环境存在时在安装前停止；旧私有配置和旧环境名不会改变目标。演练没有安装软件或访问网络。

测试未在本机安装Linux/CUDA依赖，没有访问服务器或运行模型。用户提供的旧环境诊断是已知服务器证据；新环境安装和推理结果仍未知。服务器拉取代码后按[执行手册](../../../deploy/CASE_EXTRACTION_V1_H100.md)安装新环境，成功后用H100运行10条试抽，再进行语义验收。
