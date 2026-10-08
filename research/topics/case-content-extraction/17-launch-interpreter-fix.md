# 试抽启动时误用base解释器的修复

2026-10-08，用户回传独立环境的`pip check`通过和`ready_for_gpu_preflight`结果。首轮`contract-smoke-extract-v1-001`已完成10条输入准备，但`vllm.log`显示运行时校验实际用了Python3.12.13，尚未进入GPU或模型加载。

## 原因与修改

`run-case-contract.sh`在环境检查后把Conda可执行文件所在的base/bin添加到PATH最前面。后续命令虽然用`conda run -p`指定环境，却仍通过裸命令`python`查找解释器。Conda26.1.1的激活逻辑会在原位置替换已激活环境的PATH项，前方额外插入的base/bin可能继续优先；这与服务器“检查通过、启动却使用3.12”的现象相符。[Conda激活实现](https://github.com/conda/conda/blob/26.1.1/conda/activate.py)

修复删除了base/bin的PATH注入，通过绝对`CONDA_EXE`传递同一Conda入口。检查、抽取客户端及指定prefix的vLLM启动器都显式执行所选环境的`bin/python`，包含运行时检查、权重校验及API服务进程。runtime.env同时保存Conda入口与环境prefix；运行时校验在版本检查前输出`sys.executable`和Python版本。旧v4按环境名称运行的入口行为保留。

## 验证与边界

- 116项同步回归通过，5项旧版异步测试未运行；修改后的Python通过Ruff，两个Bash入口通过语法检查。
- 新增`semantic_extraction/tests/check_launch_interpreter.sh`，使用假Conda及两套假Python模拟base优先的PATH。负对照中的裸python选错解释器；修复后的运行时检查、权重检查、服务入口及外层客户端均使用指定环境的绝对路径，环境配置传递正确。
- Shell回归不调用真实Conda激活，不访问网络、不安装依赖、不加载GPU或模型；它验证实际项目脚本的命令选择和PATH传递，不能代替服务器重试。

已有环境和80条输入继续复用，重新运行时改用`contract-smoke-extract-v1-002`保留前轮失败证据。服务器重试后仍需完成GPU/权重加载、实际抽取和语义验收，当前没有抽取效果结论。
