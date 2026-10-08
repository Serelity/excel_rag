# 开源方案核验：投诉内容的字段抽取、边界控制与人工标注

调研日期：2026-10-08。状态：本轮定向源码调研完成；未安装、运行或评测第三方项目，尚未确定本项目字段规范和模型。

本轮按“抽取方法—输出约束—标注与核验”选择 7 个公开仓库，重点回答：项目能否表达一个文本片段的边界、它属于哪个投诉事项，以及地址、对象、诉求等字段之间的关联。它们是可借鉴的工程候选。GitHub 星数、组织名、README 中的“生产可用”字样均不作为本项目效果或企业部署的证据。

## 1. 版本、许可与维护快照

下表的源码结论均绑定固定提交。`releases/latest` 的返回值只记录当时 GitHub 的发布信息；默认分支提交与发布标签不等同。7 个仓库在 metadata 快照中均未归档。固定提交时间和发布频率可以帮助安排依赖核验，不能单独证明软件可靠性。

| 编号 | 仓库与核验提交 | 本次提交时间（UTC 日期） | GitHub 最新发布元数据 | 仓库代码许可 |
|---|---|---|---|---|
| O01 | [google/langextract · `24298305d6`](https://github.com/google/langextract/tree/24298305d6998061fadd01080def4288c653d243) | 2026-10-06 | [v1.7.0](https://github.com/google/langextract/releases/tag/v1.7.0) · 2026-09-13 | [Apache-2.0](https://github.com/google/langextract/blob/24298305d6998061fadd01080def4288c653d243/LICENSE) |
| O02 | [PaddlePaddle/PaddleNLP · `3f87dac9b7`](https://github.com/PaddlePaddle/PaddleNLP/tree/3f87dac9b719f75399f92f8bf634ae2ef0611832) | 2026-05-23 | [v3.0.0-beta3](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v3.0.0-beta3) · 2024-12-16 | [Apache-2.0](https://github.com/PaddlePaddle/PaddleNLP/blob/3f87dac9b719f75399f92f8bf634ae2ef0611832/LICENSE) |
| O03 | [urchade/GLiNER · `f3c945702b`](https://github.com/urchade/GLiNER/tree/f3c945702b9b5cde381c27b95310c47909bcfd80) | 2026-10-05 | [v0.2.29](https://github.com/urchade/GLiNER/releases/tag/v0.2.29) · 2026-09-08 | [Apache-2.0](https://github.com/urchade/GLiNER/blob/f3c945702b9b5cde381c27b95310c47909bcfd80/LICENSE) |
| O04 | [567-labs/instructor · `e12f8b4920`](https://github.com/567-labs/instructor/tree/e12f8b49203b0c1f253d27c1e709d0a09b9fc5a8) | 2026-09-11 | [v1.17.0](https://github.com/567-labs/instructor/releases/tag/v1.17.0) · 2026-09-09 | [MIT](https://github.com/567-labs/instructor/blob/e12f8b49203b0c1f253d27c1e709d0a09b9fc5a8/LICENSE) |
| O05 | [dottxt-ai/outlines · `c52af8472c`](https://github.com/dottxt-ai/outlines/tree/c52af8472c6fe8a607a733ec11b709476af77a96) | 2026-08-24 | [1.3.3](https://github.com/dottxt-ai/outlines/releases/tag/1.3.3) · 2026-08-06 | [Apache-2.0](https://github.com/dottxt-ai/outlines/blob/c52af8472c6fe8a607a733ec11b709476af77a96/LICENSE) |
| O06 | [HumanSignal/label-studio · `8ef0b5ee3d`](https://github.com/HumanSignal/label-studio/tree/8ef0b5ee3d84217e742823f27646c040506f4420) | 2026-10-07 | [1.23.2](https://github.com/HumanSignal/label-studio/releases/tag/1.23.2) · 2026-09-29 | [Apache-2.0](https://github.com/HumanSignal/label-studio/blob/8ef0b5ee3d84217e742823f27646c040506f4420/LICENSE) |
| O07 | [doccano/doccano · `ab6b765ea3`](https://github.com/doccano/doccano/tree/ab6b765ea326d07b2b806a4280cd097565e62321) | 2026-01-17 | [v1.8.5](https://github.com/doccano/doccano/releases/tag/v1.8.5) · 2026-01-11 | [MIT](https://github.com/doccano/doccano/blob/ab6b765ea326d07b2b806a4280cd097565e62321/LICENSE) |

许可均核对了固定提交的 `LICENSE` 文件。代码许可不自动覆盖模型权重、第三方依赖、云模型服务或商业版功能。PaddleNLP 的标签名称包含 `beta3`，虽然 API 元数据 `prerelease=false`，仍不能据此把它表述为稳定正式版。

Instructor 的旧入口 `instructor-ai/instructor` 已重定向到 `567-labs/instructor`；引用使用当前归属，缓存目录保留旧请求名称。

## 2. 已核实能力与可复用部分

### O01 · google/langextract：抽取结果回到原文位置

**证据与机制。** [README](https://github.com/google/langextract/blob/24298305d6998061fadd01080def4288c653d243/README.md) 以提示词和少量示例定义抽取类别及属性，支持本地 Ollama 接入、长文分块和结果可视化。[Extraction 数据结构](https://github.com/google/langextract/blob/24298305d6998061fadd01080def4288c653d243/langextract/core/data.py)包含 `extraction_text`、`char_interval`、`alignment_status`、`group_index` 和 `attributes`。[对齐实现](https://github.com/google/langextract/blob/24298305d6998061fadd01080def4288c653d243/langextract/resolver.py)区分精确、部分和模糊对齐，并允许找不到位置；README 明确说明无法回到输入原文的结果可能得到 `char_interval=None`。

**可借鉴部分。** 将“抽到了什么”与“原文哪一段支持它”一起保存，给复核页面提供高亮。`attributes` 和分组标识提供表达角色及分组的接口；具体投诉事项的划分及角色含义仍要由领域说明、示例和核验规则提供。

**边界与限制。** 原文存在“甲小区”只能证明文本提到了甲小区，不能证明它是事发地。模糊对齐成功也不能视为原文逐字一致。对同名地点反复出现、跨句指代、一条投诉多个事项的情况，需要逐项验证定位与绑定；可视化和对齐本身不解决这些语义问题。

**中文证据三层。** 框架层：[Unicode/CJK 分词测试源码](https://github.com/google/langextract/blob/24298305d6998061fadd01080def4288c653d243/tests/tokenizer_test.py)覆盖中文字符切分；这是读到的测试，未执行。权重层：可接不同模型，[本地模型文档](https://github.com/google/langextract/blob/24298305d6998061fadd01080def4288c653d243/examples/ollama/README.md)特别说明权重有各自许可，尚未选定并验证中文投诉模型。项目实测层：无。

### O02 · PaddlePaddle/PaddleNLP：中文 schema 驱动抽取与训练流程

**证据与机制。** [UIE Taskflow 指南](https://github.com/PaddlePaddle/PaddleNLP/blob/3f87dac9b719f75399f92f8bf634ae2ef0611832/slm/applications/information_extraction/taskflow_text.md)提供中文实体、关系、事件和分类示例；用户可通过自然语言标签、嵌套 schema 指定父子抽取目标。[Taskflow 源码](https://github.com/PaddlePaddle/PaddleNLP/blob/3f87dac9b719f75399f92f8bf634ae2ef0611832/paddlenlp/taskflow/information_extraction.py)可核对片段结果的 `start`、`end`、`probability` 以及关系层级。[文本抽取训练指南](https://github.com/PaddlePaddle/PaddleNLP/blob/3f87dac9b719f75399f92f8bf634ae2ef0611832/slm/applications/information_extraction/text/README.md)连接人工标注、数据转换、微调和评估，并说明关系/事件任务的默认评估按阶段分别统计。

仓库另有 [PP-UIE 大模型路线](https://github.com/PaddlePaddle/PaddleNLP/blob/3f87dac9b719f75399f92f8bf634ae2ef0611832/llm/application/information_extraction/README.md)，文档列出 0.5B、1.5B、7B、14B 中英文模型，示例返回文本字段及关系。不能把较早的 span 抽取路线所具有的字符偏移，直接视为 PP-UIE 每个结果都具备的保证。

**可借鉴部分。** 用自然语言定义抽取目标、用父子关系表达实体所属事项或关系对象，并保留未出现目标的负例。标注到训练/评估的数据通路值得借鉴；评估时仍需额外统计“地址是否绑定到了正确事项”，因为各阶段单独正确不保证组合正确。

**边界与限制。** 嵌套 schema 是表示关系的方法，不能替项目决定“夜间施工”与“施工造成的噪声”是否应拆成两项，也不能自动建立“居住地”和“事发地”的一致解释。该仓库使用 Paddle 生态，本项目现有 PyTorch 检索环境是否适配需另行验证。

**中文证据三层。** 框架层：官方中文 schema 与中文输入/输出例子明确。权重层：官方文档列出 `uie-base`、UIE-M 及 `paddlenlp/PP-UIE-*` 等模型标识；本轮未下载、加载或验证具体权重许可，模型卡网络请求未成功。项目实测层：无，其他数据集上的文档分数不迁移为投诉效果。

**名称澄清。** 文献中的 ACL 2022 UIE《Unified Structure Generation for Universal Information Extraction》、PaddleNLP 的 Taskflow 工程实现、PP-UIE 模型系列和某个下载权重是不同的证据对象。引用一个 UIE 论文不能替代对工程代码、训练数据、权重版本及许可的逐项核验。

### O03 · urchade/GLiNER：开放标签片段抽取，以及较新的关系能力

**证据与机制。** [当前 README](https://github.com/urchade/GLiNER/blob/f3c945702b9b5cde381c27b95310c47909bcfd80/README.md)和[架构文档](https://github.com/urchade/GLiNER/blob/f3c945702b9b5cde381c27b95310c47909bcfd80/docs/architectures.md)除了原始开放标签 NER，还列出联合实体/关系抽取 RelEx；[使用文档](https://github.com/urchade/GLiNER/blob/f3c945702b9b5cde381c27b95310c47909bcfd80/docs/usage.md)提供实体与关系标签输入、结果中 head/tail 实体索引等接口。[关系层源码](https://github.com/urchade/GLiNER/blob/f3c945702b9b5cde381c27b95310c47909bcfd80/gliner/modeling/multitask/relations_layers.py)存在实体对及关系表示相关组件。另一个 [multitask 关系管道](https://github.com/urchade/GLiNER/blob/f3c945702b9b5cde381c27b95310c47909bcfd80/gliner/multitask/relation_extraction.py)按先 NER、再构造关系标签进行第二次抽取的方式工作，应与 joint RelEx 区分。

**可借鉴部分。** 作为生成式大模型之外的一类片段抽取候选：测试有限标签下的地址、设施、时间等原文片段；当验证角色关系时，应明确使用哪一种关系模型/管道，而不是仅靠实体标签列表。当前仓库提供更广的能力，并不意味着原始 2024 GLiNER 论文已经验证了这些新能力。

**边界与限制。** 找到两个地点和一个问题，不等于正确连接“哪个地点对应哪个事项”。模型分数和阈值是排序/筛选机制，不能当成跨领域已经校准的可信概率。源码中的分词后端、模型结构以及具体权重必须一起确定后才能讨论中文表现。仓库的流式 NER 功能也不能作为完整投诉实时处理链路已验证的证据。

**中文证据三层。** 框架层：[分词实现](https://github.com/urchade/GLiNER/blob/f3c945702b9b5cde381c27b95310c47909bcfd80/gliner/data_processing/tokenizer.py)包含 Jieba、HanLP 后端和 Stanza 的中文语言分支。权重层：README 列出 multilingual 与 RelEx checkpoint；本轮尝试访问 multilingual 模型卡失败，未核验具体权重的中文任务指标、下载可用性及独立许可。项目实测层：无。中文分词支持、跨语言权重、中文投诉角色识别是三件分别需要证据的事。

### O04 · 567-labs/instructor：输出校验与失败重试

**证据与机制。** [校验文档](https://github.com/567-labs/instructor/blob/e12f8b49203b0c1f253d27c1e709d0a09b9fc5a8/docs/concepts/validation.md)支持 Pydantic 字段约束及自定义校验器；[实际重试实现](https://github.com/567-labs/instructor/blob/e12f8b49203b0c1f253d27c1e709d0a09b9fc5a8/instructor/v2/core/retry.py)处理解析/校验错误、重试次数和失败信息。旧路径 `instructor/core/retry.py` 只是兼容性重导出，本轮已沿调用位置核验实际实现。[原文引句示例](https://github.com/567-labs/instructor/blob/e12f8b49203b0c1f253d27c1e709d0a09b9fc5a8/docs/examples/exact_citations.md)通过在输入中寻找引文，移除不存在的引文及没有依据的事实项。

**可借鉴部分。** 检查字段类型、必填/可空规则、枚举、字符偏移有效性和字段之间的显式约束；把失败结果及重试记录保留下来。原文定位可作为校验层的一部分。

**边界与限制。** Pydantic 校验通过，只说明实现的约束通过；它不自动验证“该地址真的是事发地”。某句话在原文中存在，也不证明它支持模型归纳的结论。重试可能继续输出同一种语义错误。文档还明确说明自动执行异步自定义校验器不受支持，若需要异步外部核验，应在应用层处理。

**中文证据三层。** 框架层：字符串及自定义原文检查可表达中文内容，所读路径没有提供中文投诉语义准确率证据。权重层：这是调用和校验框架，无独立抽取权重；能力及条款取决于接入模型。项目实测层：无。

### O05 · dottxt-ai/outlines：生成时约束输出形式

**证据与机制。** [输出类型文档](https://github.com/dottxt-ai/outlines/blob/c52af8472c6fe8a607a733ec11b709476af77a96/docs/features/core/output_types.md)支持 Python/Pydantic 类型、选项、JSON Schema、正则和语法；[logits processor 文档](https://github.com/dottxt-ai/outlines/blob/c52af8472c6fe8a607a733ec11b709476af77a96/docs/features/advanced/logits_processors.md)说明在受支持的可控制本地后端上，将输出限制转为生成时 token 约束。[schema 转换源码](https://github.com/dottxt-ai/outlines/blob/c52af8472c6fe8a607a733ec11b709476af77a96/src/outlines/types/json_schema_utils.py)包含枚举、常量、可空类型及对象转换处理。

**可借鉴部分。** 在字段边界确定后，约束输出键名、类型与有限选项，减少解析失败。未知情况能否输出空值，取决于我们制定的字段规则及所选后端的支持。

**边界与限制。** 输出一个格式合法的“事发地”字符串，仍可能选中了居住地。不同 API/本地后端的约束能力并不完全一致；不能把本地 logits 控制能力无条件推广到所有模型服务。结构约束与语义核验需要分别评估，延迟和生成完整率也要实测。

**中文证据三层。** 框架层：类型/语法是表示机制，实际中文生成依赖模型、tokenizer 和后端。权重层：框架不提供被本项目验证的抽取权重。项目实测层：无，V100/H100 上的运行和性能均未测试。

### O06 · HumanSignal/label-studio：记录人工片段与关系判断

**证据与机制。** [关系标注模板](https://github.com/HumanSignal/label-studio/blob/8ef0b5ee3d84217e742823f27646c040506f4420/docs/source/templates/relation_extraction.md)用 `Labels` 标记文本片段，再用 `Relations` 给片段之间的关系加标签。这样可以显式记录某地址与某个问题片段之间的连接，而不只是各自圈出几个词。

**可借鉴部分。** 用于小规模边界试标、收集分歧案例和保留原文依据；模板可表达字段与关系，项目还需写明何时划分事项、如何处理重复地点、哪些关联可标为无法判断。

**边界与限制。** [一致性指标文档](https://github.com/HumanSignal/label-studio/blob/8ef0b5ee3d84217e742823f27646c040506f4420/docs/source/guide/agreement_metrics.md)明确标记 `tier: enterprise`。本轮不能把该页面列出的内置一致性分析全部计入 Apache 许可的社区版；若使用开源版本，可先基于导出结果独立计算所需指标。工具提供标注界面，边界共识仍来自独立标注、讨论和裁决。

**中文证据三层。** 框架层：文本和标签可配置；本轮未运行中文渲染、偏移与导入导出检查。权重层：不适用，标注工具不等于抽取模型。项目实测层：无。

### O07 · doccano/doccano：文本片段、关系及可配置重叠

**证据与机制。** [数据模型](https://github.com/doccano/doccano/blob/ab6b765ea326d07b2b806a4280cd097565e62321/backend/labels/models.py)保存 span 的起止偏移，通过项目 `allow_overlapping` 控制重叠，并记录关系两端。[关系测试源码](https://github.com/doccano/doccano/blob/ab6b765ea326d07b2b806a4280cd097565e62321/backend/labels/tests/test_relation.py)检查端点必须来自同一条 example。[中文界面字符串](https://github.com/doccano/doccano/blob/ab6b765ea326d07b2b806a4280cd097565e62321/frontend/i18n/zh/projects/annotation.js)和 README 多语言说明提供了中文 UI 的直接依据。

**可借鉴部分。** 以相对简单的文本标注界面记录片段及关联关系；用重复/重叠的人工案例讨论“同一原文片段能否承担多个字段角色”。对投诉数据是否允许重叠，仍应由领域边界说明决定。

**边界与限制。** 已核对的[指标接口](https://github.com/doccano/doccano/blob/ab6b765ea326d07b2b806a4280cd097565e62321/backend/metrics/views.py)主要涉及进度及标签分布，不能据此声称存在完整的标注者一致性分析。`example_overlapping.jsonl` 缓存样例本身只是重复 span，因此本报告对可配置重叠的判断主要依据数据模型，不把样例当成任意 UI 交互都已验证的证据。最近默认分支提交在 2026-01，说明本快照的更新节奏较慢，不足以判断项目已停止维护。

**中文证据三层。** 框架层：中文 UI 字符串与多语言文档可核验，混合中文、英文、emoji 的偏移往返还需运行检查。权重层：不适用。项目实测层：无。

## 3. 对字段判断边界的直接启示

| 需要弄清的边界 | 哪类工程能力有帮助 | 开源能力仍无法代替的工作 |
|---|---|---|
| 原文片段与归纳后的问题名称 | O01 的原文定位；O02/O03 的片段输出；O04 的原文校验 | 决定哪些输出允许概括，哪些必须逐字可定位，分别评估 |
| 地址文字与地址角色 | O02 的嵌套关系、O03 的关系路线、O06/O07 的人工关系标注 | 区分事发地点、居住地、办理机构地点；保留无法确定的关系 |
| 一个问题的多个描述与多个独立事项 | O01 的属性/分组及 O06/O07 的片段关联界面 | 编写事项拆分原则与成对反例，并核对标注分歧 |
| 未提供、明确否定、猜测和已解决 | O04/O05 的可空类型/选项和校验扩展 | 确定这些状态是否在事项层或具体事实层判断，以及证据跨度 |
| 多个字段是否属于同一事项 | 关系抽取与有指向的人工标注 | 单字段 F1 之外，测量连接是否正确、是否把两个事项拼接 |
| 格式成功与语义正确 | O04 的结果校验、O05 的生成约束、O01 的原文定位 | 分别统计格式失败、无原文依据、角色错误、遗漏事项和绑定错误 |

以上是由已核验能力导出的调研问题，未把它们定为本项目最终字段名、枚举或数据结构。没有一个仓库直接给出适用于本项目投诉语料的完整边界定义。

## 4. 下一轮验证应收集的证据

先用少量私有投诉和明确标注为虚构的边界例子，验证人对事项、字段和角色能否形成可重复的判断。保留“住在 A、投诉 B”“同一地址多次出现”“同一事项先解决后复发”“多个地点对应多个问题”等案例的分歧原因；不把这些例子直接扩写成已经定稿的 schema。

待边界初稿形成，再按同一输入和人工参考结果比较三类候选：片段/关系抽取、带原文依据的 LLM 抽取，以及必要的二者组合。输出校验和生成约束作为独立变量，避免把 JSON 成功率提升误认为语义抽取提升。

具体权重需单独登记版本、来源、中文任务依据、许可、最大输入与截断方式、依赖、失败行为及硬件测量；V100 与 H100 的运行结果分别记录。无需为了调研先安装所有框架或更改现有检索环境。

## 5. 核验记录与已知缺口

- 结构化来源登记：[repositories.json](sources/repositories.json)，包括 O01–O07、固定提交、已读路径、文件 SHA256、来源链接、能力与中文证据层级。
- 检索/访问记录：[repository-search-log.jsonl](sources/repository-search-log.jsonl)。本轮为有目的的候选选择，未声称遍历全部 GitHub 项目或作穷尽排名。
- GitHub 元数据、目录树、README 和源码已缓存于被忽略的 `.cache/repositories/`。每个 `reviewed_paths` 条目都核对本地文件与成功获取日志的 SHA256；仅存在于 tree 的路径不算已经读过。
- 前次 raw.githubusercontent.com 部分路由超时；恢复时复用已成功缓存，仅通过 GitHub Contents API 补齐缺少的固定提交许可和关键实现。失败记录保留，未改写成成功。
- 当前“维护”判断只基于提交/发布/归档元数据；未审计项目安全、CI 健康度、问题响应时长或实际企业使用。
- 尚未确认具体模型权重在本项目服务器上的下载、加载、许可与中文投诉表现。模型卡访问失败明确保留为证据缺口。

本报告的证据足以划分可借鉴的工程机制及其局限；字段最终边界、模型选型和检索收益需由后续本项目数据验证决定。
