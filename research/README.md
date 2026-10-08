# 项目调研资料

本目录保存本项目的方法与方案调研：研究问题、检索记录、文献与开源项目证据、比较分析，以及仍需本项目数据验证的问题。

当前已进入投诉原文抽取规范的第一轮开发。依据80条人工复核形成了[可执行规范v1](topics/case-content-extraction/12-extraction-spec-v1.md)，包括字段边界、提示词、输出结构和离线校验；其效果和模型选择仍待试抽验证。资料核验日期：2026-10-08。

本轮进一步完成[H100试抽适配器](topics/case-content-extraction/14-h100-adapter-delivery.md)，环境固定为用户指定的`civic-rag-retrieval`；已通过本地模拟测试，服务器实际试抽待执行。

先读[扩展综合](topics/case-content-extraction/04-synthesis-and-next-questions.md)和[实践案例对照](topics/case-content-extraction/05-practice-cases.md)。本次增加了 9 个深读案例，说明企业客服、政务热线和相关抽取研究怎样做成、取得什么结果、哪些经验能迁移到投诉历史案例查询；再从[主题目录](topics/case-content-extraction/README.md)查原始证据。

| 主题 | 入口 | 状态 |
| --- | --- | --- |
| 投诉内容的字段边界、方法与模型 | [case-content-extraction](topics/case-content-extraction/README.md) | 20篇文献、9个实践案例；80条人工复核完成，第一版规范与校验工具已形成，尚未进行新模型试抽 |

通用调研方法、记录模板与各主题资料分别保存；引用必须能够追溯到原始论文、官方文档或明确版本的仓库文件。

2026-10-08 已继续到本地开发样例研究：[80 条字段用途与边界草案](topics/case-content-extraction/10-field-boundary-study.md)。初版逐条草案与241段原文证据保存在私有数据目录，作为后续复核的起点；没有新增模型或检索效果结论。

同日已导入用户完成的80条人工复核：[复核发现与修订落实](topics/case-content-extraction/11-human-review-findings.md)。70条标为同意、10条标为修改，合计13条包含意见；意见均已落实，B015对象边界修正已获用户确认。现有246段证据，初版草案作为历史快照保留。

| 项目级资料 | 用途 |
| --- | --- |
| [调研方法](METHODS.md) | 来源选择、检索、证据核验、反证与方案比较方法 |
| [主题记录模板](templates/topic-template.md) | 为后续数据治理、检索、排序等调研复用 |

目录约定：`topics/` 中每个主题独立保存报告；主题内的 `sources/` 保存来源和检索记录；`.cache/` 保存临时阅读副本并由 Git 忽略。真实投诉和标注继续存放于项目私有数据目录。
