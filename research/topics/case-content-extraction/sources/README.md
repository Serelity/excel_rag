# 来源与核验记录

本目录存放首轮及实践补充的可追溯资料。来源文件中的读取范围和限制，是理解报告结论的必要部分。

| 文件 | 用途 |
| --- | --- |
| `papers.json` | 文献元数据、来源 ID、可访问状态、已读范围与适用限制 |
| `references.bib` | 根据已核对元数据生成的论文引用 |
| `repositories.json` | 开源仓库、版本、许可、维护情况、具体证据路径 |
| `guidelines.json` | 标注与语义规范的来源、版本、已读范围 |
| `literature-search-log.jsonl` | 学术检索和正文核验的实际请求记录 |
| `repository-search-log.jsonl` | 仓库发现、版本与能力核验记录 |
| `boundary-search-log.jsonl` | 标注规范及边界资料的实际访问记录 |
| `papers-validation.json` | 学术记录格式与证据字段校验结果 |
| `papers-deduplicated.json` | 基于 DOI / arXiv 等精确标识的保守去重结果及来源索引 |
| `verification.json` | 首轮 15 篇文献、7 个仓库、5 份规范的检查快照；不表示已检查后续新增材料 |
| `enterprise-practice.json` | C01/C02：Uber、AliMe 的流程、指标、短证据与阅读位置 |
| `civic-practice.json` | C04–C06：湛江、张家口、海淀的业务方/建设方材料与效果口径 |
| `applied-practice.json` | C07/C08：QJE 客服生产研究、Snorkel 合作抽取的具体证据 |
| `opensource-practice.json` | C10/C11：JetBrains、Airbus，以及未纳入主要结论的 N26/Bosch 线索 |
| `practice-repositories.json` | O08/O09：Rasa/Haystack 固定提交与 6 个已读路径 |
| `enterprise-papers.json`、`enterprise-references.bib`、对应 `*-validation.json` / `*-deduplicated.json` | P18–P20：COTA、AliMe Assist、AliMe KBQA 的正式书目与作者公开全文；单独校验结果 |
| `applied-papers.json`、`applied-references.bib`、对应校验结果 | P16/P17：QJE 正式版、Snorkel 期刊扩展版书目与引用 |
| `papers-expanded.json`、`papers-expanded-references.bib`、`papers-expanded-validation.json`、`papers-expanded-deduplicated.json` | P16–P20 的联合登记、格式校验和保守去重；首轮 `papers.json` 不被覆盖 |
| `enterprise-practice-search-log.jsonl`、`civic-practice-search-log.jsonl`、`applied-practice-search-log.jsonl`、`opensource-practice-search-log.jsonl` | 实践补充的实际查询、获取与失败记录 |
| `papers-expanded-validation.json`、`papers-expanded-deduplicated.json` | 对上述三份论文登记合计 20 条记录的联合格式校验与保守去重 |
| `practice-verification.json` | 实践补充的数量、链接、哈希、引用、证据审查及未完成验证范围 |

报告中的 `Pxx` 对应论文，`Oxx` 对应开源项目，`Gxx` 对应规范资料，`Cxx` 对应实践案例。案例 ID 分工时预留，空号不表示另有未交付案例；实际纳入 C01/C02/C04/C05/C06/C07/C08/C10/C11，共 9 个。多个 URL 可能属于同一来源；请求次数、独立来源数、已阅读全文数不能混用。

空引用量表示本轮未核验计数；空 PDF 路径表示未交付 PDF，不代表没有开放版本。失败访问保留在日志中，不能当作来源不存在的证据。成功响应也需结合身份核对和已读范围使用。

临时 API 响应、HTML、公开全文阅读副本保存在上一级被 Git 忽略的 `.cache/`。提交资料以元数据、定位链接和自行综合为主。

核验工具为 academic-search 技能中的 `academic-records.mjs`（Node.js）：先对 `papers.json` 执行 `validate`，再执行 `dedupe`，结果保存为上述两个文件。工具检查记录格式和保守重复，不判断论文结论是否成立。其他资料检查及人工式交叉审查范围见 `verification.json`。

实践补充另对三份论文登记的组合使用同一工具，保留原首轮校验文件。关键数值均附来源段落、表格、分母/基线或未披露说明；论文与部署材料分开登记，不把版本、转载或多个实验表重复当作独立应用。
