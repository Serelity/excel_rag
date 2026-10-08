# 投诉内容结构化：文献证据与方法边界

完成日期：2026-10-08（Asia/Shanghai）。本文件为首轮定向选读和证据地图，用于讨论字段边界以及字段要求怎样影响方法选择。最终字段、标注规则和部署模型仍待本项目验证。

## 1. 本轮范围及证据质量

核验了 15 篇论文的身份：6 篇期刊论文和 9 篇会议论文。其中 **14 篇有内容依据：11 篇选读了相关正文，3 篇仅读摘要；P04 仅核验书目信息，列为背景候选，不支持方法结论**。期刊包括 *Computational Linguistics*、*IEEE Transactions on Knowledge and Data Engineering*、*Frontiers of Computer Science*、*Applied Intelligence* 等；会议包括 ACL、NAACL、EMNLP-IJCNLP、AAAI 和 ICLR。未核验期刊分区、影响因子或 CCF 版本，因此本报告不提供这些排名。

检索沿四组问题展开：NER/事件/生成式信息抽取综述、标注与字段边界、统一及开放信息抽取、投诉识别与中文处理。先用 Crossref 标题查询发现候选，再核对 DOI、出版商页、ACL Anthology、arXiv 的论文信息与开放正文。实际查询、访问失败及时间保存在 [检索日志](sources/literature-search-log.jsonl)。正式引用的完整作者、来源与阅读范围在 [papers.json](sources/papers.json)，引用格式在 [references.bib](sources/references.bib)。

这是针对当前两项问题的有界调研，包含经典工作与截至 2024 年的代表性方法，**未穷尽 2025—2026 年的新模型和文献**。没有执行预注册的全数据库筛选或 PRISMA 系统综述；不能把这组来源称为截至 2026 年的最新 SOTA 排行榜。开放 PDF 缓存通过解析检查，仅用于核验本报告涉及的片段，下载整篇不代表逐页精读。

检索日志保留机器记录的 UTC 时间；前次检索发生在 2026-10-07 16 时 UTC，对应中国时间 2026-10-08 零时。元数据 `fetched_at` 按 Asia/Shanghai 显示，逐字段 `checked_at` 保留实际原始时戳。

## 2. 首先需要解决的任务边界

文献能支持的最直接结论是：**字段名不足以定义抽取任务，字段的语义、跨度、角色和取值规则需要成为可检验的规范。** GoLLIE 举例说明，同样叫 `Person`，ACE05 与 CoNLL03 是否标注代词就不同；模型需要了解标签定义，而不仅是标签名称（P10，PDF 第 2—3 页）。标注一致性综述进一步指出，选择标注单元和确定标签是不同环节，两者都可能产生分歧（P03，§4.3）。

对于本项目，下面各层是待比较的任务表达方式。这里列的是研究问题及其能力要求，不是已确定的数据 schema。

| 层次 | 要回答的问题 | 对本项目的意义 | 主要依据 |
|---|---|---|---|
| 原文提及和跨度 | 哪些字符构成地点、对象、问题表述或诉求？是否包含修饰词、否定词？ | 可回到原文核对，明确长地址和问题短语的起止位置 | P02 摘要；P03 §4.3；P12 §2.1 |
| 类型或类别 | 该提及属于什么类型？是否允许一项属于多个层级？ | “物业公司”是主体，“物业问题”是类别；两者取值和评价方式不同 | P05 §2；P10 §6 |
| 关系和语义角色 | 哪个地点属于哪个问题？某单位是被投诉方、办理方还是转述者？ | 识别出地名之后，仍需判定“住址/事发地”及其关联事项 | P08 §2.1—2.2；P09 Figure 3；P13 §2.1 |
| 多事项记录 | 一段文本有几个事项，哪些属性属于同一事项？共享地点如何表示？ | 两个问题不能只放进一个无关联实体列表；同一事项跨句出现时也不能逐句孤立抽取 | P14 Figure 2、§3；P03 §4.3 |
| 归一化或派生表示 | 原文短语是否映射到类别/标准地址？是否再生成核心问题摘要？ | 原文、标准值和摘要应能区分来源；标准化可能需要词表或地址库 | P05 §2、§4.4；对本项目的推论，尚需本地验证 |

最后一层尤其容易被“抽取”这个词掩盖：从原文返回“某路 3 号”与补全行政区划不是同一输出要求；复制问题证据与写一句核心问题摘要也不是同一评价任务。当前文献说明了不同任务的结构，**并未证明本项目需要哪些派生字段，也未证明归纳后的输入必然比完整原文检索更好**。

## 3. 五项可用于后续设计的发现

### 3.1 字段定义必须与真实标注结果一致

P10 的错误分析发现，某数据集的指南要求标年份，但实际标注漏掉年份；还有粗粒度 `Person` 与细粒度 `Scientist` 的排斥关系含糊。模型遵循指南时，甚至可能因金标准与指南不一致而被判错（PDF 第 9 页）。这支持我们先用少量投诉检查“人是否理解一致”，再评价模型；不能把指南含糊造成的分歧全部归为模型能力不足。

P03 则说明，标注者一致可以支持可靠性，但一致并不保证规则捕捉到了正确业务概念（PDF 第 3 页）。因此未来需要同时记录规则版本、分歧例子和裁决理由。对固定分类可以讨论 κ/α；对自由跨度和多事项结构，应先定义比较单元和匹配规则，不能直接套一个 κ 数值作为全部质量指标。

### 3.2 地点/对象识别与事项角色关联需要分别评价

P08 将信息抽取分成 `spotting`（定位信息片段）与 `associating`（关联片段）；P09 也区分文本片段结构和语义关联。P13 在文档级关系抽取中将共指、实体链接、关系与证据句分开。由此可提出本项目假设：模型即使找对所有地址，也可能把居住地误当事发地，或把甲事项地址分给乙事项。

P14 明确定义 entity mention、event role、event argument 和 event record，并展示跨句散落、多条事件记录共享部分论元的情况。其结构思想与“一条投诉有多个问题”相关；金融公告的具体字段和事件类型不能直接移入投诉数据。

### 3.3 “必须有依据”和“允许为空”会改变方法与训练数据

P13 的关系标注要求选取支持句，并要求关系反映在文档中，不依赖外部世界知识。P11 的负类型采样将文本中不存在的类型对应到空列表；P12 的负实体类型实验显示，假阳性与漏检会随训练采样策略改变。对本项目的启示是，后续评估需要覆盖“确实未提及”和“角色无法判断”的输入，不能只测试字段齐全的好例子。

同时，证据字符串出现在原文中，仅证明该字符串存在；它是否支持该角色或结论，还需语义核对。例如同一地址在原文出现多次，字符对齐必须定位正确位置。P08 的 §4.1 说明其生成的片段还需通过匹配回映到 offset，这也是独立的误差来源。

### 3.4 字段要求应先于模型规模决定候选方法

P05 将 NER、关系、事件任务及提示、约束生成、监督微调分别讨论。P12 证明实体片段与类型匹配可以使用较小的双向编码模型；P10 提供了遵循复杂指南的生成式方案，但仍报告模糊标签和预训练偏好造成的失败。两者任务和训练设置不同，不能用跨论文一个总 F1 给本项目排序。

P12 的中文 MultiCONER 结果也显示，论文里的英语 GLiNER 与多语言版本差距明显，多语言版本仍与该数据集的有监督基线存在差距（Table 3，PDF 第 6 页）。这只说明**语言和训练设置是选型条件**，不构成对最新 GLiNER 版本或本项目中文地址任务的结论。

### 3.5 投诉分类研究不能代替字段抽取证据

P07 明确区分投诉行为与负面情绪，并介绍定义、独立标注、先校准后裁决的过程。但它解决英文社交媒体上的投诉二分类。P06 则面向市民投诉文本分类，领域相关，但本轮仅能读摘要，未取得全文和公开数据。

因此二者可以帮助澄清“什么算投诉”和领域局限，不能直接支持地址角色、办理状态、多事项结构等字段设计，也不能把其分类提升幅度当作当前检索效果预期。

## 4. 字段要求与方法的证据映射

| 如果后续确认有这项要求 | 可比较的方法方向 | 必须检验的失败模式 | 文献依据与边界 |
|---|---|---|---|
| 返回原文中的实体或短语，要求精确跨度 | 字符/序列标注、span-type 匹配、抽取式问答 | 错误切分、漏修饰词、重叠片段、重复词对齐 | P12/P15；适用于片段，不自动解决事项分组 |
| 需要把地点、对象、诉求关联到特定事项 | 关系/事件抽取、统一结构预测、结构生成 | 属性串项、共指错误、跨句遗漏、多事项合并 | P08/P09/P13/P14；需本项目事项规则 |
| 标签定义经常变化且需要自然语言指南 | 带定义和正反例的提示、指南遵循微调 | 只按标签常识抽取、忽略排除条件、过粗/细类别冲突 | P10；其结果依赖特定训练，不能直接外推到任意模型 |
| 希望生成可直接检索的简短问题 | 有证据约束的生成/摘要，保留原文作对照 | 删除否定、时间或对象；加入未出现事实 | P05 可支持生成方法风险；检索增益尚无本项目证据 |
| 文本缺字段时允许空值或不确定 | 阴性例、空输出、拒绝/复核路径 | 强行填满、把“不确定”误写为不存在 | P08/P11/P12；原文未提及与无法判断的业务区分仍需确定 |
| 输出必须满足固定 JSON 结构 | 结构化约束解码及结构校验 | 格式有效但字段语义错误 | P05 §4.4 讨论约束；格式和语义需分开检查 |
| 地址需要标准名称、层级或坐标 | 原文地点抽取后接规范库/消歧步骤 | 同名地点、错误补全、旧地名、角色误判 | 本组论文没有中文政务地址标准化的直接验证，需另补规范与领域证据 |

这些是候选路径，尚未决定应采用单模型、多个步骤或规则与模型组合。将所有字段都交给同一个生成模型也是待验证选项之一，而不是文献已经给出的默认结论。

## 5. 逐篇证据卡

### P01 — Nadeau & Sekine（2007），NERC 综述

**文献**：David Nadeau, Satoshi Sekine. *A survey of named entity recognition and classification*. **Lingvisticae Investigationes**，期刊论文。DOI：[10.1075/li.30.1.03nad](https://doi.org/10.1075/li.30.1.03nad)。

**已核验/阅读**：Crossref 标题、作者、年份、期刊、DOI及摘要。摘要覆盖 1991—2006 年的语言、实体类型、领域、特征与评价问题，提到精确及复杂匹配评价。**用途**：提供实体识别任务与评价维度的背景。**局限**：未读正文，不能从摘要推导投诉专用规则或当前最佳模型。

### P02 — Li 等（2022），深度 NER 综述

**文献**：Jing Li, Aixin Sun, Jianglei Han, Chenliang Li. *A Survey on Deep Learning for Named Entity Recognition*. **IEEE Transactions on Knowledge and Data Engineering**，期刊论文。DOI：[10.1109/TKDE.2020.2981314](https://doi.org/10.1109/TKDE.2020.2981314)；[作者预印本](https://arxiv.org/abs/1812.09449)。

**已核验/阅读**：Crossref 正式卷期与 arXiv 的作者、标题、关联 DOI；仅读 arXiv 摘要。正式卷期年记 2022，预印本最早为 2018、修订到 2020。**用途**：输入表示、上下文编码器与标签解码器可分别选择；NER 的对象是带语义类型的实体提及。**局限**：本轮没有阅读综述中的详细比较，不以 DOI 中的 2020 替代卷期年，也不把它视为近期模型排名。

### P03 — Artstein & Poesio（2008），标注一致性

**文献**：Ron Artstein, Massimo Poesio. *Survey Article: Inter-Coder Agreement for Computational Linguistics*. **Computational Linguistics** 34(4)，555–596，期刊论文。DOI：[10.1162/coli.07-034-R2](https://doi.org/10.1162/coli.07-034-R2)；[ACL 正文入口](https://aclanthology.org/J08-4004/)。

**已读定位**：PDF 第 3 页（印刷页 557）可靠性与有效性；第 26 页（580）§4.3 “Marking Boundaries and Unitizing”；第 32 页（586）跨度重叠/距离选择的例子。**用途**：先明确标注单元，再评价类别和边界的一致性；一致不能自动等价为正确。**局限**：不同标注任务需不同匹配假设，不能直接给所有字段统一 κ 阈值。

### P04 — Hogenboom 等（2016），事件抽取背景候选

**文献**：Frederik Hogenboom, Flavius Frasincar, Uzay Kaymak, Franciska de Jong, Emiel Caron. *A Survey of event extraction methods from text for decision support systems*. **Decision Support Systems**，期刊论文。DOI：[10.1016/j.dss.2016.02.006](https://doi.org/10.1016/j.dss.2016.02.006)。

**状态：仅身份核验（background_metadata_only）**。Crossref 未给摘要；出版商访问 403，机构页补查中有 404 和论文身份不匹配结果，均已记录。没有据此判断“没有开放版本”。**本篇不支持本报告任何方法或字段结论**，保留作后续取得可用正文时的阅读线索。

### P05 — Xu 等（2024），生成式信息抽取综述

**文献**：Derong Xu, Wei Chen, Wenjun Peng, Chao Zhang, Tong Xu, Xiangyu Zhao, Xian Wu, Yefeng Zheng, Yang Wang, Enhong Chen. *Large language models for generative information extraction: a survey*. **Frontiers of Computer Science** 18，186357，期刊综述。DOI：[10.1007/s11704-024-40555-y](https://doi.org/10.1007/s11704-024-40555-y)；[出版商开放 PDF](https://link.springer.com/content/pdf/10.1007/s11704-024-40555-y.pdf)。

**已读定位**：PDF 第 2/4 页、§2 的 NER/RE/EE 定义；第 9—10 页 §4.2—4.7 的提示、零样本、约束生成、少样本和微调；第 11 页 §6—7 的评价与局限。**用途**：字段要求应映射到不同任务与学习条件；结构格式、内容真实性、语言/领域迁移分别需要证据。**局限**：这是多来源、多设置的综述；论文表格不能构成同一条件下的本地模型排名，尚无本项目验证。

### P06 — Wang 等（2023），市民投诉分类

**文献**：Yuanhang Wang, Yonghua Zhou, Yiduo Mei. *A joint attention enhancement network for text classification applied to citizen complaint reporting*. **Applied Intelligence** 53，19255–19265，期刊论文。DOI：[10.1007/s10489-023-04490-y](https://doi.org/10.1007/s10489-023-04490-y)；[出版商页面](https://link.springer.com/article/10.1007/s10489-023-04490-y)。

**已读定位**：公开 Abstract、Data Availability、订阅提示。**用途**：是市民投诉领域的分类应用线索，采用注意力增强和关键词信息。**局限**：未读收费正文、未核验数据细节；数据因市民隐私未公开。摘要中的性能提升没有在本轮复核，不能外推为结构抽取、地址判定或检索收益。

### P07 — Preoțiuc-Pietro 等（2019），投诉与情感的区别

**文献**：Daniel Preoțiuc-Pietro, Mihaela Gaman, Nikolaos Aletras. *Automatically Identifying Complaints in Social Media*. **ACL 2019**，会议论文。DOI：[10.18653/v1/P19-1495](https://doi.org/10.18653/v1/P19-1495)；[ACL 页面](https://aclanthology.org/P19-1495/)。

**已读定位**：PDF 第 1 页 §1/Table 1 的投诉与情感区别；第 3 页 §3.2 的两位独立标注者、100 条校准后丢弃、正式标注与裁决；第 8—9 页 Tables 6/8/9 的任务及领域比较。**用途**：不能只用情绪强弱代替问题识别；可参考定义和校准流程。**局限**：英文 Twitter 二分类，每条含一个投诉行为即标投诉；不处理中文长投诉的字段和事项分组。

### P08 — Lu 等（2022），UIE

**文献**：Yaojie Lu, Qing Liu, Dai Dai, Xinyan Xiao, Hongyu Lin, Xianpei Han, Le Sun, Hua Wu. *Unified Structure Generation for Universal Information Extraction*. **ACL 2022**，会议论文。DOI：[10.18653/v1/2022.acl-long.395](https://doi.org/10.18653/v1/2022.acl-long.395)；[ACL 页面](https://aclanthology.org/2022.acl-long.395/)。

**已读定位**：PDF 第 3—4 页 §2.1—2.2 的 SEL/SSI；第 6 页 §3.3 的 rejection mechanism 和 §4.1 的评估；第 7—8 页 Table 2—5。**用途**：把“取哪些片段”“片段间什么关系”和“按哪个 schema 输出”分开；缺失类型也需训练信号。**局限**：生成片段还要回映 offset，文本重复时可能错位；论文结果与同名工程框架的某个权重不是同一证据。

### P09 — Lou 等（2023），USM

**文献**：Jie Lou, Yaojie Lu, Dai Dai, Wei Jia, Hongyu Lin, Xianpei Han, Le Sun, Hua Wu. *Universal Information Extraction as Unified Semantic Matching*. **AAAI 2023**，会议论文。DOI：[10.1609/aaai.v37i11.26563](https://doi.org/10.1609/aaai.v37i11.26563)；[官方 PDF](https://ojs.aaai.org/index.php/AAAI/article/download/26563/26335)。

**已读定位**：PDF 第 1—3 页 structuring/conceptualizing、Figure 3、token linking；第 6 页跨类型/监督设置。**用途**：结构抽取可以表达为片段与标签/片段之间的关联，适合与自由生成路径比较。**局限**：其连续跨度和关系结构不自动解决原文外的标准化或摘要；Crossref 的宽泛 `journal-article` 已按实际会议来源更正为会议论文。

### P10 — Sainz 等（2024），GoLLIE

**文献**：Oscar Sainz, Iker García-Ferrero, Rodrigo Agerri, Oier Lopez de Lacalle, German Rigau, Eneko Agirre. *GoLLIE: Annotation Guidelines improve Zero-Shot Information-Extraction*. **ICLR 2024**，会议论文；[arXiv:2310.03668](https://arxiv.org/abs/2310.03668)。

**已读定位**：PDF 第 2 页 `Person` 标注定义差异；第 3 页 §3.1 的指南表示；第 7—8 页结果和消融；第 8—9 页 §6 的标签/指南错误分析。**用途**：这是“字段边界应先明确”的直接证据，涵盖标签名歧义、排除条件、粗细类型冲突与指南/标注不一致。**局限**：特定微调方案提高指南遵循，模糊定义依旧困难；不能推成字段越多越好或大模型越大越可靠。初版预印本 2023；会议年 2024 由 arXiv Comments 和 PDF 首页核验，未取得 OpenReview API 记录。

### P11 — Zhou 等（2024），UniversalNER

**文献**：Wenxuan Zhou, Sheng Zhang, Yu Gu, Muhao Chen, Hoifung Poon. *UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition*. **ICLR 2024**，会议论文；[arXiv:2308.03279](https://arxiv.org/abs/2308.03279)。

**已读定位**：PDF 第 3—4 页 §3.1—3.2 的英语 Pile 输入、蒸馏、负类型空列表；第 8 页 §5.4 的负采样和数据集模板分析。**用途**：把一个任务蒸馏到较小模型、让不存在的类型能输出空值，是可比较的工程路径。**局限**：蒸馏数据包含模型产生的标签，不能当独立人工金标准；训练和主任务以英语 NER 为主，不能证明中文投诉角色关联能力。

### P12 — Zaratiana 等（2024），GLiNER

**文献**：Urchade Zaratiana, Nadi Tomeh, Pierre Holat, Thierry Charnois. *GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer*. **NAACL 2024**，会议论文。DOI：[10.18653/v1/2024.naacl-long.300](https://doi.org/10.18653/v1/2024.naacl-long.300)；[ACL 页面](https://aclanthology.org/2024.naacl-long.300/)。

**已读定位**：PDF 第 2—3 页 §2.1—2.3 的 span/type 匹配及 flat/nested 解码；第 6 页 §4.2/Table 3 的跨语言；第 8 页 §5.3 的负实体采样。**用途**：高频、可精确定位的实体字段可以比较轻量模型，并把跨度、重叠、语言和阴性例纳入条件。**局限**：论文是 NER 任务，未给事项关系或问题摘要能力；2024 结果不代表 2026 仓库全部版本的表现。

### P13 — Yao 等（2019），DocRED

**文献**：Yuan Yao, Deming Ye, Peng Li, Xu Han, Yankai Lin, Zhenghao Liu, Zhiyuan Liu, Lixin Huang, Jie Zhou, Maosong Sun. *DocRED: A Large-Scale Document-Level Relation Extraction Dataset*. **ACL 2019**，会议论文。DOI：[10.18653/v1/P19-1074](https://doi.org/10.18653/v1/P19-1074)；[ACL 页面](https://aclanthology.org/P19-1074/)。

**已读定位**：PDF 第 3 页 §2.1 的 Stage 2—4（提及/共指、实体链接、关系及支持句）；第 4 页 §3 的跨句推理分析。**用途**：为原文片段、指代关系、关系判断与支持证据提供分层参考。**局限**：数据来自英语 Wikipedia/Wikidata；其关系分类和推理范围不是投诉领域的最终规则，文档内推理也不等于可任意补全外部事实。

### P14 — Zheng 等（2019），Doc2EDAG

**文献**：Shun Zheng, Wei Cao, Wei Xu, Jiang Bian. *Doc2EDAG: An End-to-End Document-level Framework for Chinese Financial Event Extraction*. **EMNLP-IJCNLP 2019**，会议论文。DOI：[10.18653/v1/D19-1032](https://doi.org/10.18653/v1/D19-1032)；[ACL 页面](https://aclanthology.org/D19-1032/)。

**已读定位**：PDF 第 1—2 页的多事件/论元散落问题与 Figure 2；第 3 页 §3 的 entity mention、event role、event argument、event record 定义。**用途**：强调一段文本可以对应多条有各自角色的事件记录，单纯把所有实体合并成集合会丢失结构。**局限**：金融公告的预定义事件、规范表达和远程监督条件与热线投诉不同；不是本项目事项拆分的直接模型验证。

### P15 — Zhang & Yang（2018），中文实体边界

**文献**：Yue Zhang, Jie Yang. *Chinese NER Using Lattice LSTM*. **ACL 2018**，会议论文。DOI：[10.18653/v1/P18-1144](https://doi.org/10.18653/v1/P18-1144)；[ACL 页面](https://aclanthology.org/P18-1144/)。

**已读定位**：PDF 第 1—3 页的分词/字符边界、§3 模型与 BIOES；第 5 页 Table 1 的数据范围。**用途**：中文抽取不能照搬英语按空格分词的跨度假设；切词和实体界限会影响后续结构。**局限**：2018 架构与对应基准提供方法背景，不能作为当前选型或投诉地址准确率证据。

## 6. 尚待回答的知识缺口

1. **中文政务投诉的字段规范**：当前 14 篇内容来源中，没有一篇直接验证本项目完整的“事项—问题—地点角色—对象—诉求—状态”方案；还需行业规范、标注指南和真实数据分歧检查。
2. **地址角色和标准化**：NER 只覆盖地点提及的一部分，居住地/事发地角色、行政区补全、别名和地理消歧需独立证据；本轮不能直接指定哪种地址算法。
3. **多事项、否定、转述和办理状态**：本轮结构思想可参考事件抽取，但这些字段的取值、冲突处理和证据范围仍属待定业务问题，不能因文献术语相近就视为已有标准。
4. **抽取对历史案例检索的实际收益**：需保留原文全文检索作为比较对象，逐项增加结构信息并观察同一批案例的人工相关性判断。抽取得分、JSON 合法率和检索收益需要分别报告。
5. **模型与运行条件**：需要在字段任务确定后，针对中文、输入长度、可用标注量、吞吐、显存和部署许可比较模型；论文中的跨任务平均分不能直接选出本项目最优模型。

适合接下来的产出是一份可修订的“边界问题清单”：给每个候选概念列出包含/排除条件、跨度、归属、缺失情况和容易分歧的人工例子，再用小批双人标注检查是否可稳定执行。该步骤检验字段是否定义清楚，随后才有依据比较具体方法与模型。
