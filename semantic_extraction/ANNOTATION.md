# 浏览器标注

这一步建立抽取评测所需的参考答案。当前 20 条已用于 prompt 调试，
属于开发集；后续报告检索提升时，需要另选未用于调试的测试集。

## 打开现有 20 条

本机已生成 `data/evaluation/semantic-gold-v1.annotation.html`，
用 Chrome 或 Edge 打开即可。页面包含这 20 条原文；本机工作表没有附带服务器模型结果。
页面完全离线，不需要启动服务或 GPU。

## 服务器生成页面

在已有工作表基础上生成，命令使用固定 Conda 环境名：

```bash
cd /seu_share/home/huangkai/220243809/12345/excel_rag
conda run --no-capture-output -n civic-rag-extract python -m semantic_extraction.annotation --worksheet data/evaluation/semantic-gold-v1.worksheet.jsonl --output data/evaluation/semantic-gold-v1.annotation.html
```

将生成的 HTML 下载到自己的电脑，再用浏览器打开。全部资源已包含在 HTML 中。
命令拒绝覆盖已有页面，需要重新生成时请使用新文件名。

## 填写与保存

1. 阅读左侧原文，填写实际标注人和当前状态。模型结果默认折叠。
2. 添加当前问题或历史背景问题，填写问题名称、需要的知识和原文证据。
   每个证据框对应一个连续片段，可以包含换行。证据必须直接来自原文。
3. 明确表达本次诉求的问题勾选“直接表达本次来电的当前诉求”。
4. 点击“确认本条并继续”。缺少必填项或证据不匹配时页面给出提示。
   再次编辑已确认记录会将它恢复为待确认。
5. 随时点“导出草稿”保存；下次打开页面后，点“导入已保存工作表”继续。
   浏览器里的修改**不会自动写回 HTML 或原 JSONL**。请确认下载完成再关闭页面。
6. 全部确认后点“导出全部已确认标注”，得到
   `semantic-gold-v1.completed.worksheet.jsonl`。

导入时会核对工单 ID、行号、原文和内容哈希，避免错用另一批样本。
原文中的 HTML、脚本和命令只以文本显示。

状态判断以正文为依据。“自主挂机，故办结”没有证明实质问题已解决，
可选 `unclear`（信息不足），保留零问题并在备注解释。
“此前说已解决，现在仍没办好”则应根据当前诉求标注，不能只匹配“已解决”三个字。

## 校验并冻结

将导出的已确认工作表放入服务器的 `data/evaluation/`，运行：

```bash
conda run --no-capture-output -n civic-rag-extract python -m semantic_extraction.evaluation finalize-gold --input data/derived/qwen3-pilot-v2-2000.jsonl --worksheet data/evaluation/semantic-gold-v1.completed.worksheet.jsonl --output data/evaluation/semantic-gold-v1.jsonl
```

后端会再次核对原始数据和证据。随后按 [评测说明](EVALUATION.md) 进行候选问题对齐与评分。
页面处理的是 gold 标注工作表，候选结果的一对一对齐仍使用独立的 adjudication 工作表。
所有含数据的生成文件都放在被 Git 忽略的 `data/` 下。
