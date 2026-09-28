"use strict";
(() => {
  const statuses = {active:"仍需处理", resolved:"已解决且无新诉求", withdrawal:"申请撤单", consultation:"咨询", follow_up:"跟进处理", unclear:"信息不足"};
  const polarities = {occurred:"原文按已发生陈述", possible:"怀疑 / 尚未确认", negated:"明确否定", consultation:"咨询"};
  const scopes = {active_retrieval_issues:"active-issues", background_issues:"background-issues"};
  const $ = id => document.getElementById(id);
  const initial = JSON.parse($("worksheet-data").textContent);
  const identity = new Map(initial.map(r => [r.source_id, r]));
  let rows = structuredClone(initial), position = 0, dirty = false;
  const length = text => Array.from(text).length;
  const row = () => rows[position];
  function message(text) { $("message").textContent = text; }
  function options(select, values, selected) {
    select.replaceChildren();
    for (const [value, label] of Object.entries(values)) {
      const option = new Option(label, value); option.selected = value === selected; select.add(option);
    }
  }
  function button(text, action) {
    const el = document.createElement("button"); el.textContent = text; el.onclick = action; return el;
  }
  function changed() {
    row().review_status = "pending"; dirty = true;
    $("save-state").textContent = "有未导出的修改，请导出保存。";
    $("errors").textContent = ""; navigation();
  }
  function navigation() {
    $("progress").textContent = "已确认 " + rows.filter(r => r.review_status === "complete").length + " / " + rows.length;
    $("records").replaceChildren(...rows.map((r, index) => {
      const el = button((r.review_status === "complete" ? "✓" : "○") + " " + (index + 1) + " · " + r.source_row, () => { position = index; render(); });
      el.setAttribute("aria-current", String(position === index)); return el;
    }));
    $("previous").disabled = position === 0;
    $("next").disabled = position === rows.length - 1;
  }
  function issueField(card, label, value, update, maxLength) {
    const holder = document.createElement("label"); holder.textContent = label;
    const input = document.createElement("input"); input.value = value; input.maxLength = maxLength;
    input.oninput = () => { update(input.value); changed(); };
    holder.append(input); card.append(holder);
  }
  function renderIssues(scope) {
    const target = $(scopes[scope]); target.replaceChildren();
    row().gold[scope].forEach((issue, index) => {
      const card = document.createElement("section"); card.className = "issue";
      const heading = document.createElement("strong"); heading.textContent = "问题 " + (index + 1); card.append(heading);
      issueField(card, "问题名称", issue.label, v => issue.label = v, 100);
      issueField(card, "需要什么知识来回答", issue.required_knowledge_need, v => issue.required_knowledge_need = v, 100);
      const label = document.createElement("label"); label.textContent = "确定性 / 咨询属性";
      const select = document.createElement("select"); options(select, polarities, issue.expected_polarity);
      select.onchange = () => { issue.expected_polarity = select.value; changed(); }; label.append(select); card.append(label);
      const currentLabel = document.createElement("label"), check = document.createElement("input"); check.type = "checkbox";
      check.checked = issue.is_current_request; check.disabled = scope === "background_issues";
      check.onchange = () => { issue.is_current_request = check.checked; changed(); };
      currentLabel.append(check, document.createTextNode("直接表达本次来电的当前诉求")); card.append(currentLabel);
      const evidenceLabel = document.createElement("p"); evidenceLabel.textContent = "必要证据（从左侧原文复制，每项一个连续片段）"; card.append(evidenceLabel);
      issue.evidence_quotes.forEach((quote, qi) => {
        const wrap = document.createElement("div"); wrap.className = "quote";
        const input = document.createElement("textarea"); input.value = quote; input.setAttribute("aria-label", "问题 " + (index + 1) + " 证据 " + (qi + 1));
        input.oninput = () => { issue.evidence_quotes[qi] = input.value; changed(); };
        wrap.append(input, button("移除证据", () => { issue.evidence_quotes.splice(qi, 1); changed(); renderIssues(scope); })); card.append(wrap);
      });
      const add = button("＋ 证据", () => { issue.evidence_quotes.push(""); changed(); renderIssues(scope); });
      add.disabled = issue.evidence_quotes.length >= 8; card.append(add);
      const controls = document.createElement("footer");
      const other = scope === "active_retrieval_issues" ? "background_issues" : "active_retrieval_issues";
      controls.append(button(scope === "active_retrieval_issues" ? "移到背景" : "移到当前", () => {
        if (row().gold[other].length >= 12) { message("目标分组已达到 12 项。请在备注记录额外问题。"); return; }
        row().gold[scope].splice(index, 1); row().gold[other].push(issue);
        if (other === "background_issues") issue.is_current_request = false;
        changed(); render();
      }), button("删除问题", () => {
        if (!confirm("删除这个问题及其证据？")) return;
        row().gold[scope].splice(index, 1); changed(); renderIssues(scope);
      }));
      card.append(controls); target.append(card);
    });
  }
  function addIssue(scope) {
    if (row().gold[scope].length >= 12) { message("该分组已达到 12 项。请在备注记录额外问题。"); return; }
    const ids = new Set(Object.keys(scopes).flatMap(s => row().gold[s].map(i => i.issue_id)));
    let n = 1; while (ids.has("issue-" + n)) n++;
    row().gold[scope].push({issue_id:"issue-" + n, label:"", expected_polarity:"occurred", is_current_request:false, required_knowledge_need:"", evidence_quotes:[""]});
    changed(); renderIssues(scope);
  }
  function validate(r) {
    const errors = [], ids = new Set();
    if (!r.annotator_id.trim() || length(r.annotator_id.trim()) > 100) errors.push("请填写标注人（1–100 字）。");
    if (!Object.hasOwn(statuses, r.gold.case_status)) errors.push("请选择有效状态。");
    if (length(r.gold.annotation_notes.trim()) > 2000) errors.push("备注超过 2000 字。");
    for (const scope of Object.keys(scopes)) {
      if (r.gold[scope].length > 12) errors.push("每个分组最多 12 个问题。");
      for (const [index, issue] of r.gold[scope].entries()) {
        const prefix = (scope === "background_issues" ? "背景" : "当前") + "问题 " + (index + 1) + "：";
        if (!/^[a-z0-9][a-z0-9_-]{0,63}$/.test(issue.issue_id) || ids.has(issue.issue_id)) errors.push(prefix + "问题标识无效或重复。");
        ids.add(issue.issue_id);
        for (const field of ["label", "required_knowledge_need"]) if (!issue[field].trim() || length(issue[field].trim()) > 100) errors.push(prefix + "请完整填写名称与知识需求（各 1–100 字）。");
        if (!Object.hasOwn(polarities, issue.expected_polarity)) errors.push(prefix + "确定性无效。");
        if (scope === "background_issues" && issue.is_current_request) errors.push(prefix + "背景问题不能标为当前诉求。");
        if (!issue.evidence_quotes.length || issue.evidence_quotes.length > 8) errors.push(prefix + "需要 1–8 项证据。");
        for (const quote of issue.evidence_quotes) {
          if (!quote.trim() || length(quote.trim()) > 300 || !r.case_content.includes(quote.trim())) errors.push(prefix + "证据须为原文中的连续片段，长度 1–300 字。");
        }
      }
    }
    return errors;
  }
  function render() {
    navigation(); $("errors").textContent = "";
    $("source-title").textContent = "第 " + (position + 1) + " 条 · 原始行 " + row().source_row;
    $("source-id").textContent = "工单记录 ID：" + row().source_id;
    $("source").textContent = row().case_content;
    $("prediction-box").open = false;
    $("prediction").textContent = row().prediction ? JSON.stringify(row().prediction, null, 2) : "本工作表没有附带模型结果。";
    $("annotator").value = row().annotator_id; options($("case-status"), statuses, row().gold.case_status);
    $("notes").value = row().gold.annotation_notes;
    Object.keys(scopes).forEach(renderIssues);
  }
  function exportFile(complete) {
    if (complete) {
      const invalid = rows.findIndex(r => r.review_status !== "complete" || validate(r).length);
      if (invalid >= 0) { position = invalid; render(); $("errors").textContent = ["本条尚未通过确认。", ...validate(row())].join("\n"); return; }
    }
    const blob = new Blob([rows.map(r => JSON.stringify(r)).join("\n") + "\n"], {type:"application/x-ndjson;charset=utf-8"});
    const url = URL.createObjectURL(blob), a = document.createElement("a");
    a.href = url; a.download = complete ? "semantic-gold-v1.completed.worksheet.jsonl" : "semantic-gold-v1.draft.worksheet.jsonl";
    document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 10000);
    dirty = false; $("save-state").textContent = "已发起下载，请确认文件保存成功。";
    message(complete ? "标注已导出。下一步运行 finalize-gold 校验原文并冻结参考答案。" : "草稿已导出。下次打开页面后导入该文件即可继续。");
  }
  async function importFile(file) {
    try {
      if (dirty && !confirm("导入将替换本页未导出的修改，是否继续？")) return;
      // Split physical LF lines only; U+2028/U+2029 are legal within JSON strings.
      const imported = (await file.text()).replace(/^\uFEFF/, "").split("\n").filter(l => l.trim()).map(JSON.parse);
      if (imported.length !== initial.length) throw new Error("记录数与本工作表不同。");
      const seen = new Set();
      for (const r of imported) {
        const original = identity.get(r.source_id);
        if (!original || seen.has(r.source_id) || r.annotation_version !== original.annotation_version || r.source_row !== original.source_row || r.content_sha256 !== original.content_sha256 || r.case_content !== original.case_content) throw new Error("工单身份、原文或版本不匹配。");
        seen.add(r.source_id);
        if (!["pending","complete"].includes(r.review_status) || typeof r.annotator_id !== "string" || !r.gold || typeof r.gold.annotation_notes !== "string" || !Object.hasOwn(statuses,r.gold.case_status)) throw new Error("标注格式无效。");
        for (const scope of Object.keys(scopes)) {
          if (!Array.isArray(r.gold[scope])) throw new Error("问题列表格式无效。");
          for (const i of r.gold[scope]) {
            if (!i || ["issue_id","label","expected_polarity","required_knowledge_need"].some(k => typeof i[k] !== "string") || typeof i.is_current_request !== "boolean" || !Array.isArray(i.evidence_quotes) || i.evidence_quotes.some(q => typeof q !== "string")) throw new Error("问题字段格式无效。");
          }
        }
        if (r.review_status === "complete" && validate(r).length) throw new Error("已确认记录未通过字段或证据检查。");
        r.prediction = structuredClone(original.prediction);
      }
      const indexed = new Map(imported.map(r => [r.source_id, r]));
      rows = initial.map(r => indexed.get(r.source_id)); position = 0; dirty = false; render();
      $("save-state").textContent = "已载入保存的进度。"; message("导入成功。");
    } catch (error) { message("导入失败：" + error.message); }
  }
  $("annotator").oninput = e => { row().annotator_id = e.target.value; changed(); };
  $("case-status").onchange = e => { row().gold.case_status = e.target.value; changed(); };
  $("notes").oninput = e => { row().gold.annotation_notes = e.target.value; changed(); };
  $("add-active").onclick = () => addIssue("active_retrieval_issues");
  $("add-background").onclick = () => addIssue("background_issues");
  $("previous").onclick = () => { position--; render(); };
  $("next").onclick = () => { position++; render(); };
  $("complete").onclick = () => {
    const errors = validate(row()); $("errors").textContent = errors.join("\n"); if (errors.length) return;
    row().review_status = "complete"; dirty = true;
    $("save-state").textContent = "有未导出的修改，请导出保存。";
    if (position < rows.length - 1) position++;
    render();
  };
  $("export-draft").onclick = () => exportFile(false);
  $("export-complete").onclick = () => exportFile(true);
  $("import-button").onclick = () => $("import-file").click();
  $("import-file").onchange = e => { const file = e.target.files[0]; if (file) importFile(file); e.target.value = ""; };
  window.addEventListener("beforeunload", e => { if (dirty) { e.preventDefault(); e.returnValue = ""; } });
  render();
})();
