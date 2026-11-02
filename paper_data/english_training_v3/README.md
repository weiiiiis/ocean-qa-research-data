# 英文补训候选数据 v3

**仅对齐评测器的用户消息末尾指令；没有新增或删除知识事实，没有训练模型。**

本版来自已复核V2，保留完整300条候选、240条训练／60条验证划分、60个知识family，以及全部system、问题、assistant答案、来源和标签。V1、V2均不修改。

闭卷示例的user末尾追加：

```text
No reference passages were supplied. Do not add bracketed citations.
```

带资料的正例和证据不足示例，user末尾统一追加：

```text
Answer the question directly using the relevant passages, and cite their exact identifiers. If they do not support an answer, state that the supplied passages are insufficient.
```

两种追加文本与V2评测器末尾指令一致，用于减少训练与推理格式差异。已有的passages-only规则保持不变。闭卷答案仍正常回答已知事实而不添加引用；证据不足答案只说明给定片段不能支持回答。数据格式修订不代表引用行为已经被模型修复，须实际开发评测后判断。

`manifest.json`记录V2父文件SHA256、未改动的系统提示哈希、全部例子的完整token长度及输出哈希。`facts.tsv`、原文事实、family划分、审核表和修订记录按V2逐字复制；`provenance_v2/`保留V2溯源快照。专家审核仍未完成。

重生成仅使用本地CPU tokenizer，不加载语言模型：

```powershell
.venv/Scripts/python.exe paper_data/english_training_v3/prepare_v3.py
```

训练器只读取 `messages`，不得把provenance或标签拼进模型输入。300条仍只有60个知识family，不能用作训练后的独立评测集。
