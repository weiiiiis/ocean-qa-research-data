# 英文补训候选数据 v2

**仅更新系统提示词；尚未训练模型。**

此版本复制最终已复核的 `english_training_v1`：保留相同的300条候选、60个知识family、240／60划分，以及全部问题、答案、用户提示、来源、证据片段和人工待审核状态。唯一内容改动是将每条 `messages[0].content` 换成 `paper_eval/english.py` 中的 `OPEN_SYSTEM_V2`；完整聊天模板的token长度重新计数。

V2对应 `prompt_profile=evidence_v2`，强调英文简洁回答、通常不超过60词、按资料语境理解问题、逐字复制所给引用ID、资料不足时限定为给定资料不足，以及没有来源材料时不添加引用。它移除了V1中的虚构引用编号示意。用户提示中原有的 `Reference passages:`、`Question:` 和 passages-only约束保持原样。

- `candidates_300.jsonl`：300示例，60闭卷、180有证据、60证据不足。
- `train_candidates_240.jsonl`：48个family的240条训练候选。
- `validation_candidates_60.jsonl`：另12个family的60条验证候选。
- `facts.tsv`、`source_facts_60.jsonl`、`family_split.json`、来源审核表及 `revision_history/`：从最终V1逐字复制的溯源材料。
- `provenance_v1/`：V1的manifest、README、生成器和旧系统提示词快照，仅作溯源。
- `manifest.json`：V1父文件哈希、V2系统提示哈希、各文件完整token长度及输出哈希。
- `prepare_v2.py`：复现脚本，只加载本地CPU tokenizer，不载模型或训练。

```powershell
.venv/Scripts/python.exe paper_data/english_training_v2/prepare_v2.py
```

再次生成时会核对V1父文件与V2提示是否改变；若已改变，使用新版本目录。训练器只读取 `messages`，不能把溯源字段拼进模型输入。300个格式示例仍只有60个知识family；来源限制与专家待审核状态均沿用V1，不能当作独立专家认证或新评测成绩。
