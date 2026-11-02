# 新英文训练来源与Pirá题干相似性审计

仅使用CPU，2个PyTorch线程；审计单位为60条原生来源事实，涵盖原问题与已有改写。没有按300条格式示例重复计数。

## 结果

| 范围 | 题数 | 原文精确相同 | 规范化相同 | 问题语义≥0.82 | 其中实际训练48族语义flag | QA辅助语义flag |
|---|---:|---:|---:|---:|---:|---:|
| development | 101 | 0 | 0 | 0 | 0 | 0 |
| official_test | 227 | 0 | 0 | 0 | 0 | 0 |

主语义分数是题干对原问题/已有改写的最大余弦，阈值0.82；原问题单独分数和Q+A辅助分数分开保存。Q+A辅助命中不自动并入主flag。规范化为Unicode NFKC、casefold后去非文字/数字及下划线。

每题最相近的来源问题、答案、教材section和页码在两个nearest_review.csv中。training48列只看48个实际训练族；其余12族属于内部验证，未算作实际训练。

**相似概念不是自动认定的泄漏。未触发阈值也不能证明完全独立。** 本审计只覆盖这60个新来源事实，不覆盖旧适配器全部训练历史或未知基座预训练。所有人工决定、姓名与备注栏为空；未删题、改标签、改训练内容或划分。

正式test227只读取id、question、source_group作为训练前污染审计；未利用正式测试答案、选项或模型表现，也没有把测试内容写入训练集或检索语料。

## 复现

```powershell
.venv\Scripts\python.exe paper_data/english_training_v2/overlap_audit/audit_sources.py
```

manifest.json记录脚本、模型和全部输入哈希、线程、阈值与方法；audit.json保留逐题记录，cosine_matrices.npz保留所有成对分数。
