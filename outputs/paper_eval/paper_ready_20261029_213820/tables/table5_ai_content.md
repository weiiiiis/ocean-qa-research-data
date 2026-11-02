# 表5：AI 内容审查 / Table 5: AI content audit

| 指标 / Metric | v1 | v2 |
| --- | --- | --- |
| 错误拒答 / Erroneous refusal | 3/101 (2.97%) | 8/101 (7.92%) |
| 恰当拒答 / Appropriate refusal | 6/101 (5.94%) | 4/101 (3.96%) |
| 含材料未支持陈述 / Any unsupported claim | 9/101 (8.91%) | 10/101 (9.90%) |
| 含与材料矛盾陈述 / Any contradicted claim | 3/101 (2.97%) | 9/101 (8.91%) |
| 含无法判断陈述 / Any uncertain claim | 2/101 (1.98%) | 1/101 (0.99%) |
| 引用关联完整支持 / Fully supported citation associations | 83/101 (82.18%) | 81/101 (80.20%) |
| 引用关联部分支持 / Partially supported associations | 8/101 (7.92%) | 6/101 (5.94%) |
| 引用不支持或矛盾 / Unsupported/contradicted associations | 8/101 (7.92%) | 12/101 (11.88%) |
| 引用不适用（无标记）/ No citation, not applicable | 2/101 (1.98%) | 2/101 (1.98%) |
| 原报告要点加权覆盖 / Point-weighted coverage | 104/141 (73.76%) | 93/141 (65.96%) |
| 每题等权覆盖均值 / Equal-question coverage mean | 75.125/97 (77.45%) | 67.875/97 (69.97%) |
| 有遗漏的可评价回答 / Answers with omission | 22/97 (22.68%) | 27/97 (27.84%) |
| 支持的标记 / Supported citation markers | 97/116 (83.62%) | 95/113 (84.07%) |
| 直接回答 / Direct answer | 83/101 (82.18%) | 77/101 (76.24%) |

| v2 内容判断 / Content verdict | 题数 / Questions |
| --- | --- |
| v2_better | 5/101 (4.95%) |
| v2_worse | 17/101 (16.83%) |
| equivalent | 76/101 (75.25%) |
| tradeoff | 1/101 (0.99%) |
| uncertain | 2/101 (1.98%) |

表5 / Table 5. 开发集单 AI 内容审查（仅 v1/v2 RAG）。101 题、202 回答，91 来源组；160 token 上限。各回答指标分母 101；97 题有可评要点，共 141 点，另 4 题覆盖率 NA。覆盖计分 C=1、P=0.5、O=0；必要要点由 AI 主观定义，可能包含辅助细节。原报告将每个要点等权，敏感性分析则先逐题计算覆盖率再按 97 题等权，分子为比例之和。标记支持以 v1 的 116 个与 v2 的 113 个已评标记为分母，与回答层支持及可解析标记为不同指标。无标记不算全部支持；材料未支持不等于科学事实错误。矛盾包含“材料没有答案”之类元陈述。

This is descriptive, exploratory evidence from one conversational AI reviewer, whose exact model/version was unrecorded and who had prior exposure to identities and some results. Anonymous display did not make it fully independent or blind. No new labels, confidence intervals, expert scores, or significance tests were created. Point-weighted and question-weighted rates use distinct denominators and weights; the fractional numerators are coverage-score sums, not counts of correct answers. Citation support is relative to supplied excerpts, not independent-source verification.

来源 / Sources: `ai_content_audit_20261028_175905/answer_reviews_unblinded.jsonl`, `question_evidence_checklist.jsonl`, `paired_reviews.csv`; exact source paths are in CSV. Original labels are preserved in `audit/ai_labels/`. Formal-test 40-question AI evaluation is separate (Table 2 supplement).
