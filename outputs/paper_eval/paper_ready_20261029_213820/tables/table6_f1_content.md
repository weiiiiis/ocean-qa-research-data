# 表6：F1变化与 AI 内容判断 / Table 6: F1 changes and AI verdicts

| F1变化 / Direction | v2_better | v2_worse | equivalent | tradeoff | uncertain | 该行占全体 / All questions |
| --- | --- | --- | --- | --- | --- | --- |
| increase | 3/24 (12.5%) | 5/24 (20.8%) | 14/24 (58.3%) | 1/24 (4.2%) | 1/24 (4.2%) | 24/101 |
| decrease | 2/35 (5.7%) | 11/35 (31.4%) | 22/35 (62.9%) | 0/35 (0.0%) | 0/35 (0.0%) | 35/101 |
| unchanged | 0/42 (0.0%) | 1/42 (2.4%) | 40/42 (95.2%) | 0/42 (0.0%) | 1/42 (2.4%) | 42/101 |

分母 / Denominators: cells use the corresponding F1-direction row (24 increased, 35 decreased, 42 unchanged); last column uses all 101 questions. All original five content labels are retained. Increase/decrease is v2 minus v1 citation-stripped original-reference token F1, tolerance 1e-12. F1 is lexical overlap, not factual accuracy. No new AI reassessment was performed. Mixed and uncertain outcomes are not pooled with gains or losses. These are the same 101 development questions under two RAG stages, not independent test observations; 91 source groups. No confidence intervals were computed under the audit protocol. A v2-worse verdict can reflect omissions, refusal, citation support or content, rather than a new core factual error.

来源 / Sources: AI `paired_reviews.csv` linked to original `dev_supplement_20261028_164300/per_question_metrics.jsonl`; exact F1 values independently matched. Single AI, model/version unrecorded, prior result exposure; exploratory.
