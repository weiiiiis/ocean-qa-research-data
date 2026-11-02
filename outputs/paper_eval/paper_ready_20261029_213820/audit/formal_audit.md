# 正式数据复核

mcqa: 908 raw records, 227 unique original questions, 4 complete conditions; all stored per-item metrics and all 4 slices × 2 groupings × summaries/paired bootstrap reproduced exactly.

openqa: 908 raw records, 227 unique original questions, 4 complete conditions; all stored per-item metrics and all 4 slices × 2 groupings × summaries/paired bootstrap reproduced exactly.

AI40: all 4-condition/5-metric means, denominators and 40/32-source-group CIs/paired differences reproduced exactly from original adopted item labels.

## 冻结哈希

| category | path | status |
| --- | --- | --- |
| data_sha256 | D:\weitiao\ocean-chat\paper_data\external_benchmarks\official_openqa_en_227.jsonl | match |
| data_sha256 | D:\weitiao\ocean-chat\paper_data\external_benchmarks\official_mcqa_en_227.jsonl | match |
| data_sha256 | D:\weitiao\ocean-chat\paper_data\external_benchmarks\corpus_pira_chunks_en.jsonl | match |
| data_sha256 | D:\weitiao\ocean-chat\paper_data\external_benchmarks\official_english_stage_manifest.json | match |
| data_sha256 | D:\weitiao\ocean-chat\paper_data\external_benchmarks\overlap_pretest_decisions_20261017.json | match |
| code_sha256 | D:\weitiao\ocean-chat\paper_eval\english.py | match |
| code_sha256 | D:\weitiao\ocean-chat\paper_eval\inference.py | match |
| code_sha256 | D:\weitiao\ocean-chat\paper_eval\common.py | match |
| code_sha256 | D:\weitiao\ocean-chat\paper_eval\retrieval.py | match |
| code_sha256 | D:\weitiao\ocean-chat\paper_eval\bm25_retrieval.py | match |
| code_sha256 | D:\weitiao\ocean-chat\paper_eval\english_report.py | match |
| code_sha256 | D:\weitiao\ocean-chat\paper_eval\official_english_report.py | match |
| code_sha256 | D:\weitiao\ocean-chat\paper_eval\export_official_blind_subset.py | match |
| model_files_sha256 | D:\weitiao\ocean-chat\outputs\ocean_qwen3_english_grounded_v2\epoch_1\adapter_config.json | match |
| model_files_sha256 | D:\weitiao\ocean-chat\outputs\ocean_qwen3_english_grounded_v2\epoch_1\adapter_model.safetensors | match |
| protocol | D:\weitiao\ocean-chat\paper_data\frozen_english_v1_20261017\protocol.json | match |
| mcqa_run | D:\weitiao\ocean-chat\outputs\paper_eval\en_test_mcqa_frozen_v1_20261017\run_manifest.json | match |
| mcqa_run | D:\weitiao\ocean-chat\outputs\paper_eval\en_test_mcqa_frozen_v1_20261017\answers.jsonl | match |
| mcqa_run | D:\weitiao\ocean-chat\outputs\paper_eval\en_test_mcqa_frozen_v1_20261017\selected_questions.jsonl | match |
| openqa_run | D:\weitiao\ocean-chat\outputs\paper_eval\en_test_openqa_frozen_v1_20261017\run_manifest.json | match |
| openqa_run | D:\weitiao\ocean-chat\outputs\paper_eval\en_test_openqa_frozen_v1_20261017\answers.jsonl | match |
| openqa_run | D:\weitiao\ocean-chat\outputs\paper_eval\en_test_openqa_frozen_v1_20261017\selected_questions.jsonl | match |


## 差异与解释

逐题原始预测、原报告JSON和中文汇总的关键指标一致；中文汇总四舍五入至一位小数，不以舍入数直接相减。正式MCQA无效格式与严格答错分开；F1分子是分数和，不是正确题数。正式报告expert_review=pending保留原状态；后续采纳AI评分不能改写成人评完成。两任务ID交集为227，不可合并为454。

原报告accuracy是严格单字母且正确；mc_prefix_sensitivity_accuracy为主选项准确率；不能把前者称为事实正确率。引用标记编号可解析不说明陈述得到证据支持。主来源组通过已知EID/归一化文本合并，无法保证整篇来源独立；144组为将corpus2合并的依赖敏感性假设。AI40三个分片属于同一AI流程，不是3位评审。

没有重跑模型或改变任何标签；没有读取多GB基座权重；小适配器权重及数据、代码、协议按冻结记录逐项核验，完整值见frozen_hash_validation.csv。
