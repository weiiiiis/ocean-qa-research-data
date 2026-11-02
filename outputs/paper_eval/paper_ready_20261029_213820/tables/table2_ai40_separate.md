# 正式测试的独立呈现部分：40题AI评价

| condition | metric | value | ci_low | ci_high | numerator | denominator | n_questions | source_groups |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base | factual_quality | 1.025 | 0.65 | 1.4 | 41.0 | 40.0 | 40 | 40 |
| Base | completeness | 0.95 | 0.525 | 1.375 | 38.0 | 40.0 | 40 | 40 |
| Base | faithfulness | nan | nan | nan | 0.0 | 0.0 | 40 | 40 |
| Base | citation_support | nan | nan | nan | 0.0 | 0.0 | 40 | 40 |
| Base | citation_coverage | nan | nan | nan | 0.0 | 0.0 | 40 | 40 |
| Base+RAG | factual_quality | 3.125 | 2.675 | 3.525 | 125.0 | 40.0 | 40 | 40 |
| Base+RAG | completeness | 3.275 | 2.8 | 3.7 | 131.0 | 40.0 | 40 | 40 |
| Base+RAG | faithfulness | 3.325 | 2.975 | 3.625 | 133.0 | 40.0 | 40 | 40 |
| Base+RAG | citation_support | 0.6231884057971014 | 0.5016525423728814 | 0.7872551782443284 | 129.0 | 207.0 | 40 | 40 |
| Base+RAG | citation_coverage | 0.7898089171974523 | 0.7083333333333334 | 0.8620797413793102 | 124.0 | 157.0 | 40 | 40 |
| FT | factual_quality | 1.4 | 0.975 | 1.85 | 56.0 | 40.0 | 40 | 40 |
| FT | completeness | 1.175 | 0.725 | 1.675 | 47.0 | 40.0 | 40 | 40 |
| FT | faithfulness | nan | nan | nan | 0.0 | 0.0 | 40 | 40 |
| FT | citation_support | nan | nan | nan | 0.0 | 0.0 | 40 | 40 |
| FT | citation_coverage | nan | nan | nan | 0.0 | 0.0 | 40 | 40 |
| FT+RAG | factual_quality | 3.25 | 2.825 | 3.625 | 130.0 | 40.0 | 40 | 40 |
| FT+RAG | completeness | 3.3 | 2.85 | 3.7 | 132.0 | 40.0 | 40 | 40 |
| FT+RAG | faithfulness | 3.375 | 3.025 | 3.675 | 135.0 | 40.0 | 40 | 40 |
| FT+RAG | citation_support | 0.7623762376237624 | 0.6538157894736842 | 0.860469881814716 | 77.0 | 101.0 | 40 | 40 |
| FT+RAG | citation_coverage | 0.7524752475247525 | 0.6428571428571429 | 0.8532110091743119 | 76.0 | 101.0 | 40 | 40 |

注：这不是新增独立测试集，也不是人评。40题为206道answer_in_text=1题中的预先选定子集，每组40，质量评分160条，两个RAG条件证据评分80条；40主来源组，32保守组。事实答题质量、必要完整性、忠实性为0—4均分；前两者分母40回答，忠实性仅RAG适用。citation_support为受支持的陈述—引用关联/所有关联；citation_coverage为有支持引用的陈述/需引用陈述，均使用合计数而非逐回答等权。已按原协议2000次、seed=20261018、共享来源组抽样逐项复现区间。区间不包含AI评价误差。单AI流程，无独立人类评分；精确评审模型版本未记录。

来源：outputs/paper_eval/ai_evaluation_adopted_20261018/per_answer_ai_scores.json 与 analysis.json。
