# AI-based evaluation

## Methods draft

We performed an AI-based evaluation on a fixed subset of 40 Pirá questions selected before the official model outputs were generated. The subset contained 31 questions from corpus 1 and nine from corpus 2, with one question per primary source group. All questions belonged to the 206 records with a published `answer_in_text` label of 1. Each of the four configurations contributed one answer per question, yielding 160 answer-quality evaluations. Evidence-related evaluation covered the two retrieval-augmented configurations, yielding 80 evaluations.

A single interactive Codex AI evaluation workflow assigned separate ordinal scores from 0 to 4 for factual answer quality, essential-point completeness, and faithfulness to the supplied evidence. Three parallel grading shards divided the records; they were parts of one evaluation workflow and were not treated as independent raters. Quality assessment used the question, the released reference answer, and its supporting passage. Faithfulness assessment used only the passages actually supplied to the answering model. Published references were treated as potentially fallible. An abstention supplying none of the requested facts received zero answer-quality and completeness scores, while an appropriately qualified abstention could receive a high faithfulness score when the supplied evidence was insufficient.

The workflow recorded item-level rationales, reference ambiguities, and claim–citation checks. AI internal calibration reviewed 14 borderline answers and revised nine answers before the final scores were archived. Explicit configuration labels were omitted from the scoring packets and were joined to the archived ratings for analysis. The presence of retrieved passages and differences in answer style could nevertheless reveal aspects of the configuration. No independent human ratings or inter-rater agreement statistics were obtained. The exact judge model revision and decoding settings were not independently logged in this interactive workflow.

Citation support was calculated as the pooled number of fully supported claim–citation links divided by all assessed links. Citation coverage was calculated as the pooled number of claims with a supporting citation divided by all claims requiring citation. These semantic judgments were distinct from automatic source-ID format checks. Exploratory 95% intervals were calculated using 2,000 paired source-cluster bootstrap resamples with seed 20261018. The main analysis used 40 source groups; a conservative sensitivity analysis used 32 groups by combining unresolved corpus-2 sources. These intervals describe variation across the selected questions and source-group assumptions, not uncertainty in the AI judge itself.

## Results draft

Retrieval improved AI-rated answer quality on the selected subset. Mean factual answer-quality scores were 1.025 for the base model, 1.400 for LoRA, 3.125 for the base model with retrieval, and 3.250 for LoRA with retrieval. Corresponding completeness scores were 0.950, 1.175, 3.275, and 3.300. Adding retrieval increased factual answer quality by 2.100 points for the base model (95% interval, 1.550–2.626) and by 1.850 points for LoRA (1.250–2.400).

The additional LoRA effect under retrieval was small and uncertain. Compared with the base model with retrieval, LoRA with retrieval differed by 0.125 points in factual answer quality (95% interval, −0.100 to 0.425), 0.025 points in completeness (−0.250 to 0.325), and 0.050 points in faithfulness (−0.250 to 0.375). AI-assessed citation support was 76.24% (77/101 links) for LoRA with retrieval and 62.32% (129/207) for the base model with retrieval. The paired difference was 13.92 percentage points (−4.35 to 29.88). Citation coverage was 75.25% (76/101 claims) and 78.98% (124/157), respectively. Thus, the higher citation-support point estimate did not establish a consistent additional LoRA benefit, and citation coverage did not increase.

These findings are AI-generated assessments of a 40-question subset. They do not constitute independent expert validation or an evaluation of all 227 test questions. The original full-test automatic metrics and references were retained separately.

## 中文说明与术语

一句话论点：在固定40题的AI评价中，RAG提高答题质量；微调在已有RAG之上的额外质量与引用支持增益仍不确定。这与全227题的选择题格式改善、开放问答词语重合度改善分别报告。

| 固定术语 | 含义与使用边界 |
|---|---|
| AI-based evaluation | AI评价，不写成人工评估或专家盲评 |
| factual answer quality | AI按0—4分评价题目所求事实的答题质量，不换算事实准确率 |
| essential-point completeness | 必要要点完整性，不要求开放题穷举原文所有细节 |
| faithfulness to supplied evidence | 对模型实际收到资料的忠实性，不以Gold原文补证据 |
| citation support | AI判断的陈述—引用关联支持率，与编号可解析率分开 |
| citation coverage | AI判断的有支持引用陈述数／需引用陈述数 |
| LoRA | low-rank adaptation；本实验中的已冻结英文适配器 |
| RAG | retrieval-augmented generation；本实验中的已冻结BM25检索配置 |

## 论断与证据

- RAG提高AI答题质量：同40题配对差异区间为正；仅支持这一AI样本内的比较。
- 微调额外提高RAG条件下的事实质量：当前差值区间跨0，不作确定性结论。
- 微调引用支持点估计较高：77/101对129/207；配对区间跨0，且引用覆盖率点估计下降。
- 真实人类评审认可这些逐条分数：没有该证据，不作此表述。用户转述老师同意采用AI建议，是采用方式的授权，不是逐题人评记录。

## 缺失记录与使用范围

精确评审模型版本和解码参数没有独立留档；现有可追溯材料包括评分输入、评分尺度、逐条理由、AI内部校对记录、最终分数与哈希。当前不补造这些缺失参数，不报告双人一致性。本文段是AI评价方法与结果草稿，不是整篇论文或期刊已接受声明。
