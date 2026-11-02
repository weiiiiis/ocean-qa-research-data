# 指标定义与统计单位 / Metric definitions and statistical units

本字典用于表1—6及图1—4。CSV中0—1原始值显示为百分比时乘100；两个比例或F1均值的差乘100后单位为**百分点（percentage points, pp）**，不是相对百分比。质量均分差单位为0—4量尺的分值（score points）。CSV分数和可以为小数，不能称正确题数。

## 自动任务指标 / Automatic task metrics

| 中文 / English | 定义、分子与分母 / Definition and denominator | 解释边界 / Interpretation |
|---|---|---|
| 选项准确率 / Option-selection accuracy | 冻结规则从回答开头解析A—E选项，正确题数/全部227正式题。代码键`mc_prefix_sensitivity_accuracy`；不可解析按错误计。The frozen leading-option parser is identical across conditions. | 衡量选择公开标签的能力，不要求完全单字母输出。Not a free-form factual accuracy measure. |
| 严格单字母且正确 / Strict single-letter accuracy | 原始答案去空白后必须恰为合法单字母且等于金标；满足两者题数/227。代码键`accuracy`。 | 同时受内容选择与格式影响；不可单独解释为知识增益。Conflates option correctness and format. |
| 格式遵循率 / Format compliance | 满足单字母合法输出的题数/227；`1−mc_format_invalid`。答案可以选错。 | 与严格准确率不同；仅判输出约束。Assesses formatting irrespective of gold correctness. |
| 去引用词面F1 / Citation-stripped token F1 | 先用冻结完整引用正则去标记，再按SQuAD式小写、去英文标点/冠词/合并空白；以token多重集合交集算每题precision/recall及调和均值。仅用`reference_answer_origin`；均值=各题F1分数和/N。正式N=227，开发每条件N=101。 | 与原参考的词面重合，不是事实正确率。未闭合标记不被正则去除。Lexical overlap, not factual correctness; malformed incomplete markers may remain. |
| 原始F1 / Raw token F1 | 相同SQuAD式计算，保留引用字符串；分子为题级F1之和，分母N。 | 可受标记文本与答案长度影响。Retains citation text and is sensitive to wording/length. |
| 去引用EM / Citation-stripped exact match | 去完整引用后归一化文本与原参考完全相同的题数/N。 | 不要求原字符完全相同，不能把EM缺失当全部事实错误。Normalized match, not factual truth. |
| Gold来源召回@3 / Gold-source recall@3 | 公开预期来源身份在实际提供的top-3片段中命中的问题数/有可用来源的RAG问题数；正式202/227。无RAG不适用。 | 来源检索命中，不保证片段含答案，也不等于答题正确。Retrieval identity hit, not answer correctness or excerpt sufficiency. |
| 实际输出token / Actual output tokens | 保存的`output_tokens_including_eos`之和/实际回答数；包括EOS。 | 上限相同不保证均值或分布相同。Equal budgets do not ensure equal realized lengths. |
| 实际词数 / Actual word count | 完整解码答案按空白分词；默认含引用，另存去完整引用词数。均值按题等权。 | 不等同模型token数。Whitespace words differ from model tokens. |
| 预算截断 / Hit token limit | `hit_token_limit=True`的回答数/N，即到达生成上限且未正常EOS结束。 | 可能包含尚未闭合引用，不直接证明正文不完整。A termination diagnostic, not a semantic completeness label. |

## 引用的三层含义 / Three distinct levels of citation evidence

| 指标 / Metric | 定义 / Definition | 分母与不可推论之处 / Denominator and limits |
|---|---|---|
| 有引用标记回答 / Answers containing markers | 至少存在一个正则匹配完整标记的回答数/N。逗号/分号分开的ID分别计为标记。 | 无标记并不自动构成引用语义不忠实；需看是否应引用。 |
| 可解析标记比例 / Resolved-ID ratio | 能对应当题实际提供chunk ID的完整标记总数/全部完整标记总数。标记计数加权，非逐回答比例均值。 | 0/0为NA，不能写100%。无RAG没有提供片段，出现的ID无法映射。An existing ID does not entail its associated claim. |
| 至少一标记且全部可解析 / At least one marker and all IDs resolved | 有标记且所有完整标记ID均有效的回答数/N。没有引用时此回答指标为false。 | 与标记合计比例及语义支持不同。 |
| 至少一编号无法映射 / Any unresolved ID | 含至少一个完整标记ID无法在实际片段中找到的回答数/N。 | 仅是形式/指向诊断。 |
| 方括号未闭合 / Unclosed bracket | 开发审计记录的未闭合方括号回答数/N，与完整正则标记另列。 | 可解析完整标记达到100%时仍可存在此问题。A complete-marker denominator excludes incomplete strings. |
| AI引用支持率（正式AI40）/ AI40 citation support | 被AI判断得到引用材料支持的“陈述—引用关联”合计数/全部关联数。Base+RAG=129/207，FT+RAG=77/101。 | 关联总数加权；不同于自动标记数，不将编号存在当支持。 |
| AI引用覆盖率（正式AI40）/ AI40 supported citation coverage | 有至少一个支持引用的需引用陈述数/全部需引用陈述数。Base+RAG=124/157，FT+RAG=76/101。 | 描述回答已说出的需引用陈述得到支持的比例，非回答覆盖全部科学事实或全部必要要点。 |
| AI支持标记率（开发AI101）/ AI101 supported marker rate | 审查中完整支持的实际标记数/已评实际标记数：v1=97/116，v2=95/113。 | 与AI40的陈述—引用关联定义不同，不能混池。Not pooled with AI40 association counts. |
| AI回答层引用支持 / Answer-level citation association status | 保留完整支持、部分支持、不支持/矛盾、不适用等原标签，按对应回答数/101。 | 无标记不算完整支持；分类或多标签规则以原审查记录为准。 |

片段内支持（entailment within supplied excerpts）、回答已陈述事实的证据覆盖（evidence coverage of stated claims）、独立来源核实的科学事实正确性（scientific correctness verified against independent sources）是三个不同目标。当前数据主要覆盖前两者，不能用一个替代另一个。

## AI内容指标 / AI content metrics

| 指标 / Metric | 定义与分母 / Definition and denominator | 边界 / Limits |
|---|---|---|
| 正式AI事实答题质量 / AI40 answer quality | 原始`ai_factual_correctness`的0—4评分和/40回答；每条件40。 | 量尺均值不是正确题比例；纯拒答未给所问事实时为0。This is AI-judged quality, not human factual accuracy. |
| 正式AI必要完整性 / AI40 essential completeness | 原始`ai_essential_completeness`的0—4评分和/40回答。 | 与开发集逐要点覆盖不是同一量尺。 |
| 正式AI证据忠实性 / AI40 evidence faithfulness | 原始证据忠实性0—4评分和/40个RAG回答。无RAG不适用。 | 资料不足时恰当拒答可忠实但未完成题目。 |
| 资料充分性 / Evidence sufficiency | 开发101题先于答案建立充分/部分充分/不足/不确定标签，题数/101。 | 相对实际给定窗口，不是对整个文献世界可答性的裁决。 |
| 错误拒答 / Erroneous refusal | AI认为给定材料足以或部分足以回答但回答错误地拒绝；回答数/101，保留充分性与争议。 | 不输出完整答案不自动算拒答；新增/消除须由同题转移计算。 |
| 恰当拒答 / Appropriate refusal | 给定资料确实不足时，AI判断拒绝越界合理；回答数/101。 | 忠于证据不等于完成所问事实。 |
| 至少一个未支持/矛盾/未知陈述 / Any unsupported, contradicted or uncertain claim | 每种状态分别统计含至少一该状态陈述的回答数/101；类别可重叠。 | 未支持是相对给定材料；矛盾包括“资料未给答案”的元陈述，不能称同数核心科学错误。 |
| 要点等权覆盖 / Point-weighted coverage | 对AI预先定义的可判要点，以C=1、P=0.5、O=0计分，U剔除并单列；合计得分/(C+P+O要点数)。本次每阶段141点，v1=104/141，v2=93/141。 | 要点多的题权重大；AI最低要点定义主观，0.5为描述约定。 |
| 每题等权覆盖 / Equal-question coverage | 先按题计算上述覆盖分数，再对97道有可评要点题等权平均；分子为各题比例之和。v1=75.125/97，v2=67.875/97。4题NA不置零。 | 是对权重选择的敏感性分析，不新增标签或验证AI要点集合。 |
| 内容胜负 / Paired content verdict | 同题v1/v2 RAG回答比较，保留v2_better/v2_worse/equivalent/tradeoff/uncertain五类，题数/101。 | 综合内容、遗漏、拒答与引用；不能把worse当同数核心事实错误。 |
| F1×内容交叉 / F1-by-content crosstab | 原始题级F1差以1e−12容差分升/降/不变，与五类内容标签交叉；总分母101，同时报告各F1行分母24/35/42。 | 不将不确定当0或把混合当改善/损失；下降并不自动等于内容损失。 |

## 样本和统计约定 / Samples and statistical conventions

- **正式自动：** 227同题、181主来源组，144保守组敏感性；2000次来源组有放回bootstrap，seed=20261017。组内题保留，均值按题加权；配对差由共有同题及同来源组计算。另保留225/206/21题切片，不替换227题主结果。
- **开发自动：** 101同题、91主来源组；10000次共享抽样矩阵，seed=20261028，90合并来源组敏感性。全部为探索性逐项区间，未作多重比较校正。
- **正式AI40：** 40预选同题、40主来源组/32保守组；2000次共享来源组抽样，seed=20261018。分数按回答等权，支持/覆盖按陈述关联合计；区间条件于固定AI标签，不包含评价误差。
- **开发AI101：** 仅v1/v2 RAG，101同题、202回答、91来源组；不计算CI或显著性，保留原标签。单AI、精确版本未记录、先前结果暴露，不称独立盲评。
- **缺失和区间：** 分母0或不适用记NA；不确定单列。区间跨0意味着证据不足，不证明没有效应。比例差显示为百分点，不能把原始0—1差值直接标成百分比数值。

These are distinct, paired evaluation scopes. Neither task duplication nor repeated development analyses create additional independent questions. All source paths and exact numerical denominators are recorded in the table CSVs and data inventory; no original labels or references were changed.
