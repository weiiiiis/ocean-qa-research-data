# 复核与验收报告

本轮仅重算既有预测和既有AI标签，没有训练、付费评价、模型推理或正式测试重新评估。全部新增科研产物位于本目录。执行起点：2026-10-29 21:38:20（Asia/Shanghai）；完成时间见audit/runtime.json。

## 数据、范围与复现

- 正式MCQA/OpenQA各227题×4条件，任务共享227个原始ID；181主组/144保守组。逐题指标、均值、原区间及配对差值均复现，含225/206/21切片。
- 正式AI40从206题answer_in_text=1中事前预选，160质量/80证据评分；40主组/32保守组统计逐项复现，与AI101分开。
- 开发10条件各101题、共1010输出，91主组/90合并支撑文本组。逐条重新计算F1、长度、引用和截断，复现162项配对区间及162项敏感性区间；最大浮点误差约3.6e−15，不构成数值冲突。
- AI101只有v1/v2 RAG160条件，101配对/202回答；保留141要点、97可评题/4NA。原始标签逐条关联回答与F1；141点摘录均可在实际文本定位。统计未创建新内容标签或CI。
- 正式冻结协议、代码、数据和适配器哈希比对一致，见audit/frozen_hash_validation.csv。根清单保护178个源文件，交付时逐项复核无改变。没有对基础模型大权重重新计算哈希，不将其描述为本轮完整认证。

## 原材料差异及采用依据

没有发现原始预测、逐题指标和指定汇总中的实质数值冲突。保留以下口径区别，避免造成表观冲突：

1. 原差值表偶用“%”表示差值，本交付统一写百分点；原CSV部分以0—1保存，显示前×100。正式F1差4.2百分点与35.7−31.4得到4.3的表观差异来自先舍入：采用未舍入逐题差值4.2434百分点。
2. 正式FT/v2不是历史中文应用模型，亦非v1 epoch_2。开发补充重新统一实际评估提示，不把历史提示不同的成绩混入。
3. 开发91 EID组与90支撑文本合并组是同一101题的不同聚类；AI证据窗口还涉及更多检索来源，不能把检索来源数当独立问题组数。
4. 原AI报告覆盖采用要点等权104/141→93/141；本交付新增每题等权75.125/97→67.875/97为敏感性，不覆盖原标签。分子分别为加权覆盖分数和题级比例之和。
5. AI目录名含20261028，但实际审查跨2026-10-28/29续跑至完成；本交付如实注明，不据此假设同一模型快照。
6. MCQA内部字段accuracy实际上是严格单字母正确率，mc_prefix_sensitivity_accuracy为宽松开头选项正确率；主表已显式重命名。两者均以全部227题为分母。
7. 生成160上限下OpenQA与MCQA16上限的截断是不同任务指标。引用ID有效只检验完整标记，未闭合残片另计；不因100%完整标记有效而宣称引用无问题。

采用依据：原始回答及冻结解析协议优先，已有逐题分析用于双向核对；经逐项复现的配对区间优先于从四舍五入均值重算差值。AI标签原样保留，争议列明而不以本轮文字修改原判。

## 正文与结论审核

正式、开发、AI内容三个独立子任务对中英文正文进行了只读交叉审核，数字全部与所负责表格一致；主代理完成空格、分母和评价范围澄清。主要主张均有claim_evidence_map.csv条目，包含30条主张与不支持的推断。正文区分直接显示、提示与尚待验证；F1/格式/ID解析/片段支持/独立科学验证分开。未把开发集写成新测试集，未把AI写成人工评价，未将17个综合更差标签写成17个核心事实错误。

## 图与文件验收

5幅图各有SVG/PDF/600dpi PNG及底层CSV，图注中英齐备。PNG逐幅目视检查；SVG经XML解析并确认保留文本；PDF打开、提取文本并由Python渲染预览。数据范围、轴量纲与样本标注清楚，配对内容图不加虚构区间。图4细线重合不代表只有可见线数，已在图注明确。

自动验收逐项结果：

| check                                    | passed   | detail                                                                                                                                                                                            |
|:-----------------------------------------|:---------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| official counts                          | True     |                                                                                                                                                                                                   |
| formal arithmetic option_accuracy        | True     |                                                                                                                                                                                                   |
| formal arithmetic strict_letter_accuracy | True     |                                                                                                                                                                                                   |
| formal arithmetic format_compliance      | True     |                                                                                                                                                                                                   |
| formal arithmetic stripped_f1            | True     |                                                                                                                                                                                                   |
| dev six conditions                       | True     |                                                                                                                                                                                                   |
| budget eight conditions                  | True     |                                                                                                                                                                                                   |
| AI proportion arithmetic                 | True     |                                                                                                                                                                                                   |
| cross-tab total                          | True     |                                                                                                                                                                                                   |
| cross-tab verdict marginal               | True     |                                                                                                                                                                                                   |
| required cases                           | True     |                                                                                                                                                                                                   |
| case F1 arithmetic                       | True     |                                                                                                                                                                                                   |
| claim map                                | True     | ['claim_id', 'claim_zh', 'claim_en', 'evidence_table_field_or_case', 'dataset_conditions_and_sample_scope', 'evidence_type', 'support_strength', 'permitted_wording_zh', 'limitations_to_retain'] |
| source immutability                      | True     | 178 source files; changed=[]                                                                                                                                                                      |
| fig1_formal exports                      | True     | (4389, 3249); dpi=(599.9988, 599.9988); SVG text objects=64; PDF text chars=479                                                                                                                   |
| fig1_formal source CSV                   | True     |                                                                                                                                                                                                   |
| fig2_stages exports                      | True     | (4389, 3189); dpi=(599.9988, 599.9988); SVG text objects=55; PDF text chars=455                                                                                                                   |
| fig2_stages source CSV                   | True     |                                                                                                                                                                                                   |
| fig3_f1_content exports                  | True     | (4381, 1929); dpi=(599.9988, 599.9988); SVG text objects=35; PDF text chars=278                                                                                                                   |
| fig3_f1_content source CSV               | True     |                                                                                                                                                                                                   |
| fig4_paired exports                      | True     | (4390, 2109); dpi=(599.9988, 599.9988); SVG text objects=25; PDF text chars=301                                                                                                                   |
| fig4_paired source CSV                   | True     |                                                                                                                                                                                                   |
| figS1_ai40 exports                       | True     | (4389, 2109); dpi=(599.9988, 599.9988); SVG text objects=32; PDF text chars=333                                                                                                                   |
| figS1_ai40 source CSV                    | True     |                                                                                                                                                                                                   |
| figure1 table alignment                  | True     |                                                                                                                                                                                                   |
| figure2 table alignment                  | True     |                                                                                                                                                                                                   |
| figure3 table alignment                  | True     |                                                                                                                                                                                                   |
| figure4 coverage alignment               | True     |                                                                                                                                                                                                   |
| manuscript key numeric trace             | True     |                                                                                                                                                                                                   |

## 尚未解决的问题

精确AI版本/解码参数、先验接触、主观要点、缺少独立专家与完整来源验证、单模型单轨迹、额外训练与目标/提示/采样混杂、预算不等长、正式v1对照缺失，均保留为证据缺口。这些不阻止描述现有正式指标及开发行为取舍，但限制因果互补、普遍事实质量或效率论点。详细最小补充设计见limitations_and_gaps.md。本交付不是独立重复实验或AI标签真值认证。
