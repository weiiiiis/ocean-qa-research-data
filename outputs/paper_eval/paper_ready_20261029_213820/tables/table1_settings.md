# 表1 / Table 1. Experimental settings and evidence scope

| Evidence set           | Sample and grouping                           | Conditions                                   | Budget (tokens)              | Interpretation boundary                         |
|:-----------------------|:----------------------------------------------|:---------------------------------------------|:-----------------------------|:------------------------------------------------|
| Official test          | 227 original questions; 181/144 groups        | Base/Base+RAG/FT/FT+RAG                      | MCQA16;OpenQA160             | Primary; two tasks share question IDs           |
| Development            | 101 questions;91 EID/90 merged-passage groups | Base/v1/v2; with/without RAG                 | 160 for all;96 Base/v2 only  | Exploratory;10 cells/1010 answers               |
| Official AI evaluation | 40 questions;40/32 groups                     | 4 conditions;160 quality/80 evidence ratings | Existing official160 outputs | One AI process; scores0-4 and citation counts   |
| Development AI audit   | 101 pairs/202 answers;91 groups               | v1+RAG versus v2+RAG only                    | Existing160 outputs          | One AI judge;97 evaluable questions/141 points  |
| English SFT            | 240/60 examples;48/12 fact families           | Original base -> v1 epoch1 -> v2 epoch1      | Training limit1024           | v2:336 draws;no new families;v1 epoch2 excluded |

正式FT统一指英文v2/epoch_1；开发v1指英文v1/epoch_1。所有评估采用同一本地Qwen3-4B-Instruct、4-bit NF4/FP16；LoRA r=8、alpha=16、dropout=0.05，q_proj/v_proj。BM25配置为top-3、来源去重、k1=1.5、b=0.75，688支撑文本/1507窗口，上下文1000 token、输入2048 token；只用问题干检索，参考和来源标签不入提示。贪心生成，repetition_penalty=1.05。

主分析对来源组配对抽样并保留组内同题：正式2000次seed20261017；正式AI2000次seed20261018；开发10000次seed20261028、按题加权、逐项未调整区间。91→90是合并相同支撑文本的敏感性，不是新增数据。来源组不是已证实独立完整文档。

样本、指标定义、配置和原始文件逐实验见 `table1_settings.csv` 和 `../data_inventory.csv`。正式227题不是454独立题；101题不是新独立测试集；AI40与AI101分别报告。

English note: FT denotes the final English v2 epoch1 adapter; development v1 denotes English v1 epoch1. The official tasks share 227 original questions. Development comparisons are exploratory. The 40-question official AI evaluation and 101-question development audit are distinct; neither is expert human evaluation. Exact paths and per-cell definitions are supplied in the CSV.
