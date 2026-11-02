# ocean-qa-research-data

Research data, evaluation results, figure source data, and code for studying QLoRA fine-tuning and retrieval-augmented generation in ocean-science question answering.

整理日期为 2026-11-02。本包按 2026-10-29 论文初稿使用的实验整理，包含训练数据、正式测试及开发集逐条结果、AI 评分、图表源数据和复现代码。复制时保留项目内目录层级。

## 从哪里查看

| 内容 | 位置 |
|---|---|
| 具体上传清单、图表对应关系 | 根目录的《论文数据上传清单.md》 |
| 英文训练、验证、来源和家族划分 | `paper_data/english_training_v2/`、`paper_data/english_training_v3/` |
| 外部基准题、检索语料、下载及筛选脚本 | `paper_data/external_benchmarks/` |
| 正式冻结协议和历史哈希 | `paper_data/frozen_english_v1_20261017/` |
| 正式选择题、开放问答逐条回答 | `outputs/paper_eval/en_test_mcqa_frozen_v1_20261017/`、`en_test_openqa_frozen_v1_20261017/`（同父目录） |
| 正式测试 AI40 评分 | `outputs/paper_eval/ai_evaluation_adopted_20261018/` |
| 开发集逐条结果 | `outputs/paper_eval/dev_supplement_20261028_164300/` |
| 开发集 AI 内容评价 | `outputs/paper_eval/ai_content_audit_20261028_175905/` |
| 表格 CSV、绘图源 CSV、SVG 图件与分析脚本 | `outputs/paper_eval/paper_ready_20261029_213820/` |
| 训练配置、训练输入快照、训练指标和模型卡 | `outputs/ocean_qwen3_english_grounded_v1/`、`outputs/ocean_qwen3_english_grounded_v2/` |
| 评测代码 | `paper_eval/` |
| 本次复制的源文件与副本 SHA256 | 根目录的《文件校验清单.csv》 |

正式两项任务使用同 227 道原始题，每项四组共 908 条回答。开发补充为 101 题、10 条件、1,010 条回答。模型第一阶段使用训练数据 v2，续训使用数据 v3；正式适配器对应模型 v2/epoch_1。正式 AI40 与开发集 AI101 分别保存。

## 本地复核

317 份源文件均逐字复制，并核对源文件与副本 SHA256 相同。文件校验清单只登记复制的源文件，本 README 和核对脚本为本次新增说明。

在本文件夹运行 PowerShell 命令即可再次检查副本完整性。

```powershell
.\核对文件完整性.ps1
```

本次整理验证了文件复制完整性和主要 JSON/JSONL 的格式及记录数。模型推理和统计分析的历史结果保留在原报告中。

## 数据使用和复现说明

- 本包保留了原始题干、检索原文、AI 评分和运行记录。公开前按《论文数据上传清单.md》第 4 节核对署名及第三方摘录的公开范围。Pirá 原始数据卡和固定上游下载清单已随包保留；英文教材衍生数据的署名和改动说明仍需补齐。
- 部分脚本、清单和训练配置含原项目的本机绝对路径。迁移使用时应传入新根目录或调整路径。冻结清单和记录的原始哈希对应原件；若生成公开修订副本，另写改动记录和新哈希。
- 代码依赖见两份 requirements 文件；论文分析和绘图脚本另使用 numpy、pandas、matplotlib 等库。本包提供了历史代码副本，跨目录执行前应核对各脚本的根目录参数和依赖。
- 模型权重另行提供时，给出基座来源和版本、v1/epoch_1 与 v2/epoch_1 适配器及校验值。所需权重身份可从冻结清单和模型卡查到。
- 本包选取截至 10 月 29 日初稿所用材料。10 月 30 日补强实验若进入最终稿，再按最终采用结果补充。

本仓库保留实验材料的来源记录及历史状态，后续修订应记录改动及新哈希。
