# LLM v2.1 人工参照评测

本目录用于建立 120 条“单审核者人工参照”，不是人际信度或绝对金标。编码构念、维度和确定性映射冻结于 `config/ai_rubric_v2_1.yaml`。

## 两阶段交付纪律

1. 先填写 `blind_development.xlsx` 的 60 条开发集并返回；开发集可用于最多 3 轮提示词调整。
2. v2.1 提示词、模型、schema、thinking 设置与指纹冻结并提交后，再填写和返回 `blind_holdout.xlsx`。
3. 锁定留出集只评测一次。若未达到发布门槛，不得用它继续调整 v2.1；v2 保持正式结果，并建立新的 v2.2 划分。

## 人工字段

- `human_score`：0–3。
- `technology_role`：`none | auxiliary | core`。
- `strict_ai`：`true | false`。
- `human_confidence`：`high | medium | low`。
- `human_evidence`：正分必填，必须复制原文中的一段连续短语。
- `human_note`：可写边界判断，建议说明为什么不是相邻等级。

确定映射为：`none → 0`；`auxiliary → 1`；`core + false → 2`；`core + true → 3`。任何不一致都会在导入时被拒绝。

## 防泄漏

盲审表只含随机 `review_id`、岗位、岗位描述和岗位标签，不含公司、年份、原始 ID、模型分数、模型理由或抽样原因。开发集与留出集互不重叠。随机 ID 到原始 ID 的映射只在本机被忽略的 `private/` 目录中生成，人工审核完成后才会随最终评测账本公开。

## 发布门槛

- 开发集：精确一致率 ≥85%，主阈值一致率 ≥95%，二次加权 κ ≥0.85。
- 锁定留出集：精确一致率 ≥80%，主阈值一致率 ≥90%，二次加权 κ ≥0.80，evidence/schema 合法率 100%。
- 严格 AI 的 precision、recall、F1 单独报告；人工严格 AI 阳性少于 5 条时只报告数量与个案。
- 冻结提示词在开发集运行 3 次，主阈值稳定率须 ≥95%，结构化输出成功率须为 100%。

## 复现命令

```bash
uv run ra-task prepare-human-eval --seed 20260629
```

该命令从已提交的 v2 输出、v1–v2 差异和同模型重测账本确定性重建两份盲审表。
