# 验证报告

## 结论

- status: PASS
- release_status: formal
- files_checked: 22
- canonical_ads: 573
- matched_ads: 538
- listed_companies: 378
- unmatched_ads: 35
- provisional_labels: 0
- reliability_sample: 120
- prompt_version: 2.1.0

## 完整性检查

必填字段无空值；主键唯一；未匹配公司均有原因；所有正分标签均有原文证据；120 条同模型盲重测中的 17 条分歧均已完成上下文裁决。

## 可选字段空值 / NA

下列空值来自原始数据或条件字段，不属于不完整交付：

- `cleaned_ads.岗位标签`：254 个空值
- `cleaned_ads.待遇`：65 个空值
- `cleaned_ads.学历`：34 个空值
- `cleaned_ads.所在城市`：30 个空值
- `company_matches.industry`：63 个空值
- `ai_scores.audit_score`：453 个空值

## 未完成项

无。
