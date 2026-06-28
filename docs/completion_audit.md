# RA 任务逐条完成审计

本文档把原始任务 1–10 与加分项逐条映射到可运行代码和已提交产出。数值以最终验证通过的产出为准。

| 要求 | 完成证据 | 验收结果 |
|---|---|---|
| 1. 读入广告并报告行数 | `src/ra_task/cleaning.py::clean_ads`、`reports/ra_task_report.md` | 原始 612 条 |
| 2. 清理格式乱码 | `clean_text`、`tests/test_cleaning.py`、`data/processed/cleaned_ads.csv` | 全部文本字段无 `<$&数字&$>` 残留 |
| 3. 识别重复并说明标准 | `clean_ads`、`data/interim/duplicate_map.csv`、报告“数据与清洗” | 除 ID 外 9 个业务字段完全相同；612 → 573，移除 39 条，38 个重复组 |
| 4. 匹配上市公司并统计 | `src/ra_task/matching.py`、`outputs/company_matches.csv` | 广告数和上市公司数在报告/验证报告动态生成 |
| 5. 子公司/分支归并 | `config/company_aliases.csv`、`artifacts/review/company_match_review.csv`、报告“公司匹配” | 平安寿险/财险、万科物业、招行分行及已复核曾用名均显式留痕 |
| 6. 说明 false positive 并举例 | `tests/test_matching.py`、报告“公司匹配” | 记录并修正“岭南园林 → 园林股份”；拒绝“中关村科技租赁 → 中关村” |
| 7. AI/数字化评分 | `config/ai_rubric.yaml`、`src/ra_task/llm_labeling.py`、`outputs/ai_scores.csv` | 573 条均有 0–3 分、理由、置信度与来源哈希；正分均有原文证据 |
| 8. 按年汇总与可视化 | `outputs/annual_ai_share.csv`、`outputs/figures/annual_ai_share.png`、报告“年度结果” | 2014–2025 年、三个阈值、主指标 Wilson 95% CI 和年度样本量 |
| 9. 3–5 句发现 | 报告“发现” | 4 句描述性发现，无因果外推 |
| 10. 数据局限 | 报告“数据局限” | 覆盖非随机样本、早期小样本、构成变化及 LLM 边界 |
| 加分：LLM API 与稳定性 | `artifacts/llm/v2/*.jsonl`、`artifacts/review/reliability_*`、`artifacts/review/v1_v2_*` | DeepSeek 正式编码；120 条目标化+分层盲重测；所有 18 条分歧上下文裁决；明确不宣称独立人工信度；v1→v2 敏感性完整留存 |
| 加分：稳健匹配 | `company_aliases.csv`、`company_match_candidates.csv` | 仅接受精确或审核规则；模糊候选不自动接受；香港/退市/不确定公司保持未匹配 |
| 加分：工程与 Git | `pyproject.toml`、`tests/`、`.github/workflows/ci.yml`、`artifacts/manifests/` | CLI、锁文件、离线正式缓存复现、自动测试、秘密扫描、SHA-256 清单及分阶段 Git 历史 |

## 提交要求

- 可运行代码、README、原始/中间/最终数据、匹配结果、年度表与图、HTML 报告均已纳入仓库。
- `dist/ra_task_submission.zip` 是可直接提交的完整包。
- 工具/库与大致用时同时写在 README 和分析报告。
- `verification_report.md` 区分必填字段与可选/条件字段的空值，并明确列出未完成项。
