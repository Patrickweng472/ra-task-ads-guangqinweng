# 上市公司招聘广告中的 AI / 数字技术含量

本项目是清华大学经济管理学院研究助理筛选任务的完整、可复现实现。它清洗 612 条 51job 招聘广告，将广告主归并至上市母公司，并使用 DeepSeek V4 Pro 对岗位的 AI / 数字技术含量进行 0–3 级编码。

## 核心方法

- 清除 `<$&数字&$>` 网页格式乱码，兼容时间戳和纯日期格式。
- 按除 `id` 外的全部业务字段识别重复，保留来源 ID 映射。
- 公司匹配采用精确匹配、经审核母公司规则和唯一证券简称；模糊相似度只生成候选。
- 评分定义：0 无实质技术，1 辅助数字工具，2 数字技术为核心，3 明确 AI / 模型 / 高级算法。
- 主年度指标为得分不低于 2，同时报告得分不低于 1 和严格等于 3。

## 运行

1. 安装 [uv](https://docs.astral.sh/uv/) 并执行 `uv sync --all-groups`。
2. 在线正式编码前设置环境变量 `DEEPSEEK_API_KEY`，执行 `uv run ra-task run`。
3. 无密钥时可执行 `uv run ra-task run --offline` 验证完整流水线；其标签会明确标记为 provisional。
4. 执行 `uv run pytest` 和 `uv run ra-task verify` 完成验证。

## 主要产出

- `outputs/company_matches.csv`：公司匹配状态、方法、置信度和未匹配原因。
- `outputs/ai_scores.csv`：得分、证据、理由、置信度和模型来源。
- `outputs/annual_ai_share.csv`：2014—2025 年年度占比和 Wilson 95% 区间。
- `reports/ra_task_report.html`：可直接阅读的中文报告。
- `dist/ra_task_submission.zip`：完整提交包。

## 可追溯性与限制

原始、中间和最终数据、匹配候选、标签缓存、运行元数据、文件哈希、测试与报告均纳入 Git。API 密钥和模型思维链不会持久化。样本不是按年份随机抽取，早期年份样本很少，因此结果是描述性证据而不是总体趋势或因果估计。

## 工具

Python、pandas、NumPy、RapidFuzz、OpenAI SDK、Pydantic、Matplotlib、pytest、uv、Quarto、Git 和 GitHub Actions。实际流水线运行时间记录在 `artifacts/manifests/run_metadata.json`。

