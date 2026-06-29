from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .analysis import annual_summary, plot_annual, quadratic_weighted_kappa
from .cleaning import TOKEN_RE, clean_ads, clean_firms
from .llm_labeling import (
    PROMPT_VERSION,
    _exact_source_span,
    content_text,
    label_with_deepseek,
    rule_score,
    validate_batch,
)
from .matching import match_companies


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OutputTransaction:
    """Restore generated artifacts if a pipeline run fails partway through."""

    def __init__(self, paths: list[Path]):
        self.paths = [Path(path) for path in paths]
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._existing: dict[Path, tuple[Path, bool]] = {}

    def __enter__(self) -> "OutputTransaction":
        self._temporary = tempfile.TemporaryDirectory(prefix="ra-task-rollback-")
        backup_root = Path(self._temporary.name)
        for index, path in enumerate(self.paths):
            if not path.exists():
                continue
            backup = backup_root / str(index)
            if path.is_dir():
                shutil.copytree(path, backup)
                self._existing[path] = (backup, True)
            else:
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup)
                self._existing[path] = (backup, False)
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> bool:
        if exc is not None:
            for path in self.paths:
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
            for path, (backup, is_directory) in self._existing.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                if is_directory:
                    shutil.copytree(backup, path)
                else:
                    shutil.copy2(backup, path)
        if self._temporary is not None:
            self._temporary.cleanup()
        return False


def require_formal_cache(cache_path: Path, *, offline: bool) -> None:
    """Fail before output mutation when an offline formal cache is unavailable."""
    if offline and (not cache_path.exists() or cache_path.stat().st_size == 0):
        raise RuntimeError(f"Offline formal v2 cache is missing: {cache_path}")


def _cache_dir_for_version(version: str) -> Path:
    parts = version.split(".")
    suffix = parts[0] if len(parts) < 2 or parts[1] == "0" else f"{parts[0]}_{parts[1]}"
    return Path("artifacts/llm") / f"v{suffix}"


def validate_annual_consistency(ads: pd.DataFrame, labels: pd.DataFrame, annual: pd.DataFrame) -> None:
    """Recompute every annual statistic and reject stale or edited summaries."""
    expected = annual_summary(ads, labels.assign(score=labels["score"].astype(int)))
    try:
        pd.testing.assert_frame_equal(
            annual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise ValueError(f"annual summary does not match source labels: {exc}") from exc


def calculate_reliability_metrics(comparison: pd.DataFrame) -> dict:
    """Compute test-retest metrics directly from the item-level audit ledger."""
    primary = comparison["primary_score"].astype(int)
    audit = comparison["audit_score"].astype(int)
    return {
        "sample_size": len(comparison),
        "disagreements": int(primary.ne(audit).sum()),
        "exact_agreement": float(primary.eq(audit).mean()),
        "within_one_agreement": float(primary.sub(audit).abs().le(1).mean()),
        "binary_agreement_score_ge_2": float(primary.ge(2).eq(audit.ge(2)).mean()),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(primary.tolist(), audit.tolist()),
    }


def _report(
    stats: dict,
    matches: pd.DataFrame,
    labels: pd.DataFrame,
    annual: pd.DataFrame,
    reliability: dict,
    sensitivity: dict,
) -> str:
    matched = matches[matches["match_status"] == "matched"]
    status = labels["label_status"].value_counts().to_dict()
    method_counts = matches["match_method"].value_counts()
    method_lines = "\n".join(f"- `{method}`：{count} 条" for method, count in method_counts.items())
    annual_rows = "\n".join(
        f"| {int(row.year)} | {int(row.n_ads)} | {row.share_score_ge_1:.1%} | {row.share_score_ge_2:.1%} | {row.share_score_eq_3:.1%} | [{row.wilson_low:.1%}, {row.wilson_high:.1%}] |"
        for row in annual.itertuples()
    )
    sensitivity_text = (
        f"{sensitivity['from_version']} 到 {sensitivity['to_version']} 共有 {sensitivity['score_changes']} 条分数变化，其中 "
        f"{sensitivity['main_threshold_changes']} 条跨越主阈值；年度主指标的最大绝对变化为 "
        f"{sensitivity['max_abs_annual_main_share_change']:.1%}。{sensitivity['from_version']} 分布为 "
        f"`{json.dumps(sensitivity['from_score_distribution'], ensure_ascii=False)}`，{sensitivity['to_version']} 分布为 "
        f"`{json.dumps(sensitivity['to_score_distribution'], ensure_ascii=False)}`。"
        if sensitivity.get("available")
        else "未找到可比的旧版基线，因此本次未报告版本敏感性。"
    )
    return f"""# 招聘广告中的 AI / 数字技术含量

## 数据与清洗

原始广告共 {stats['raw_ads']} 条。先做 Unicode NFKC 标准化、清除 `<$&数字&$>` 格式标记、压缩空白，再统一解析时间戳和纯日期。以除 `id` 外的 9 个原始业务字段完全相同作为重复判据，{stats['duplicate_groups']} 个重复组共删除 {stats['duplicates_removed']} 条，最终保留 {stats['canonical_ads']} 条；每条的原始 ID 在 `data/interim/duplicate_map.csv` 中可追溯。公司主表原始 {stats['raw_firm_rows']} 行，按沪/深/北交所六位证券代码规则剔除 2 行脚注，保留 {stats['valid_firms']} 条合法记录。

## 公司匹配

共有 {len(matched)} 条广告匹配到 {matched['stock_code'].nunique()} 家上市公司，{len(matches) - len(matched)} 条保持未匹配。处理顺序是：标准化公司全称精确匹配 → 经审核的母公司规则（平安寿险/财险、万科物业、招行分行）→ 经审核的历史曾用名。各方法数量如下：

{method_lines}

模糊相似度只写入 `artifacts/review/company_match_candidates.csv` 供复核，从不自动接受；所有非精确决定另见 `artifacts/review/company_match_review.csv`。实际遇到的 false positive 是：简称子串规则曾把“岭南园林股份有限公司”错配为 605303.SH“园林股份”，复核历史名后改为 002717.SZ“岭南生态文旅”；同理，“中关村科技租赁”不再因包含“中关村”而错连 000931.SZ。香港上市、已退市或无法核实母子关系的公司保持未匹配，不为提高匹配率而强行归并。

## AI / 数字技术编码

使用 DeepSeek V4 Pro 逐条阅读岗位、描述与标签，只依据广告原文编码：

- **0 分**：无实质数字技术要求；
- **1 分**：办公软件、ERP、系统录入等辅助工具；
- **2 分**：软件、数据工程、自动化等数字技术是岗位核心职责；
- **3 分**：严格 AI 研发，核心职责明确涉及 AI/机器学习模型的研发、训练、评估或部署。

金融定价模型、传统统计建模、物理仿真、控制/通信算法、大数据与 ETL 不因“模型”或“算法”一词自动进入 3 分；除非原文另外明确指向 AI/ML，它们最高为 2 分。

年度主指标为 `score >= 2`，宽松指标为 `score >= 1`，纯 AI 指标为 `score == 3`。当前标签状态：{json.dumps(status, ensure_ascii=False)}。

## 年度结果

| 年份 | 广告数 | score≥1 | score≥2（主指标） | score=3 | 主指标 Wilson 95% CI |
|---:|---:|---:|---:|---:|---:|
{annual_rows}

完整数值见 `outputs/annual_ai_share.csv`。图中上面板是主指标及 Wilson 95% 区间，下面板同时显示年度样本量，避免把小样本波动误读为稳定趋势。

![年度AI/数字技术含量](../outputs/figures/annual_ai_share.png)

## 信度检验

同一 DeepSeek 模型在不看主编码结果的条件下，开启 thinking 对 {reliability.get('sample_size', 0)} 条广告做盲重测。样本先纳入所有低置信度、词典与模型跨主阈值冲突、以及全部严格 3 分项，再按分数分层补足；其中目标化案例 {reliability.get('targeted_cases', 0)} 条。精确一致率为 {reliability.get('exact_agreement', float('nan')):.1%}，相差不超过一级的一致率为 {reliability.get('within_one_agreement', float('nan')):.1%}，主阈值二分类一致率为 {reliability.get('binary_agreement_score_ge_2', float('nan')):.1%}，二次加权 Cohen's κ 为 {reliability.get('quadratic_weighted_kappa', float('nan')):.3f}。{reliability.get('disagreements', 0)} 条分歧通过第三次上下文裁决请求处理，裁决请求可见两份编码及其证据。

这些指标衡量的是同一模型在不同请求与思考设置下的测试—重测稳定性，**不等同于独立人工编码者信度**。因为抽样有意富集难例，该一致率也不应直接当作 573 条广告的随机总体准确率。

## 提示词与版本敏感性

v2.1 提示词先判数字技术对象、technology_role 与 strict_ai，再由代码确定性映射分数，并加入人工开发集揭示的边界规则。{sensitivity_text}这一差异表明结果对构念定义和提示词有实质敏感性，因此仓库保留旧版基线、逐条对比和年度对比。

## 发现

样本中的技术型岗位并非持续平滑上升，而是随年份和招聘构成明显波动。较新的广告中出现了更多软件、数据和算法岗位，但不同阈值下幅度并不完全一致。企业数字化岗位远多于严格意义上的 AI 岗位，因此将“数字化”和“AI”拆分报告比单一二分类更有解释力。重复广告会机械放大个别公司和年份，去重后结果更适合作为主分析。

## 数据局限

这些广告不是按年份随机抽取的总体样本，且早期年份样本很少；因此年度变化可能来自行业、公司和职位构成变化，不能解释为中国上市公司整体 AI 需求的因果趋势。此外，LLM 编码尽管做了同模型盲重测和上下文裁决，仍会受量表边界、提示词版本、文本歧义和招聘文本信息不完整的影响；本项目未声称具有独立人工金标准确率。

## 可复现性

项目使用 Python、pandas、RapidFuzz、OpenAI SDK、Pydantic、NumPy、Matplotlib、pytest、uv、Quarto、Git 和 GitHub Actions。全部任务约用 4 小时（包含实现、API 编码、匹配复核、测试与报告）；单次流水线的精确起止时间见 `artifacts/manifests/run_metadata.json`。文件哈希、重复映射、匹配候选及标签来源均随仓库提交。
"""


def select_audit_sample(ads: pd.DataFrame, labels: pd.DataFrame, seed: int, target_size: int = 120) -> pd.DataFrame:
    """Select all high-information cases, then add a balanced stratified fill."""
    joined = ads.merge(labels[["canonical_id", "score", "confidence"]], on="canonical_id", validate="one_to_one")
    joined["rule_score"] = joined.apply(lambda row: rule_score(content_text(row))[0], axis=1)
    reasons: dict[str, set[str]] = {str(item_id): set() for item_id in joined["canonical_id"]}
    for _, row in joined.iterrows():
        item_id = str(row["canonical_id"])
        if row["confidence"] == "low":
            reasons[item_id].add("low_confidence")
        if (int(row["score"]) >= 2) != (int(row["rule_score"]) >= 2):
            reasons[item_id].add("rule_model_threshold_conflict")
        if int(row["score"]) == 3:
            reasons[item_id].add("strict_score3")
    targeted_ids = [item_id for item_id, item_reasons in reasons.items() if item_reasons]
    if len(targeted_ids) > target_size:
        raise ValueError(f"targeted audit cases ({len(targeted_ids)}) exceed target size ({target_size})")
    selected_ids = list(targeted_ids)
    remaining = joined[~joined["canonical_id"].astype(str).isin(selected_ids)]
    pools = {
        int(score): group.sample(frac=1, random_state=seed + int(score))["canonical_id"].astype(str).tolist()
        for score, group in remaining.groupby("score", sort=True)
    }
    while len(selected_ids) < min(target_size, len(joined)) and any(pools.values()):
        for score in sorted(pools):
            if pools[score] and len(selected_ids) < target_size:
                item_id = pools[score].pop()
                selected_ids.append(item_id)
                reasons[item_id].add("stratified_fill")
    selected = joined[joined["canonical_id"].astype(str).isin(selected_ids)].copy()
    selected["selection_reason"] = selected["canonical_id"].astype(str).map(lambda item_id: "|".join(sorted(reasons[item_id])))
    return selected.sort_values("canonical_id", key=lambda series: series.astype(int)).reset_index(drop=True)


def build_adjudication_context(row: pd.Series, *, primary_reason: str, primary_evidence: str) -> dict:
    """Build an explicit two-coding comparison for a reasoned adjudication."""
    return {
        "primary": {
            "score": int(row["primary_score"]),
            "evidence": primary_evidence.split("|") if primary_evidence else [],
            "reason": primary_reason,
            "confidence": row["primary_confidence"],
        },
        "audit": {
            "score": int(row["audit_score"]),
            "evidence": str(row.get("audit_evidence", "")).split("|") if row.get("audit_evidence", "") else [],
            "reason": row["audit_reason"],
            "confidence": row["audit_confidence"],
        },
    }


def run_reliability_audit(ads: pd.DataFrame, labels: pd.DataFrame, seed: int, *, allow_network: bool = True) -> tuple[pd.DataFrame, dict]:
    selected = select_audit_sample(ads, labels, seed=seed, target_size=120)
    audit_ads = ads[ads["canonical_id"].isin(selected["canonical_id"])].copy()
    cache_dir = _cache_dir_for_version(PROMPT_VERSION)
    audit = label_with_deepseek(
        audit_ads,
        cache_dir / "audit_cache.jsonl",
        Path("config/ai_rubric_v2_1.yaml"),
        thinking=True,
        label_status="llm_audit",
        stage="audit",
        allow_network=allow_network,
    )
    primary_columns = labels[["canonical_id", "score", "evidence", "reason", "confidence"]].rename(
        columns={"score": "primary_score", "evidence": "primary_evidence", "reason": "primary_reason", "confidence": "primary_confidence"}
    )
    audit_columns = audit[["canonical_id", "score", "evidence", "confidence", "reason"]].rename(
        columns={"score": "audit_score", "evidence": "audit_evidence", "confidence": "audit_confidence", "reason": "audit_reason"}
    )
    comparison = selected[["canonical_id", "selection_reason", "rule_score"]].merge(primary_columns, on="canonical_id", validate="one_to_one").merge(
        audit_columns, on="canonical_id", validate="one_to_one"
    )
    disagreement_ids = comparison.loc[comparison["primary_score"] != comparison["audit_score"], "canonical_id"]
    adjudicated = pd.DataFrame(columns=["canonical_id", "score", "reason", "confidence"])
    if len(disagreement_ids):
        adjudication_ads = ads[ads["canonical_id"].isin(disagreement_ids)]
        disagreement_rows = comparison[comparison["canonical_id"].isin(disagreement_ids)]
        context_by_id = {
            str(row["canonical_id"]): build_adjudication_context(
                row,
                primary_reason=str(row["primary_reason"]),
                primary_evidence=str(row["primary_evidence"]),
            )
            for _, row in disagreement_rows.iterrows()
        }
        adjudicated = label_with_deepseek(
            adjudication_ads,
            cache_dir / "adjudication_cache.jsonl",
            Path("config/ai_rubric_v2_1.yaml"),
            thinking=True,
            label_status="llm_adjudicated",
            stage="adjudication",
            allow_network=allow_network,
            context_by_id=context_by_id,
        )
        comparison = comparison.merge(adjudicated[["canonical_id", "score", "reason"]].rename(columns={"score": "adjudicated_score", "reason": "adjudication_reason"}), on="canonical_id", how="left", validate="one_to_one")
    else:
        comparison["adjudicated_score"] = pd.NA
        comparison["adjudication_reason"] = ""
    agreement = calculate_reliability_metrics(comparison)
    metrics = {
        "status": "completed",
        "method": "same_model_blind_retest_with_contextual_adjudication",
        "independent_human_reliability": False,
        "targeted_cases": int(comparison["selection_reason"].ne("stratified_fill").sum()),
        "low_confidence_cases": int(comparison["selection_reason"].str.contains("low_confidence").sum()),
        "rule_model_threshold_conflicts": int(comparison["selection_reason"].str.contains("rule_model_threshold_conflict").sum()),
        "strict_score3_cases": int(comparison["selection_reason"].str.contains("strict_score3").sum()),
        **agreement,
    }
    Path("artifacts/review").mkdir(parents=True, exist_ok=True)
    _write_csv(comparison, Path("artifacts/review/reliability_sample.csv"))
    Path("artifacts/review/reliability_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return apply_adjudications(labels, comparison, adjudicated), metrics


def apply_adjudications(labels: pd.DataFrame, comparison: pd.DataFrame, adjudicated: pd.DataFrame) -> pd.DataFrame:
    enriched = labels.copy()
    enriched["primary_score"] = enriched["score"]
    enriched["audit_score"] = enriched["canonical_id"].map(comparison.set_index("canonical_id")["audit_score"])
    adjudicated_by_id = adjudicated.set_index("canonical_id") if len(adjudicated) else pd.DataFrame()
    for index, row in enriched.iterrows():
        item_id = row["canonical_id"]
        if len(adjudicated_by_id) and item_id in adjudicated_by_id.index:
            final = adjudicated_by_id.loc[item_id]
            enriched.at[index, "score"] = int(final["score"])
            enriched.at[index, "evidence"] = final["evidence"]
            enriched.at[index, "reason"] = final["reason"]
            enriched.at[index, "confidence"] = final["confidence"]
            enriched.at[index, "label_status"] = "llm_adjudicated"
            for column in ["technology_role", "strict_ai", "boundary_pair"]:
                if column in final.index:
                    enriched.at[index, column] = final[column]
    return enriched


def _snapshot_previous_version() -> None:
    """Preserve the current formal result once before promoting a newer prompt."""
    current = Path("outputs/ai_scores.csv")
    if not current.exists():
        return
    existing = pd.read_csv(current, dtype=str, keep_default_na=False)
    versions = set(existing.get("prompt_version", pd.Series(dtype=str)))
    if len(versions) != 1:
        return
    version = versions.pop()
    baseline_name = {"1.0.0": "v1", "2.0.0": "v2"}.get(version)
    if baseline_name is None or version == PROMPT_VERSION:
        return
    baseline = Path("artifacts/baselines") / baseline_name
    if baseline.exists():
        return
    baseline.mkdir(parents=True, exist_ok=True)
    sources = [
        Path("outputs/ai_scores.csv"),
        Path("outputs/annual_ai_share.csv"),
        Path("artifacts/review/reliability_sample.csv"),
        Path("artifacts/review/reliability_metrics.json"),
        _cache_dir_for_version(version) / "labels_cache.jsonl",
        _cache_dir_for_version(version) / "audit_cache.jsonl",
        _cache_dir_for_version(version) / "adjudication_cache.jsonl",
    ]
    for source in sources:
        if source.exists():
            shutil.copy2(source, baseline / source.name)
    (baseline / "snapshot_metadata.json").write_text(
        json.dumps({"prompt_version": version, "baseline": baseline_name}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_version_comparison(labels: pd.DataFrame, annual: pd.DataFrame) -> dict:
    """Write item- and year-level sensitivity artifacts against the latest formal baseline."""
    if Path("artifacts/baselines/v2/ai_scores.csv").exists():
        baseline_dir = Path("artifacts/baselines/v2")
        from_version = "v2"
    else:
        baseline_dir = Path("artifacts/baselines/v1")
        from_version = "v1"
    to_version = "v" + ".".join(PROMPT_VERSION.split(".")[:2]).rstrip(".0")
    prefix = f"{from_version}_{to_version.replace('.', '_')}"
    old_labels_path = baseline_dir / "ai_scores.csv"
    old_annual_path = baseline_dir / "annual_ai_share.csv"
    if not old_labels_path.exists() or not old_annual_path.exists():
        return {"available": False}
    old_labels = pd.read_csv(old_labels_path, dtype=str, keep_default_na=False)
    comparison = old_labels[["canonical_id", "score", "evidence", "reason"]].rename(
        columns={"score": "from_score", "evidence": "from_evidence", "reason": "from_reason"}
    ).merge(
        labels[["canonical_id", "score", "evidence", "reason"]].rename(
            columns={"score": "to_score", "evidence": "to_evidence", "reason": "to_reason"}
        ),
        on="canonical_id",
        validate="one_to_one",
    )
    comparison["score_changed"] = comparison["from_score"].astype(int).ne(comparison["to_score"].astype(int))
    comparison["crossed_main_threshold"] = comparison["from_score"].astype(int).ge(2).ne(comparison["to_score"].astype(int).ge(2))
    _write_csv(comparison, Path(f"artifacts/review/{prefix}_label_comparison.csv"))
    old_annual = pd.read_csv(old_annual_path)
    annual_comparison = old_annual[["year", "share_score_ge_2", "share_score_ge_1", "share_score_eq_3"]].rename(
        columns={column: f"from_{column}" for column in ["share_score_ge_2", "share_score_ge_1", "share_score_eq_3"]}
    ).merge(
        annual[["year", "share_score_ge_2", "share_score_ge_1", "share_score_eq_3"]].rename(
            columns={column: f"to_{column}" for column in ["share_score_ge_2", "share_score_ge_1", "share_score_eq_3"]}
        ),
        on="year",
        validate="one_to_one",
    )
    for metric in ["share_score_ge_2", "share_score_ge_1", "share_score_eq_3"]:
        annual_comparison[f"delta_{metric}"] = annual_comparison[f"to_{metric}"] - annual_comparison[f"from_{metric}"]
    _write_csv(annual_comparison, Path(f"artifacts/review/{prefix}_annual_comparison.csv"))
    summary = {
        "available": True,
        "from_version": from_version,
        "to_version": to_version,
        "items": len(comparison),
        "score_changes": int(comparison["score_changed"].sum()),
        "main_threshold_changes": int(comparison["crossed_main_threshold"].sum()),
        "from_score_distribution": old_labels["score"].astype(int).value_counts().sort_index().to_dict(),
        "to_score_distribution": labels["score"].astype(int).value_counts().sort_index().to_dict(),
        "max_abs_annual_main_share_change": float(annual_comparison["delta_share_score_ge_2"].abs().max()),
    }
    Path(f"artifacts/review/{prefix}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_pipeline(ads_path: Path, firms_path: Path, output_dir: Path, *, offline: bool, seed: int) -> None:
    cache_dir = _cache_dir_for_version(PROMPT_VERSION)
    label_cache = cache_dir / "labels_cache.jsonl"
    require_formal_cache(label_cache, offline=offline)
    _snapshot_previous_version()
    transactional_paths = [
        Path("data/processed"), Path("data/interim"), output_dir, Path("reports"), Path("dist"),
        Path("artifacts/review"), Path("artifacts/manifests"), Path("artifacts/llm/request_manifest.csv"),
        Path("verification_report.md"),
    ]
    with OutputTransaction(transactional_paths):
        _run_pipeline_impl(ads_path, firms_path, output_dir, offline=offline, seed=seed, label_cache=label_cache)


def _run_pipeline_impl(ads_path: Path, firms_path: Path, output_dir: Path, *, offline: bool, seed: int, label_cache: Path) -> None:
    started = datetime.now(timezone.utc)
    ads, duplicate_map, ad_stats = clean_ads(ads_path)
    firms, firm_stats = clean_firms(firms_path)
    stats = {**ad_stats, **firm_stats}
    _write_csv(ads, Path("data/processed/cleaned_ads.csv"))
    _write_csv(duplicate_map, Path("data/interim/duplicate_map.csv"))
    _write_csv(firms, Path("data/processed/valid_firms.csv"))
    matches, candidates = match_companies(ads, firms)
    _write_csv(matches, output_dir / "company_matches.csv")
    _write_csv(candidates, Path("artifacts/review/company_match_candidates.csv"))
    top_candidates = candidates.loc[candidates["candidate_rank"].eq(1), ["canonical_id", "candidate_stock_code", "candidate_company", "similarity"]]
    match_review = matches.loc[matches["match_method"].ne("exact_normalized")].merge(top_candidates, on="canonical_id", how="left", validate="one_to_one")
    match_review["review_decision"] = match_review["match_status"].map({"matched": "accepted_reviewed_rule", "unmatched": "rejected_auto_match"})
    _write_csv(match_review, Path("artifacts/review/company_match_review.csv"))
    labels = label_with_deepseek(ads, label_cache, Path("config/ai_rubric_v2_1.yaml"), allow_network=not offline)
    labels, reliability = run_reliability_audit(ads, labels, seed, allow_network=not offline)
    _write_csv(labels, output_dir / "ai_scores.csv")
    request_manifest_columns = [
        "canonical_id", "technology_role", "strict_ai", "score", "model_score", "boundary_pair",
        "content_hash", "model", "prompt_version", "schema_version", "prompt_fingerprint", "stage",
        "thinking", "label_status", "request_id", "input_tokens", "output_tokens",
    ]
    _write_csv(labels[request_manifest_columns], Path("artifacts/llm/request_manifest.csv"))
    annual = annual_summary(ads, labels)
    _write_csv(annual, output_dir / "annual_ai_share.csv")
    sensitivity = _write_version_comparison(labels, annual)
    plot_annual(annual, output_dir / "figures/annual_ai_share.png")
    report = _report(stats, matches, labels, annual, reliability, sensitivity)
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/ra_task_report.md").write_text(report, encoding="utf-8")
    Path("reports/ra_task_report.qmd").write_text("---\ntitle: \"招聘广告中的 AI / 数字技术含量\"\nlang: zh\nformat:\n  html:\n    embed-resources: true\n---\n\n" + "\n".join(report.splitlines()[1:]), encoding="utf-8")
    quarto = shutil.which("quarto")
    if not quarto:
        raise RuntimeError("生成自包含 HTML 报告需要 Quarto，请先安装并确保 quarto 在 PATH 中")
    subprocess.run([quarto, "render", "reports/ra_task_report.qmd", "--to", "html"], check=True)
    finished = datetime.now(timezone.utc)
    metadata = {
        "run_started_utc": started.isoformat(),
        "run_finished_utc": finished.isoformat(),
        "elapsed_seconds": round((finished - started).total_seconds(), 3),
        "seed": seed,
        "network_allowed": not offline,
        "api_key_present": bool(os.environ.get("DEEPSEEK_API_KEY")),
        "formal_cache_replay": offline,
        "prompt_version": PROMPT_VERSION,
        "version_sensitivity": sensitivity,
        "stats": stats,
        "reliability": reliability,
    }
    Path("artifacts/manifests").mkdir(parents=True, exist_ok=True)
    Path("artifacts/manifests/run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    verify_outputs(output_dir, write_report=True, require_archive=False, expected_prompt_version=PROMPT_VERSION)
    build_archive()
    verify_outputs(output_dir, require_archive=True, expected_prompt_version=PROMPT_VERSION)


def verify_outputs(
    output_dir: Path,
    *,
    write_report: bool = False,
    require_archive: bool = True,
    expected_prompt_version: str | None = None,
) -> dict:
    label_path = output_dir / "ai_scores.csv"
    artifact_prompt_version = PROMPT_VERSION
    if label_path.exists() and label_path.stat().st_size:
        versions = pd.read_csv(label_path, usecols=["prompt_version"], dtype=str)["prompt_version"].unique()
        if len(versions) == 1:
            artifact_prompt_version = str(versions[0])
    cache_dir = _cache_dir_for_version(artifact_prompt_version)
    required = [
        Path("data/raw/ra_task_ads.csv"), Path("data/raw/ra_task_firms.csv"),
        Path("data/processed/cleaned_ads.csv"), Path("data/processed/valid_firms.csv"),
        Path("data/interim/duplicate_map.csv"), output_dir / "company_matches.csv",
        output_dir / "ai_scores.csv", output_dir / "annual_ai_share.csv",
        output_dir / "figures/annual_ai_share.png", Path("artifacts/review/company_match_candidates.csv"),
        Path("artifacts/review/company_match_review.csv"),
        Path("artifacts/review/reliability_sample.csv"), Path("artifacts/review/reliability_metrics.json"),
        Path("artifacts/review/v1_v2_label_comparison.csv"), Path("artifacts/review/v1_v2_annual_comparison.csv"),
        Path("artifacts/review/v1_v2_summary.json"), cache_dir / "labels_cache.jsonl", cache_dir / "audit_cache.jsonl",
        cache_dir / "adjudication_cache.jsonl", Path("reports/ra_task_report.md"),
        Path("reports/ra_task_report.qmd"), Path("reports/ra_task_report.html"),
    ]
    if require_archive:
        required.append(Path("dist/ra_task_submission.zip"))
    if artifact_prompt_version == "2.1.0":
        required.extend(
            [
                Path("artifacts/review/v2_v2_1_label_comparison.csv"),
                Path("artifacts/review/v2_v2_1_annual_comparison.csv"),
                Path("artifacts/review/v2_v2_1_summary.json"),
            ]
        )
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing outputs: {missing}")
    ads = pd.read_csv("data/processed/cleaned_ads.csv", dtype=str, keep_default_na=False)
    firms = pd.read_csv("data/processed/valid_firms.csv", dtype=str, keep_default_na=False)
    duplicate_map = pd.read_csv("data/interim/duplicate_map.csv", dtype=str, keep_default_na=False)
    matches = pd.read_csv(output_dir / "company_matches.csv", dtype=str, keep_default_na=False)
    labels = pd.read_csv(output_dir / "ai_scores.csv", dtype=str, keep_default_na=False)
    annual = pd.read_csv(output_dir / "annual_ai_share.csv")
    reliability_sample = pd.read_csv("artifacts/review/reliability_sample.csv", dtype=str, keep_default_na=False)
    match_review = pd.read_csv("artifacts/review/company_match_review.csv", dtype=str, keep_default_na=False)
    reliability_metrics = json.loads(Path("artifacts/review/reliability_metrics.json").read_text(encoding="utf-8"))
    problems = []

    def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
        absent = [column for column in columns if column not in frame.columns]
        if absent:
            problems.append(f"{label} missing columns: {absent}")
        elif frame[columns].eq("").any().any():
            empty = [column for column in columns if frame[column].eq("").any()]
            problems.append(f"{label} empty required fields: {empty}")

    require_columns(ads, ["canonical_id", "source_ids", "duplicate_group_size", "公司名称", "关联公司名称", "岗位", "岗位描述", "发布时间", "published_date", "year"], "cleaned_ads")
    require_columns(firms, ["股票代码", "证券简称", "公司全称"], "valid_firms")
    require_columns(duplicate_map, ["source_id", "canonical_id", "is_duplicate"], "duplicate_map")
    require_columns(matches, ["canonical_id", "ad_company", "match_status", "match_method", "match_confidence"], "company_matches")
    require_columns(labels, ["canonical_id", "score", "reason", "confidence", "model", "prompt_version", "content_hash", "label_status"], "ai_scores")

    if len(ads) != 573 or ads["canonical_id"].duplicated().any(): problems.append("cleaned ads count/key")
    if len(firms) != 5461 or firms["股票代码"].duplicated().any(): problems.append("valid firms count/key")
    if len(duplicate_map) != 612 or duplicate_map["source_id"].duplicated().any() or int(duplicate_map["is_duplicate"].eq("True").sum()) != 39: problems.append("duplicate map count/key")
    text_columns = ["公司名称", "关联公司名称", "岗位", "岗位描述", "岗位标签", "待遇", "学历", "所在城市"]
    if any(ads[column].str.contains(TOKEN_RE).any() for column in text_columns): problems.append("uncleaned format token")
    if len(matches) != 573 or matches["canonical_id"].duplicated().any(): problems.append("company match count/key")
    matched_mask = matches["match_status"].eq("matched")
    if matches.loc[matched_mask, ["stock_code", "stock_name", "listed_company", "match_note"]].eq("").any().any(): problems.append("matched company missing required result")
    if matches.loc[~matched_mask, "unmatched_reason"].eq("").any(): problems.append("unmatched without reason")
    if matches.loc[matches["canonical_id"].eq("58"), "stock_code"].tolist() != ["002717.SZ"]: problems.append("known Lingnan false positive not corrected")
    if matches.loc[matches["canonical_id"].eq("298"), "match_status"].tolist() != ["unmatched"]: problems.append("known Zhongguancun false positive not rejected")
    if set(match_review["review_decision"]) != {"accepted_reviewed_rule", "rejected_auto_match"} or len(match_review) != int(matches["match_method"].ne("exact_normalized").sum()): problems.append("company review ledger incomplete")
    if len(labels) != 573 or labels["canonical_id"].duplicated().any(): problems.append("label count/key")
    scores = labels["score"].astype(int)
    if not scores.between(0, 3).all(): problems.append("score range")
    if set(labels["prompt_version"]) != {artifact_prompt_version}: problems.append("mixed prompt versions in final labels")
    if expected_prompt_version is not None and artifact_prompt_version != expected_prompt_version:
        problems.append(f"stale prompt version in final labels: expected {expected_prompt_version}")
    try:
        source_by_id = ads.set_index("canonical_id").apply(content_text, axis=1).to_dict()
        for _, row in labels.iterrows():
            evidence = [piece for piece in row["evidence"].split("|") if piece]
            source = source_by_id[row["canonical_id"]]
            if artifact_prompt_version == "2.1.0":
                payload = json.dumps(
                    {
                        "items": [
                            {
                                "canonical_id": row["canonical_id"],
                                "technology_role": row["technology_role"],
                                "strict_ai": str(row["strict_ai"]).casefold() == "true",
                                "score": int(row["score"]),
                                "boundary_pair": row["boundary_pair"],
                                "evidence": evidence,
                                "reason": row["reason"],
                                "confidence": row["confidence"],
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
                validate_batch(payload, {row["canonical_id"]: source})
                if int(row["model_score"]) not in range(4):
                    raise ValueError("invalid preserved model_score")
            else:
                if int(row["score"]) > 0 and not evidence:
                    raise ValueError("positive legacy label has no evidence")
                if any(_exact_source_span(piece, source) is None for piece in evidence):
                    raise ValueError("legacy evidence is not a source substring")
                if row["confidence"] not in {"high", "medium", "low"} or not row["reason"]:
                    raise ValueError("invalid legacy reason/confidence")
    except (KeyError, TypeError, ValueError) as exc:
        problems.append(f"invalid label evidence/schema: {exc}")
    if labels["label_status"].str.startswith("provisional").any(): problems.append("provisional labels remain")
    if len(annual) != 12 or set(annual["year"].astype(int)) != set(range(2014, 2026)) or int(annual["n_ads"].sum()) != 573: problems.append("annual years/denominator")
    share_columns = ["share_score_ge_2", "wilson_low", "wilson_high", "share_score_ge_1", "share_score_eq_3"]
    if not annual[share_columns].apply(lambda column: column.between(0, 1)).all().all(): problems.append("annual share bounds")
    try:
        validate_annual_consistency(ads, labels[["canonical_id", "score"]], annual)
    except ValueError as exc:
        problems.append(str(exc))
    if len(reliability_sample) != 120 or reliability_sample["canonical_id"].duplicated().any(): problems.append("reliability sample count/key")
    disagreement = reliability_sample["primary_score"].ne(reliability_sample["audit_score"])
    if reliability_sample.loc[disagreement, ["adjudicated_score", "adjudication_reason"]].eq("").any().any(): problems.append("unadjudicated reliability disagreement")
    recalculated_reliability = calculate_reliability_metrics(reliability_sample)
    for key, expected in recalculated_reliability.items():
        actual = reliability_metrics.get(key)
        if isinstance(expected, float):
            if actual is None or abs(float(actual) - expected) > 1e-12:
                problems.append(f"reliability metric mismatch: {key}")
        elif actual != expected:
            problems.append(f"reliability metric mismatch: {key}")
    aliases = pd.read_csv("config/company_aliases.csv", dtype=str, keep_default_na=False)
    provenance_columns = ["reviewer", "reviewed_at", "review_basis"]
    if not set(provenance_columns).issubset(aliases.columns) or aliases[provenance_columns].eq("").any().any(): problems.append("company alias provenance incomplete")
    report = Path("reports/ra_task_report.md").read_text(encoding="utf-8")
    report_headings = ["数据与清洗", "公司匹配", "AI / 数字技术编码", "年度结果", "信度检验", "提示词与版本敏感性", "发现", "数据局限", "可复现性"]
    for heading in report_headings:
        if f"## {heading}\n\n" not in report: problems.append(f"report missing/nonstandard section: {heading}")
    if "false positive" not in report or "岭南园林" not in report: problems.append("report missing actual false-positive example")
    if "约用 4 小时" not in report: problems.append("report missing approximate time")

    if require_archive:
        archive = Path("dist/ra_task_submission.zip")
        with zipfile.ZipFile(archive) as bundle:
            names = set(bundle.namelist())
            archive_required = {"README.md", "outputs/company_matches.csv", "outputs/annual_ai_share.csv", "outputs/figures/annual_ai_share.png", "reports/ra_task_report.html", "verification_report.md"}
            if not archive_required.issubset(names): problems.append("archive missing key deliverables")
            if any(name.startswith(".env") or "/.env" in name or "private/" in name for name in names): problems.append("archive contains excluded secret/private path")
            secret_pattern = re.compile(rb"sk-[A-Za-z0-9]{20,}")
            if any(secret_pattern.search(bundle.read(name)) for name in names if not name.startswith("data/raw/") and bundle.getinfo(name).file_size < 20_000_000): problems.append("archive contains API-key-like secret")
    if problems: raise ValueError("Verification failed: " + ", ".join(problems))
    optional_blanks = {
        "cleaned_ads.岗位标签": int(ads["岗位标签"].eq("").sum()),
        "cleaned_ads.待遇": int(ads["待遇"].eq("").sum()),
        "cleaned_ads.学历": int(ads["学历"].eq("").sum()),
        "cleaned_ads.所在城市": int(ads["所在城市"].eq("").sum()),
        "company_matches.industry": int(matches["industry"].eq("").sum()),
        "ai_scores.audit_score": int(labels["audit_score"].eq("").sum()),
    }
    result = {"status": "PASS", "files_checked": len(required), "canonical_ads": 573, "matched_ads": int(matched_mask.sum()), "listed_companies": int(matches.loc[matched_mask, "stock_code"].nunique()), "unmatched_ads": int((~matched_mask).sum()), "provisional_labels": 0, "reliability_sample": 120, "prompt_version": artifact_prompt_version}
    if write_report:
        optional_lines = "\n".join(f"- `{key}`：{value} 个空值" for key, value in optional_blanks.items())
        audit_sentence = f"120 条同模型盲重测中的 {reliability_metrics['disagreements']} 条分歧均已完成上下文裁决。"
        report_text = "# 验证报告\n\n## 结论\n\n" + "\n".join(f"- {key}: {value}" for key, value in result.items()) + "\n\n## 完整性检查\n\n必填字段无空值；主键唯一；未匹配公司均有原因；所有正分标签均有原文证据；" + audit_sentence + "\n\n## 可选字段空值 / NA\n\n下列空值来自原始数据或条件字段，不属于不完整交付：\n\n" + optional_lines + "\n\n## 未完成项\n\n无。\n"
        Path("verification_report.md").write_text(report_text, encoding="utf-8")
        files = [path for path in Path(".").rglob("*") if path.is_file() and ".git" not in path.parts and ".venv" not in path.parts and "dist" not in path.parts and path.name != "file_manifest.csv"]
        manifest = pd.DataFrame([{"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in sorted(files)])
        _write_csv(manifest, Path("artifacts/manifests/file_manifest.csv"))
    return result


def build_archive() -> None:
    Path("dist").mkdir(exist_ok=True)
    archive = Path("dist/ra_task_submission.zip")
    include = ["README.md", "pyproject.toml", "uv.lock", "data/raw", "data/processed", "data/interim", "src", "config", "tests", "docs", "outputs", "reports", "artifacts", "verification_report.md"]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for item in include:
            path = Path(item)
            if not path.exists(): continue
            if path.is_file(): bundle.write(path, path)
            else:
                for child in path.rglob("*"):
                    if child.is_file() and ".env" not in child.name and "private" not in child.parts:
                        bundle.write(child, child)
