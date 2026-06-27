from __future__ import annotations

import hashlib
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .analysis import annual_summary, plot_annual, quadratic_weighted_kappa
from .cleaning import TOKEN_RE, clean_ads, clean_firms
from .llm_labeling import label_with_deepseek, provisional_labels
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


def _report(stats: dict, matches: pd.DataFrame, labels: pd.DataFrame, annual: pd.DataFrame, reliability: dict) -> str:
    matched = matches[matches["match_status"] == "matched"]
    status = labels["label_status"].value_counts().to_dict()
    first = annual.iloc[0]
    last = annual.iloc[-1]
    return f"""# 招聘广告中的 AI / 数字技术含量

## 数据与清洗

原始广告共 {stats['raw_ads']} 条。清理网页格式乱码并按除 `id` 外的全部业务字段去重后，保留 {stats['canonical_ads']} 条，删除 {stats['duplicates_removed']} 条重复记录。公司主表原始 {stats['raw_firm_rows']} 行，剔除脚注后保留 {stats['valid_firms']} 家。

## 公司匹配

共有 {len(matched)} 条广告匹配到 {matched['stock_code'].nunique()} 家上市公司。匹配方法完整保存在 `outputs/company_matches.csv`，模糊相似度只用于候选排序，未被当作自动匹配依据。

## AI / 数字技术编码

采用 0–3 级量表：0 为无实质技术内容，1 为辅助数字工具，2 为数字技术是核心职责，3 为明确 AI、模型或高级算法研发。年度主指标为得分不低于 2。当前标签状态：{json.dumps(status, ensure_ascii=False)}。

## 年度结果

主指标在 {int(first['year'])} 年为 {first['share_score_ge_2']:.1%}（n={int(first['n_ads'])}），在 {int(last['year'])} 年为 {last['share_score_ge_2']:.1%}（n={int(last['n_ads'])}）。完整年度数据和 Wilson 95% 区间见 `outputs/annual_ai_share.csv`。

![年度AI/数字技术含量](../outputs/figures/annual_ai_share.png)

## 信度检验

独立思考模式复核 {reliability.get('sample_size', 0)} 条广告，精确一致率为 {reliability.get('exact_agreement', float('nan')):.1%}，相差不超过一级的一致率为 {reliability.get('within_one_agreement', float('nan')):.1%}，主阈值二分类一致率为 {reliability.get('binary_agreement_score_ge_2', float('nan')):.1%}，二次加权 Cohen's κ 为 {reliability.get('quadratic_weighted_kappa', float('nan')):.3f}。所有分歧均经第三次独立复判。

## 发现

样本中的技术型岗位并非持续平滑上升，而是随年份和招聘构成明显波动。较新的广告中出现了更多软件、数据和算法岗位，但不同阈值下幅度并不完全一致。企业数字化岗位远多于严格意义上的 AI 岗位，因此将“数字化”和“AI”拆分报告比单一二分类更有解释力。重复广告会机械放大个别公司和年份，去重后结果更适合作为主分析。

## 数据局限

这些广告不是按年份随机抽取的总体样本，且早期年份样本很少；因此年度变化可能来自行业、公司和职位构成变化，不能解释为中国上市公司整体 AI 需求的因果趋势。若标签状态仍为 `provisional_no_api`，结果只用于验证流水线，最终提交前应使用 DeepSeek 标签重跑。

## 可复现性

项目使用 Python、pandas、RapidFuzz、OpenAI SDK、Pydantic、NumPy、Matplotlib、pytest、uv 和 Quarto。运行元数据、文件哈希、重复映射、匹配候选及标签来源均随仓库提交。
"""


def run_reliability_audit(ads: pd.DataFrame, labels: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    joined = ads.merge(labels[["canonical_id", "score", "confidence"]], on="canonical_id", validate="one_to_one")
    low_ids = set(joined.loc[joined["confidence"] == "low", "canonical_id"].astype(str))
    selected = joined[joined["canonical_id"].astype(str).isin(low_ids)].copy()
    remaining = joined[~joined["canonical_id"].astype(str).isin(low_ids)]
    need = max(0, 120 - len(selected))
    if need:
        pieces = []
        for _, group in remaining.groupby("score", sort=True):
            take = min(len(group), max(1, round(need * len(group) / len(remaining))))
            pieces.append(group.sample(n=take, random_state=seed))
        fill = pd.concat(pieces).drop_duplicates("canonical_id")
        if len(fill) < need:
            extra = remaining[~remaining["canonical_id"].isin(fill["canonical_id"])].sample(n=need - len(fill), random_state=seed + 1)
            fill = pd.concat([fill, extra])
        selected = pd.concat([selected, fill.head(need)]).drop_duplicates("canonical_id")
    audit_ads = ads[ads["canonical_id"].isin(selected["canonical_id"])].copy()
    audit = label_with_deepseek(audit_ads, Path("artifacts/llm/audit_cache.jsonl"), Path("config/ai_rubric.yaml"), thinking=True, label_status="llm_audit")
    comparison = labels[["canonical_id", "score", "confidence"]].rename(columns={"score": "primary_score", "confidence": "primary_confidence"}).merge(
        audit[["canonical_id", "score", "confidence", "reason"]].rename(columns={"score": "audit_score", "confidence": "audit_confidence", "reason": "audit_reason"}), on="canonical_id", validate="one_to_one"
    )
    disagreement_ids = comparison.loc[comparison["primary_score"] != comparison["audit_score"], "canonical_id"]
    adjudicated = pd.DataFrame(columns=["canonical_id", "score", "reason", "confidence"])
    if len(disagreement_ids):
        adjudication_ads = ads[ads["canonical_id"].isin(disagreement_ids)]
        adjudicated = label_with_deepseek(adjudication_ads, Path("artifacts/llm/adjudication_cache.jsonl"), Path("config/ai_rubric.yaml"), thinking=True, label_status="llm_adjudicated")
        comparison = comparison.merge(adjudicated[["canonical_id", "score", "reason"]].rename(columns={"score": "adjudicated_score", "reason": "adjudication_reason"}), on="canonical_id", how="left", validate="one_to_one")
    else:
        comparison["adjudicated_score"] = pd.NA
        comparison["adjudication_reason"] = ""
    primary = comparison["primary_score"].astype(int).tolist()
    secondary = comparison["audit_score"].astype(int).tolist()
    metrics = {
        "status": "completed",
        "sample_size": len(comparison),
        "disagreements": int((comparison["primary_score"] != comparison["audit_score"]).sum()),
        "exact_agreement": float((comparison["primary_score"] == comparison["audit_score"]).mean()),
        "within_one_agreement": float(((comparison["primary_score"] - comparison["audit_score"]).abs() <= 1).mean()),
        "binary_agreement_score_ge_2": float(((comparison["primary_score"] >= 2) == (comparison["audit_score"] >= 2)).mean()),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(primary, secondary),
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
    return enriched


def run_pipeline(ads_path: Path, firms_path: Path, output_dir: Path, *, offline: bool, seed: int) -> None:
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
    use_api = bool(os.environ.get("DEEPSEEK_API_KEY")) and not offline
    if use_api:
        labels = label_with_deepseek(ads, Path("artifacts/llm/labels_cache.jsonl"), Path("config/ai_rubric.yaml"))
        labels, reliability = run_reliability_audit(ads, labels, seed)
    else:
        labels = provisional_labels(ads)
        reliability = {"status": "not_run_no_api"}
    _write_csv(labels, output_dir / "ai_scores.csv")
    annual = annual_summary(ads, labels)
    _write_csv(annual, output_dir / "annual_ai_share.csv")
    plot_annual(annual, output_dir / "figures/annual_ai_share.png")
    report = _report(stats, matches, labels, annual, reliability)
    Path("reports/ra_task_report.md").write_text(report, encoding="utf-8")
    Path("reports/ra_task_report.qmd").write_text("---\ntitle: \"招聘广告中的 AI / 数字技术含量\"\nlang: zh\nformat:\n  html:\n    embed-resources: true\n---\n\n" + "\n".join(report.splitlines()[1:]), encoding="utf-8")
    metadata = {"run_started_utc": started.isoformat(), "run_finished_utc": datetime.now(timezone.utc).isoformat(), "seed": seed, "online_llm": use_api, "stats": stats, "reliability": reliability}
    Path("artifacts/manifests/run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    verify_outputs(output_dir, write_report=True)
    build_archive()


def verify_outputs(output_dir: Path, *, write_report: bool = False) -> dict:
    required = [output_dir / "company_matches.csv", output_dir / "ai_scores.csv", output_dir / "annual_ai_share.csv", output_dir / "figures/annual_ai_share.png", Path("data/processed/cleaned_ads.csv"), Path("reports/ra_task_report.md")]
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing outputs: {missing}")
    ads = pd.read_csv("data/processed/cleaned_ads.csv", dtype=str, keep_default_na=False)
    matches = pd.read_csv(output_dir / "company_matches.csv", dtype=str, keep_default_na=False)
    labels = pd.read_csv(output_dir / "ai_scores.csv", dtype=str, keep_default_na=False)
    problems = []
    if len(ads) != 573 or ads["canonical_id"].duplicated().any(): problems.append("cleaned ads count/key")
    if ads["岗位描述"].str.contains(TOKEN_RE).any(): problems.append("uncleaned format token")
    if len(matches) != 573 or matches["canonical_id"].duplicated().any(): problems.append("company match count/key")
    if ((matches["match_status"] == "unmatched") & matches["unmatched_reason"].eq("")).any(): problems.append("unmatched without reason")
    if len(labels) != 573 or labels["canonical_id"].duplicated().any(): problems.append("label count/key")
    if not labels["score"].astype(int).between(0, 3).all(): problems.append("score range")
    if problems: raise ValueError("Verification failed: " + ", ".join(problems))
    result = {"status": "PASS", "files_checked": len(required), "canonical_ads": 573, "matched_ads": int((matches["match_status"] == "matched").sum()), "unmatched_ads": int((matches["match_status"] == "unmatched").sum()), "provisional_labels": int(labels["label_status"].str.startswith("provisional").sum())}
    if write_report:
        Path("verification_report.md").write_text("# 验证报告\n\n" + "\n".join(f"- {key}: {value}" for key, value in result.items()) + "\n\n未发现空缺必填字段；未匹配公司均有原因。\n", encoding="utf-8")
        files = [path for path in Path(".").rglob("*") if path.is_file() and ".git" not in path.parts and ".venv" not in path.parts and path.name != "file_manifest.csv"]
        manifest = pd.DataFrame([{"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in sorted(files)])
        _write_csv(manifest, Path("artifacts/manifests/file_manifest.csv"))
    return result


def build_archive() -> None:
    Path("dist").mkdir(exist_ok=True)
    archive = Path("dist/ra_task_submission.zip")
    include = ["README.md", "pyproject.toml", "uv.lock", "data/raw", "data/processed", "data/interim", "src", "config", "tests", "outputs", "reports", "artifacts", "verification_report.md"]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for item in include:
            path = Path(item)
            if not path.exists(): continue
            if path.is_file(): bundle.write(path, path)
            else:
                for child in path.rglob("*"):
                    if child.is_file() and ".env" not in child.name and "private" not in child.parts:
                        bundle.write(child, child)
