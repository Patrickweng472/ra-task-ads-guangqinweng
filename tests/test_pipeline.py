from pathlib import Path
import shutil

import pandas as pd
import pytest

import ra_task.pipeline as pipeline
from ra_task.pipeline import apply_adjudications, verify_outputs


def test_committed_outputs_verify_when_present() -> None:
    if Path("outputs/ai_scores.csv").exists():
        result = verify_outputs(Path("outputs"))
        assert result["status"] == "PASS"


def test_adjudication_replaces_score_and_explanation_atomically() -> None:
    primary = pd.DataFrame([{"canonical_id": "1", "score": 1, "evidence": "系统", "reason": "主编码", "confidence": "medium", "label_status": "llm_primary"}])
    comparison = pd.DataFrame([{"canonical_id": "1", "audit_score": 3}])
    final = pd.DataFrame([{"canonical_id": "1", "score": 2, "evidence": "软件开发", "reason": "复判理由", "confidence": "high"}])
    result = apply_adjudications(primary, comparison, final).iloc[0]
    assert (result["score"], result["evidence"], result["reason"], result["confidence"], result["label_status"]) == (2, "软件开发", "复判理由", "high", "llm_adjudicated")


def test_audit_selection_always_includes_high_information_cases() -> None:
    assert hasattr(pipeline, "select_audit_sample")
    ads = pd.DataFrame([
        {"canonical_id": "1", "岗位": "销售", "岗位描述": "使用ERP", "岗位标签": ""},
        {"canonical_id": "2", "岗位": "数据工程师", "岗位描述": "建设数据平台", "岗位标签": ""},
        {"canonical_id": "3", "岗位": "机器学习工程师", "岗位描述": "训练深度学习模型", "岗位标签": "AI"},
        {"canonical_id": "4", "岗位": "行政", "岗位描述": "文件管理", "岗位标签": ""},
    ])
    labels = pd.DataFrame([
        {"canonical_id": "1", "score": 0, "confidence": "low"},
        {"canonical_id": "2", "score": 0, "confidence": "high"},
        {"canonical_id": "3", "score": 3, "confidence": "high"},
        {"canonical_id": "4", "score": 0, "confidence": "high"},
    ])
    selected = pipeline.select_audit_sample(ads, labels, seed=7, target_size=3)
    assert set(selected["canonical_id"]) == {"1", "2", "3"}
    reasons = selected.set_index("canonical_id")["selection_reason"]
    assert "low_confidence" in reasons["1"]
    assert "rule_model_threshold_conflict" in reasons["2"]
    assert "strict_score3" in reasons["3"]


def test_adjudication_payload_contains_both_blind_codings() -> None:
    assert hasattr(pipeline, "build_adjudication_context")
    row = pd.Series({"canonical_id": "1", "primary_score": 1, "primary_confidence": "medium", "audit_score": 2, "audit_confidence": "high", "audit_reason": "技术是核心"})
    context = pipeline.build_adjudication_context(row, primary_reason="工具仅辅助", primary_evidence="ERP")
    assert context["primary"]["score"] == 1
    assert context["audit"]["score"] == 2
    assert context["primary"]["reason"] == "工具仅辅助"


def test_annual_validation_recomputes_values_instead_of_only_checking_bounds() -> None:
    assert hasattr(pipeline, "validate_annual_consistency")
    ads = pd.DataFrame({"canonical_id": ["1", "2"], "year": [2025, 2025]})
    labels = pd.DataFrame({"canonical_id": ["1", "2"], "score": [0, 2]})
    annual = pipeline.annual_summary(ads, labels)
    annual.loc[0, "share_score_ge_2"] = 0.75
    with pytest.raises(ValueError, match="annual summary"):
        pipeline.validate_annual_consistency(ads, labels, annual)


def test_output_transaction_restores_previous_artifact_after_failure(tmp_path: Path) -> None:
    assert hasattr(pipeline, "OutputTransaction")
    artifact = tmp_path / "result.txt"
    artifact.write_text("stable", encoding="utf-8")
    with pytest.raises(RuntimeError):
        with pipeline.OutputTransaction([artifact]):
            artifact.write_text("partial", encoding="utf-8")
            raise RuntimeError("render failed")
    assert artifact.read_text(encoding="utf-8") == "stable"


def test_submission_archive_excludes_nested_candidate_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("README.md").write_text("submission", encoding="utf-8")
    candidate = Path("artifacts/candidates/v2_1")
    candidate.mkdir(parents=True)
    (candidate / "nested.zip").write_bytes(b"not a real zip")

    pipeline.build_archive()

    with pipeline.zipfile.ZipFile("dist/ra_task_submission.zip") as archive:
        assert "README.md" in archive.namelist()
        assert not any(name.startswith("artifacts/candidates/") for name in archive.namelist())


def test_snapshot_preserves_formal_v2_before_v2_1_rerun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    Path("outputs").mkdir()
    Path("outputs/ai_scores.csv").write_text("canonical_id,prompt_version,score\n1,2.0.0,1\n", encoding="utf-8")
    Path("outputs/annual_ai_share.csv").write_text(
        "year,share_score_ge_2,share_score_ge_1,share_score_eq_3\n2025,0,1,0\n", encoding="utf-8"
    )

    pipeline._snapshot_previous_version()

    assert Path("artifacts/baselines/v2/ai_scores.csv").exists()
    assert Path("artifacts/baselines/v2/annual_ai_share.csv").exists()


def test_version_comparison_uses_v2_baseline_for_v2_1_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    baseline = Path("artifacts/baselines/v2")
    baseline.mkdir(parents=True)
    pd.DataFrame(
        [{"canonical_id": "1", "score": 1, "evidence": "Excel", "reason": "辅助"}]
    ).to_csv(baseline / "ai_scores.csv", index=False)
    pd.DataFrame(
        [{"year": 2025, "share_score_ge_2": 0.0, "share_score_ge_1": 1.0, "share_score_eq_3": 0.0}]
    ).to_csv(baseline / "annual_ai_share.csv", index=False)
    labels = pd.DataFrame(
        [{"canonical_id": "1", "score": 2, "evidence": "数据平台", "reason": "核心"}]
    )
    annual = pd.DataFrame(
        [{"year": 2025, "share_score_ge_2": 1.0, "share_score_ge_1": 1.0, "share_score_eq_3": 0.0}]
    )

    summary = pipeline._write_version_comparison(labels, annual)

    assert summary["from_version"] == "v2"
    assert summary["to_version"] == "v2.1"
    assert summary["main_threshold_changes"] == 1
    assert Path("artifacts/review/v2_v2_1_label_comparison.csv").exists()


def test_offline_mode_fails_before_outputs_when_formal_cache_is_absent(tmp_path: Path) -> None:
    assert hasattr(pipeline, "require_formal_cache")
    with pytest.raises(RuntimeError, match="formal v2 cache"):
        pipeline.require_formal_cache(tmp_path / "missing.jsonl", offline=True)


@pytest.mark.skipif(
    shutil.which("quarto") is None or not Path("artifacts/llm/v2_1/labels_cache.jsonl").exists(),
    reason="Quarto and formal v2.1 caches are required for the full delivery pipeline",
)
def test_full_offline_pipeline_rebuilds_delivery_from_formal_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    for relative in ["config", "data/raw", "artifacts/llm/v2_1", "artifacts/baselines/v1", "artifacts/baselines/v2"]:
        shutil.copytree(project_root / relative, tmp_path / relative)
    shutil.copy2(project_root / "README.md", tmp_path / "README.md")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    pipeline.run_pipeline(
        Path("data/raw/ra_task_ads.csv"),
        Path("data/raw/ra_task_firms.csv"),
        Path("outputs"),
        offline=True,
        seed=20260627,
    )

    assert pipeline.verify_outputs(Path("outputs"))["status"] == "PASS"
    metadata = (tmp_path / "artifacts/manifests/run_metadata.json").read_text(encoding="utf-8")
    assert '"formal_cache_replay": true' in metadata
    assert '"api_key_present": false' in metadata


def test_report_names_strict_ai_and_same_model_retest_honestly() -> None:
    stats = {
        "raw_ads": 2, "duplicate_groups": 0, "duplicates_removed": 0, "canonical_ads": 2,
        "raw_firm_rows": 2, "valid_firms": 2,
    }
    matches = pd.DataFrame(
        [
            {"match_status": "matched", "stock_code": "000001.SZ", "match_method": "exact_normalized"},
            {"match_status": "unmatched", "stock_code": "", "match_method": "unmatched"},
        ]
    )
    labels = pd.DataFrame(
        [
            {"score": 0, "label_status": "llm_primary"},
            {"score": 3, "label_status": "llm_adjudicated"},
        ]
    )
    annual = pd.DataFrame(
        [
            {
                "year": 2025, "n_ads": 2, "share_score_ge_1": 0.5, "share_score_ge_2": 0.5,
                "share_score_eq_3": 0.5, "wilson_low": 0.1, "wilson_high": 0.9,
            }
        ]
    )
    reliability = {
        "sample_size": 2, "disagreements": 1, "exact_agreement": 0.5,
        "within_one_agreement": 1.0, "binary_agreement_score_ge_2": 0.5,
        "quadratic_weighted_kappa": 0.4, "targeted_cases": 1,
        "low_confidence_cases": 0, "rule_model_threshold_conflicts": 1, "strict_score3_cases": 1,
    }
    sensitivity = {
        "available": True, "score_changes": 1, "main_threshold_changes": 1,
        "from_version": "v2", "to_version": "v2.1",
        "from_score_distribution": {0: 1, 2: 1}, "to_score_distribution": {0: 1, 3: 1},
        "max_abs_annual_main_share_change": 0.1,
    }

    report = pipeline._report(stats, matches, labels, annual, reliability, sensitivity)

    assert "严格 AI 研发" in report
    assert "同一 DeepSeek 模型" in report
    assert "不等同于独立人工编码者信度" in report
    assert "金融定价模型" in report
    assert "版本敏感性" in report
    assert "锁定留出集尚未完成" in report
    assert "第三次独立复判" not in report
