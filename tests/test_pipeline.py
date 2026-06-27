from pathlib import Path

import pandas as pd

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
