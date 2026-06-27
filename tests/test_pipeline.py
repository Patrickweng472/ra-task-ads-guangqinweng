from pathlib import Path

import pandas as pd

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
