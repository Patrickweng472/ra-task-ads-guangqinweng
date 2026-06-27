import json

import pandas as pd
import pytest

from ra_task.llm_labeling import content_hash, provisional_labels, validate_batch


def test_validate_batch_checks_ids_and_evidence() -> None:
    payload = json.dumps({"items": [{"canonical_id": "1", "score": 3, "evidence": ["机器学习"], "reason": "核心职责", "confidence": "high"}]}, ensure_ascii=False)
    labels = validate_batch(payload, {"1": "岗位要求机器学习经验"})
    assert labels[0].score == 3
    with pytest.raises(ValueError):
        validate_batch(payload, {"2": "岗位要求机器学习经验"})


def test_provisional_labels_are_explicit() -> None:
    ads = pd.DataFrame([{"canonical_id": "1", "岗位": "算法工程师", "岗位描述": "负责机器学习模型", "岗位标签": "AI"}])
    labels = provisional_labels(ads)
    assert labels.loc[0, "score"] == 3
    assert labels.loc[0, "label_status"] == "provisional_no_api"
    assert len(content_hash("x")) == 64

