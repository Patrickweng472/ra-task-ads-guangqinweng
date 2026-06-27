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
    zero = json.dumps({"items": [{"canonical_id": "1", "score": 0, "evidence": [], "reason": "没有技术内容", "confidence": "high"}]}, ensure_ascii=False)
    assert validate_batch(zero, {"1": "普通销售岗位"})[0].score == 0
    paraphrase = json.dumps({"items": [{"canonical_id": "1", "score": 2, "evidence": ["不存在的改写"], "reason": "技术岗位", "confidence": "medium"}]}, ensure_ascii=False)
    assert validate_batch(paraphrase, {"1": "岗位：软件工程师\n描述：开发系统"})[0].evidence == ["软件工程师"]


def test_provisional_labels_are_explicit() -> None:
    ads = pd.DataFrame([{"canonical_id": "1", "岗位": "算法工程师", "岗位描述": "负责机器学习模型", "岗位标签": "AI"}])
    labels = provisional_labels(ads)
    assert labels.loc[0, "score"] == 3
    assert labels.loc[0, "label_status"] == "provisional_no_api"
    assert len(content_hash("x")) == 64


def test_identical_text_can_belong_to_distinct_ads() -> None:
    ads = pd.DataFrame([
        {"canonical_id": "1", "岗位": "销售", "岗位描述": "维护客户", "岗位标签": ""},
        {"canonical_id": "2", "岗位": "销售", "岗位描述": "维护客户", "岗位标签": ""},
    ])
    labels = provisional_labels(ads)
    assert labels["content_hash"].nunique() == 1
    assert labels["canonical_id"].is_unique
