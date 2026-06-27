import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

import ra_task.llm_labeling as llm
from ra_task.llm_labeling import content_hash, label_with_deepseek, provisional_labels, validate_batch


def test_validate_batch_checks_ids_and_evidence() -> None:
    payload = json.dumps({"items": [{"canonical_id": "1", "score": 3, "evidence": ["机器学习"], "reason": "核心职责", "confidence": "high"}]}, ensure_ascii=False)
    labels = validate_batch(payload, {"1": "岗位要求机器学习经验"})
    assert labels[0].score == 3
    with pytest.raises(ValueError):
        validate_batch(payload, {"2": "岗位要求机器学习经验"})
    zero = json.dumps({"items": [{"canonical_id": "1", "score": 0, "evidence": [], "reason": "没有技术内容", "confidence": "high"}]}, ensure_ascii=False)
    assert validate_batch(zero, {"1": "普通销售岗位"})[0].score == 0
    paraphrase = json.dumps({"items": [{"canonical_id": "1", "score": 2, "evidence": ["不存在的改写"], "reason": "技术岗位", "confidence": "medium"}]}, ensure_ascii=False)
    with pytest.raises(ValueError, match="evidence"):
        validate_batch(paraphrase, {"1": "岗位：软件工程师\n描述：开发系统"})


def test_generic_title_cannot_silently_replace_supporting_evidence() -> None:
    payload = json.dumps({"items": [{"canonical_id": "1", "score": 1, "evidence": ["销售"], "reason": "需使用办公软件", "confidence": "medium"}]}, ensure_ascii=False)
    with pytest.raises(ValueError, match="support"):
        validate_batch(payload, {"1": "岗位：销售\n描述：熟练使用办公软件\n标签："})


def test_v2_prompt_defines_ai_only_boundary_and_confidence() -> None:
    rubric = yaml.safe_load(Path("config/ai_rubric.yaml").read_text(encoding="utf-8"))
    prompt = llm._system_prompt(rubric, thinking=False)
    for phrase in ["金融定价模型", "物理仿真", "通信算法", "为什么不是相邻等级", "low"]:
        assert phrase in prompt
    assert len(prompt) >= 1200


def test_prompt_fingerprint_changes_with_rubric_stage_and_thinking() -> None:
    assert hasattr(llm, "prompt_fingerprint")
    rubric = {"version": "2", "scores": {0: {"definition": "none"}}}
    primary = llm.prompt_fingerprint(rubric, stage="primary", thinking=False)
    assert primary != llm.prompt_fingerprint({**rubric, "version": "3"}, stage="primary", thinking=False)
    assert primary != llm.prompt_fingerprint(rubric, stage="audit", thinking=False)
    assert primary != llm.prompt_fingerprint(rubric, stage="primary", thinking=True)


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


def test_complete_formal_cache_replays_without_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ads = pd.DataFrame([{"canonical_id": "1", "岗位": "数据工程师", "岗位描述": "开发数据平台", "岗位标签": "SQL"}])
    text = "岗位：数据工程师\n描述：开发数据平台\n标签：SQL"
    rubric = tmp_path / "rubric.yaml"
    rubric.write_text(Path("config/ai_rubric.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    rubric_data = yaml.safe_load(rubric.read_text(encoding="utf-8"))
    fingerprint = llm.prompt_fingerprint(rubric_data, stage="primary", thinking=False)
    record = {"canonical_id": "1", "score": 2, "evidence": "数据平台", "reason": "数字技术是核心职责，而不是辅助工具", "confidence": "medium", "model": llm.MODEL, "prompt_version": llm.PROMPT_VERSION, "schema_version": llm.SCHEMA_VERSION, "prompt_fingerprint": fingerprint, "stage": "primary", "thinking": False, "content_hash": content_hash(text, fingerprint), "label_status": "llm_primary"}
    cache = tmp_path / "cache.jsonl"
    cache.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    result = label_with_deepseek(ads, cache, rubric, allow_network=False)
    assert result.loc[0, "score"] == 2
    assert result.loc[0, "label_status"] == "llm_primary"


def test_stale_or_foreign_cache_record_is_not_accepted(tmp_path: Path) -> None:
    ads = pd.DataFrame([{"canonical_id": "1", "岗位": "数据工程师", "岗位描述": "开发数据平台", "岗位标签": "SQL"}])
    text = "岗位：数据工程师\n描述：开发数据平台\n标签：SQL"
    stale = {"canonical_id": "1", "score": 2, "evidence": "数据平台", "reason": "数字技术是核心职责", "confidence": "high", "model": "foreign-model", "prompt_version": "old", "content_hash": content_hash(text), "label_status": "llm_primary"}
    cache = tmp_path / "cache.jsonl"
    cache.write_text(json.dumps(stale, ensure_ascii=False) + "\n", encoding="utf-8")
    rubric = tmp_path / "rubric.yaml"
    rubric.write_text(Path("config/ai_rubric.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Offline cache is incomplete"):
        label_with_deepseek(ads, cache, rubric, allow_network=False)


def test_empty_response_retries_and_writes_valid_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ads = pd.DataFrame([{"canonical_id": "1", "岗位": "算法工程师", "岗位描述": "开发机器学习模型", "岗位标签": "AI"}])
    valid = json.dumps({"items": [{"canonical_id": "1", "score": 3, "evidence": ["机器学习"], "reason": "AI研发", "confidence": "high"}]}, ensure_ascii=False)

    class Completions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_: object) -> object:
            self.calls += 1
            content = "" if self.calls == 1 else valid
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=None)

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    rubric = tmp_path / "rubric.yaml"
    rubric.write_text("scores: {}\n", encoding="utf-8")
    monkeypatch.setattr("ra_task.llm_labeling.time.sleep", lambda _: None)
    result = label_with_deepseek(ads, tmp_path / "cache.jsonl", rubric, client=client)
    assert completions.calls == 2
    assert result.loc[0, "score"] == 3


def test_incomplete_offline_cache_fails_explicitly(tmp_path: Path) -> None:
    ads = pd.DataFrame([{"canonical_id": "1", "岗位": "销售", "岗位描述": "维护客户", "岗位标签": ""}])
    rubric = tmp_path / "rubric.yaml"
    rubric.write_text("scores: {}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Offline cache is incomplete"):
        label_with_deepseek(ads, tmp_path / "missing.jsonl", rubric, allow_network=False)
