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


def test_v2_1_schema_requires_dimensions_and_derives_consistent_score() -> None:
    source = "岗位：联合质量工程师\n描述：协助AI项目推进和跨部门沟通\n标签：质量"
    payload = json.dumps(
        {
            "items": [
                {
                    "canonical_id": "1",
                    "technology_role": "auxiliary",
                    "strict_ai": False,
                    "score": 1,
                    "boundary_pair": "1_vs_2",
                    "evidence": ["协助AI项目推进"],
                    "reason": "AI只作为项目背景，质量协调仍是主业，不承担模型、数据或系统技术工作。",
                    "confidence": "high",
                }
            ]
        },
        ensure_ascii=False,
    )

    label = validate_batch(payload, {"1": source})[0]

    assert label.technology_role == "auxiliary"
    assert label.strict_ai is False
    assert label.boundary_pair == "1_vs_2"
    assert llm.derive_score(label.technology_role, label.strict_ai) == label.score


def test_v2_1_schema_rejects_score_dimension_conflicts_and_extra_fields() -> None:
    base = {
        "canonical_id": "1",
        "technology_role": "auxiliary",
        "strict_ai": False,
        "score": 2,
        "boundary_pair": "1_vs_2",
        "evidence": ["协助AI项目推进"],
        "reason": "只协调AI项目，不承担技术工作。",
        "confidence": "medium",
    }
    source = {"1": "岗位：质量工程师\n描述：协助AI项目推进\n标签：质量"}
    with pytest.raises(ValueError, match="conflicts"):
        validate_batch(json.dumps({"items": [base]}, ensure_ascii=False), source)
    with pytest.raises(ValueError):
        validate_batch(json.dumps({"items": [{**base, "score": 1, "unexpected": "leak"}]}, ensure_ascii=False), source)


def test_v2_1_prompt_encodes_review_discovered_boundaries_without_copying_ads() -> None:
    rubric = yaml.safe_load(Path("config/ai_rubric_v2_1.yaml").read_text(encoding="utf-8"))
    prompt = llm._system_prompt(rubric, thinking=False)
    for phrase in [
        "先判 technology_role",
        "物理、电气、机械、工艺、模拟或半导体技术本身不自动等于数字技术",
        "仅协助 AI 项目",
        "业务部门参与系统需求、测试或维护",
        "普通硬件产品测试",
    ]:
        assert phrase in prompt
    assert "联合质量工程师(JQE)" not in prompt
    assert llm.PROMPT_VERSION == "2.1.0"
    assert llm.SCHEMA_VERSION == "2.1.0"


def test_generic_title_cannot_silently_replace_supporting_evidence() -> None:
    payload = json.dumps({"items": [{"canonical_id": "1", "score": 1, "evidence": ["销售"], "reason": "需使用办公软件", "confidence": "medium"}]}, ensure_ascii=False)
    with pytest.raises(ValueError, match="support"):
        validate_batch(payload, {"1": "岗位：销售\n描述：熟练使用办公软件\n标签："})


def test_up_to_five_exact_supporting_phrases_are_allowed() -> None:
    evidence = ["网络运行", "ERP系统", "数据备份", "系统维护"]
    source = "岗位：网络管理员\n描述：负责网络运行、ERP系统、数据备份和系统维护\n标签：IT"
    payload = json.dumps({"items": [{"canonical_id": "1", "score": 2, "evidence": evidence, "reason": "系统运维是核心而非辅助使用", "confidence": "high"}]}, ensure_ascii=False)
    assert validate_batch(payload, {"1": source})[0].evidence == evidence


def test_excess_valid_evidence_is_deterministically_capped_at_five() -> None:
    evidence = [f"证据{i}" for i in range(7)]
    source = "岗位：系统实施\n描述：" + "、".join(evidence) + "\n标签：IT"
    payload = json.dumps(
        {
            "items": [
                {
                    "canonical_id": "1",
                    "score": 2,
                    "evidence": evidence,
                    "reason": "系统实施是核心职责，不是辅助工具使用",
                    "confidence": "high",
                }
            ]
        },
        ensure_ascii=False,
    )
    assert validate_batch(payload, {"1": source})[0].evidence == evidence[:5]


def test_single_evidence_positive_label_cannot_claim_high_confidence() -> None:
    payload = json.dumps(
        {
            "items": [
                {
                    "canonical_id": "1",
                    "score": 1,
                    "evidence": ["熟练使用财务软件"],
                    "reason": "软件是财务工作的辅助工具，不是技术核心产出",
                    "confidence": "high",
                }
            ]
        },
        ensure_ascii=False,
    )
    label = validate_batch(payload, {"1": "岗位：会计\n描述：熟练使用财务软件\n标签：财务"})[0]
    assert label.confidence == "medium"


def test_punctuation_normalized_evidence_is_repaired_to_exact_source_span() -> None:
    source = (
        "岗位：天猫运营\n"
        "描述：熟悉淘宝店铺运营广告投放的形式与方法,"
        "对淘宝钻石展位(CPM),直通车(CPC),淘宝各项活动平台,如聚划算,淘金币,相关类目主题活动等熟悉\n"
        "标签：电商"
    )
    model_evidence = (
        "熟悉淘宝店铺运营广告投放的形式与方法，"
        "对淘宝钻石展位（CPM），直通车（CPC），淘宝各项活动平台，"
        "如聚划算，淘金币，相关类目主题活动等熟悉"
    )
    payload = json.dumps(
        {
            "items": [
                {
                    "canonical_id": "1",
                    "score": 1,
                    "evidence": [model_evidence],
                    "reason": "广告投放工具是运营辅助能力，而非技术开发核心产出",
                    "confidence": "high",
                }
            ]
        },
        ensure_ascii=False,
    )
    repaired = validate_batch(payload, {"1": source})[0].evidence[0]
    assert repaired in source
    assert repaired == source.split("描述：", 1)[1].split("\n标签：", 1)[0]


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


def test_offline_replay_compacts_cache_to_current_valid_records(tmp_path: Path) -> None:
    ads = pd.DataFrame([{"canonical_id": "1", "岗位": "数据工程师", "岗位描述": "开发数据平台", "岗位标签": "SQL"}])
    text = "岗位：数据工程师\n描述：开发数据平台\n标签：SQL"
    rubric = tmp_path / "rubric.yaml"
    rubric.write_text(Path("config/ai_rubric.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    fingerprint = llm.prompt_fingerprint(yaml.safe_load(rubric.read_text(encoding="utf-8")), stage="primary", thinking=False)
    valid = {
        "canonical_id": "1", "score": 2, "evidence": "开发数据平台", "reason": "数据平台开发是核心职责，不是辅助使用", "confidence": "high",
        "model": llm.MODEL, "prompt_version": llm.PROMPT_VERSION, "schema_version": llm.SCHEMA_VERSION,
        "prompt_fingerprint": fingerprint, "stage": "primary", "thinking": False,
        "content_hash": content_hash(text, fingerprint), "label_status": "llm_primary",
    }
    stale = {**valid, "canonical_id": "obsolete", "content_hash": "stale"}
    cache = tmp_path / "cache.jsonl"
    cache.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in [stale, valid]) + "\n", encoding="utf-8")

    result = label_with_deepseek(ads, cache, rubric, allow_network=False)

    assert result.loc[0, "confidence"] == "medium"
    compacted = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines()]
    assert [row["canonical_id"] for row in compacted] == ["1"]
    assert compacted[0]["confidence"] == "medium"


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
