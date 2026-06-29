import json
from pathlib import Path

import pandas as pd

import ra_task.prompt_evaluation as pe
from ra_task.prompt_evaluation import evaluate_predictions, prepare_development_reference, stability_metrics


def _completed_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "review_id": "DEV-A",
                "岗位": "质量工程师",
                "岗位描述": "协助AI项目推进和跨部门沟通",
                "岗位标签": "质量",
                "human_score": "1",
                "technology_role": "auxiliary",
                "strict_ai": "false",
                "human_confidence": "high",
                "human_evidence": "原文明示‘协助AI项目推进’，但没有模型或系统技术职责，因此为1分。",
                "human_note": "",
            },
            {
                "review_id": "DEV-B",
                "岗位": "系统工程师",
                "岗位描述": "开发数据平台并维护数据库",
                "岗位标签": "数据",
                "human_score": "2",
                "technology_role": "core",
                "strict_ai": "false",
                "human_confidence": "high",
                "human_evidence": "开发数据平台并维护数据库",
                "human_note": "数据系统是主要产出",
            },
        ]
    )


def test_prepare_development_reference_normalizes_and_joins_private_mapping(tmp_path: Path) -> None:
    review = tmp_path / "review.xlsx"
    _completed_rows().to_excel(review, sheet_name="待审核", index=False)
    mapping = tmp_path / "mapping.csv"
    pd.DataFrame(
        [
            {"review_id": "DEV-A", "canonical_id": "10", "split": "development"},
            {"review_id": "DEV-B", "canonical_id": "20", "split": "development"},
        ]
    ).to_csv(mapping, index=False)
    output = tmp_path / "reference.csv"

    ledger = prepare_development_reference(review, mapping, output)

    assert output.exists()
    assert ledger["canonical_id"].tolist() == ["10", "20"]
    assert ledger.loc[0, "human_evidence"] == "协助AI项目推进"
    assert "因此为1分" in ledger.loc[0, "human_note"]


def test_evaluate_predictions_records_threshold_error_direction_and_metrics() -> None:
    reference = pd.DataFrame(
        {
            "canonical_id": ["10", "20", "30"],
            "human_score": [1, 2, 3],
            "technology_role": ["auxiliary", "core", "core"],
            "strict_ai": ["False", "false", "True"],
        }
    )
    predictions = pd.DataFrame(
        {
            "canonical_id": ["10", "20", "30"],
            "score": [2, 1, 3],
            "model_score": [0, 1, 3],
            "label_status": ["llm_adjudicated", "llm_primary", "llm_primary"],
            "technology_role": ["core", "auxiliary", "core"],
            "strict_ai": ["False", "false", "True"],
            "evidence": ["AI项目", "数据平台", "机器学习"],
            "reason": ["误判核心", "误判辅助", "正确"],
            "confidence": ["medium", "medium", "high"],
        }
    )

    comparison, report = evaluate_predictions(reference, predictions)

    assert comparison["error_type"].tolist() == ["threshold_false_positive", "threshold_false_negative", "match"]
    assert report["metrics"]["exact_agreement"] == 1 / 3
    assert report["metrics"]["binary_agreement_score_ge_2"] == 1 / 3
    assert report["error_counts"]["threshold_false_positive"] == 1
    assert report["schema_validity_rate"] == 1.0


def test_stability_metrics_require_all_three_frozen_runs_to_agree_per_item() -> None:
    runs = [
        pd.DataFrame({"canonical_id": ["10", "20"], "score": [1, 2]}),
        pd.DataFrame({"canonical_id": ["10", "20"], "score": [1, 2]}),
        pd.DataFrame({"canonical_id": ["10", "20"], "score": [1, 1]}),
    ]

    metrics = stability_metrics(runs)

    assert metrics["trials"] == 3
    assert metrics["exact_score_all_three"] == 0.5
    assert metrics["main_threshold_all_three"] == 0.5
    assert metrics["passes_main_threshold_stability"] is False


def test_development_and_frozen_stability_artifacts_build_offline(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    eval_dir = tmp_path / "eval"
    private = eval_dir / "private"
    private.mkdir(parents=True)
    _completed_rows().to_excel(eval_dir / "blind_development.xlsx", sheet_name="待审核", index=False)
    pd.DataFrame(
        [
            {"review_id": "DEV-A", "canonical_id": "10", "split": "development"},
            {"review_id": "DEV-B", "canonical_id": "20", "split": "development"},
        ]
    ).to_csv(private / "review_id_map.csv", index=False)
    data_dir = tmp_path / "data/processed"
    data_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"canonical_id": "10", "岗位": "质量工程师", "岗位描述": "协助AI项目推进和跨部门沟通", "岗位标签": "质量"},
            {"canonical_id": "20", "岗位": "系统工程师", "岗位描述": "开发数据平台并维护数据库", "岗位标签": "数据"},
        ]
    ).to_csv(data_dir / "cleaned_ads.csv", index=False)
    rubric = tmp_path / "rubric.yaml"
    rubric.write_text("version: 2.1.0\nscores: {}\n", encoding="utf-8")
    predictions = pd.DataFrame(
        {
            "canonical_id": ["10", "20"],
            "technology_role": ["auxiliary", "core"],
            "strict_ai": [False, False],
            "score": [1, 2],
            "model_score": [1, 2],
            "boundary_pair": ["1_vs_2", "1_vs_2"],
            "evidence": ["协助AI项目推进", "开发数据平台并维护数据库"],
            "reason": ["协调为辅助", "数据平台是核心"],
            "confidence": ["high", "high"],
            "request_id": ["req-a", "req-b"],
            "input_tokens": [100, 100],
            "output_tokens": [20, 20],
        }
    )
    monkeypatch.setattr(pe, "label_with_deepseek", lambda *args, **kwargs: predictions.copy())

    development = pe.run_development_round(
        3, offline=True, eval_dir=eval_dir, cache_dir=tmp_path / "cache", rubric_path=rubric
    )
    trial_2 = pe.run_stability_trial(
        2, offline=True, eval_dir=eval_dir, cache_dir=tmp_path / "cache", rubric_path=rubric
    )
    trial_3 = pe.run_stability_trial(
        3, offline=True, eval_dir=eval_dir, cache_dir=tmp_path / "cache", rubric_path=rubric
    )

    assert development["passes_all_release_targets"] is True
    assert trial_2["schema_validity_rate"] == 1.0
    assert trial_3["schema_validity_rate"] == 1.0
    summary = json.loads((eval_dir / "frozen_stability_summary.json").read_text(encoding="utf-8"))
    assert summary["main_threshold_all_three"] == 1.0
    assert summary["schema_validity_all_trials"] is True
