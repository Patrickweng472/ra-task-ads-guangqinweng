from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from .human_evaluation import classification_metrics, normalize_completed_review, validate_human_labels
from .llm_labeling import _system_prompt, derive_score, label_with_deepseek, prompt_fingerprint


DEFAULT_EVAL_DIR = Path("artifacts/evals/llm_v2_1")
DEFAULT_CACHE_DIR = Path("artifacts/llm/v2_1")
DEFAULT_RUBRIC_PATH = Path("config/ai_rubric_v2_1.yaml")
DEEPSEEK_PRICING_SOURCE = "https://api-docs.deepseek.com/zh-cn/quick_start/pricing"
DEEPSEEK_V4_PRO_CACHE_MISS_INPUT_CNY_PER_MILLION = 3.0
DEEPSEEK_V4_PRO_OUTPUT_CNY_PER_MILLION = 6.0


def prepare_development_reference(
    review_xlsx: Path,
    mapping_csv: Path,
    output_csv: Path,
) -> pd.DataFrame:
    """Build a machine-valid development reference without changing the review workbook."""
    review = pd.read_excel(review_xlsx, sheet_name="待审核", dtype=str).fillna("")
    normalized = normalize_completed_review(review)
    validated = validate_human_labels(normalized)
    mapping = pd.read_csv(mapping_csv, dtype=str, keep_default_na=False)
    required_mapping = {"review_id", "canonical_id", "split"}
    missing = required_mapping.difference(mapping.columns)
    if missing:
        raise ValueError(f"review mapping missing columns: {sorted(missing)}")
    development = mapping.loc[mapping["split"].eq("development"), ["review_id", "canonical_id", "split"]]
    ledger = validated.merge(development, on="review_id", validate="one_to_one")
    if len(ledger) != len(validated) or ledger["canonical_id"].duplicated().any():
        raise ValueError("development review and private mapping do not form a complete one-to-one ledger")
    ledger["canonical_id"] = ledger["canonical_id"].astype(str)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return ledger


def _prediction_schema_is_valid(row: pd.Series) -> bool:
    try:
        score = int(row["score"])
        derived = derive_score(str(row["technology_role"]), bool(row["strict_ai"]))  # type: ignore[arg-type]
        evidence = str(row.get("evidence", "")).strip()
        return score == derived and (score == 0 or bool(evidence))
    except (KeyError, TypeError, ValueError):
        return False


def _error_type(human_score: int, model_score: int) -> str:
    if human_score == model_score:
        return "match"
    if human_score < 2 <= model_score:
        return "threshold_false_positive"
    if model_score < 2 <= human_score:
        return "threshold_false_negative"
    if human_score == 3 or model_score == 3:
        return "strict_ai_false_positive" if model_score == 3 else "strict_ai_false_negative"
    return "within_threshold_over" if model_score > human_score else "within_threshold_under"


def evaluate_predictions(reference: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required_reference = {"canonical_id", "human_score", "technology_role", "strict_ai"}
    required_predictions = {
        "canonical_id", "score", "technology_role", "strict_ai", "evidence", "reason", "confidence"
    }
    if missing := required_reference.difference(reference.columns):
        raise ValueError(f"reference missing columns: {sorted(missing)}")
    if missing := required_predictions.difference(predictions.columns):
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    if reference["canonical_id"].duplicated().any() or predictions["canonical_id"].duplicated().any():
        raise ValueError("reference and predictions must have unique canonical_id values")
    reference_ids = set(reference["canonical_id"].astype(str))
    prediction_ids = set(predictions["canonical_id"].astype(str))
    if reference_ids != prediction_ids:
        raise ValueError("prediction IDs do not exactly match the human reference")

    human = reference.copy()
    model = predictions.copy()
    human["canonical_id"] = human["canonical_id"].astype(str)
    model["canonical_id"] = model["canonical_id"].astype(str)
    schema_valid = model.apply(_prediction_schema_is_valid, axis=1)
    if not schema_valid.all():
        bad_ids = model.loc[~schema_valid, "canonical_id"].tolist()
        raise ValueError(f"invalid v2.1 prediction schema for IDs: {bad_ids}")

    if "model_score" in model.columns:
        if not pd.to_numeric(model["model_score"]).eq(pd.to_numeric(model["score"])).all():
            raise ValueError("preserved model_score conflicts with the validated model score")
        model = model.drop(columns=["model_score"])

    model = model.rename(
        columns={
            "score": "model_score",
            "technology_role": "model_technology_role",
            "strict_ai": "model_strict_ai",
            "evidence": "model_evidence",
            "reason": "model_reason",
            "confidence": "model_confidence",
        }
    )
    comparison = human.merge(model, on="canonical_id", validate="one_to_one")
    comparison["human_score"] = comparison["human_score"].astype(int)
    comparison["model_score"] = comparison["model_score"].astype(int)
    comparison["score_delta_model_minus_human"] = comparison["model_score"] - comparison["human_score"]
    comparison["error_type"] = comparison.apply(
        lambda row: _error_type(int(row["human_score"]), int(row["model_score"])), axis=1
    )
    metrics = classification_metrics(comparison["human_score"], comparison["model_score"])
    error_counts = comparison["error_type"].value_counts().to_dict()
    usage = {"input_tokens": None, "output_tokens": None}
    if "request_id" in comparison.columns:
        requests = comparison.drop_duplicates("request_id")
        usage = {
            "input_tokens": int(pd.to_numeric(requests.get("input_tokens"), errors="coerce").fillna(0).sum()),
            "output_tokens": int(pd.to_numeric(requests.get("output_tokens"), errors="coerce").fillna(0).sum()),
        }
    estimated_cost_cny = None
    if usage["input_tokens"] is not None and usage["output_tokens"] is not None:
        estimated_cost_cny = round(
            usage["input_tokens"] * DEEPSEEK_V4_PRO_CACHE_MISS_INPUT_CNY_PER_MILLION / 1_000_000
            + usage["output_tokens"] * DEEPSEEK_V4_PRO_OUTPUT_CNY_PER_MILLION / 1_000_000,
            6,
        )
    report = {
        "metrics": metrics,
        "error_counts": {str(key): int(value) for key, value in error_counts.items()},
        "schema_validity_rate": float(schema_valid.mean()),
        "token_usage": usage,
        "estimated_max_cost_cny_without_cache_discount": estimated_cost_cny,
        "pricing_source": DEEPSEEK_PRICING_SOURCE,
        "release_targets": {
            "exact_agreement": metrics["exact_agreement"] >= 0.85,
            "binary_agreement_score_ge_2": metrics["binary_agreement_score_ge_2"] >= 0.95,
            "quadratic_weighted_kappa": metrics["quadratic_weighted_kappa"] >= 0.85,
            "schema_validity": bool(schema_valid.all()),
        },
    }
    report["passes_all_release_targets"] = all(report["release_targets"].values())
    return comparison, report


def stability_metrics(runs: list[pd.DataFrame]) -> dict:
    if len(runs) != 3:
        raise ValueError("frozen stability requires exactly three trials")
    normalized = []
    expected_ids: set[str] | None = None
    for trial, frame in enumerate(runs, start=1):
        if not {"canonical_id", "score"}.issubset(frame.columns) or frame["canonical_id"].duplicated().any():
            raise ValueError(f"invalid stability trial {trial}")
        current = frame[["canonical_id", "score"]].copy()
        current["canonical_id"] = current["canonical_id"].astype(str)
        current["score"] = current["score"].astype(int)
        ids = set(current["canonical_id"])
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError("stability trial IDs do not match")
        normalized.append(current.rename(columns={"score": f"score_trial_{trial}"}))
    comparison = normalized[0]
    for frame in normalized[1:]:
        comparison = comparison.merge(frame, on="canonical_id", validate="one_to_one")
    score_columns = [f"score_trial_{trial}" for trial in range(1, 4)]
    exact_stable = comparison[score_columns].nunique(axis=1).eq(1)
    thresholds = comparison[score_columns].ge(2).astype(int)
    threshold_stable = thresholds.nunique(axis=1).eq(1)
    result = {
        "trials": 3,
        "sample_size": len(comparison),
        "exact_score_all_three": float(exact_stable.mean()),
        "main_threshold_all_three": float(threshold_stable.mean()),
        "exact_score_unstable_items": int((~exact_stable).sum()),
        "main_threshold_unstable_items": int((~threshold_stable).sum()),
        "passes_main_threshold_stability": bool(threshold_stable.mean() >= 0.95),
    }
    return result


def run_development_round(
    round_number: int,
    *,
    offline: bool = False,
    eval_dir: Path = DEFAULT_EVAL_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    rubric_path: Path = DEFAULT_RUBRIC_PATH,
) -> dict:
    if round_number not in {1, 2, 3}:
        raise ValueError("development round must be 1, 2, or 3")
    reference_path = eval_dir / "development_human_reference.csv"
    reference = prepare_development_reference(
        eval_dir / "blind_development.xlsx",
        eval_dir / "private/review_id_map.csv",
        reference_path,
    )
    ads = pd.read_csv("data/processed/cleaned_ads.csv", dtype=str, keep_default_na=False)
    ads["canonical_id"] = ads["canonical_id"].astype(str)
    requested = ads.loc[ads["canonical_id"].isin(reference["canonical_id"])].copy()
    if len(requested) != len(reference):
        raise ValueError("cleaned ads do not contain the complete development reference")

    round_dir = eval_dir / f"development_round_{round_number}"
    round_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"development_round_{round_number}.jsonl"
    predictions = label_with_deepseek(
        requested,
        cache_path,
        rubric_path,
        stage="development",
        label_status="llm_v2_1_development",
        allow_network=not offline,
    )
    comparison, report = evaluate_predictions(reference, predictions)
    predictions.to_csv(round_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(round_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    (round_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rubric = yaml.safe_load(rubric_path.read_text(encoding="utf-8"))
    prompt = _system_prompt(rubric, thinking=False, stage="development")
    (round_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    metadata = {
        "round": round_number,
        "prompt_fingerprint": prompt_fingerprint(rubric, stage="development", thinking=False),
        "cache_path": str(cache_path),
        "offline": offline,
        "sample_size": len(reference),
    }
    (round_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run_stability_trial(
    trial: int,
    *,
    offline: bool = False,
    eval_dir: Path = DEFAULT_EVAL_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    rubric_path: Path = DEFAULT_RUBRIC_PATH,
) -> dict:
    if trial not in {2, 3}:
        raise ValueError("stability trial must be 2 or 3; development round 3 is frozen trial 1")
    reference = pd.read_csv(eval_dir / "development_human_reference.csv", dtype=str, keep_default_na=False)
    ads = pd.read_csv("data/processed/cleaned_ads.csv", dtype=str, keep_default_na=False)
    ads["canonical_id"] = ads["canonical_id"].astype(str)
    requested = ads.loc[ads["canonical_id"].isin(reference["canonical_id"])].copy()
    output_dir = eval_dir / f"frozen_stability_trial_{trial}"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = label_with_deepseek(
        requested,
        cache_dir / f"frozen_stability_trial_{trial}.jsonl",
        rubric_path,
        stage="development",
        label_status="llm_v2_1_stability",
        allow_network=not offline,
    )
    comparison, report = evaluate_predictions(reference, predictions)
    predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    (output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rubric = yaml.safe_load(rubric_path.read_text(encoding="utf-8"))
    fingerprint = prompt_fingerprint(rubric, stage="development", thinking=False)
    metadata = {"trial": trial, "prompt_fingerprint": fingerprint, "offline": offline, "sample_size": len(reference)}
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    trial_paths = [
        eval_dir / "development_round_3/predictions.csv",
        eval_dir / "frozen_stability_trial_2/predictions.csv",
        eval_dir / "frozen_stability_trial_3/predictions.csv",
    ]
    metadata_paths = [
        eval_dir / "development_round_3/run_metadata.json",
        eval_dir / "frozen_stability_trial_2/run_metadata.json",
        eval_dir / "frozen_stability_trial_3/run_metadata.json",
    ]
    if all(path.exists() for path in trial_paths + metadata_paths):
        fingerprints = {
            json.loads(path.read_text(encoding="utf-8"))["prompt_fingerprint"] for path in metadata_paths
        }
        if len(fingerprints) != 1:
            raise ValueError("frozen stability trials used different prompt fingerprints")
        summary = stability_metrics(
            [pd.read_csv(path, dtype=str, keep_default_na=False) for path in trial_paths]
        )
        summary["prompt_fingerprint"] = fingerprints.pop()
        summary["schema_validity_all_trials"] = all(
            json.loads(path.with_name("metrics.json").read_text(encoding="utf-8"))["schema_validity_rate"] == 1.0
            for path in trial_paths
        )
        (eval_dir / "frozen_stability_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return report
