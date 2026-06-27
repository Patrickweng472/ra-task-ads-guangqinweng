from pathlib import Path


def test_ci_runs_full_offline_pipeline_and_enforces_coverage() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "ra-task run --offline" in workflow
    assert "--cov-fail-under=80" in workflow
    assert "quarto-actions/setup" in workflow
