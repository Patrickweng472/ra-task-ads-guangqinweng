from pathlib import Path

from ra_task.pipeline import verify_outputs


def test_committed_outputs_verify_when_present() -> None:
    if Path("outputs/ai_scores.csv").exists():
        result = verify_outputs(Path("outputs"))
        assert result["status"] == "PASS"

