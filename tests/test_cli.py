import ra_task.cli as cli
from ra_task.cli import build_parser


def test_offline_flag_is_available() -> None:
    args = build_parser().parse_args(["run", "--offline"])
    assert args.command == "run"
    assert args.offline is True


def test_human_evaluation_command_has_locked_default_seed() -> None:
    args = build_parser().parse_args(["prepare-human-eval"])
    assert args.command == "prepare-human-eval"
    assert args.seed == 20260629


def test_v2_1_development_evaluation_command_is_versioned_and_offline_capable() -> None:
    args = build_parser().parse_args(["evaluate-v2-1-development", "--round", "2", "--offline"])
    assert args.command == "evaluate-v2-1-development"
    assert args.round == 2
    assert args.offline is True


def test_v2_1_stability_command_accepts_only_followup_trials() -> None:
    args = build_parser().parse_args(["evaluate-v2-1-stability", "--trial", "2", "--offline"])
    assert args.command == "evaluate-v2-1-stability"
    assert args.trial == 2
    assert args.offline is True


def test_main_dispatches_all_commands(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "run_pipeline", lambda *args, **kwargs: calls.append(("run", kwargs)))
    monkeypatch.setattr(cli, "verify_outputs", lambda *args, **kwargs: calls.append(("verify", kwargs)))
    monkeypatch.setattr(cli, "prepare_human_evaluation", lambda *args, **kwargs: calls.append(("prepare", kwargs)))
    monkeypatch.setattr(cli, "run_development_round", lambda *args, **kwargs: calls.append(("development", kwargs)))
    monkeypatch.setattr(cli, "run_stability_trial", lambda *args, **kwargs: calls.append(("stability", kwargs)))

    assert cli.main(["run", "--offline"]) == 0
    assert cli.main(["verify"]) == 0
    assert cli.main(["prepare-human-eval"]) == 0
    assert cli.main(["evaluate-v2-1-development", "--round", "3", "--offline"]) == 0
    assert cli.main(["evaluate-v2-1-stability", "--trial", "2", "--offline"]) == 0

    assert [name for name, _ in calls] == ["run", "verify", "prepare", "development", "stability"]
