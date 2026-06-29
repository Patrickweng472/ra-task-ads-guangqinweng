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
