from ra_task.cli import build_parser


def test_offline_flag_is_available() -> None:
    args = build_parser().parse_args(["run", "--offline"])
    assert args.command == "run"
    assert args.offline is True


def test_human_evaluation_command_has_locked_default_seed() -> None:
    args = build_parser().parse_args(["prepare-human-eval"])
    assert args.command == "prepare-human-eval"
    assert args.seed == 20260629
