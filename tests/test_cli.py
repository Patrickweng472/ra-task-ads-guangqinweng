from ra_task.cli import build_parser


def test_offline_flag_is_available() -> None:
    args = build_parser().parse_args(["run", "--offline"])
    assert args.command == "run"
    assert args.offline is True

