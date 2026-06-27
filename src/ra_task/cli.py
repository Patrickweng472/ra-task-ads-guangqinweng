from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline, verify_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ra-task", description="招聘广告数据分析流水线")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="运行完整分析")
    run.add_argument("--ads", type=Path, default=Path("data/raw/ra_task_ads.csv"))
    run.add_argument("--firms", type=Path, default=Path("data/raw/ra_task_firms.csv"))
    run.add_argument("--output-dir", type=Path, default=Path("outputs"))
    run.add_argument("--offline", action="store_true", help="禁止 API 调用并使用缓存/规则标签")
    run.add_argument("--seed", type=int, default=20260627)
    verify = sub.add_parser("verify", help="验证已生成的交付物")
    verify.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        run_pipeline(args.ads, args.firms, args.output_dir, offline=args.offline, seed=args.seed)
    else:
        verify_outputs(args.output_dir, require_archive=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
