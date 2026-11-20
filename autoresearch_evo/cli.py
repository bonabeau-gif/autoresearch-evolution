"""Command-line entry point for the autonomous evolutionary loop."""
import argparse
import datetime as dt
import os
from pathlib import Path

from .core import Evolution, initialize
from .dashboard import build_dashboard


def main(argv=None):
    parser = argparse.ArgumentParser(prog="evo-research", description="Evolve a population of train.py commits")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Create an experiment branch and initialize TSV state")
    init.add_argument("--tag", default=dt.datetime.now().strftime("%b%d").lower())
    init.add_argument("--config", type=Path, default=Path("config/evolution.json"))
    init.add_argument("--allow-missing-data", action="store_true", help="Initialization only: skip data validation")
    init.add_argument("--allow-no-gpu", action="store_true", help="Initialization only: skip CUDA check")
    run = sub.add_parser("run", help="Run or resume autonomous evolution")
    run.add_argument("--provider", choices=("openai", "mock"), default="openai")
    run.add_argument("--model", default=os.getenv("OPENAI_MODEL", ""))
    run.add_argument("--max-generations", type=int, default=0, help="0 = until convergence/failure")
    run.add_argument("--hours", type=float, default=0, help="0 = no wall-time limit")
    run.add_argument("--cuda-device", default="0", help="CUDA_VISIBLE_DEVICES value")
    run.add_argument("--train-command", default="uv run train.py", help="Testing override")
    run.add_argument("--smoke-command", default="uv run python -c 'import train'", help="Testing override")
    sub.add_parser("dashboard", help="Rebuild standalone interactive HTML progress view")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "init":
        config = args.config if args.config.is_absolute() else root / args.config
        initialize(root, args.tag, config, args.allow_missing_data, args.allow_no_gpu)
    elif args.command == "run":
        experiment = Evolution(root, args.provider, args.model, args.train_command, args.smoke_command, args.cuda_device)
        reason = experiment.run(max_generations=args.max_generations, max_hours=args.hours)
        print(f"Evolution stopped: {reason}")
    elif args.command == "dashboard":
        print(build_dashboard(root))


if __name__ == "__main__":
    main()
