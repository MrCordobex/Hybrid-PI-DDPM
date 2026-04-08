from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from tfm_shells.sampling.guided import run_guided_sampling
from tfm_shells.training.train_architect import train_architect
from tfm_shells.training.train_engineer import train_engineer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TFM shell research CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    architect = subparsers.add_parser("architect", help="Train the architect DDPM")
    architect.add_argument("--config", default="configs/architect.yaml")

    engineer = subparsers.add_parser("engineer", help="Train the engineer surrogate")
    engineer.add_argument("--config", default="configs/engineer.yaml")

    sample = subparsers.add_parser("sample", help="Run physics-guided sampling")
    sample.add_argument("--config", default="configs/sample_guided.yaml")

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "architect":
        train_architect(Path(args.config))
        return
    if args.command == "engineer":
        train_engineer(Path(args.config))
        return
    if args.command == "sample":
        run_guided_sampling(Path(args.config))
        return
    raise ValueError(f"Unsupported command: {args.command}")
