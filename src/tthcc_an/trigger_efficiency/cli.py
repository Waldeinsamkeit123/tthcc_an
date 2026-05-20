from __future__ import annotations

import argparse
from pathlib import Path

from tthcc_an.trigger_efficiency.config import load_trigger_efficiency_config
from tthcc_an.trigger_efficiency.workflow import run_trigger_efficiency


REPO_ROOT = Path(__file__).resolve().parents[3]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Signal-only AK8/PFHT trigger-efficiency study for ttHcc/ttHbb Pepper ntuples."
    )
    parser.add_argument("--config", required=True, help="Path to trigger-efficiency JSON config.")
    parser.add_argument("--outdir", default=None, help="Optional output directory override.")
    parser.add_argument(
        "--max-files-per-sample",
        type=int,
        default=None,
        help="Optional file limit for smoke tests.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = load_trigger_efficiency_config(
        args.config,
        REPO_ROOT,
        outdir=args.outdir,
        max_files_per_sample=args.max_files_per_sample,
    )
    summary = run_trigger_efficiency(config)
    print("Trigger-efficiency study finished.")
    print(f"Output directory: {summary['outdir']}")
    print(f"Table: {summary['tables'][0]}")
    print(f"Plots: {len(summary['plots'])}")

