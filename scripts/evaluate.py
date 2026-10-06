"""Run or resume the registered Experiment 001 evaluation for one model.

Usage:
  python scripts/evaluate.py --run <vector run folder> --dry-run
      Stub judge on a small slice (joy and trust, 3 prompts per split, seed 0).
      Checks the whole pipeline in minutes, for free. Nothing is a result.
  python scripts/evaluate.py --run <vector run folder>
      Full registered evaluation with the stub judge. Free; nothing is a result.
  python scripts/evaluate.py --run <vector run folder> --paid --effort <level>
      Claude Fable 5.1 judge per docs/judge_protocol.md. Asks for a typed
      confirmation, with the call count and estimated cost, before every batch.
  Add --tables-only (with the same other flags) to rebuild tables from saved records.

Re-running the same command resumes an interrupted run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # src layout, no install step needed

from asb.evaluation import JUDGE_EFFORTS, run_evaluation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True,
                        help="Path B extraction run folder of the model to evaluate")
    parser.add_argument("--dry-run", action="store_true",
                        help="stub judge on a small slice, to test the pipeline")
    parser.add_argument("--paid", action="store_true",
                        help="use the Claude Fable 5.1 judge (costs money; asks first)")
    parser.add_argument("--effort", choices=JUDGE_EFFORTS, default=None,
                        help="judge effort for --paid, fixed by the protocol's pilot")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="split each batch into fixed chunks if memory requires (default: whole split)")
    parser.add_argument("--tables-only", action="store_true",
                        help="only rebuild tables from saved records")
    args = parser.parse_args()

    run_evaluation(Path(args.run), dry_run=args.dry_run, paid=args.paid, effort=args.effort,
                   chunk_size=args.chunk_size, tables_only=args.tables_only)


if __name__ == "__main__":
    main()
