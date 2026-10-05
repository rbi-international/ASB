"""Single-command experiment entry point.

Usage: python scripts/run_experiment.py --experiment 001
Writes full provenance to experiments/experiment_XXX/runs/<timestamp>/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # src layout, no install step needed

from asb.extraction import DEFAULT_SEED, run_extraction  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", required=True, choices=["001"],
                        help="experiment number; 002 and 003 land with their own code")
    parser.add_argument("--model", default=None,
                        help="model id from configs/models.yaml (default: Qwen2.5-1.5B-Instruct)")
    # No --layer option: the layer is fixed at half the decoder depth
    # (Experiment 001 README, Amendment 1), so it is not a run setting.
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"random seed, recorded in provenance (default: {DEFAULT_SEED})")
    args = parser.parse_args()

    if args.experiment == "001":
        run_extraction(model_id=args.model, seed=args.seed)


if __name__ == "__main__":
    main()
