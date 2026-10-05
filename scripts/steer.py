"""Manual steering check: one prompt, one category, a few coefficients.

Usage: python scripts/steer.py --category joy --prompt "Describe your morning."
       python scripts/steer.py --category fear --prompt "..." --coefficients 0 0.5 1 --out steer.jsonl

Uses the latest Path B extraction run unless --run is given. Prints each
coefficient's output so the shift can be eyeballed. For manual testing only:
SSR and SCR are measured by evaluation.py, not here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # src layout, no install step needed

from asb.extraction import DEFAULT_SEED  # noqa: E402
from asb.steering import GenerationSettings, Steerer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--category", required=True, help="Plutchik category to steer toward")
    parser.add_argument("--prompt", required=True, help="user prompt to answer")
    parser.add_argument("--coefficients", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0],
                        help="fractions of the typical residual norm (default: 0 0.25 0.5 1)")
    parser.add_argument("--run", default=None,
                        help="extraction run folder (default: latest Path B run)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"seed for every coefficient (default: {DEFAULT_SEED})")
    parser.add_argument("--max-new-tokens", type=int, default=GenerationSettings.max_new_tokens,
                        help=f"default: {GenerationSettings.max_new_tokens}")
    parser.add_argument("--greedy", action="store_true", help="greedy decoding instead of sampling")
    parser.add_argument("--out", default=None, help="append one JSON record per generation here")
    args = parser.parse_args()

    settings = GenerationSettings(max_new_tokens=args.max_new_tokens, do_sample=not args.greedy)
    steerer = Steerer(Path(args.run) if args.run else None)
    print(f"run: {steerer.run_dir}")
    print(f"model: {steerer.model_id}, layer {steerer.layer}, "
          f"residual scale {steerer.residual_scale:.2f}, seed {args.seed}")

    for coefficient in args.coefficients:
        record = steerer.generate([args.prompt], args.category, coefficient,
                                  seed=args.seed, settings=settings)[0]
        print()
        print(f"--- {args.category} x {coefficient:g} (added norm {record['added_norm']:.2f}, "
              f"{record['steered_decode_steps']} steered steps)")
        print(record["output"])
        if args.out:
            with open(args.out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
