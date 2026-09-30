"""Diagnostics for a saved Experiment 001 extraction run.

Usage: python scripts/analyze_extraction.py [run_dir]

Reads raw_activations.pt from a run folder (default: the latest run under
experiments/experiment_001_baseline_replication/runs/) and reports, per
category, two things that say whether the contrastive pairs are well written:

1. Norm against pair count. The difference-in-means norm is recomputed from
   subsamples of 3, 5, 10, 15 and 20 pairs. The norm of a difference of sample
   means carries a noise term that shrinks roughly as 1/sqrt(n), so it falls
   as pairs are added whatever the pair quality. What matters is whether the
   curve is flattening by the largest count, which says the pair count is
   sufficient.

2. Split-half direction agreement, against the category's own flip-null. The
   pairs are split in half at random, a difference-in-means vector is built
   from each half, and the cosine between the halves is reported. This asks
   whether the pairs agree on a direction, which is what actually steers. A
   category can show a healthy norm while its pairs disagree, if one or two
   extreme sentences dominate; the cosine catches that and the norm does not.

   The same cosine is then recomputed with the emotion and neutral labels
   randomly flipped within each pair, which destroys the emotion signal and
   leaves the noise floor for this category's own data. The verdict is whether
   the real cosine sits clearly above that null, reported in standard
   deviations of the null. There is no fixed threshold to clear: what counts
   as a high cosine depends on how widely the prompts vary in topic, which is
   category-specific and set by how the pairs were written.

What this does not say: nothing here measures steering effectiveness. A
category clearing its null means its vector reflects a real, consistent
difference between the poles, not that the vector steers generation. That is
SSR's job.

No GPU, no model reload, no plotting. Reads the run folder only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_001_RUNS = ROOT / "experiments" / "experiment_001_baseline_replication" / "runs"

DEFAULT_COUNTS = (3, 5, 10, 15, 20)
DEFAULT_SUBSAMPLES = 25
DEFAULT_SPLITS = 25
DEFAULT_SEED = 0


def latest_run(runs_dir: Path = EXPERIMENT_001_RUNS) -> Path:
    """The most recent run folder that actually holds raw activations."""
    if not runs_dir.exists():
        raise FileNotFoundError(f"no runs directory: {runs_dir}")
    candidates = sorted(
        (d for d in runs_dir.iterdir() if (d / "raw_activations.pt").exists()),
        key=lambda d: d.name,
    )
    if not candidates:
        raise FileNotFoundError(f"no run folder with raw_activations.pt under {runs_dir}")
    return candidates[-1]


def load_activations(run_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load raw per-prompt activations as float64 numpy arrays, shaped (pairs, hidden)."""
    path = run_dir / "raw_activations.pt"
    if not path.exists():
        raise FileNotFoundError(f"no raw activations in {run_dir}")
    raw = torch.load(path, map_location="cpu", weights_only=True)

    out: dict[str, dict[str, np.ndarray]] = {}
    for category, sides in raw.items():
        arrays = {}
        for side in ("emotion", "neutral"):
            if side not in sides:
                raise ValueError(f"{path}: category '{category}' has no '{side}' activations")
            arrays[side] = sides[side].to(torch.float64).numpy()
        if arrays["emotion"].shape != arrays["neutral"].shape:
            raise ValueError(
                f"{path}: category '{category}' has mismatched pole shapes "
                f"{arrays['emotion'].shape} vs {arrays['neutral'].shape}"
            )
        out[category] = arrays
    return out


def diff_in_means(emotion: np.ndarray, neutral: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """The difference-in-means vector over the pairs selected by `idx`.

    Indices select pairs, so the emotion and neutral poles stay paired.
    """
    return emotion[idx].mean(axis=0) - neutral[idx].mean(axis=0)


def norm_vs_count(
    emotion: np.ndarray,
    neutral: np.ndarray,
    counts: tuple[int, ...],
    subsamples: int,
    rng: np.random.Generator,
) -> dict[int, tuple[float, float, bool]]:
    """Mean and spread of the vector norm at each pair count.

    Returns {count: (mean norm, standard deviation, exact)} where `exact` marks
    the count that uses every pair, so there is nothing to sample over. Counts
    above the number of available pairs are left out.
    """
    n_pairs = emotion.shape[0]
    results: dict[int, tuple[float, float, bool]] = {}
    for count in counts:
        if count > n_pairs:
            continue
        if count == n_pairs:
            norm = np.linalg.norm(diff_in_means(emotion, neutral, np.arange(n_pairs)))
            results[count] = (float(norm), 0.0, True)
            continue
        norms = [
            np.linalg.norm(
                diff_in_means(emotion, neutral, rng.choice(n_pairs, size=count, replace=False))
            )
            for _ in range(subsamples)
        ]
        results[count] = (float(np.mean(norms)), float(np.std(norms)), False)
    return results


def split_half_cosines(
    emotion: np.ndarray,
    neutral: np.ndarray,
    splits: int,
    rng: np.random.Generator,
    flip: bool = False,
) -> np.ndarray | None:
    """Cosines between vectors built from random halves, one per split.

    With flip=True, the emotion and neutral labels are swapped at random within
    each pair before each split. Every pair still contributes the same sentences
    and the same magnitude, only the sign of its contribution changes, so this
    keeps the noise and destroys the emotion signal. That gives the null this
    category's own data produces.

    Returns None when there are fewer than 4 pairs, since a half of one pair
    says nothing. An odd pair count puts the extra pair in the first half.
    """
    n_pairs = emotion.shape[0]
    if n_pairs < 4:
        return None

    differences = emotion - neutral  # one vector per pair
    half = n_pairs - n_pairs // 2  # first half takes the odd pair
    cosines = []
    for _ in range(splits):
        deltas = differences
        if flip:
            signs = rng.choice((-1.0, 1.0), size=n_pairs)
            deltas = differences * signs[:, None]
        order = rng.permutation(n_pairs)
        a = deltas[order[:half]].mean(axis=0)
        b = deltas[order[half:]].mean(axis=0)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            continue
        cosines.append(float(np.dot(a, b) / denom))
    if not cosines:
        return None
    return np.asarray(cosines)


def report(
    activations: dict[str, dict[str, np.ndarray]],
    counts: tuple[int, ...],
    subsamples: int,
    splits: int,
    seed: int,
) -> None:
    """Print the per-category table. One RNG, seeded once, for the whole report."""
    rng = np.random.default_rng(seed)

    header = f"{'category':<14}{'pairs':>6}  " + "".join(f"{f'n={c}':>10}" for c in counts)
    header += f"{'real cos':>12}{'flip-null':>16}{'null p95':>10}{'p':>8}"
    print(header)
    print("-" * len(header))

    skipped: list[str] = []
    for category, sides in activations.items():
        emotion, neutral = sides["emotion"], sides["neutral"]
        n_pairs = emotion.shape[0]
        norms = norm_vs_count(emotion, neutral, counts, subsamples, rng)
        real = split_half_cosines(emotion, neutral, splits, rng)
        null = split_half_cosines(emotion, neutral, splits, rng, flip=True)

        row = f"{category:<14}{n_pairs:>6}  "
        for count in counts:
            if count not in norms:
                row += f"{'-':>10}"
                continue
            mean, _std, exact = norms[count]
            row += f"{mean:>9.3f}{'*' if exact else ' '}"
        if real is None or null is None:
            row += f"{'-':>12}{'-':>16}{'-':>10}{'-':>8}"
            skipped.append(category)
        else:
            # Share of the null at or above the real cosine. The null can be
            # bimodal when one direction dominates, so a standard-deviation gap
            # would mislead; this share does not.
            p = float(np.mean(null >= real.mean()))
            row += f"{real.mean():>12.3f}"
            row += f"{null.mean():>10.3f} +-{null.std():>4.2f}"
            row += f"{np.quantile(null, 0.95):>10.3f}"
            row += f"{p:>8.2f}"
        print(row)

    print()
    print(f"seed {seed}, {subsamples} subsamples per count, {splits} random splits")
    print("* uses every pair, so there is no sampling spread at that count")
    print("- count exceeds the pairs available, or fewer than 4 pairs for the cosine")
    if skipped:
        print(f"split-half needs at least 4 pairs, skipped: {', '.join(skipped)}")
    print()
    print("Reading it: the norm falls as pairs are added whatever the pair quality, so")
    print("look for the curve flattening, not for a particular value. For the cosine,")
    print("compare each category against its own flip-null rather than a fixed number.")
    print("p is the share of null splits at or above the real cosine, so small p means")
    print("the pairs agree on a direction the shuffle cannot produce. A category sitting")
    print("on its null (p near 1, real cosine inside the null spread) has pairs that do")
    print("not agree. Clearing the null says the vector is real, not that it steers")
    print("generation, which is SSR's job and is not measured here.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", nargs="?", default=None,
                        help="run folder to analyse (default: latest under experiment 001)")
    parser.add_argument("--counts", type=int, nargs="+", default=list(DEFAULT_COUNTS),
                        help=f"pair counts for the norm curve (default: {' '.join(map(str, DEFAULT_COUNTS))})")
    parser.add_argument("--subsamples", type=int, default=DEFAULT_SUBSAMPLES,
                        help=f"random subsamples per count (default: {DEFAULT_SUBSAMPLES})")
    parser.add_argument("--splits", type=int, default=DEFAULT_SPLITS,
                        help=f"random splits for the cosine (default: {DEFAULT_SPLITS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"seed, printed with the table (default: {DEFAULT_SEED})")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run()
    print(f"run: {run_dir}")
    activations = load_activations(run_dir)
    report(activations, tuple(sorted(set(args.counts))), args.subsamples, args.splits, args.seed)


if __name__ == "__main__":
    main()
