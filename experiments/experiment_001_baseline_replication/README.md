# Experiment 001: Baseline Replication of the Basic vs Nuanced Steering Reliability Gap

## Hypothesis
On Qwen2.5-1.5B-Instruct, difference-in-means emotion steering vectors achieve a measurably higher Steering Success Rate (SSR) for basic categories (joy, sadness, anger, fear, surprise, disgust) than for nuanced categories (trust, anticipation), consistent with the reliability gap reported in the 2025 emotion-steering literature.

## Motivation
ASB's headline hypothesis (category-differentiated decay under fine-tuning) presupposes that the pre-fine-tuning reliability gap is reproducible in our setup. If the field's own baseline cannot be reproduced here, the benchmark design must be revisited before any original experiment runs. This experiment is the foundation stone and a deliberate de-risking step.

## Expected Outcome
- Primary: mean SSR(basic) > mean SSR(nuanced) with a non-trivial effect size.
- Acceptable alternative: gap absent but overall SSR high, which would itself be a reportable divergence from prior work and would trigger a taxonomy-split review before Experiment 002.

## Baselines
- Unsteered model outputs on identical prompts (floor).
- Prompt-based emotion instruction ("write this in a joyful tone") as the non-mechanistic comparator.

## Variables
- Independent: emotion category (8 levels), steering coefficient (swept), injection layer (swept once, then fixed).
- Dependent: SSR, Semantic Coherence Retention (SCR).
- Controlled: model (Qwen2.5-1.5B-Instruct, 4-bit), prompt set, seed set, generation parameters.

## Metrics
- SSR: fraction of generations judged as expressing the target category (judge protocol in `docs/judge_protocol.md`, to be written before first run).
- SCR: judged coherence of steered output relative to unsteered output on the same prompt.

## Statistical Validation
- Per-category SSR with 95% bootstrap confidence intervals over prompts and seeds.
- Basic vs nuanced comparison: Mann-Whitney U on per-prompt SSR, effect size reported as rank-biserial correlation.
- Minimum 3 seeds per condition.

## Threats to Validity
- Judge reliability: single-judge scoring may bias SSR; mitigate with position-randomized prompts and a fixed rubric, and report judge model and version.
- Layer/coefficient selection: tuning these on the same prompts used for evaluation would leak; use a held-out prompt split for selection.
- Category imbalance in extraction data: equal per-category extraction sets enforced in config.
- Small model idiosyncrasy: results here inform go/no-go only; cross-model claims wait for the full grid.

## Provenance
Each run writes `runs/<timestamp>/` containing: resolved config, seeds, environment freeze, git commit hash, raw generations, judge outputs, and computed metrics. Figures are generated from saved logs only.

## Single-Command Reproduction
```
python scripts/run_experiment.py --experiment 001
```
(Entry point to be implemented; this file is the pre-registration.)
