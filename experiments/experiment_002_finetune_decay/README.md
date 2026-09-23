# Experiment 002: Steering Decay Under Neutral and Contradictory Fine-Tuning (Go/No-Go)

## Hypothesis
QLoRA fine-tuning degrades emotion-steering effectiveness on Qwen2.5-1.5B-Instruct more in the contradictory arm than in the neutral arm, and within each arm the decay rate differs between basic and nuanced Plutchik categories (nuanced decays faster). Following the Re-Align workshop paper, decay in the neutral arm is expected to be small.

## Motivation
This is ASB's first original data point and the project's go/no-go gate. Prior work (ICLR 2026 Re-Align Workshop) shows embedded behavioral steering degrades when training data contradicts the steered behavior, but tested only generic behaviors (refusal, brevity), not affective categories, and did not analyze category-differentiated decay. If category-differentiated decay exists, ASB has a headline finding; if decay is flat and uniform, the benchmark's central axis must be redesigned before further investment.

Decision (2026-09-22): the Re-Align workshop paper found that embedded steering persists when the fine-tuning data does not contradict the steered behavior, and degrades only when it does. A neutral-only design would therefore likely produce little or no decay and partly repeat that result. Experiment 002 runs two arms, neutral and contradictory, recorded as a deliberate amendment to the README "Locked v1.0 Scope". The neutral arm tests whether the Re-Align finding holds for affective steering; the contradictory arm is where per-category differences are expected to show.

## Arms
- Neutral arm: emotionally neutral instruction data, emotion-content-filtered.
- Contradictory arm: the neutral arm's dataset with 50% of examples replaced by affect-flattening examples (emotion-inviting prompts paired with deliberately affect-neutral responses). Both arms use the same total size, step budget, and hyperparameters, so the only difference is the replaced fraction. The 50% fraction is pre-registered here and logged in the run config before the run; rationale: 50% is chosen as strong pressure against steered affect that should plausibly keep the model functional. This is a reasoned choice, not a tested one; the sanity check that fine-tuning improves the target task in both arms (Statistical Validation) is the guard if it proves too aggressive. The construction procedure is also fixed and logged before the run.
- Per-Plutchik-opposite training arms are deferred to v2.0 because they would confound category with dataset; the affect-flattening design avoids this by holding fine-tuning pressure uniform across all eight categories.

## Expected Outcome
- Go: visibly different decay slopes between basic and nuanced categories across checkpoints in at least one arm, in either direction (nuanced-faster is hypothesized, basic-faster would be equally reportable).
- No-Go: statistically indistinguishable, near-zero slopes across all categories in both arms after the full fine-tuning run.
- A near-zero neutral arm alongside a decaying contradictory arm is a reportable result (it extends the Re-Align finding to affective steering), not a No-Go.

## Baselines
- Checkpoint 0 (pre-fine-tuning) SSR per category from Experiment 001, shared by both arms.
- Steering-free generation quality at each checkpoint in each arm (to separate general capability drift from steering-specific decay).

## Variables
- Independent: data arm (neutral, contradictory), fine-tuning steps (checkpoints at 0%, 33%, 66%, 100% of a fixed budget), emotion category.
- Dependent: SSR per category per checkpoint per arm, decay slope per category per arm, SCR per checkpoint per arm.
- Controlled: model, steering vectors (frozen from Experiment 001, re-applied at each checkpoint), extraction data, prompt set, seeds, QLoRA hyperparameters, total dataset size and step budget (identical across arms).

## Metrics
- SSR per category per checkpoint, per arm.
- Decay slope: least-squares slope of SSR over checkpoints, per category, per arm.
- SCR per checkpoint, per arm.

## Statistical Validation
- Slope confidence intervals via bootstrap over prompts and seeds.
- Basic vs nuanced slope comparison: Mann-Whitney U on per-category slopes, effect size reported, per arm.
- Arm comparison: per-category slope difference (contradictory minus neutral) with bootstrap confidence intervals.
- Sanity check: verify fine-tuning actually improved the target task in both arms, otherwise "decay" is meaningless.
- Sanity check: verify the contradictory arm applies pressure (affect-flattening examples are learned, measured on held-out affect-flattening prompts); if not, the arm is invalid rather than evidence of stability.

## Threats to Validity
- Confound: general capability drift vs steering-specific decay; mitigated by the steering-free baseline at each checkpoint.
- Vector staleness: frozen vectors may misalign with shifted activation geometry rather than the concept being forgotten; report both frozen-vector SSR and re-extracted-vector SSR at the final checkpoint to distinguish these.
- Dataset leakage: neutral instruction data must be filtered for emotional content; filtering procedure logged.
- Contradictory-arm construction: affect-flattening examples may differ from the neutral data in length, topic, or style as well as affect; matching procedure logged, and the replaced fraction held fixed.
- Single model: findings gate further work only; grid runs generalize later.

## Provenance
Identical run-folder discipline to Experiment 001, with the arm recorded in each run's config.

## Single-Command Reproduction
```
python scripts/run_experiment.py --experiment 002
```
