# Experiment 002: Steering Decay Under Neutral Fine-Tuning (Go/No-Go)

## Hypothesis
QLoRA fine-tuning on emotionally neutral instruction data degrades emotion-steering effectiveness on Qwen2.5-1.5B-Instruct, and the decay rate differs between basic and nuanced Plutchik categories (nuanced decays faster).

## Motivation
This is ASB's first original data point and the project's go/no-go gate. Prior work (ICLR 2026 Re-Align Workshop) shows embedded behavioral steering degrades when training data contradicts the steered behavior, but tested only generic behaviors (refusal, brevity), not affective categories, and did not analyze category-differentiated decay. If category-differentiated decay exists, ASB has a headline finding; if decay is flat and uniform, the benchmark's central axis must be redesigned before further investment.

Note: the Re-Align workshop paper found that embedded steering persists when the fine-tuning data does not contradict the steered behavior, and degrades only when it does. This directly affects our neutral-data design, since emotionally neutral fine-tuning data applies little contradicting pressure and may therefore produce little or no decay. Design decision pending: neutral-only fine-tuning vs adding a contradictory-data arm.

## Expected Outcome
- Go: visibly different decay slopes between basic and nuanced categories across checkpoints, in either direction (nuanced-faster is hypothesized, basic-faster would be equally reportable).
- No-Go: statistically indistinguishable, near-zero slopes across all categories after the full fine-tuning run.

## Baselines
- Checkpoint 0 (pre-fine-tuning) SSR per category from Experiment 001.
- Steering-free generation quality at each checkpoint (to separate general capability drift from steering-specific decay).

## Variables
- Independent: fine-tuning steps (checkpoints at 0%, 33%, 66%, 100% of a fixed budget), emotion category.
- Dependent: SSR per category per checkpoint, decay slope per category, SCR per checkpoint.
- Controlled: model, steering vectors (frozen from Experiment 001, re-applied at each checkpoint), extraction data, prompt set, seeds, QLoRA hyperparameters, fine-tuning dataset (neutral instruction data, emotion-content-filtered).

## Metrics
- SSR per category per checkpoint.
- Decay slope: least-squares slope of SSR over checkpoints, per category.
- SCR per checkpoint.

## Statistical Validation
- Slope confidence intervals via bootstrap over prompts and seeds.
- Basic vs nuanced slope comparison: Mann-Whitney U on per-category slopes, effect size reported.
- Sanity check: verify fine-tuning actually improved the target task, otherwise "decay" is meaningless.

## Threats to Validity
- Confound: general capability drift vs steering-specific decay; mitigated by the steering-free baseline at each checkpoint.
- Vector staleness: frozen vectors may misalign with shifted activation geometry rather than the concept being forgotten; report both frozen-vector SSR and re-extracted-vector SSR at the final checkpoint to distinguish these.
- Dataset leakage: neutral instruction data must be filtered for emotional content; filtering procedure logged.
- Single model: findings gate further work only; grid runs generalize later.

## Provenance
Identical run-folder discipline to Experiment 001.

## Single-Command Reproduction
```
python scripts/run_experiment.py --experiment 002
```
