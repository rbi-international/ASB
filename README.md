# ASB: Affective Stability Benchmark

A benchmark for measuring the stability of emotion-steering interventions in large language models under real-world deployment stressors, fully reproducible on consumer hardware (single 6 GB GPU).

**Status**: v1.0 in active development. Scope below is locked; see [Scope Policy](#scope-policy).

## Motivation

Recent work shows that emotional behavior in LLMs can be located and controlled through activation-level interventions (emotion circuits, steering vectors, latent affective manifolds). Separately, work on generic behavioral steering shows that such interventions can degrade under downstream fine-tuning. No unified standard currently measures how *stable* affective control is, per emotion category, under the stressors real deployments apply to models. ASB provides that measurement layer.

## Research Questions (v1.0)

- RQ1: How reliably can each of the eight Plutchik emotion categories be steered in current small open-weight instruction models?
- RQ2: Does steering effectiveness decay under downstream fine-tuning, both on emotionally neutral data and on data that contradicts the steered affect, and does the decay rate differ between basic and nuanced emotion categories?
- RQ3: Does post-training quantization (FP16 vs 8-bit vs 4-bit) degrade steering effectiveness, and does the degradation differ by category?

## Locked v1.0 Scope

### Emotion taxonomy
Plutchik 8 primary categories: joy, sadness, anger, fear, trust, disgust, surprise, anticipation.
Category split for decay analysis (rationale in `configs/emotions.yaml`):
- Basic (Ekman-overlapping): joy, sadness, anger, fear, surprise, disgust
- Nuanced (Plutchik-specific): trust, anticipation

### Models (all runnable in 4-bit on 6 GB VRAM)
| Model | Family | Params |
|---|---|---|
| Qwen2.5-1.5B-Instruct | Qwen | 1.5B |
| Llama-3.2-1B-Instruct | Llama | 1B |
| Llama-3.2-3B-Instruct | Llama | 3B |
| gemma-2-2b-it | Gemma | 2B |
| Phi-3.5-mini-instruct | Phi | 3.8B |

### Stability axes
1. Fine-tuning decay: steering effectiveness re-measured at fixed checkpoints during QLoRA fine-tuning, in two data arms:
   - Neutral arm: emotionally neutral instruction data.
   - Contradictory arm: the same data with a fixed fraction (50%, pre-registered) replaced by affect-flattening examples (emotion-inviting prompts answered in a deliberately affect-neutral register), which pushes against steered affect. Construction and fraction are fixed in `experiments/experiment_002_finetune_decay/README.md` before the run.
2. Quantization degradation: steering effectiveness compared across FP16, 8-bit, and 4-bit inference.

Scope amendment (2026-09-22): the contradictory arm was added to axis 1 deliberately. The ICLR 2026 Re-Align Workshop paper on embedded steering under fine-tuning found steering persists when fine-tuning data does not contradict the steered behavior (see `docs/related_work.md`), so a neutral-only design risks a null result that repeats existing work. This is the only change to the locked scope; the Scope Policy below still applies.

### Metrics (three, no more)
1. Steering Success Rate (SSR) per emotion category, judged.
2. Semantic Coherence Retention (SCR) of steered outputs.
3. Decay slope per category (change in SSR per fine-tuning checkpoint).

## Scope Policy

The following are explicitly deferred to v2.0 and will not be added to v1.0 regardless of how tempting they become:
- Long-context affective drift
- Paraphrase / prompt-perturbation robustness
- Indic-language affective evaluation (Hindi, Punjabi)
- Intensity tiers within Plutchik categories (e.g., serenity, joy, ecstasy)
- Per-category opposite-affect fine-tuning arms
- Fine-tuning data-fraction sweep
- Steering-vector layer selection (v1.0 fixes a single mid-layer, layer 14 for Qwen2.5-1.5B; sweeping layers is deferred)

Any pull request or experiment expanding v1.0 beyond the locked scope will be declined with a pointer to this section.

## Repository Structure

```
ASB/
  configs/        model list and emotion taxonomy (single source of truth)
  src/asb/        extraction, steering, evaluation, metrics
  experiments/    one folder per experiment, full provenance per run
  results/        generated figures and tables (from logs, never hand-made)
  docs/           related work notes and design decisions
  scripts/        single-command entry points per experiment
```

## Reproducibility

Every experiment is reproducible from a single command on a machine with one 6 GB GPU and 32 GB RAM. Full run provenance (config, seed, environment, git commit) is saved per run under `experiments/experiment_XXX/runs/`. Install from `requirements.lock.txt` rather than `requirements.txt`, since it pins the exact versions the results were produced with and the behavior of transformers and torch APIs varies across versions.

## Roadmap

- v1.0 (target: Apr 2027): scope above, paper submission (benchmark + findings).
- v2.0: deferred axes above, community model requests.

## Citation

See `CITATION.cff`. Paper preprint link will appear here on release.

## License

MIT. See `LICENSE`.
