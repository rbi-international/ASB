# Experiment 001: Baseline Replication of the Basic vs Nuanced Steering Reliability Gap

Amended 2026-10-04 before any steered generation: extraction now measures the model expressing each emotion (Path B). See [Amendment 1](#amendment-1-2026-10-04-extraction-measures-the-model-expressing-the-emotion-path-b).

Amended 2026-10-06 before any evaluation run: steering strength grid, selection rule, generation settings, prompt set, baselines and judging are fixed. See [Amendment 2](#amendment-2-2026-10-06-evaluation-settings-fixed-before-any-evaluation-run).

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
- Independent: emotion category (8 levels), steering coefficient (swept). Injection layer is fixed, not swept (corrected in Amendment 1).
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

## Amendment 1 (2026-10-04): extraction measures the model expressing the emotion (Path B)

Adopted before any steered generation was produced. It replaces the extraction convention used by the first full extraction run and fixes the injection rule for `steering.py`. Nothing else in this pre-registration changes.

### Rationale
The first full run (Path A) placed each sentence in the user turn and read the residual stream at the first-generated-token position: the last token of the chat-templated prompt with `add_generation_prompt=True`, following the Re-Align workshop paper (Appendix B, p14). A vector read there measures how the model's state differs when a *user* has expressed the emotion. SSR judges whether the *model's output* expresses it. A vector of the first kind may steer the model toward responding to an emotion (consoling, reassuring) instead of expressing it, so SSR would measure a property the vector was not built to encode. Path B extracts the model producing the emotional content itself.

Grounding in the Tier 2 methodology papers:
- **Contrast design from CAA** (Rimsky et al., ACL 2024). Each pair shares the same question and differs only in the answer letter written into the model's own response, and activations are read at the answer-letter position (p3, Eq. 1). Path B keeps this structure: same context, different assistant output.
- **Token pooling from RepE's function procedure** (Zou et al., 2023). For functions, the text is placed in the output field of a `USER: ... / ASSISTANT: <output>` template, and representations are collected from each token of the response, because the model engages with the function when generating every new token (p9 to 10, Eq. 2). RepE's concept procedure is different: the stimulus is placed inside a question and the last token is read (Eq. 1). That is what RepE uses for emotion, and it measures the model perceiving the concept, which is closer to Path A. Path B treats expressing an emotion as a function. One difference from RepE: RepE varies the user instruction and holds the output fixed, while Path B holds the instruction fixed and varies the output, as CAA does.

### Prompt presentation
- Every sentence, emotion and neutral, is the assistant turn of a two-turn chat that follows one fixed user turn. The exact user string is:

  ```
  Write one sentence.
  ```

  It is identical for all 320 prompts and both poles, so it cancels in the difference. It names no emotion and fits both first-person and third-person sentences. It is recorded verbatim in provenance.
- The chat template is applied with each model's own tokenizer and no added system message. Where a template inserts its own default system prompt (Qwen2.5), that prompt is kept, because steering applies the template the same way.
- The 160 pairs in `configs/emotion_prompts.yaml` are reused unchanged; only their presentation changes. The Path B run must record the same `prompts_sha256` as the Path A run (`4bd11430e899...`).
- **First person and third person are mixed on purpose.** Some sentences are first person ("I recoiled when I saw the worms...") and some third person ("She wrinkled her nose at the sour smell of the rag"). They stay mixed because steered output can express an emotion either as the speaker's own state or through narration, SSR will credit both, and the vector should therefore cover both.

### Fixed template date
The Llama 3.2 chat template inserts the current date into a system header unless a date is supplied, which would make the prompt and its activations depend on the day of the run. For every model whose chat template reads a date variable, the date is fixed to `26 Jul 2024`, the Llama 3.2 template's own fallback value. The same fixed date is used for extraction and for steering. Extraction checks each template for a date variable instead of assuming which models have one; the check is expected to fire for the two Llama 3.2 models only. Provenance records the fixed date (or that none applied) and the full rendered text of one templated example, so the exact context the model saw can be inspected.

### Read position
- Layer unchanged: the output of decoder block 14 (0-based), captured by a forward hook on that block.
- Per prompt: the mean of the layer output over the sentence's own tokens. Excluded: everything before the sentence (any system text, the user turn, the assistant header) and the template tokens after it (end of turn). The span is located by checking that the full templated sequence starts with exactly the token ids of the same conversation templated up to the assistant header (`add_generation_prompt=True`). Extraction fails loudly if that check fails or the end-of-turn token cannot be found.
- Steering vector per category: the mean of the per-prompt means for the emotion pole minus the same for the neutral pole, as before.
- **Why the mean and not the last token.** The last token of these sentences is nearly always ".", while the emotional content sits mid-sentence. The vector will be added at every generated token, so it should describe the average state while emotional content is being produced, which is RepE's rationale for functions. CAA reads a single token only because its behaviour is a single answer letter.
- **Saved files.** `raw_activations.pt` keeps one pooled vector per prompt (pairs by hidden size), so `scripts/analyze_extraction.py` runs unchanged. The per-token activations over each sentence span are saved separately in `token_activations.pt`, so other poolings (last token, for example) can be checked later without another forward pass.
- **Provenance strings.** `token_position`: "mean over the assistant-turn tokens of the sentence (template and end-of-turn tokens excluded)". `prompt_template`: "chat template, fixed user turn, sentence as the assistant turn". New fields hold the user string, the template date, and the rendered example.

### Injection rule for steering.py
- The vector is added at the same point it was read: the output of decoder block 14, through a forward hook on that block (not a pre-hook, which would act one block earlier).
- It is added to **generated tokens only**. No prompt position is steered, including the end of the assistant header. This is CAA's rule (p4: "every token position of the generated text after the end of the initial prompt"; p2: "all and only token positions after the original prompt"). It also matches extraction, which excludes the header from the read span.
- **One unsteered token.** The first output token is predicted from the header position, which is not steered, so each generation's first token is chosen without steering. This is one token per generation; it is documented here rather than worked around.
- Because generation uses a KV cache, each token's residual stream is computed once. Adding the vector to the new token at every step therefore steers every generated position exactly once; nothing is re-added to earlier tokens.

### Correction to the Variables list
The Variables section originally read:

> Independent: emotion category (8 levels), steering coefficient (swept), injection layer (swept once, then fixed).

That contradicted the locked v1.0 scope in the main README, which fixes a single mid-layer and defers layer sweeping to v2.0 (Scope Policy). The line now reads "Independent: emotion category (8 levels), steering coefficient (swept). Injection layer is fixed, not swept (corrected in Amendment 1)." The layer is not an independent variable in Experiment 001, and no layer sweep is run.

The fixed layer for each model is half its decoder depth (0-based index into the decoder blocks), matching layer 14 of 28 for Qwen2.5-1.5B:

| Model | Decoder layers | Fixed layer |
|---|---|---|
| Qwen2.5-1.5B-Instruct | 28 | 14 |
| Llama-3.2-1B-Instruct | 16 | 8 |
| Llama-3.2-3B-Instruct | 28 | 14 |
| gemma-2-2b-it | 26 | 13 |
| Phi-3.5-mini-instruct | 32 | 16 |

Experiment 001 itself runs on Qwen2.5-1.5B only; the other rows fix the rule in advance for later experiments on the full grid. Layer counts come from each model's published config and are checked against the loaded model at extraction time, with the layer index recorded in provenance as now.

### Threats to validity added by Path B
- **Emotion-word naming.** Extraction places hand-written sentences in the assistant turn; the model is given them and does not choose them. The vector therefore partly encodes "an emotion word is the current token", and injecting it may push the model to name emotions literally rather than shift its tone. SSR could then credit naming that a reader would not call expression. `docs/judge_protocol.md` must rule, before the first steered run, on whether naming an emotion counts as expressing it, and the rubric is applied as written. SCR flags degenerate repetition but does not settle this question.
- **Span length.** Vectors come from sentences of about 12 tokens and are applied across full generations. This is standard practice for steering vectors and is noted, not corrected.
- **Different user turns.** The user turn differs between extraction (the fixed string above) and steering (the evaluation prompts). The extraction user turn is identical on both poles, so it cancels in the difference, and the vector carries nothing from it by design.

### The Path A run
- The Path A run `runs/20261003T031322Z` (provenance `token_position`: "first_generated_token (position -1 of the templated prompt)") is superseded. It stays in `runs/` as a record and is not used for steering.
- Its analysis result (all eight categories clearing their flip-nulls, raw and centered) does not carry over. The text-level fixes to the pairs give a reason to expect Path B to pass, but the Path B run must pass on its own. Gate before `steering.py` is written: re-run extraction under Path B, then `scripts/analyze_extraction.py --centered`, and all eight categories clear their nulls.
- Vector norms are not comparable between the two runs, because averaging over tokens changes their scale. The norm curve is read within one run only.
- The earlier run `runs/20260925T024040Z` is the three-seed-pair pipeline smoke test and has no bearing on results.

### Open items, parked for evaluation.py
These are not decided by this amendment and must be settled before the first steered run:
- **Judge model**: local or API. This affects cost, and the rubric (including the naming ruling above) must be written for the chosen judge.
- **Steering-evaluation prompt set**: neutral, open-ended prompts, separate from the extraction pairs, with a held-out split for coefficient selection. It does not exist yet.

## Amendment 2 (2026-10-06): evaluation settings fixed before any evaluation run

Recorded before any evaluation run and before any judged output exists, so that no setting below can be tuned after seeing results. It resolves both open items parked at the end of Amendment 1. Nothing in Amendment 1 changes.

**Earlier exploratory steering.** Before this amendment, steering was checked by eye on Qwen2.5-1.5B and gemma-2-2b: one prompt, one seed, at strengths 0, 0.25, 0.5 and 1.0, with no judge. Those checks informed the grid below (fluent and steered at 0.5 on both models; degraded at 1.0, severely on Gemma). They are not results and are not reported as such.

### Steering strength grid
Strengths: **0, 0.25, 0.5, 0.75, 1.0**.

Each is a fraction of the model's typical residual-stream size at the fixed layer, in the units `src/asb/steering.py` already uses: the median per-token norm over the neutral sentence tokens of that model's Path B extraction run. Strength 0 attaches no steering hook, so it is the plain model.

### Choosing the strength
For each model and each category, selection uses the 20 held-out prompts only:
1. Generate at every non-zero strength, plus the unsteered outputs at strength 0, for the held-out prompts and all three seeds.
2. Judge them, and compute SSR and SCR per strength as defined in `docs/judge_protocol.md`.
3. **Eligible strengths** are the non-zero strengths whose SCR is at least **0.85**: the steered text keeps at least 85% of the matched unsteered text's coherence.
4. **Select** the eligible strength with the highest SSR. If two eligible strengths tie on SSR, select the lower one, since less intervention reaches the same effect.
5. **If no strength is eligible**, the category has no usable strength for that model. Its headline SSR is the strength-0 base rate, and it is flagged as "not steerable within the coherence floor". This counts it as a steering failure rather than hiding it, which is the honest reading for RQ1. The flag can hide two different failures: the vector does not push the emotion at all, or it pushes the emotion but the text breaks before the effect gets strong. So for any flagged category, a clearly labelled secondary figure is also reported: its best SSR across the non-zero strengths ignoring the floor, together with that strength's SCR. As with the headline, the strength is chosen on the held-out prompts (highest SSR, ties to the lower strength), and its SSR and SCR are reported on the evaluation prompts. Picking the maximum directly on the evaluation prompts would inflate it. The headline stays the strict value. The secondary figure only explains why the category failed and is never used in the basic-versus-nuanced comparison.

The selected strength is frozen. It is used unchanged for the 40 evaluation prompts, and for the same model and category in Experiments 002 and 003. Re-choosing it at each fine-tuning checkpoint or quantization level would absorb the decay those experiments measure. Selection results (SSR and SCR per strength on the held-out prompts, and the strength chosen) are written to the run record before any evaluation-prompt output is generated.

### Reporting
For each model and category:
- the full SSR and SCR curves against strength, on the 40 evaluation prompts;
- the selected strength, and the headline SSR and SCR at that strength;
- the strength-0 base rate next to every SSR.

**Shared-strength check.** Every category is also reported at one shared strength, **0.5**, fixed here in advance. The basic-versus-nuanced comparison is reported at both the selected strengths and the shared strength. If the gap appears only under per-category selection, it is reported as depending on selection, not as a property of the categories.

### Generation settings
| Setting | Value |
|---|---|
| Decoding | sampling |
| temperature | 0.7 |
| top_p | 0.9 |
| max_new_tokens | 128 |
| Seeds | **0, 1, 2** |
| Prompt format | each prompt as a single user turn through the model's chat template, with the pinned template date where one applies (Amendment 1) |

**Batches.** Generation is batched, and with sampling an output depends on the rest of its batch. So batch composition is fixed. For each (model, category, strength, seed), one batch holds every prompt of the split, in file order. If memory requires smaller batches, the split is cut into fixed-size consecutive chunks in file order, and the chunk size is recorded per run. Every generation record stores its batch size and index, as `steering.py` already does.

**Shared unsteered outputs.** Unsteered outputs (strength 0) are the same for every category, so they are generated once per (model, prompt, seed) and shared as the matched outputs for SCR.

### Prompt set
- **File:** `configs/eval_prompts.yaml`, 60 prompts, SHA-256 `f41825095b9f622112dd5a561c14bb07f2cab086082dbb823a18fac081711a1a`.
- **Split:** 40 evaluation prompts (`eval`) and 20 held-out prompts (`heldout`). The split was drawn with seed **20261006** (`random.Random(seed).shuffle` over ids 1 to 60, first 20 held out). It is fixed and is never re-drawn.
- **Provenance:** drafted by GPT-6 Astra High, a model family outside the steered grid, then reviewed by hand for neutrality, no self-reference, topical variety, and no overlap with the extraction pairs. All 60 passed review unchanged.
- **Hash check:** every evaluation run records the file's SHA-256 and stops if it differs from the value above.

### Baselines
- **Unsteered:** strength 0, the plain model, on the same prompts and seeds.
- **Prompt-based comparison:** no steering. The evaluation prompt is followed by one fixed sentence asking for the target emotion directly:

  ```
  {prompt} Write it so that it conveys {emotion}.
  ```

  Here `{emotion}` is the category name exactly as listed in `configs/emotions.yaml`. The output is judged by the same blinded request as every other output; the judge never sees the added sentence. Its SSR uses the same formula, and its SCR is matched to the unsteered output for the same prompt and seed.

### Judging
- **Rules:** all scoring follows `docs/judge_protocol.md` as committed in `98faa23`. Judge-protocol changes are made only as dated amendments in that file. Each evaluation run records the judge prompt version (the protocol's hash of its prompt template and definitions).
- **Local dry run:** a local judge is used only to test the pipeline end to end. Its records are tagged with their backend, and none of its numbers are reported as results.
- **Scoring:** results come only from Claude Fable 5.1 under the frozen protocol. The paid validation slice and its go/no-go criteria come first (protocol, Section 11).
