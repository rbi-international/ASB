# Related Work Notes (adversarial reading log)

Format per paper: (1) what they did, (2) what they explicitly could not do, (3) one sentence on how ASB differs.

## Tier 1: direct neighbors
### Does Downstream Fine-Tuning Undo Embedded Activation Steering? (ICLR 2026 Re-Align Workshop)
1. They embedded two linear steering edits directly into the weights of five instruct models (Llama-3.2-3B, Llama-3.1-8B, Llama-3-8B, Qwen3-14B, SOLAR-10.7B; Table 1, p5): refusal ablation (full orthogonalisation on Llama, optimised per-layer variable orthogonalisation on Qwen and SOLAR) and brevity amplification (optimised variable amplification), with directions estimated by difference-in-means contrastive prompting (Eq. 1 p3, Eqs. 3 to 5 p4, pp5 to 6, Appendix B p14). They then ran full-parameter SFT on Open-Orca (1,024 steps) and RLHF (PPO) on Anthropic hh-rlhf with the Skywork-Reward-V2-Llama-3.1-8B reward model (about 420 steps) (p6, Appendix D pp15 to 16). They measured behavioural preservation P (refusal rate on 100 harmful prompts, mean length on 300 long-answer prompts; Eq. 6, pp6 to 7) and weight-space vector recovery ρ (Eq. 7, p8). Under RLHF both edits held (mean P_rlhf 0.96 for ablation, 0.98 for brevity); under SFT brevity mostly held (mean P_sft 0.86) but refusal ablation degraded (mean refusal rate back to 0.64, mean P_sft 0.36, per-model range 0.09 to 0.73), which they attribute to the ~0.1% refusal completions in Open-Orca (Table 2 p7, pp7 to 8). SOLAR SFT brevity was a catastrophic failure and was excluded (Table 2, p7). Vector recovery stayed under 2% in every condition (mean ρ 0.004, 95% CI [0.001, 0.007]) and was uncorrelated with behavioural recovery (Pearson r = -0.094), so they conclude that fine-tuning routes around the edit rather than undoing it (p8, p9).
2. What they say they did not cover: quantisation and other non-gradient perturbations are "outside our experimental scope" (p5), and they name quantisation as future work because it may degrade steering by a different mechanism (p9). They used only full-weight fine-tuning, not LoRA, chosen as the worst case (p2 footnote 2, p9). They did not test models above 14B or closed-weight models (p9). They tested only two steering targets, two embedding methods, two training paradigms and one dataset per paradigm (p9). SFT and RLHF results should not be compared as training methods, because datasets and hyperparameters differ (p6). They did not compare against behaviour acquired through training alone, which they plan as future work (p10). Direction estimation and refusal classification rely on heuristics, namely contrastive differences and substring matching (p10). Model-specific factors behind resilience were not controlled (p8). What the paper does not test, although it does not list these as limitations: emotion or affective targets (the targets are refusal and brevity only, p5); inference-time offset steering (all interventions are weight-embedded, and they note ActAdd cannot be embedded, p4); intermediate training checkpoints (results are reported only at the base, steered and post-training stages, Table 2 p7); and general capability or output coherence after fine-tuning (not stated in paper; KL divergence is used only when choosing steering parameters, Appendix B p14).
3. This paper already asks and answers ASB's fine-tuning question (does steering survive routine, non-adversarial downstream fine-tuning?) with a stronger perturbation (full-parameter SFT and RLHF on 3B to 14B models, 4x H100, p15), so ASB's genuine differences are narrower: affective targets broken out per Plutchik category, inference-time difference-in-means vectors tracked across QLoRA checkpoints rather than weight-embedded edits measured once after training, a quantisation axis this paper explicitly left to future work (p9), and a 6 GB budget; note also that their finding that steering persists when training data does not push against it (pp7 to 9) predicts little decay on ASB's emotionally neutral fine-tuning data, and that QLoRA is milder than their worst case.

### Do LLMs "Feel"? Emotion Circuits Discovery and Control (arXiv 2510.11328)
1. PDF not found in papers/
2. PDF not found in papers/
3. PDF not found in papers/

### Emotions Where Art Thou (arXiv 2510.22042)
1. PDF not found in papers/
2. PDF not found in papers/
3. PDF not found in papers/

### Cross-model Transferability on Platonic Representations (Huang et al., ACL 2025)
1. PDF not found in papers/
2. PDF not found in papers/
3. PDF not found in papers/

## Tier 2: methodology
### Contrastive Activation Addition (Rimsky et al., ACL 2024)
1. PDF not found in papers/
2. PDF not found in papers/
3. PDF not found in papers/

### Representation Engineering (Zou et al., 2023)
1. PDF not found in papers/
2. PDF not found in papers/
3. PDF not found in papers/

### Analyzing the Generalization and Reliability of Steering Vectors (Tan et al., NeurIPS 2024)
1. PDF not found in papers/
2. PDF not found in papers/
3. PDF not found in papers/

### QLoRA (Dettmers et al., 2023)
1. PDF not found in papers/
2. PDF not found in papers/
3. PDF not found in papers/

### Painless Activation Steering (arXiv 2509.22739)
1. PDF not found in papers/
2. PDF not found in papers/
3. PDF not found in papers/

## Tier 3: benchmark craft
### MMLU (Hendrycks et al.)
1. PDF not found in papers/
2. PDF not found in papers/
3. PDF not found in papers/

### HELM (Liang et al., 2022)
1. PDF not found in papers/
2. PDF not found in papers/
3. PDF not found in papers/
