"""ASB steering module. Implemented for Experiment 001 (see experiments/).

Adds a category's steering vector into the residual stream during generation.

The injection rule is fixed by Experiment 001 README, Amendment 1:
- The vector is added where it was read: the output of the half-depth decoder
  block, through a forward hook on that block. A pre-hook would act one block
  earlier.
- Generated tokens only. The prompt and the assistant header are not steered,
  as in CAA. The first output token is predicted from the unsteered header
  position, so it is chosen without steering; every later token is steered.

Strength: each category vector is scaled to unit length, and the coefficient
is a fraction of the typical residual-stream norm at the injection layer. The
typical norm is the median per-token norm over the neutral sentence tokens
saved by the extraction run, so no forward pass is needed to estimate it. A
coefficient of 1.0 adds a vector as long as a typical residual-stream state,
which keeps one coefficient comparable across categories and models.
Coefficient 0 registers no hook at all, so the unsteered baseline is the plain
model.

Scoring (SSR, SCR) belongs to evaluation.py, not here.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from asb.extraction import (
    DEFAULT_SEED,
    EXPERIMENT_001_DIR,
    ROOT,
    check_depth,
    decoder_layers,
    fixed_layer,
    git_commit,
    git_is_dirty,
    load_model,
    resolve_model,
    set_seed,
    template_kwargs,
)

PATH_B_CONVENTION = "path_b"


# --- 1. Generation settings ---------------------------------------------------

@dataclass(frozen=True)
class GenerationSettings:
    """Decoding settings, recorded in full with every generation.

    These defaults are for manual testing. The values used for SSR must be
    fixed in the pre-registration before evaluation runs.
    """
    max_new_tokens: int = 128
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9


# --- 2. Loading a vector run --------------------------------------------------

def latest_path_b_run(runs_dir: Path = EXPERIMENT_001_DIR / "runs") -> Path:
    """The most recent extraction run made under the Path B convention."""
    if not runs_dir.exists():
        raise FileNotFoundError(f"no runs directory: {runs_dir}")
    candidates = []
    for d in sorted(runs_dir.iterdir(), key=lambda d: d.name):
        path = d / "provenance.json"
        if not path.exists():
            continue
        provenance = json.loads(path.read_text(encoding="utf-8"))
        if str(provenance.get("extraction_convention", "")).startswith(PATH_B_CONVENTION):
            candidates.append(d)
    if not candidates:
        raise FileNotFoundError(f"no Path B extraction run under {runs_dir}")
    return candidates[-1]


def load_vector_run(run_dir: Path) -> tuple[dict[str, torch.Tensor], dict]:
    """Load (vectors, provenance) from an extraction run, Path B only.

    A Path A vector was read at a different position and measures a different
    thing (Amendment 1), so steering refuses it rather than mixing conventions.
    """
    prov_path = run_dir / "provenance.json"
    vec_path = run_dir / "steering_vectors.pt"
    for path in (prov_path, vec_path):
        if not path.exists():
            raise FileNotFoundError(f"missing {path.name} in {run_dir}")
    provenance = json.loads(prov_path.read_text(encoding="utf-8"))
    convention = str(provenance.get("extraction_convention", ""))
    if not convention.startswith(PATH_B_CONVENTION):
        raise ValueError(
            f"{run_dir} is not a Path B extraction run (extraction_convention "
            f"{convention!r}); steering uses Path B vectors only"
        )
    vectors = torch.load(vec_path, map_location="cpu", weights_only=True)
    return {c: v.to(torch.float32) for c, v in vectors.items()}, provenance


def residual_scale(run_dir: Path) -> float:
    """Median per-token residual norm over the run's neutral sentence tokens.

    Per-token norms are used, not the pooled per-prompt vectors in
    raw_activations.pt, because the vector is added per token and averaging
    over a sentence shrinks the norm. Neutral tokens of every category are
    pooled, so the scale is one number for the model and layer. The median
    keeps a few outlier tokens from setting it.
    """
    path = run_dir / "token_activations.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing token_activations.pt in {run_dir}")
    tokens = torch.load(path, map_location="cpu", weights_only=True)
    norms = [
        t.to(torch.float32).norm(dim=-1)
        for sides in tokens.values()
        for t in sides["neutral"]
    ]
    if not norms:
        raise ValueError(f"{path} holds no neutral token activations")
    return float(torch.cat(norms).median())


# --- 3. Injection via forward hook --------------------------------------------

@contextmanager
def steering_hook(model, layer: int, delta: torch.Tensor):
    """Add `delta` to the output of decoder block `layer`, generated tokens only.

    With a KV cache, generate() makes one forward pass over the whole prompt
    and then one pass per new token, each with sequence length 1. The prompt
    pass is skipped, so neither the prompt nor the assistant header is
    steered; every later pass carries one newly generated token, which gets
    delta exactly once. Yields a one-element list counting forward passes, so
    the caller can record how many steps were steered. Fails loudly if a later
    pass is not one token wide, since that would mean the cache is off and
    positions would be steered more than once. The hook is always removed,
    including on error.
    """
    calls = [0]

    def hook(_module, _inputs, output):
        calls[0] += 1
        if calls[0] == 1:
            return None  # prompt pass: unsteered
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] != 1:
            raise RuntimeError(
                f"expected one new token per generation step, got {hidden.shape[1]}; "
                f"generation must use the KV cache"
            )
        hidden = hidden + delta.to(device=hidden.device, dtype=hidden.dtype)
        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        return hidden

    handle = decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        yield calls
    finally:
        handle.remove()


# --- 4. Steered generation ----------------------------------------------------

class Steerer:
    """A loaded model plus one Path B vector run, ready to generate.

    Loading is the slow part, so evaluation.py builds one Steerer per model
    and calls generate() for every category and coefficient.
    """

    def __init__(self, run_dir: Path | None = None):
        self.run_dir = Path(run_dir) if run_dir else latest_path_b_run()
        vectors, self.vector_provenance = load_vector_run(self.run_dir)

        entry, inference = resolve_model(self.vector_provenance["model_id"])
        self.model_id = entry["id"]
        self.layer = fixed_layer(entry)
        if self.vector_provenance.get("layer") != self.layer:
            raise ValueError(
                f"{self.run_dir} was extracted at layer {self.vector_provenance.get('layer')}, "
                f"but the fixed layer for {self.model_id} is {self.layer}"
            )
        self.quantization = inference.get("default_quantization", "nf4")
        run_quant = self.vector_provenance.get("quantization", {}).get("scheme")
        if run_quant != self.quantization:
            raise ValueError(
                f"{self.run_dir} was extracted in {run_quant!r}, steering would load "
                f"{self.quantization!r}"
            )

        self.vector_norms = {c: float(v.norm()) for c, v in vectors.items()}
        zero = [c for c, n in self.vector_norms.items() if n == 0.0]
        if zero:
            raise ValueError(f"{self.run_dir} has zero-length vectors for: {zero}")
        self.directions = {c: v / v.norm() for c, v in vectors.items()}
        self.residual_scale = residual_scale(self.run_dir)

        self.model, self.tokenizer = load_model(self.model_id, self.quantization)
        check_depth(self.model, entry)
        hidden = getattr(self.model.config, "hidden_size", None)
        if hidden is not None and hidden != self.vector_provenance.get("hidden_size"):
            raise ValueError(
                f"model hidden size {hidden} differs from the vectors' "
                f"{self.vector_provenance.get('hidden_size')}"
            )

        # Same templating as extraction, including the pinned date.
        self.template_extra = template_kwargs(self.tokenizer)
        if self.template_extra.get("date_string") != self.vector_provenance.get("template_date"):
            raise ValueError(
                f"template date {self.template_extra.get('date_string')!r} differs from "
                f"the vector run's {self.vector_provenance.get('template_date')!r}"
            )
        # Left padding keeps every row's last prompt token at the same index,
        # so each generation step adds exactly one new token per row.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def render(self, prompt: str) -> str:
        """The prompt as a single user turn, ready for the assistant to reply."""
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
            **self.template_extra,
        )

    def generate(
        self,
        prompts: list[str],
        category: str | None,
        coefficient: float,
        seed: int = DEFAULT_SEED,
        settings: GenerationSettings = GenerationSettings(),
    ) -> list[dict]:
        """Generate one steered reply per prompt, as one batch.

        Adds coefficient * residual_scale * unit vector for `category` at each
        generated token. Coefficient 0 adds no hook, so it is the plain model;
        category may then be None, for unsteered outputs shared by every
        category. Negative coefficients steer away from the category. Returns
        one record per prompt, holding the output and its full provenance.

        Outputs depend on the batch's composition (sampling order, padding),
        so batch_size and batch_index are recorded; reproducing a generation
        means re-running the same batch with the same seed.
        """
        if category is None and coefficient != 0:
            raise ValueError("a non-zero coefficient needs a category to steer toward")
        if category is not None and category not in self.directions:
            raise ValueError(f"unknown category {category!r}; have {sorted(self.directions)}")
        if not math.isfinite(coefficient):
            raise ValueError(f"coefficient must be finite, got {coefficient}")
        if not prompts:
            raise ValueError("no prompts given")

        rendered = [self.render(p) for p in prompts]
        # The template already holds any BOS token, as in extraction.
        inputs = self.tokenizer(
            rendered, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.model.device)
        gen_kwargs = {
            "max_new_tokens": settings.max_new_tokens,
            "do_sample": settings.do_sample,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if settings.do_sample:
            gen_kwargs["temperature"] = settings.temperature
            gen_kwargs["top_p"] = settings.top_p

        set_seed(seed)
        started = datetime.now(timezone.utc)
        with torch.no_grad():
            if coefficient == 0:
                output = self.model.generate(**inputs, **gen_kwargs)
                steered_steps = 0
            else:
                delta = coefficient * self.residual_scale * self.directions[category]
                with steering_hook(self.model, self.layer, delta) as calls:
                    output = self.model.generate(**inputs, **gen_kwargs)
                steered_steps = calls[0] - 1

        new_tokens = output[:, inputs["input_ids"].shape[1]:].to("cpu")
        shared = self._provenance(category, coefficient, seed, settings, steered_steps, started)
        records = []
        for i, prompt in enumerate(prompts):
            ids = new_tokens[i].tolist()
            n_generated = len(ids)
            for stop in (self.tokenizer.eos_token_id, self.tokenizer.pad_token_id):
                if stop is not None and stop in ids:
                    n_generated = min(n_generated, ids.index(stop))
            records.append({
                **shared,
                "batch_size": len(prompts),
                "batch_index": i,
                "user_prompt": prompt,
                "rendered_prompt": rendered[i],
                "output": self.tokenizer.decode(ids[:n_generated], skip_special_tokens=True),
                "generated_tokens": n_generated,
            })
        return records

    def _provenance(
        self,
        category: str | None,
        coefficient: float,
        seed: int,
        settings: GenerationSettings,
        steered_steps: int,
        started: datetime,
    ) -> dict:
        """Fields shared by every record in one generate() call."""
        import transformers  # imported here, as in extraction

        run = self.vector_provenance
        return {
            "stage": "steering",
            "model_id": self.model_id,
            "quantization": self.quantization,
            "layer": self.layer,
            "layer_indexing": "0-based index into model.model.layers",
            "category": category,
            "coefficient": coefficient,
            "coefficient_units": (
                "fraction of the median per-token residual norm at the layer, "
                "over the vector run's neutral sentence tokens"
            ),
            "residual_scale": self.residual_scale,
            "added_norm": abs(coefficient) * self.residual_scale,
            "vector_norm_raw": self.vector_norms[category] if category else None,
            "injection": (
                "forward hook on the block output; generated tokens only; "
                "first output token unsteered (Experiment 001 Amendment 1)"
            ),
            "steered_decode_steps": steered_steps,
            "vector_run": _relative(self.run_dir),
            "vector_run_git_commit": run.get("git_commit"),
            "vector_run_prompts_sha256": run.get("prompts_sha256"),
            "template_date": self.template_extra.get("date_string"),
            "seed": seed,
            "generation": asdict(settings),
            "git_commit": git_commit(),
            "git_dirty": git_is_dirty(),
            "started_utc": started.isoformat(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }


def _relative(path: Path) -> str:
    """The path relative to the repo root when inside it, else as given."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)
