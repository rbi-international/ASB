"""ASB extraction module. Implemented during Experiment 001 (see experiments/).

Difference-in-means emotion steering vectors.

For each Plutchik category we run every contrastive pair from
configs/emotion_prompts.yaml through the model, capture the residual-stream
activation at a single fixed layer (v1.0 locks one mid-layer, see README
"Scope Policy"), and take mean(emotion) minus mean(neutral). That difference
is the steering direction for the category.

Everything needed to re-run or re-analyse a run is written to the run folder:
the eight vectors, the raw per-prompt activations, and provenance.json.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
MODELS_CONFIG = ROOT / "configs" / "models.yaml"
EMOTIONS_CONFIG = ROOT / "configs" / "emotions.yaml"
PROMPTS_CONFIG = ROOT / "configs" / "emotion_prompts.yaml"
EXPERIMENT_001_DIR = ROOT / "experiments" / "experiment_001_baseline_replication"

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_LAYER = 14  # locked mid-layer for Qwen2.5-1.5B (28 decoder blocks)
DEFAULT_SEED = 0


# --- 1. Config loading --------------------------------------------------------

def load_yaml(path: Path) -> dict:
    """Read a YAML config, failing loudly if it is missing or not a mapping."""
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"config is not a mapping: {path}")
    return data


def expected_categories(emotions_config: dict) -> list[str]:
    """The eight locked Plutchik categories, basic first, then nuanced."""
    cats = emotions_config.get("categories", {})
    ordered = list(cats.get("basic", [])) + list(cats.get("nuanced", []))
    if len(ordered) != 8 or len(set(ordered)) != 8:
        raise ValueError(
            f"{EMOTIONS_CONFIG} must define 8 distinct categories, found {ordered}"
        )
    return ordered


def load_prompt_pairs(
    prompts_path: Path = PROMPTS_CONFIG,
    emotions_path: Path = EMOTIONS_CONFIG,
) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
    """Load and validate the contrastive prompt pairs.

    Every locked category must be present, every pair must carry a non-empty
    'emotion' and 'neutral' string, and no category may be empty. Any problem
    raises with the exact category and pair index so it is quick to fix by hand.
    """
    categories = expected_categories(load_yaml(emotions_path))
    raw = load_yaml(prompts_path).get("categories")
    if not isinstance(raw, dict):
        raise ValueError(f"{prompts_path} must contain a 'categories' mapping")

    missing = [c for c in categories if c not in raw]
    if missing:
        raise ValueError(f"{prompts_path} is missing categories: {missing}")
    extra = [c for c in raw if c not in categories]
    if extra:
        raise ValueError(
            f"{prompts_path} has categories outside the locked taxonomy: {extra}"
        )

    pairs: dict[str, list[dict[str, str]]] = {}
    for category in categories:
        entries = raw[category]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{prompts_path}: category '{category}' has no pairs")
        cleaned = []
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"{prompts_path}: {category}[{i}] is not a mapping")
            for key in ("emotion", "neutral"):
                value = entry.get(key)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"{prompts_path}: {category}[{i}] has an empty or missing "
                        f"'{key}' prompt"
                    )
            cleaned.append({"emotion": entry["emotion"].strip(),
                            "neutral": entry["neutral"].strip()})
        pairs[category] = cleaned

    counts = {c: len(p) for c, p in pairs.items()}
    if len(set(counts.values())) != 1:
        # Not fatal: extraction still runs, but unequal counts break the
        # equal_per_category_examples assumption in configs/emotions.yaml.
        print(f"warning: unequal pair counts per category: {counts}")
    return pairs, categories


def resolve_model_id(model_id: str | None, models_path: Path = MODELS_CONFIG) -> tuple[str, dict]:
    """Return (model id, inference settings) from configs/models.yaml.

    A model id passed on the command line must still appear in the locked grid.
    """
    config = load_yaml(models_path)
    grid = [m["id"] for m in config.get("models", []) if "id" in m]
    chosen = model_id or DEFAULT_MODEL_ID
    if chosen not in grid:
        raise ValueError(
            f"model '{chosen}' is not in the locked v1.0 grid in {models_path}: {grid}"
        )
    return chosen, config.get("inference", {})


# --- 2. Model loading ---------------------------------------------------------

def load_model(model_id: str, quantization: str = "nf4"):
    """Load the model in 4-bit (nf4, fp16 compute) plus its tokenizer.

    Only nf4 is supported here; the fp16 and int8 arms of the quantization axis
    belong to Experiment 003 and are not needed for vector extraction.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if quantization != "nf4":
        raise ValueError(f"extraction runs in nf4 only, got '{quantization}'")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def decoder_layers(model) -> list:
    """The list of decoder blocks, for hooking one of them by index."""
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise AttributeError(
            "could not find decoder layers on this model; extraction assumes a "
            "model.model.layers list (Qwen, Llama, Phi style)"
        )
    return layers


# --- 3. Activation capture via forward hook -----------------------------------

@contextmanager
def residual_stream_hook(model, layer: int):
    """Capture the residual stream leaving decoder block `layer`.

    Yields a one-element list that holds the most recent hidden state, shaped
    (batch, seq, hidden). The hook is always removed, including on error.
    """
    layers = decoder_layers(model)
    if not 0 <= layer < len(layers):
        raise IndexError(
            f"layer {layer} is out of range for this model ({len(layers)} layers)"
        )
    captured: list[torch.Tensor | None] = [None]

    def hook(_module, _inputs, output):
        # Decoder blocks return either a tensor or a tuple whose first element
        # is the hidden state.
        captured[0] = output[0] if isinstance(output, tuple) else output

    handle = layers[layer].register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


# --- 4. Per-prompt activation extraction --------------------------------------

def last_token_activation(model, tokenizer, text: str, captured: list) -> torch.Tensor:
    """Run one forward pass and return the layer's hidden vector at position -1.

    The prompt goes through the chat template as a single user turn with
    add_generation_prompt=True, so position -1 is the first-generated-token
    position (the assistant header that follows the user's text), not the
    final content token of the sentence. This is deliberate: Experiment 001
    replicates the Re-Align workshop paper, which records residual-stream
    activations at the first generated token (Appendix B, p14).

    One prompt per forward pass, so no padding is involved.
    """
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    captured[0] = None
    with torch.no_grad():
        model(**inputs)
    if captured[0] is None:
        raise RuntimeError("forward hook captured nothing; check the layer index")

    # (batch, seq, hidden) -> the final position of the only sequence.
    return captured[0][0, -1, :].detach().to("cpu", torch.float32)


# --- 5. Difference in means per category --------------------------------------

def difference_in_means(
    model,
    tokenizer,
    pairs: dict[str, list[dict[str, str]]],
    categories: list[str],
    layer: int,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
    """Compute one steering direction per category.

    Returns (vectors, raw) where vectors[category] is
    mean(emotion activations) minus mean(neutral activations), and
    raw[category]['emotion'|'neutral'] holds the stacked per-prompt vectors so
    re-analysis never needs another forward pass.
    """
    vectors: dict[str, torch.Tensor] = {}
    raw: dict[str, dict[str, torch.Tensor]] = {}

    with residual_stream_hook(model, layer) as captured:
        for category in categories:
            sides = {"emotion": [], "neutral": []}
            for pair in pairs[category]:
                for side in ("emotion", "neutral"):
                    sides[side].append(
                        last_token_activation(model, tokenizer, pair[side], captured)
                    )
            emotion = torch.stack(sides["emotion"])
            neutral = torch.stack(sides["neutral"])
            vectors[category] = emotion.mean(dim=0) - neutral.mean(dim=0)
            raw[category] = {"emotion": emotion, "neutral": neutral}
    return vectors, raw


# --- 6. Provenance and saving -------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix and log the seed for every RNG that could touch a run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    """Current commit hash, or 'unknown' outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def git_is_dirty() -> bool | None:
    """True if the working tree has uncommitted changes, None if unknown."""
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


def save_run(
    run_dir: Path,
    vectors: dict[str, torch.Tensor],
    raw: dict[str, dict[str, torch.Tensor]],
    provenance: dict,
) -> None:
    """Write vectors, raw activations, and provenance.json to the run folder."""
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(vectors, run_dir / "steering_vectors.pt")
    torch.save(raw, run_dir / "raw_activations.pt")
    with (run_dir / "provenance.json").open("w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2, sort_keys=True)
        fh.write("\n")


# --- 7. Run entry point -------------------------------------------------------

def run_extraction(
    model_id: str | None = None,
    layer: int = DEFAULT_LAYER,
    seed: int = DEFAULT_SEED,
    prompts_path: Path = PROMPTS_CONFIG,
    experiment_dir: Path = EXPERIMENT_001_DIR,
) -> Path:
    """Extract all eight steering vectors and save one timestamped run folder.

    Returns the run folder path. Prints one line per category (vector norm and
    pair count) so a run can be eyeballed for sanity without opening the files.
    """
    resolved_id, inference = resolve_model_id(model_id)
    quantization = inference.get("default_quantization", "nf4")
    pairs, categories = load_prompt_pairs(prompts_path)
    set_seed(seed)

    started = datetime.now(timezone.utc)
    print(f"model: {resolved_id} ({quantization}), layer {layer}, seed {seed}")
    model, tokenizer = load_model(resolved_id, quantization)
    try:
        vectors, raw = difference_in_means(model, tokenizer, pairs, categories, layer)
    finally:
        del model
        torch.cuda.empty_cache()
    finished = datetime.now(timezone.utc)

    run_dir = experiment_dir / "runs" / started.strftime("%Y%m%dT%H%M%SZ")
    import transformers  # imported here so config errors surface before load

    provenance = {
        "experiment": experiment_dir.name,
        "stage": "extraction",
        "model_id": resolved_id,
        "layer": layer,
        "layer_indexing": "0-based index into model.model.layers",
        "quantization": {
            "scheme": quantization,
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_use_double_quant": False,
        },
        "method": "difference_in_means",
        "token_position": "first_generated_token (position -1 of the templated prompt)",
        "prompt_template": "chat template, single user turn, add_generation_prompt=True",
        "seed": seed,
        "categories": categories,
        "pairs_per_category": {c: len(p) for c, p in pairs.items()},
        "vector_norms": {c: float(v.norm()) for c, v in vectors.items()},
        "hidden_size": int(next(iter(vectors.values())).shape[0]),
        "prompts_file": str(prompts_path.relative_to(ROOT)),
        "prompts_sha256": file_sha256(prompts_path),
        "models_config_sha256": file_sha256(MODELS_CONFIG),
        "emotions_config_sha256": file_sha256(EMOTIONS_CONFIG),
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    save_run(run_dir, vectors, raw, provenance)

    for category in categories:
        print(
            f"  {category:<13} norm {vectors[category].norm():8.3f}  "
            f"n_pairs {len(pairs[category]):3d}"
        )
    print(f"saved to {run_dir}")
    return run_dir
