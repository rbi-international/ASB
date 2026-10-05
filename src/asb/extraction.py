"""ASB extraction module. Implemented during Experiment 001 (see experiments/).

Difference-in-means emotion steering vectors.

For each Plutchik category we run every contrastive pair from
configs/emotion_prompts.yaml through the model and take mean(emotion) minus
mean(neutral) of the residual stream at a single fixed layer. That difference
is the steering direction for the category.

The vector measures the model expressing the emotion, not reacting to it
(Path B, Experiment 001 README, Amendment 1). Each sentence is placed in the
assistant turn of a chat, after one fixed user turn, so the model is the one
producing it. The activation for a prompt is the mean over the sentence's own
tokens, the way RepE reads functions; template, user turn, assistant header
and end-of-turn tokens are excluded. The layer is fixed at half the model's
decoder depth (v1.0 locks one mid-layer, see README "Scope Policy").

Everything needed to re-run or re-analyse a run is written to the run folder:
the eight vectors, the pooled per-prompt activations, the per-token sentence
activations, and provenance.json.
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
DEFAULT_SEED = 0

# Fixed by Experiment 001 Amendment 1. Changing any of these is a change to
# the pre-registration, not a tuning knob.
USER_TURN = "Write one sentence."
TEMPLATE_DATE = "26 Jul 2024"  # the Llama 3.2 template's own fallback date
# The prompts file the Path A run used. Path B must reuse the sentences
# unchanged, and this check makes the run record prove it.
PATH_A_PROMPTS_SHA256 = "4bd11430e89983b4bb4a26ef60e1a972bc3ce9de4f589c706bd142bff99c8c3a"


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


def resolve_model(model_id: str | None, models_path: Path = MODELS_CONFIG) -> tuple[dict, dict]:
    """Return (model entry, inference settings) from configs/models.yaml.

    A model id passed on the command line must still appear in the locked grid,
    and its entry must give an even decoder_layers, so half of it is exact.
    """
    config = load_yaml(models_path)
    entries = {m["id"]: m for m in config.get("models", []) if "id" in m}
    chosen = model_id or DEFAULT_MODEL_ID
    if chosen not in entries:
        raise ValueError(
            f"model '{chosen}' is not in the locked v1.0 grid in {models_path}: "
            f"{list(entries)}"
        )
    entry = entries[chosen]
    depth = entry.get("decoder_layers")
    if not isinstance(depth, int) or depth <= 0 or depth % 2:
        raise ValueError(
            f"{models_path}: '{chosen}' needs a positive even decoder_layers, got {depth!r}"
        )
    return entry, config.get("inference", {})


def fixed_layer(entry: dict) -> int:
    """The locked steering layer: half the decoder depth, 0-based (Amendment 1)."""
    return entry["decoder_layers"] // 2


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
            "model.model.layers list (Qwen, Llama, Gemma, Phi style)"
        )
    return layers


def check_depth(model, entry: dict) -> None:
    """Fail if the loaded model's depth differs from the configured one.

    The fixed layer is derived from the configured depth, so a wrong config
    value would silently move the layer.
    """
    actual = len(decoder_layers(model))
    if actual != entry["decoder_layers"]:
        raise ValueError(
            f"'{entry['id']}' has {actual} decoder layers when loaded, but "
            f"configs/models.yaml says {entry['decoder_layers']}; fix the config, "
            f"since the fixed layer is half of it"
        )


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

def template_kwargs(tokenizer) -> dict[str, str]:
    """Extra chat-template variables that pin anything run-dependent.

    Some templates (Llama 3.2) write the current date into a system header,
    which would make the prompt depend on the day of the run. Each template is
    checked for a date variable instead of assuming which models have one; if
    it reads one, the date is pinned to TEMPLATE_DATE. A template that reads
    the clock but offers no date_string override cannot be pinned and is
    rejected.
    """
    template = tokenizer.chat_template
    if not isinstance(template, str):
        raise ValueError(
            "tokenizer does not have a single chat template string, so its date "
            "handling cannot be checked"
        )
    if "date_string" not in template and "strftime_now" not in template:
        return {}
    if "date_string" not in template:
        raise ValueError(
            "chat template inserts the current date with no date_string override, "
            "so it cannot be pinned"
        )
    return {"date_string": TEMPLATE_DATE}


def render_conversation(tokenizer, sentence: str, extra: dict[str, str]) -> tuple[str, str]:
    """(prefix, full) template text for the sentence as the assistant's reply.

    prefix is the conversation up to and including the assistant header;
    full adds the sentence and whatever the template closes the turn with.
    """
    user = {"role": "user", "content": USER_TURN}
    prefix = tokenizer.apply_chat_template(
        [user], add_generation_prompt=True, tokenize=False, **extra
    )
    full = tokenizer.apply_chat_template(
        [user, {"role": "assistant", "content": sentence}], tokenize=False, **extra
    )
    return prefix, full


def sentence_span_inputs(
    tokenizer, sentence: str, extra: dict[str, str]
) -> tuple[torch.Tensor, int, int]:
    """Template the sentence as the assistant turn and locate its own tokens.

    Returns (input_ids shaped (1, seq), start, end), where input_ids[0, start:end]
    are exactly the sentence's tokens: everything before (system text, user
    turn, assistant header) and after (end of turn) is outside the span.

    Fails loudly unless all of these hold, since a misplaced span would read the
    wrong tokens without any visible error:
    - the full text starts with the prefix text, and the sentence follows the
      assistant header directly;
    - the token ids start with the prefix's own token ids, and the sentence's
      first token comes right after them, so no token straddles the boundary;
    - the sentence's tokens are contiguous and cover its characters exactly;
    - the template closes the turn with at least one token after the sentence;
    - the ids match what apply_chat_template itself produces.
    """
    if not tokenizer.is_fast:
        raise ValueError("locating the sentence span needs a fast tokenizer (offset mapping)")
    prefix, full = render_conversation(tokenizer, sentence, extra)

    if not full.startswith(prefix):
        raise ValueError(f"templated text does not start with the expected prefix for: {sentence!r}")
    start_char = len(prefix)
    end_char = start_char + len(sentence)
    if full[start_char:end_char] != sentence:
        raise ValueError(
            f"chat template does not place the sentence directly after the "
            f"assistant header: {sentence!r}"
        )
    if end_char >= len(full):
        raise ValueError(f"chat template does not close the assistant turn after: {sentence!r}")

    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    if ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError(f"token ids do not start with the prefix's token ids for: {sentence!r}")

    inside = [
        i for i, (s, e) in enumerate(offsets)
        if e > s and s >= start_char and e <= end_char
    ]
    if not inside:
        raise ValueError(f"no tokens found inside the sentence span for: {sentence!r}")
    start, end = inside[0], inside[-1] + 1
    if (
        start != len(prefix_ids)
        or inside != list(range(start, end))
        or offsets[start][0] != start_char
        or offsets[end - 1][1] != end_char
    ):
        raise ValueError(
            f"sentence tokens do not line up with the sentence characters for: {sentence!r}"
        )
    if end >= len(ids):
        raise ValueError(f"no end-of-turn token after the sentence for: {sentence!r}")

    expected = tokenizer.apply_chat_template(
        [{"role": "user", "content": USER_TURN}, {"role": "assistant", "content": sentence}],
        tokenize=True, return_dict=True, **extra,
    )["input_ids"]
    if list(expected) != ids:
        raise ValueError(
            f"tokenizing the templated text differs from apply_chat_template for: {sentence!r}"
        )
    return torch.tensor([ids]), start, end


def sentence_activations(
    model, tokenizer, sentence: str, captured: list, extra: dict[str, str]
) -> torch.Tensor:
    """Run one forward pass and return the layer's output over the sentence's tokens.

    Returns a (sentence tokens, hidden) float32 tensor on the CPU. The caller
    takes the mean for the pooled activation. One prompt per forward pass, so
    no padding is involved.
    """
    input_ids, start, end = sentence_span_inputs(tokenizer, sentence, extra)
    input_ids = input_ids.to(model.device)

    captured[0] = None
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    if captured[0] is None:
        raise RuntimeError("forward hook captured nothing; check the layer index")

    # (batch, seq, hidden) -> the sentence's positions in the only sequence.
    return captured[0][0, start:end, :].detach().to("cpu", torch.float32)


# --- 5. Difference in means per category --------------------------------------

def difference_in_means(
    model,
    tokenizer,
    pairs: dict[str, list[dict[str, str]]],
    categories: list[str],
    layer: int,
    extra: dict[str, str],
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, dict[str, torch.Tensor]],
    dict[str, dict[str, list[torch.Tensor]]],
]:
    """Compute one steering direction per category.

    Returns (vectors, raw, tokens) where vectors[category] is
    mean(emotion activations) minus mean(neutral activations);
    raw[category]['emotion'|'neutral'] stacks one pooled vector per prompt
    (the mean over that sentence's tokens), shaped (pairs, hidden), so
    scripts/analyze_extraction.py reads it unchanged; and
    tokens[category][side] lists each prompt's (sentence tokens, hidden)
    activations, so another pooling can be checked without a forward pass.
    """
    vectors: dict[str, torch.Tensor] = {}
    raw: dict[str, dict[str, torch.Tensor]] = {}
    tokens: dict[str, dict[str, list[torch.Tensor]]] = {}

    with residual_stream_hook(model, layer) as captured:
        for category in categories:
            per_token: dict[str, list[torch.Tensor]] = {"emotion": [], "neutral": []}
            for pair in pairs[category]:
                for side in ("emotion", "neutral"):
                    per_token[side].append(
                        sentence_activations(model, tokenizer, pair[side], captured, extra)
                    )
            emotion = torch.stack([t.mean(dim=0) for t in per_token["emotion"]])
            neutral = torch.stack([t.mean(dim=0) for t in per_token["neutral"]])
            vectors[category] = emotion.mean(dim=0) - neutral.mean(dim=0)
            raw[category] = {"emotion": emotion, "neutral": neutral}
            tokens[category] = per_token
    return vectors, raw, tokens


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
    tokens: dict[str, dict[str, list[torch.Tensor]]],
    provenance: dict,
) -> None:
    """Write vectors, pooled and per-token activations, and provenance.json."""
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(vectors, run_dir / "steering_vectors.pt")
    torch.save(raw, run_dir / "raw_activations.pt")
    torch.save(tokens, run_dir / "token_activations.pt")
    with (run_dir / "provenance.json").open("w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2, sort_keys=True)
        fh.write("\n")


# --- 7. Run entry point -------------------------------------------------------

def run_extraction(
    model_id: str | None = None,
    seed: int = DEFAULT_SEED,
    prompts_path: Path = PROMPTS_CONFIG,
    experiment_dir: Path = EXPERIMENT_001_DIR,
) -> Path:
    """Extract all eight steering vectors and save one timestamped run folder.

    The layer is not a parameter: it is fixed at half the model's decoder
    depth (Amendment 1). Returns the run folder path. Prints one line per
    category (vector norm and pair count) so a run can be eyeballed for sanity
    without opening the files.
    """
    entry, inference = resolve_model(model_id)
    resolved_id = entry["id"]
    layer = fixed_layer(entry)
    quantization = inference.get("default_quantization", "nf4")
    pairs, categories = load_prompt_pairs(prompts_path)

    # Checked before the model loads, so a changed prompts file fails fast.
    prompts_sha256 = file_sha256(prompts_path)
    if prompts_sha256 != PATH_A_PROMPTS_SHA256:
        raise ValueError(
            f"{prompts_path} differs from the prompts the Path A run used "
            f"(sha256 {prompts_sha256}, expected {PATH_A_PROMPTS_SHA256}); "
            f"Amendment 1 requires the sentences unchanged"
        )
    set_seed(seed)

    started = datetime.now(timezone.utc)
    print(f"model: {resolved_id} ({quantization}), layer {layer}, seed {seed}")
    model, tokenizer = load_model(resolved_id, quantization)
    try:
        check_depth(model, entry)
        extra = template_kwargs(tokenizer)
        # One rendered example, so the exact context the model saw is on record.
        rendered_example = render_conversation(
            tokenizer, pairs[categories[0]][0]["emotion"], extra
        )[1]
        vectors, raw, tokens = difference_in_means(
            model, tokenizer, pairs, categories, layer, extra
        )
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
        "layer_rule": "half the decoder depth (Experiment 001 Amendment 1)",
        "decoder_layers": entry["decoder_layers"],
        "quantization": {
            "scheme": quantization,
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_use_double_quant": False,
        },
        "method": "difference_in_means",
        "extraction_convention": "path_b (Experiment 001 Amendment 1)",
        "token_position": (
            "mean over the assistant-turn tokens of the sentence "
            "(template and end-of-turn tokens excluded)"
        ),
        "prompt_template": "chat template, fixed user turn, sentence as the assistant turn",
        "user_turn": USER_TURN,
        "template_date": extra.get("date_string"),
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest(),
        "rendered_example": rendered_example,
        "seed": seed,
        "categories": categories,
        "pairs_per_category": {c: len(p) for c, p in pairs.items()},
        "vector_norms": {c: float(v.norm()) for c, v in vectors.items()},
        "hidden_size": int(next(iter(vectors.values())).shape[0]),
        "prompts_file": str(prompts_path.relative_to(ROOT)),
        "prompts_sha256": prompts_sha256,
        "prompts_match_path_a": True,  # enforced above; the run stops otherwise
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
    save_run(run_dir, vectors, raw, tokens, provenance)

    for category in categories:
        print(
            f"  {category:<13} norm {vectors[category].norm():8.3f}  "
            f"n_pairs {len(pairs[category]):3d}"
        )
    print(f"saved to {run_dir}")
    return run_dir
