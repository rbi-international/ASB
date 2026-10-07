"""Judge validation slice (docs/judge_protocol.md, Sections 5, 8, 9 and 11).

Usage, in this order:
  python scripts/validation.py sample --source <evaluation folder>
      Draw the label sets and the test-retest subset from a full (not dry-run)
      evaluation folder's generations. Writes two blind label sheets and a
      separate key file. Costs nothing.
  python scripts/validation.py export --run <run>
      Refresh labelling_instructions.txt to the current judge prompt and write
      validation_sheet_annotator2.csv for the second annotator (Protocol
      Amendments 2 and 3). <run> is the vector run folder name.
  (label pilot_sheet.csv, then)
  python scripts/validation.py lock --run <run> --sheet pilot
  python scripts/validation.py judge --run <run> --stage pilot [--paid]
      Judge the 40 pilot items at each effort level; the protocol's rule picks
      the effort.
  (label validation_sheet.csv, then)
  python scripts/validation.py lock --run <run> --sheet validation
  python scripts/validation.py judge --run <run> --stage validation [--paid]
      Judge the 100 validation items once and the 200 test-retest items twice,
      at the chosen effort.
  python scripts/validation.py lock --run <run> --sheet validation2
      Lock the second annotator's completed sheet.
  python scripts/validation.py relabel-sheet --run <run>
      At least 3 days after the author's later lock: release the 20-item
      re-label sheet, then 'lock --sheet relabel' once it is labelled.
  python scripts/validation.py report --run <run> [--paid]
      Agreement, effort choice, test-retest, second annotator, intra-rater and
      the go/no-go table.

Without --paid the stub judge is used, so the whole workflow can be tested for
free; its numbers are never results. Every command refuses to run out of order.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # src layout, no install step needed

from asb.evaluation import (  # noqa: E402
    DEFINITIONS, DRY_RUN_ROOT, JUDGE_EFFORTS, JUDGE_PROMPT_VERSION, JUDGE_TEMPLATE, LABELS,
    FableJudge, Judge, StubJudge, append_jsonl, build_request, final_status, load_categories,
    now_utc, read_jsonl, repair_jsonl, sha256_text, verify_protocol_text, write_json_atomic,
)
from asb.extraction import file_sha256  # noqa: E402

# --- 1. Registered sizes and thresholds (protocol, Sections 5, 8, 9, 11) ----------

SAMPLE_SEED = 20261007
VALIDATION_PER_CATEGORY = 10   # each basic category
VALIDATION_SUBTLE = 20         # trust and anticipation (Protocol Amendment 1); 100 hand labels in all
PILOT_PER_CATEGORY = 5         # 40 hand labels, disjoint from the validation set
RETEST_PER_CATEGORY = 25       # 200 outputs judged twice
PILOT_TOLERANCE = 0.05         # lowest effort within 5 points of the best
THRESHOLDS = {
    "valid_response_rate_min": 0.99,
    "refusal_rate_max": 0.01,
    "refusal_cell_flag": 0.02,
    "kappa_min": 0.6,
    "retest_kappa_min": 0.7,
}
SUBTLE = ("trust", "anticipation")

# Protocol Amendment 3: second annotator and intra-rater check.
RELABEL_N = 20                   # of the 140 items the author labelled
RELABEL_SEED = 20261008
RELABEL_MIN_DAYS = 3             # after the later of the author's two locks
ANNOTATOR2_ORDER_SEED = 20261009  # row order of the second annotator's sheet
SHEETS = {  # sheet name -> (file name, item-id prefix)
    "pilot": ("pilot_sheet.csv", "P"),
    "validation": ("validation_sheet.csv", "V"),
    "validation2": ("validation_sheet_annotator2.csv", "V"),
    "relabel": ("relabel_sheet.csv", "R"),
}

VALIDATION_ROOT = ROOT / "experiments" / "experiment_001_baseline_replication" / "runs" / "validation"
KEY_NAME = "KEY_do_not_open_until_labels_locked.json"


def folder(source_run: str) -> Path:
    return VALIDATION_ROOT / source_run


# --- 2. Sampling -------------------------------------------------------------------

def stratified_draw(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Draw n items spread across strengths: shuffle each strength's items, then
    take them round-robin, so no strength dominates a category's sample."""
    by_strength: dict[float, list[dict]] = defaultdict(list)
    for g in sorted(pool, key=lambda g: g["gen_id"]):
        by_strength[g["strength"]].append(g)
    for items in by_strength.values():
        rng.shuffle(items)
    order = sorted(by_strength)
    drawn: list[dict] = []
    while len(drawn) < n and any(by_strength.values()):
        for s in order:
            if by_strength[s] and len(drawn) < n:
                drawn.append(by_strength[s].pop())
    if len(drawn) < n:
        raise ValueError(f"pool too small: wanted {n}, found {len(drawn)}")
    return drawn


def write_sheet(path: Path, items: list[tuple[str, dict]]) -> None:
    """The blind sheet: item id and text only, plus two empty columns to fill in.

    UTF-8 with a byte-order mark, so Excel opens it with the right encoding.
    """
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_ALL)
        writer.writerow(["item_id", "text", "emotion", "coherence"])
        for item_id, g in items:
            writer.writerow([item_id, g["output"], "", ""])


def labelling_instructions() -> str:
    """The judge's own instructions, so the author labels by the same rules (Section 9)."""
    listing = "\n".join(DEFINITIONS.values())
    body = JUDGE_TEMPLATE.replace("{CATEGORY_LIST}", listing).replace(
        "<text>\n{TEXT}\n</text>", "<text>\n(the text column of the sheet)\n</text>")
    return (
        "Label each row of the sheet using exactly the instructions below.\n"
        "In the emotion column write one of: " + ", ".join(LABELS) + ".\n"
        "In the coherence column write a whole number from 1 to 5.\n"
        "Do not open the key file until both sheets are locked.\n"
        "Save as 'CSV UTF-8' if you edit in Excel.\n\n" + body + "\n"
    )


def cmd_sample(source: Path) -> None:
    """Draw the validation, pilot and retest sets, disjoint, with a fixed seed."""
    source = source.resolve()
    if DRY_RUN_ROOT.resolve() in source.parents:
        raise ValueError("sample from a full evaluation folder, not a dry run")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest["slice"]["prompts_per_split"] is not None:
        raise ValueError(f"{source} is not a full run")
    out = folder(manifest["vector_run"])
    if (out / "items.json").exists():
        raise ValueError(f"{out} already has drawn items; they are drawn once and never redrawn")
    out.mkdir(parents=True, exist_ok=True)

    categories, _ = load_categories()
    gens = {g["gen_id"]: g for g in read_jsonl(source / "generations.jsonl")}
    rng = random.Random(SAMPLE_SEED)
    sets: dict[str, list[dict]] = {"validation": [], "pilot": [], "retest": []}
    for c in categories:
        pool = [g for g in gens.values() if g["kind"] == "steered" and g["category"] == c]
        n_val = VALIDATION_SUBTLE if c in SUBTLE else VALIDATION_PER_CATEGORY
        drawn = stratified_draw(pool, n_val + PILOT_PER_CATEGORY + RETEST_PER_CATEGORY, rng)
        sets["validation"] += drawn[:n_val]
        sets["pilot"] += drawn[n_val:n_val + PILOT_PER_CATEGORY]
        sets["retest"] += drawn[n_val + PILOT_PER_CATEGORY:]

    key, sheets = {}, {}
    for name, prefix in (("validation", "V"), ("pilot", "P")):
        shuffled = sets[name][:]
        rng.shuffle(shuffled)  # mix categories and strengths on the sheet
        sheets[name] = [(f"{prefix}{i:03d}", g) for i, g in enumerate(shuffled, 1)]
        write_sheet(out / f"{name}_sheet.csv", sheets[name])
        for item_id, g in sheets[name]:
            key[item_id] = {k: g[k] for k in ("gen_id", "output_sha256", "split", "kind",
                                              "category", "strength", "prompt_id", "seed")}
    write_json_atomic(out / KEY_NAME, key)
    (out / "labelling_instructions.txt").write_text(labelling_instructions(), encoding="utf-8")
    write_json_atomic(out / "items.json", {
        "source": str(source), "source_manifest": manifest, "seed": SAMPLE_SEED,
        "sets": {name: [g["gen_id"] for g in items] for name, items in sets.items()},
        "generations": {g["gen_id"]: g for items in sets.values() for g in items},
        "created_utc": now_utc(),
    })
    print(f"drew {len(sets['validation'])} validation, {len(sets['pilot'])} pilot and "
          f"{len(sets['retest'])} retest items into {out}")
    print("Label pilot_sheet.csv first, following labelling_instructions.txt. "
          f"Do not open {KEY_NAME}.")


# --- 3. Locking the labels -------------------------------------------------------------

def load_items(out: Path) -> dict:
    path = out / "items.json"
    if not path.exists():
        raise ValueError("no drawn items; run 'sample' first")
    return json.loads(path.read_text(encoding="utf-8"))


def locked_path(out: Path, sheet: str) -> Path:
    return out / f"labels_{sheet}.locked.json"


def cmd_lock(out: Path, sheet: str) -> None:
    """Validate a completed sheet and freeze it, with its hash. Never re-locked."""
    target = locked_path(out, sheet)
    if target.exists():
        raise ValueError(f"{target.name} is already locked and cannot change")
    key = json.loads((out / KEY_NAME).read_text(encoding="utf-8"))
    filename, prefix = SHEETS[sheet]
    expected = {k for k in key if k.startswith(prefix)}
    path = out / filename
    if not path.exists():
        raise ValueError(f"{filename} does not exist yet")
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    generations = load_items(out)["generations"]
    errors, labels, edited, line_endings = [], {}, [], []
    seen = [r["item_id"].strip() for r in rows]
    if sorted(seen) != sorted(expected):
        errors.append("item ids differ from the drawn set (rows added, removed or renamed)")
    for r in rows:
        item = r["item_id"].strip()
        emotion = r["emotion"].strip().lower()
        coherence = r["coherence"].strip()
        if emotion not in LABELS:
            errors.append(f"{item}: emotion {r['emotion']!r} is not one of {', '.join(LABELS)}")
        if coherence not in {"1", "2", "3", "4", "5"}:
            errors.append(f"{item}: coherence {r['coherence']!r} is not a whole number 1 to 5")
        # Raw text first: drawn outputs can themselves contain \r\n, so normalizing
        # before comparing would flag identical text. Line endings are compared
        # only once the raw text is known to differ.
        if item in key and sha256_text(r["text"]) != key[item]["output_sha256"]:
            drawn = generations[key[item]["gen_id"]]["output"]
            unify = lambda s: s.replace("\r\n", "\n").replace("\r", "\n")  # noqa: E731
            (line_endings if unify(r["text"]) == unify(drawn) else edited).append(item)
        labels[item] = {"emotion": emotion, "coherence": int(coherence) if coherence.isdigit() else None}
    if errors:
        raise ValueError(f"{path.name} is not ready to lock:\n  " + "\n  ".join(errors))
    if line_endings:
        print(f"note: the text differs from the drawn output only in line endings for "
              f"{len(line_endings)} items: {line_endings[:10]}")
    if edited:
        print(f"warning: the text differs from the drawn output beyond line endings for "
              f"{len(edited)} items: {edited[:10]}")

    write_json_atomic(target, {"sheet": sheet, "sheet_sha256": file_sha256(path),
                               "labels": labels, "locked_utc": now_utc()})
    print(f"locked {len(labels)} {sheet} labels (sheet sha256 {file_sha256(path)[:16]}...)")


def locked_labels(out: Path, sheet: str) -> dict[str, dict]:
    """Labels by generation id. Fails unless the sheet is locked."""
    path = locked_path(out, sheet)
    if not path.exists():
        raise ValueError(f"the {sheet} labels are not locked; label the sheet and run 'lock --sheet {sheet}'")
    key = json.loads((out / KEY_NAME).read_text(encoding="utf-8"))
    labels = json.loads(path.read_text(encoding="utf-8"))["labels"]
    return {key[item]["gen_id"]: v for item, v in labels.items()}


def optional_labels(out: Path, sheet: str) -> dict[str, dict] | None:
    """Locked labels by generation id, or None if that sheet is not locked yet."""
    return locked_labels(out, sheet) if locked_path(out, sheet).exists() else None


# --- 3b. Extra sheets (Protocol Amendments 2 and 3) ------------------------------------

def cmd_export(out: Path) -> None:
    """Refresh the labelling instructions and write the second annotator's sheet.

    The items were drawn once and are not redrawn; this only adds outputs.
    Instructions written before Protocol Amendment 2 are kept as a record.
    """
    items = load_items(out)
    current = labelling_instructions()
    path = out / "labelling_instructions.txt"
    if path.exists() and path.read_text(encoding="utf-8") != current:
        old = out / "labelling_instructions_v1.txt"
        if not old.exists():
            path.replace(old)
            print(f"kept the earlier instructions as {old.name}")
    path.write_text(current, encoding="utf-8")
    print(f"wrote {path.name} (judge prompt version {JUDGE_PROMPT_VERSION[:16]}...)")

    sheet2 = out / SHEETS["validation2"][0]
    if sheet2.exists():
        print(f"{sheet2.name} already exists; left unchanged")
        return
    key = json.loads((out / KEY_NAME).read_text(encoding="utf-8"))
    rows = [(item, items["generations"][meta["gen_id"]])
            for item, meta in sorted(key.items()) if item.startswith("V")]
    random.Random(ANNOTATOR2_ORDER_SEED).shuffle(rows)  # a different row order from the author's
    write_sheet(sheet2, rows)
    print(f"wrote {sheet2.name}: {len(rows)} items. Send the second annotator only this file and "
          f"{path.name}; never the key, the author's sheet, or any judge output.")


def cmd_relabel_sheet(out: Path) -> None:
    """Release the intra-rater sheet: 20 of the author's 140 items, under new ids.

    Only after both of the author's sheets are locked and at least 3 days have
    passed since the later lock, so the re-labels are not made from memory.
    The 20 are fixed by seed, so the choice cannot depend on the labels.
    """
    target = out / SHEETS["relabel"][0]
    if target.exists():
        raise ValueError(f"{target.name} already exists and is never redrawn")
    locks = []
    for sheet in ("pilot", "validation"):
        if not locked_path(out, sheet).exists():
            raise ValueError(f"lock the {sheet} sheet first")
        locks.append(json.loads(locked_path(out, sheet).read_text(encoding="utf-8"))["locked_utc"])
    from datetime import datetime, timedelta
    ready = max(datetime.fromisoformat(t) for t in locks) + timedelta(days=RELABEL_MIN_DAYS)
    if datetime.fromisoformat(now_utc()) < ready:
        raise ValueError(f"the re-label sheet is released from {ready.isoformat()} "
                         f"({RELABEL_MIN_DAYS} days after the later lock)")

    items = load_items(out)
    key = json.loads((out / KEY_NAME).read_text(encoding="utf-8"))
    rng = random.Random(RELABEL_SEED)
    chosen = rng.sample(sorted(k for k in key if k[0] in "PV"), RELABEL_N)
    rng.shuffle(chosen)
    rows = []
    for i, original in enumerate(chosen, 1):
        item = f"R{i:03d}"
        key[item] = {**key[original], "relabel_of": original}
        rows.append((item, items["generations"][key[original]["gen_id"]]))
    write_json_atomic(out / KEY_NAME, key)
    write_sheet(target, rows)
    print(f"wrote {target.name}: {len(rows)} items under new ids. Label them without "
          f"looking at your earlier sheets, then run 'lock --sheet relabel'.")


# --- 4. Judging, with one retry (protocol, Section 7) --------------------------------

def judgments_path(out: Path, backend: str) -> Path:
    (out / backend).mkdir(parents=True, exist_ok=True)
    return out / backend / "judgments.jsonl"


def state_for(path: Path, gens: list[dict], backend: str, pass_name: str) -> dict[str, list[dict]]:
    by_id = {g["gen_id"]: g for g in gens}
    state: dict[str, list[dict]] = defaultdict(list)
    for r in read_jsonl(path):
        g = by_id.get(r["gen_id"])
        if g and r["backend"] == backend and r["pass"] == pass_name \
                and r["output_sha256"] == g["output_sha256"]:
            state[r["gen_id"]].append(r)
    return state


def judge_with_retry(gens: list[dict], pass_name: str, judge: Judge, path: Path) -> None:
    repair_jsonl(path)
    sink = lambda records: append_jsonl(path, records)  # noqa: E731
    judge.resume(sink)
    for attempt, wanted in ((1, "todo"), (2, "retry")):
        state = state_for(path, gens, judge.backend, pass_name)
        todo = [build_request(g, pass_name) for g in gens
                if final_status(state.get(g["gen_id"], [])) == wanted]
        print(f"{pass_name}, attempt {attempt}: {len(todo)} to judge")
        judge.judge(todo, attempt, sink)


def make_judge(paid: bool, effort: str, out: Path) -> Judge:
    if not paid:
        return StubJudge()
    return FableJudge(effort, out / "fable" / f"pending_{effort}.json")


def verdicts_for(path: Path, gens: list[dict], backend: str, pass_name: str) -> dict[str, dict]:
    state = state_for(path, gens, backend, pass_name)
    out = {}
    for gid, records in state.items():
        ok = [r for r in records if r["status"] == "ok"]
        if ok:
            out[gid] = ok[0]
    return out


def pilot_choice(out: Path, backend: str) -> dict:
    """The protocol's rule: the lowest effort within 5 points of the best one tried."""
    items = load_items(out)
    gens = [items["generations"][g] for g in items["sets"]["pilot"]]
    labels = locked_labels(out, "pilot")
    path = judgments_path(out, backend)
    agreement = {}
    for effort in JUDGE_EFFORTS:
        v = verdicts_for(path, gens, backend, f"pilot-{effort}")
        pairs = [(labels[g]["emotion"], v[g]["emotion"]) for g in labels if g in v]
        if len(pairs) < len(gens):
            raise ValueError(f"pilot at effort {effort} is not fully judged; run 'judge --stage pilot'")
        agreement[effort] = sum(a == b for a, b in pairs) / len(pairs)
    best = max(agreement.values())
    chosen = next(e for e in JUDGE_EFFORTS if agreement[e] >= best - PILOT_TOLERANCE)
    return {"agreement": agreement, "chosen": chosen, "tolerance": PILOT_TOLERANCE}


def cmd_judge(out: Path, stage: str, paid: bool) -> None:
    verify_protocol_text()
    items = load_items(out)
    backend = "fable" if paid else "stub"
    path = judgments_path(out, backend)
    pick = lambda name: [items["generations"][g] for g in items["sets"][name]]  # noqa: E731
    if not paid:
        print("stub judge: verdicts are fake and nothing here is a result")

    if stage == "pilot":
        locked_labels(out, "pilot")  # labels must be frozen before any judging of these items
        for effort in JUDGE_EFFORTS:
            judge_with_retry(pick("pilot"), f"pilot-{effort}", make_judge(paid, effort, out), path)
        choice = pilot_choice(out, backend)
        write_json_atomic(out / backend / "pilot_choice.json", {**choice, "created_utc": now_utc()})
        print(f"pilot agreement {choice['agreement']}; chosen effort: {choice['chosen']}")
        if paid:
            print("Record the chosen effort and these pilot results in docs/judge_protocol.md "
                  "(Section 5 requires it) before running the validation stage.")
        return

    locked_labels(out, "validation")
    choice_path = out / backend / "pilot_choice.json"
    if not choice_path.exists():
        raise ValueError("run the pilot stage first; it fixes the effort")
    effort = json.loads(choice_path.read_text(encoding="utf-8"))["chosen"]
    print(f"using the pilot's chosen effort: {effort}")
    judge = make_judge(paid, effort, out)
    judge_with_retry(pick("validation"), "validation", judge, path)
    judge_with_retry(pick("retest"), "retest-1", judge, path)
    judge_with_retry(pick("retest"), "retest-2", judge, path)


# --- 5. Agreement statistics --------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def cohen_kappa(a: list, b: list, labels: list, quadratic: bool = False) -> float | None:
    """Cohen's kappa; with quadratic=True, the weighted form for ordinal scores."""
    if not a:
        return None
    k = len(labels)
    index = {lab: i for i, lab in enumerate(labels)}
    observed = np.zeros((k, k))
    for x, y in zip(a, b):
        observed[index[x], index[y]] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    i, j = np.indices((k, k))
    weights = ((i - j) ** 2) / ((k - 1) ** 2) if quadratic else (i != j).astype(float)
    denom = (weights * expected).sum()
    return None if denom == 0 else float(1 - (weights * observed).sum() / denom)


def fmt(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


# --- 6. Report ---------------------------------------------------------------------------

def cmd_report(out: Path, paid: bool) -> None:
    items = load_items(out)
    backend = "fable" if paid else "stub"
    path = judgments_path(out, backend)
    categories, groups = load_categories()
    gens = items["generations"]
    labels = locked_labels(out, "validation")
    val = verdicts_for(path, [gens[g] for g in items["sets"]["validation"]], backend, "validation")
    retest_gens = [gens[g] for g in items["sets"]["retest"]]
    r1 = verdicts_for(path, retest_gens, backend, "retest-1")
    r2 = verdicts_for(path, retest_gens, backend, "retest-2")
    choice_path = out / backend / "pilot_choice.json"
    choice = json.loads(choice_path.read_text(encoding="utf-8")) if choice_path.exists() else None

    # Agreement with the author's labels.
    judged = [g for g in labels if g in val]
    human = [labels[g]["emotion"] for g in judged]
    judge_ = [val[g]["emotion"] for g in judged]
    kappa = cohen_kappa(human, judge_, list(LABELS))
    percent = sum(a == b for a, b in zip(human, judge_)) / len(judged) if judged else None
    coh_kappa = cohen_kappa([labels[g]["coherence"] for g in judged],
                            [val[g]["coherence"] for g in judged], [1, 2, 3, 4, 5], quadratic=True)
    per_cat = {}
    for c in categories:
        h_c = [g for g in judged if labels[g]["emotion"] == c]
        j_c = [g for g in judged if val[g]["emotion"] == c]
        hit = sum(val[g]["emotion"] == c for g in h_c)
        prec = sum(labels[g]["emotion"] == c for g in j_c)
        # SSR's own question on c's drawn items: does the text express c, yes or no?
        # (Protocol Amendment 1: this is criterion 4; recall is secondary.)
        stratum = [g for g in judged if gens[g]["category"] == c]
        agree = sum((labels[g]["emotion"] == c) == (val[g]["emotion"] == c) for g in stratum)
        per_cat[c] = {
            "binary_agreement": agree / len(stratum) if stratum else None,
            "binary_ci": wilson(agree, len(stratum)), "n_stratum": len(stratum),
            "recall": hit / len(h_c) if h_c else None, "recall_ci": wilson(hit, len(h_c)),
            "n_human": len(h_c),
            "precision": prec / len(j_c) if j_c else None, "n_judge": len(j_c),
        }

    # Test-retest.
    both = [g["gen_id"] for g in retest_gens if g["gen_id"] in r1 and g["gen_id"] in r2]
    retest_kappa = cohen_kappa([r1[g]["emotion"] for g in both], [r2[g]["emotion"] for g in both], list(LABELS))
    retest_cat = {}
    for c in categories:
        ids = [g for g in both if gens[g]["category"] == c]
        a, b = [r1[g]["emotion"] for g in ids], [r2[g]["emotion"] for g in ids]
        retest_cat[c] = {
            "agreement": sum(x == y for x, y in zip(a, b)) / len(ids) if ids else None,
            "kappa": cohen_kappa(a, b, list(LABELS)),
            "coherence_exact": (sum(r1[g]["coherence"] == r2[g]["coherence"] for g in ids) / len(ids)) if ids else None,
            "coherence_mad": (sum(abs(r1[g]["coherence"] - r2[g]["coherence"]) for g in ids) / len(ids)) if ids else None,
        }

    # Every call in the slice, for response validity and refusals.
    calls = [r for r in read_jsonl(path) if r["backend"] == backend]
    n_calls = len(calls)
    valid_rate = sum(r["status"] == "ok" for r in calls) / n_calls if n_calls else None
    refusals = [r for r in calls if r["failure_type"] == "refusal"]
    refusal_rate = len(refusals) / n_calls if n_calls else None
    calls_by_cat = Counter(gens[r["gen_id"]]["category"] for r in calls if r["gen_id"] in gens)
    refusals_by_cat = Counter(gens[r["gen_id"]]["category"] for r in refusals if r["gen_id"] in gens)
    worst_cell = max((refusals_by_cat[c] / calls_by_cat[c] for c in calls_by_cat), default=0.0)

    # Go/no-go, the subtle-category check first (protocol, Section 11, as changed
    # by Protocol Amendment 1, and by Protocol Amendment 3: it must pass against
    # each annotator separately, the blind second annotator included).
    checks = []
    for annotator, annotator_labels in (("author", labels),
                                        ("annotator 2", optional_labels(out, "validation2"))):
        for c, ok, detail in criterion4_rows(annotator_labels, val, gens, categories, groups):
            checks.append((f"4. {c} binary agreement not clearly below the basic categories, "
                           f"vs {annotator}", ok, detail))
    checks += [
        ("3. overall emotion kappa >= 0.6", None if kappa is None else kappa >= THRESHOLDS["kappa_min"],
         f"kappa {fmt(kappa)}, agreement {fmt(percent)}, n {len(judged)}"),
        ("5. test-retest emotion kappa >= 0.7",
         None if retest_kappa is None else retest_kappa >= THRESHOLDS["retest_kappa_min"],
         f"kappa {fmt(retest_kappa)}, n {len(both)}"),
        ("1. schema-valid responses >= 99%",
         None if valid_rate is None else valid_rate >= THRESHOLDS["valid_response_rate_min"],
         f"{fmt(valid_rate, 3)} of {n_calls} calls"),
        ("2. refusals < 1% overall, no category above 2%",
         None if refusal_rate is None else (refusal_rate < THRESHOLDS["refusal_rate_max"]
                                            and worst_cell <= THRESHOLDS["refusal_cell_flag"]),
         f"{len(refusals)} refusals ({fmt(refusal_rate, 3)}); worst category {fmt(worst_cell, 3)}"),
    ]
    verdict = ("GO" if all(ok is True for _, ok, _ in checks) else "NO-GO")

    lines = [
        "# Judge validation report",
        "",
        "Real results: Claude Fable 5.1." if paid else
        "NOT A RESULT. Stub judge: fake verdicts, for testing this workflow only.",
        f"Judge prompt version {JUDGE_PROMPT_VERSION[:16]}..., built {now_utc()}.",
        "",
        f"## Go/no-go: {verdict}",
        "",
        "| Check | Pass | Detail |",
        "|---|---|---|",
        *[f"| {name} | {'yes' if ok else ('n/a' if ok is None else 'NO')} | {detail} |"
          for name, ok, detail in checks],
        "",
        "A check that cannot be assessed counts as not passed.",
        "",
        "## Effort pilot",
        "",
        ("not run" if choice is None else
         "agreement by effort: " + ", ".join(f"{e} {fmt(a)}" for e, a in choice["agreement"].items())
         + f"; chosen: {choice['chosen']} (lowest within {choice['tolerance']:.0%} of the best)"),
        "",
        "## Agreement with the author's labels (validation set)",
        "",
        f"Overall: kappa {fmt(kappa)}, percent agreement {fmt(percent)}, coherence weighted kappa "
        f"{fmt(coh_kappa)}, n {len(judged)}.",
        "",
        "Primary (criterion 4): binary agreement on each category's drawn items (20 for trust "
        "and anticipation, 10 for each basic category), on SSR's "
        "question (does the text express the target emotion, yes or no).",
        "",
        "| Category | Binary agreement | 95% CI | n |",
        "|---|---|---|---|",
        *[f"| {c} | {fmt(v['binary_agreement'])} | {fmt(v['binary_ci'][0])} to "
          f"{fmt(v['binary_ci'][1])} | {v['n_stratum']} |" for c, v in per_cat.items()],
        "",
        "Secondary, never used for go/no-go (Protocol Amendment 1). Recall is out of the items "
        "the author labelled with that emotion, so it depends on how often outputs read as it.",
        "",
        "| Category | Recall | 95% CI | n (author) | Precision | n (judge) |",
        "|---|---|---|---|---|---|",
        *[f"| {c} | {fmt(v['recall'])} | {fmt(v['recall_ci'][0])} to {fmt(v['recall_ci'][1])} | "
          f"{v['n_human']} | {fmt(v['precision'])} | {v['n_judge']} |" for c, v in per_cat.items()],
        "",
        "## Test-retest (second judgment of the same requests)",
        "",
        f"Overall emotion kappa {fmt(retest_kappa)}, n {len(both)}.",
        "",
        "| Target category | Emotion agreement | Kappa | Coherence exact | Coherence mean abs diff |",
        "|---|---|---|---|---|",
        *[f"| {c} | {fmt(v['agreement'])} | {fmt(v['kappa'])} | {fmt(v['coherence_exact'])} | "
          f"{fmt(v['coherence_mad'])} |" for c, v in retest_cat.items()],
        "",
        *second_annotator_section(out, categories, gens, labels, val),
        *intra_rater_section(out),
        "## Cross-judge (GPT-6 Astra High)",
        "",
        "Pre-registered secondary analysis (Protocol Amendment 3); not run yet. Never ground "
        "truth, never used for go/no-go.",
        "",
    ]
    report = out / backend / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:16 + len(checks)]))
    print(f"full report: {report}")


def criterion4_rows(annotator: dict | None, judged: dict, gens: dict, categories: list[str],
                    groups: dict[str, str]) -> list[tuple[str, bool | None, str]]:
    """Criterion 4 against one annotator: (category, passed or None, detail) for trust and
    anticipation. None (not assessable) when that annotator's labels are not locked."""
    if annotator is None:
        return [(c, None, "not assessable: the second annotator's labels are not locked")
                for c in SUBTLE]
    agreement = {}
    for c in categories:
        stratum = [g for g in annotator if g in judged and gens[g]["category"] == c]
        agreement[c] = binary_agreement(stratum, judged, annotator, c)
    basic = [k / n for c, (k, n) in agreement.items() if groups[c] == "basic" and n]
    median_basic = float(np.median(basic)) if basic else None
    rows = []
    for c in SUBTLE:
        k, n = agreement[c]
        lo, hi = wilson(k, n)
        if hi is None or median_basic is None:
            rows.append((c, None, f"not assessable: {n} judged items drawn from {c}"))
        else:
            rows.append((c, hi >= median_basic,
                         f"binary agreement {fmt(k / n)} (95% CI {fmt(lo)} to {fmt(hi)}, n {n}) "
                         f"vs basic median {fmt(median_basic)}"))
    return rows


def binary_agreement(ids: list[str], a: dict, b: dict, c: str) -> tuple[int, int]:
    """(matches, n) on SSR's question for category c: is the label c, yes or no."""
    return sum((a[g]["emotion"] == c) == (b[g]["emotion"] == c) for g in ids), len(ids)


def second_annotator_section(out: Path, categories: list[str], gens: dict,
                             author: dict, judged: dict) -> list[str]:
    """Human-human agreement and the judge against each annotator (Protocol Amendment 3).

    Reported next to criterion 4 for context; go/no-go stays computed against
    the author's labels, as registered.
    """
    second = optional_labels(out, "validation2")
    head = ["## Second annotator (Protocol Amendment 3)", ""]
    if second is None:
        return head + ["Not available yet: the second annotator's sheet is not locked.", ""]

    both = [g for g in author if g in second]
    hh = cohen_kappa([author[g]["emotion"] for g in both], [second[g]["emotion"] for g in both], list(LABELS))
    hh_pct = sum(author[g]["emotion"] == second[g]["emotion"] for g in both) / len(both) if both else None
    with_judge = [g for g in both if g in judged]
    j2 = cohen_kappa([second[g]["emotion"] for g in with_judge],
                     [judged[g]["emotion"] for g in with_judge], list(LABELS))
    j1 = cohen_kappa([author[g]["emotion"] for g in with_judge],
                     [judged[g]["emotion"] for g in with_judge], list(LABELS))
    rows = []
    for c in categories:
        stratum = [g for g in both if gens[g]["category"] == c]
        judged_stratum = [g for g in stratum if g in judged]
        k_hh, n_hh = binary_agreement(stratum, author, second, c)
        k_j1, n_j1 = binary_agreement(judged_stratum, judged, author, c)
        k_j2, n_j2 = binary_agreement(judged_stratum, judged, second, c)
        kappa_c = cohen_kappa([author[g]["emotion"] for g in stratum],
                              [second[g]["emotion"] for g in stratum], list(LABELS))
        lo, hi = wilson(k_hh, n_hh)
        rows.append(f"| {c} | {fmt(k_j1 / n_j1 if n_j1 else None)} | {fmt(k_j2 / n_j2 if n_j2 else None)} | "
                    f"{fmt(k_hh / n_hh if n_hh else None)} ({fmt(lo)} to {fmt(hi)}) | {fmt(kappa_c)} | {n_hh} |")
    return head + [
        f"Author vs second annotator: kappa {fmt(hh)}, percent agreement {fmt(hh_pct)}, n {len(both)}.",
        f"Judge vs author: kappa {fmt(j1)}. Judge vs second annotator: kappa {fmt(j2)} "
        f"(n {len(with_judge)}).",
        "",
        "Binary agreement on SSR's question, per category's drawn items. Context for "
        "criterion 4 only: go/no-go stays computed against the author, and this table never "
        "turns a failed criterion into a pass.",
        "",
        "| Category | Judge vs author | Judge vs annotator 2 | Author vs annotator 2 (95% CI) | "
        "Author vs annotator 2 kappa | n |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
    ]


def intra_rater_section(out: Path) -> list[str]:
    """The author's self-agreement on the 20 re-labelled items (Protocol Amendment 3)."""
    relabel = optional_labels(out, "relabel")
    head = ["## Intra-rater check (Protocol Amendment 3)", ""]
    if relabel is None:
        return head + ["Not available yet: the re-label sheet is not locked.", ""]
    first = {**locked_labels(out, "pilot"), **locked_labels(out, "validation")}
    ids = [g for g in relabel if g in first]
    a, b = [first[g]["emotion"] for g in ids], [relabel[g]["emotion"] for g in ids]
    return head + [
        f"Self-agreement on {len(ids)} items: kappa {fmt(cohen_kappa(a, b, list(LABELS)))}, "
        f"percent agreement {fmt(sum(x == y for x, y in zip(a, b)) / len(ids) if ids else None)}, "
        f"coherence weighted kappa "
        f"{fmt(cohen_kappa([first[g]['coherence'] for g in ids], [relabel[g]['coherence'] for g in ids], [1, 2, 3, 4, 5], quadratic=True))}.",
        "",
    ]


# --- 7. Entry point ------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sample", help="draw the label and retest sets")
    p.add_argument("--source", required=True, help="full evaluation folder (runs/evaluation/<run>/<backend>)")
    for name in ("export", "relabel-sheet", "lock", "judge", "report"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True, help="vector run name the validation folder belongs to")
        if name == "lock":
            p.add_argument("--sheet", required=True, choices=tuple(SHEETS))
        if name == "judge":
            p.add_argument("--stage", required=True, choices=("pilot", "validation"))
        if name in ("judge", "report"):
            p.add_argument("--paid", action="store_true", help="Claude Fable 5.1 (asks before paying)")
    args = parser.parse_args()

    if args.command == "sample":
        cmd_sample(Path(args.source))
    elif args.command == "export":
        cmd_export(folder(args.run))
    elif args.command == "relabel-sheet":
        cmd_relabel_sheet(folder(args.run))
    elif args.command == "lock":
        cmd_lock(folder(args.run), args.sheet)
    elif args.command == "judge":
        cmd_judge(folder(args.run), args.stage, args.paid)
    else:
        cmd_report(folder(args.run), args.paid)


if __name__ == "__main__":
    main()
