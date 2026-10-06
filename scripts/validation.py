"""Judge validation slice (docs/judge_protocol.md, Sections 5, 8, 9 and 11).

Usage, in this order:
  python scripts/validation.py sample --source <evaluation folder>
      Draw the label sets and the test-retest subset from a full (not dry-run)
      evaluation folder's generations. Writes two blind label sheets and a
      separate key file. Costs nothing.
  (label pilot_sheet.csv, then, with <run> the vector run folder name)
  python scripts/validation.py lock --run <run> --sheet pilot
  python scripts/validation.py judge --run <run> --stage pilot [--paid]
      Judge the 40 pilot items at each effort level; the protocol's rule picks
      the effort.
  (label validation_sheet.csv, then)
  python scripts/validation.py lock --run <run> --sheet validation
  python scripts/validation.py judge --run <run> --stage validation [--paid]
      Judge the 100 validation items once and the 200 test-retest items twice,
      at the chosen effort.
  python scripts/validation.py report --run <run> [--paid]
      Agreement, effort choice, test-retest and the go/no-go table.

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
    expected = {k for k in key if k.startswith("V" if sheet == "validation" else "P")}
    path = out / f"{sheet}_sheet.csv"
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    errors, labels, edited = [], {}, []
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
        if item in key and sha256_text(r["text"].replace("\r\n", "\n")) != key[item]["output_sha256"]:
            edited.append(item)
        labels[item] = {"emotion": emotion, "coherence": int(coherence) if coherence.isdigit() else None}
    if errors:
        raise ValueError(f"{path.name} is not ready to lock:\n  " + "\n  ".join(errors))
    if edited:
        print(f"warning: the text differs from the drawn output for {len(edited)} items "
              f"(often only spreadsheet line-ending changes): {edited[:10]}")

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
    # by Protocol Amendment 1: binary agreement on each category's drawn items).
    basic_agreement = sorted(per_cat[c]["binary_agreement"] for c in categories
                             if groups[c] == "basic" and per_cat[c]["binary_agreement"] is not None)
    median_basic = float(np.median(basic_agreement)) if basic_agreement else None
    subtle_rows = []
    for c in SUBTLE:
        lo, hi = per_cat[c]["binary_ci"]
        if hi is None or median_basic is None:
            subtle_rows.append((c, None, f"not assessable: {per_cat[c]['n_stratum']} judged items drawn from {c}"))
        else:
            subtle_rows.append((c, hi >= median_basic,
                                f"binary agreement {fmt(per_cat[c]['binary_agreement'])} "
                                f"(95% CI {fmt(lo)} to {fmt(hi)}, n {per_cat[c]['n_stratum']}) "
                                f"vs basic median {fmt(median_basic)}"))
    checks = [(f"4. {c} binary agreement not clearly below the basic categories", ok, detail)
              for c, ok, detail in subtle_rows]
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
    ]
    report = out / backend / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:16 + len(checks)]))
    print(f"full report: {report}")


# --- 7. Entry point ------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sample", help="draw the label and retest sets")
    p.add_argument("--source", required=True, help="full evaluation folder (runs/evaluation/<run>/<backend>)")
    for name in ("lock", "judge", "report"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True, help="vector run name the validation folder belongs to")
        if name == "lock":
            p.add_argument("--sheet", required=True, choices=("pilot", "validation"))
        if name == "judge":
            p.add_argument("--stage", required=True, choices=("pilot", "validation"))
        if name in ("judge", "report"):
            p.add_argument("--paid", action="store_true", help="Claude Fable 5.1 (asks before paying)")
    args = parser.parse_args()

    if args.command == "sample":
        cmd_sample(Path(args.source))
    elif args.command == "lock":
        cmd_lock(folder(args.run), args.sheet)
    elif args.command == "judge":
        cmd_judge(folder(args.run), args.stage, args.paid)
    else:
        cmd_report(folder(args.run), args.paid)


if __name__ == "__main__":
    main()
