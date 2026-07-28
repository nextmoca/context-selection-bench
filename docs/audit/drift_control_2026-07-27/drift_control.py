"""GATE 2 drift control: re-run a seeded sample of items that ARE present in
ruler_repl_v1 and compare new scores against their originals.

Merges nothing. Writes only to the scratch out-dir. Aborts on budget breach.
"""
from __future__ import annotations
import json, os, random, sys, time
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from csbench.suites.ruler.harness import build_arm, run_item, _budget_spec
from csbench.suites.ruler.data import load_examples
from csbench.model_client import build_gemini_client

ITEMS = REPO / "runs" / "ruler_repl_v1" / "items"
DATA = REPO / "data" / "ruler"
OUT = Path(os.environ.get("DRIFT_OUT", "runs/drift_control_out"))
OUT.mkdir(parents=True, exist_ok=True)

DRIFT_SEED = 20260727
N_ITEMS = 50                      # x 3 arms = 150 evaluations
ARMS = ["full_context", "needlepath", "compresr"]
MODEL = "gemini-3.1-pro-preview"
MAX_OUT = 256
OP = "np-2026-07-r1"
NP_URL = "http://127.0.0.1:8799"

# budget guard, armed at the HIGHER of the two recorded rates (list price)
RATE_IN, RATE_OUT = 2.00, 12.00   # USD / 1M tokens, ai.google.dev retrieved 2026-07-27
BUDGET_USD = float(os.environ.get("BENCH_BUDGET_USD", "8"))

AFFECTED = [  # (length, task) cells that lost positions
    ("8k","niah_multikey_1"),("8k","niah_multikey_3"),("8k","niah_multiquery"),
    ("8k","niah_multivalue"),("8k","niah_single_1"),("8k","niah_single_2"),("8k","niah_single_3"),
    ("16k","niah_multikey_1"),("16k","niah_multikey_3"),("16k","niah_multiquery"),
    ("16k","niah_multivalue"),("16k","niah_single_1"),("16k","niah_single_2"),("16k","niah_single_3"),
]
UNTOUCHED = [("8k","qa_1"),("8k","vt"),("16k","cwe"),("16k","niah_multikey_2")]


def original_rows():
    """(length, arm, task) -> {position: row}, duplicates collapsed."""
    out = defaultdict(dict)
    for p in ITEMS.glob("*/*/*.jsonl"):
        length, arm, task = p.parts[-3], p.parts[-2], p.name[:-6]
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[(length, arm, task)].setdefault(int(r["position"]), r)
    return out


def pick_items(orig):
    rng = random.Random(DRIFT_SEED)
    chosen = []
    per_aff = 36 // len(AFFECTED) + 1     # ~3 per affected cell
    for cell in AFFECTED:
        surv = sorted(orig[(cell[0], "full_context", cell[1])])
        chosen += [(cell[0], cell[1], p) for p in rng.sample(surv, min(per_aff, len(surv)))]
    per_unt = max(1, (N_ITEMS - len(chosen)) // len(UNTOUCHED))
    for cell in UNTOUCHED:
        surv = sorted(orig[(cell[0], "full_context", cell[1])])
        chosen += [(cell[0], cell[1], p) for p in rng.sample(surv, min(per_unt, len(surv)))]
    rng.shuffle(chosen)
    return chosen[:N_ITEMS]


def main():
    orig = original_rows()
    items = pick_items(orig)
    print(f"drift control: {len(items)} items x {len(ARMS)} arms = {len(items)*len(ARMS)} evaluations")
    print(f"seed={DRIFT_SEED}  budget=${BUDGET_USD:.2f} armed at ${RATE_IN}/1M in, ${RATE_OUT}/1M out\n", flush=True)

    client = build_gemini_client(timeout_ms=120000)
    budget = _budget_spec(OP)
    arms = {a: build_arm(a, needlepath_url=NP_URL, operating_point=OP) for a in ARMS}

    ex_cache = {}
    def examples_for(length, task):
        if (length, task) not in ex_cache:
            ex_cache[(length, task)] = load_examples(DATA / length, task, limit=100)
        return ex_cache[(length, task)]

    spent_in = spent_out = 0.0
    results = []
    t0 = time.time()
    for n, (length, task, pos) in enumerate(items, 1):
        ex = examples_for(length, task)[pos - 1]
        for arm_name in ARMS:
            o = orig[(length, arm_name, task)].get(pos)
            if o is None:
                continue
            cost = (spent_in * RATE_IN + spent_out * RATE_OUT) / 1e6
            if cost > BUDGET_USD:
                print(f"\nBUDGET BREACH at ${cost:.2f} > ${BUDGET_USD:.2f} -- ABORTING", flush=True)
                json.dump(results, (OUT / "drift_rows.json").open("w"), indent=2)
                return 2
            try:
                row = run_item(arms[arm_name], arm_name, ex, position=pos, client=client,
                               model=MODEL, max_output_tokens=MAX_OUT, budget=budget,
                               operating_point=OP)
            except Exception as e:
                print(f"  ERROR {length}/{task}/{arm_name}/p{pos}: {type(e).__name__}: {e}", flush=True)
                continue
            spent_in += row.get("input_tokens", 0); spent_out += row.get("output_tokens", 0)
            results.append({
                "length": length, "task": task, "arm": arm_name, "position": pos,
                "index_orig": o["index"], "index_new": row["index"],
                "score_orig": float(o.get("score", 0.0)), "score_new": float(row.get("score", 0.0)),
                "correct_orig": bool(o.get("correct")), "correct_new": bool(row.get("correct")),
                "fallback_orig": bool(o.get("fallback_used")), "fallback_new": bool(row.get("fallback_used")),
                "tokens_after_orig": o.get("tokens_after"), "tokens_after_new": row.get("tokens_after"),
            })
        if n % 10 == 0:
            c = (spent_in * RATE_IN + spent_out * RATE_OUT) / 1e6
            print(f"  [{n}/{len(items)}] spent ${c:.2f}  elapsed {time.time()-t0:.0f}s", flush=True)

    json.dump(results, (OUT / "drift_rows.json").open("w"), indent=2)
    final = (spent_in * RATE_IN + spent_out * RATE_OUT) / 1e6
    print(f"\ncompleted {len(results)} evaluations, spent ${final:.2f} "
          f"({spent_in/1e6:.3f}M in, {spent_out/1e6:.4f}M out)")
    print(f"at recorded $0.80/1M blended that is ${(spent_in+spent_out)/1e6*0.80:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
