"""GATE 2 analysis: per-arm agreement, signed shift, paired test."""
import json, sys
import numpy as np
from scipy.stats import wilcoxon
from collections import defaultdict

import os
P = os.environ.get("DRIFT_ROWS", "runs/drift_control_out/drift_rows.json")
rows = json.load(open(P))
print(f"drift-control evaluations: {len(rows)}\n")

# identity sanity: did we re-run the SAME underlying item?
bad = [r for r in rows if r["index_orig"] != r["index_new"]]
print(f"index identity mismatches (must be 0): {len(bad)}")
if bad:
    for b in bad[:5]:
        print("   ", b["length"], b["task"], b["arm"], b["position"], b["index_orig"], b["index_new"])
    print("\nIDENTITY FAILURE - the re-run did not hit the same items. STOP.")
    sys.exit(2)

by = defaultdict(list)
for r in rows:
    by[r["arm"]].append(r)

print(f"\n{'arm':<14}{'n':>5}{'exact score':>13}{'exact EM':>10}{'mean shift':>12}{'wilcoxon p':>12}{'verdict':>12}")
print("-" * 78)
overall = {}
for arm in ("full_context", "needlepath", "compresr"):
    rs = by.get(arm, [])
    if not rs:
        continue
    so = np.array([r["score_orig"] for r in rs]); sn = np.array([r["score_new"] for r in rs])
    eo = np.array([r["correct_orig"] for r in rs]); en = np.array([r["correct_new"] for r in rs])
    agree_s = float((so == sn).mean()); agree_e = float((eo == en).mean())
    d = sn - so; shift = float(d.mean())
    if np.allclose(d, 0):
        p = 1.0
    else:
        try: p = float(wilcoxon(d).pvalue)
        except ValueError: p = 1.0
    negligible = agree_s >= 0.95 and abs(shift) < 0.02 and p > 0.05
    overall[arm] = negligible
    print(f"{arm:<14}{len(rs):>5}{agree_s:>12.1%}{agree_e:>10.1%}{shift:>+12.4f}{p:>12.4f}"
          f"{'negligible' if negligible else 'MATERIAL':>12}")

# fallback-rate stability (needlepath/compresr behavioural drift)
print(f"\n{'arm':<14}{'fallback orig':>15}{'fallback new':>14}{'delta':>9}")
for arm in ("needlepath", "compresr"):
    rs = by.get(arm, [])
    if not rs: continue
    fo = np.mean([r["fallback_orig"] for r in rs]); fn = np.mean([r["fallback_new"] for r in rs])
    print(f"{arm:<14}{fo:>15.1%}{fn:>14.1%}{fn-fo:>+9.1%}")

# per-stratum view: affected vs untouched cells
UNT = {("8k","qa_1"),("8k","vt"),("16k","cwe"),("16k","niah_multikey_2")}
print(f"\n{'stratum':<12}{'arm':<14}{'n':>5}{'exact score':>13}{'mean shift':>12}")
for lab, pred in (("affected", lambda r: (r["length"], r["task"]) not in UNT),
                  ("untouched", lambda r: (r["length"], r["task"]) in UNT)):
    for arm in ("full_context", "needlepath", "compresr"):
        rs = [r for r in by.get(arm, []) if pred(r)]
        if not rs: continue
        so = np.array([r["score_orig"] for r in rs]); sn = np.array([r["score_new"] for r in rs])
        print(f"{lab:<12}{arm:<14}{len(rs):>5}{float((so==sn).mean()):>12.1%}{float((sn-so).mean()):>+12.4f}")

print("\n" + "=" * 78)
if all(overall.values()):
    print("GATE 2: PASS on all three arms -> merge authorized")
elif not overall.get("full_context", True) or not overall.get("needlepath", True):
    print("GATE 2: GEMINI/ANSWERING-MODEL DRIFT -> STOP AND ESCALATE. Do not merge.")
else:
    print("GATE 2: COMPRESR-ONLY DRIFT -> STOP AND ESCALATE with numbers. Maintainer chooses.")
