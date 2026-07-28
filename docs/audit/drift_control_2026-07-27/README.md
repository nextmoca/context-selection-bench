# Drift control, 2026-07-27

Artifacts behind the decision NOT to re-evaluate the 396 rows destroyed by the
resume-overwrite defect (see the paper's defect-disclosure section and
`docs/audit/HARNESS_AUDIT_2026-07.md`).

## What this measured

50 items that survived in `ruler_repl_v1`, selected under seed 20260727 and
stratified across the 14 affected task/length cells plus 4 untouched control
cells, re-run across all three arms on 2026-07-27 under the original
configuration (`gemini-3.1-pro-preview`, `max_output_tokens=256`, temperature 0,
operating point `np-2026-07-r1`, compresr 2.8.2 / `latte_v1` / no target ratio).
150 evaluations. Each new score is compared against that item's original.

## Result

Item-level exact agreement on the continuous score was 80.0% (full context),
88.0% (needlepath), 92.0% (compresr); on binarized correctness 88.0%, 94.0%,
94.0%. Mean signed shifts were -0.018, -0.035, -0.055. No paired test reached
significance (Wilcoxon p = 0.60, 0.24, 0.13). Fallback rates were unchanged
(needlepath 48.0% both times, compresr 0% both times).

## Scope limits

n = 50 per arm cannot separate ordinary non-determinism at temperature 0 from a
genuine model-side change. The threshold used to call the drift "material" was
chosen by us, not derived from a test; on the paired tests nothing is
significant. These numbers are reported descriptively and no inferential claim
rests on them. What they supported was a decision: re-evaluating the missing
items would have replaced a disclosed, characterised population limitation with
an undisclosed one, a population whose rows were measured on two dates with a
known 80-92% item-level agreement between them.

## Files

- `drift_rows.sha256` - checksum of `drift_rows.json`, the 150 paired
  evaluations carrying each item's original and new score, correctness,
  fallback flag and both `index` values (an identity precheck confirmed every
  re-run hit the same underlying item). The rows themselves follow this
  repository's standing convention for per-item outputs: object storage plus a
  SHA-256 manifest, never git.
- `drift_summary.json` - per-arm agreement, shifts and Wilcoxon p-values
- `drift_control.py` - selection and execution
- `drift_analyze.py` - the analysis reproducing the numbers above
