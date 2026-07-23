"""Pre-run Gemini API cost estimator + spend-confirmation gate for the RULER
matched-protocol benchmark.

This module is pure arithmetic over real, already-generated token counts; it
never calls a model/completion API itself. It sizes the token/dollar cost of a
full RULER run so a human can review the figure and explicitly confirm the
spend before the run driver spends anything.

The RULER matrix this estimator is built for::

    13 tasks x 2 context lengths (8k, 16k) x 100 examples x 3 arms
    = 2,600 cases x 3 arms = 7,800 real Gemini completions.

---------------------------------------------------------------------------
Real per-case completion count
---------------------------------------------------------------------------

Each RULER case issues exactly THREE real ``call_gemini`` completions -- one
per arm under test, each on that arm's rendered prompt::

    full        -> the full, unreduced baseline prompt
    needlepath  -> the Needlepath-selected context prompt
    llmlingua2  -> the LLMLingua-2-compressed prompt

An arm's selection/compression step does NOT itself call Gemini and so is
correctly NOT counted here: the Needlepath arm resolves its selection on its
hosted endpoint (``arm.select(request) -> ContextResponse``), and the
LLMLingua-2 arm compresses locally with an in-process model. ``call_gemini``
also does not make a separate token-counting call: real prompt/completion
token counts come from the completion response's own ``usage_metadata`` (with
a local ``estimate_tokens`` fallback only when that is absent). Any dataset
generation done to size the RULER corpus to 8k/16k is sunk cost already spent,
not part of this run's spend, and is correctly NOT counted here.

So: 2,600 cases x 3 completions/case = 7,800 total Gemini completions for the
full 13-task x 2-length x 100-example x 3-arm plan.

---------------------------------------------------------------------------
PRICING -- gemini-3.1-pro-preview
---------------------------------------------------------------------------

Checked live 2026-07-06 against the official Gemini Developer API pricing page
(ai.google.dev/gemini-api/docs/pricing). ``gemini-3.1-pro-preview`` is directly
and currently listed there::

    Standard tier, prompts <= 200k tokens:
        input:  $2.00 / 1M tokens
        output: $12.00 / 1M tokens
    Standard tier, prompts > 200k tokens:
        input:  $4.00 / 1M tokens
        output: $18.00 / 1M tokens

Every RULER case here (8k or 16k context) is far under the 200k threshold, so
the <=200k tier applies throughout this estimator. This is a
directly-documented, current figure for the exact model being used -- no tier
substitution or "closest documented model" caveat is needed. The caveats that
DO apply, stated plainly:

  1. ``gemini-3.1-pro-preview`` is a **preview** model -- preview pricing can
     change with little notice. Re-check the live pricing page immediately
     before the run actually spends, not just once here.
  2. Token counts below use this repository's own deterministic
     ``estimate_tokens`` heuristic (word/punctuation + char-based estimate),
     the same fallback ``call_gemini`` uses when a live response's
     ``usage_metadata`` is absent -- it is NOT Gemini's real tokenizer, so real
     spend can differ from this estimate in either direction. It is a sizing
     tool, not an invoice.
  3. Input-token sizing below is a **conservative upper bound**: it assumes NO
     context reduction for the needlepath and llmlingua2 arms (i.e. the same
     real, full baseline-prompt token count is charged for all three arms). The
     Needlepath arm is configured with a bounded selection budget which should
     reduce its real input well below this bound in most cases (except when a
     full-context fallback triggers and it reverts to the full baseline
     prompt); LLMLingua-2's real compression ratio is only known once the run
     executes. "Assume no reduction" is the documented, explicitly-labeled
     placeholder for both non-baseline arms. Real spend should be at or below
     this estimate for the two compressed arms, not above it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping, Sequence

from csbench.model_client import DEFAULT_MODEL
from csbench.suites.ruler.data import OfficialRulerExample, load_examples
from csbench.tokenizing import estimate_tokens

RULER_MODEL_NAME = DEFAULT_MODEL

# The RULER v1 task set this benchmark runs (13 tasks).
DEFAULT_RULER_TASKS: Sequence[str] = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)

# Default on-disk locations for the fetched RULER JSONL corpora, one per context
# length. Upstream data is not redistributed in this repository; a fetch-script
# lands it in a git-ignored ``data/`` directory (see ``datasets/README.md``).
# Callers may override ``data_roots`` to point elsewhere.
DEFAULT_DATA_ROOT = Path("data")
DEFAULT_DATA_ROOTS: Mapping[str, Path] = {
    "8k": DEFAULT_DATA_ROOT / "ruler_official_8k",
    "16k": DEFAULT_DATA_ROOT / "ruler_official_16k",
}

PRICING_CHECKED_DATE = "2026-07-06"
PRICING_TIER_LABEL = "<=200k tokens/prompt"
GEMINI_3_1_PRO_INPUT_USD_PER_1M_TOKENS = 2.00
GEMINI_3_1_PRO_OUTPUT_USD_PER_1M_TOKENS = 12.00
# Recorded for completeness / a future >200k regime; not used by this
# estimator since every RULER case here is 8k or 16k context.
GEMINI_3_1_PRO_INPUT_USD_PER_1M_TOKENS_GT_200K = 4.00
GEMINI_3_1_PRO_OUTPUT_USD_PER_1M_TOKENS_GT_200K = 18.00

PRICING_SOURCE_NOTE = (
    f"PRICING SOURCE: ai.google.dev/gemini-api/docs/pricing (official Gemini Developer "
    f"API pricing page), checked {PRICING_CHECKED_DATE}. {RULER_MODEL_NAME}, "
    f"{PRICING_TIER_LABEL} tier: ${GEMINI_3_1_PRO_INPUT_USD_PER_1M_TOKENS}/1M input, "
    f"${GEMINI_3_1_PRO_OUTPUT_USD_PER_1M_TOKENS}/1M output. This IS the exact preview "
    f"model this benchmark calls (no closest-tier substitution needed) -- but it is "
    f"still a PREVIEW model, so re-verify current pricing immediately before the run "
    f"actually spends, not just once here. Token counts are this repository's own "
    f"`estimate_tokens` heuristic, not Gemini's real tokenizer -- real spend can differ "
    f"from this estimate. Needlepath/llmlingua2 input-token sizing below assumes NO "
    f"context reduction (a conservative, explicitly-labeled UPPER BOUND) since neither "
    f"arm's real achieved reduction is known until the run executes -- see module "
    f"docstring."
)

# Three completions per case: the full-context baseline, the Needlepath arm, and
# the LLMLingua-2 arm (csbench.arms.FullContextArm / NeedlepathArm /
# llmlingua2.Llmlingua2Arm). Only these three make real Gemini completions.
RULER_METHODS: Sequence[str] = ("full", "needlepath", "llmlingua2")
RULER_LENGTH_LABELS: Sequence[str] = ("8k", "16k")
RULER_COMPLETIONS_PER_CASE = len(RULER_METHODS)  # 3
DEFAULT_EXAMPLES_PER_CELL = 100  # examples per (task, length) cell
DEFAULT_MAX_OUTPUT_TOKENS = 256  # matches the run's --max-output-tokens default


def baseline_prompt(example: OfficialRulerExample) -> str:
    """The full, unreduced baseline prompt for one RULER example -- the official
    input text followed by its answer prefix. This is the token-sizing basis for
    every arm's conservative upper-bound input estimate."""
    return example.input_text + example.answer_prefix


def load_real_ruler_examples(
    length_label: str,
    task: str,
    *,
    data_roots: Mapping[str, Path] = DEFAULT_DATA_ROOTS,
    limit: int = DEFAULT_EXAMPLES_PER_CELL,
) -> List[OfficialRulerExample]:
    """Load the real, already-generated RULER JSONL rows for one
    (length, task) cell -- the same ``load_examples`` the real run uses, not a
    second invented loader."""
    return load_examples(data_roots[length_label], task, limit=limit)


def estimate_case_input_tokens(example: OfficialRulerExample) -> int:
    """Conservative per-arm input-token estimate for one real example: the real,
    full baseline prompt's token count via this repository's own
    ``estimate_tokens`` -- used for ALL THREE arms (see module docstring's
    "conservative upper bound" caveat: needlepath/llmlingua2 are assumed to
    achieve NO reduction for this headline estimate)."""
    return estimate_tokens(baseline_prompt(example))


@dataclass(frozen=True)
class TaskLengthCostEstimate:
    """Estimated cost for one (length_label, task) cell across all
    ``RULER_METHODS``, aggregated over its real, already-generated examples."""

    length_label: str
    task: str
    n_examples: int
    total_completions: int
    total_input_tokens: int
    total_output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float


def estimate_task_length_cost(
    length_label: str,
    task: str,
    *,
    data_roots: Mapping[str, Path] = DEFAULT_DATA_ROOTS,
    limit: int = DEFAULT_EXAMPLES_PER_CELL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    methods: Sequence[str] = RULER_METHODS,
) -> TaskLengthCostEstimate:
    examples = load_real_ruler_examples(length_label, task, data_roots=data_roots, limit=limit)
    n_examples = len(examples)
    n_methods = len(methods)
    per_example_input_tokens = [estimate_case_input_tokens(example) for example in examples]

    total_completions = n_examples * n_methods
    total_input_tokens = sum(per_example_input_tokens) * n_methods
    total_output_tokens = total_completions * max_output_tokens

    input_cost_usd = total_input_tokens / 1_000_000 * GEMINI_3_1_PRO_INPUT_USD_PER_1M_TOKENS
    output_cost_usd = total_output_tokens / 1_000_000 * GEMINI_3_1_PRO_OUTPUT_USD_PER_1M_TOKENS

    return TaskLengthCostEstimate(
        length_label=length_label,
        task=task,
        n_examples=n_examples,
        total_completions=total_completions,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        input_cost_usd=input_cost_usd,
        output_cost_usd=output_cost_usd,
        total_cost_usd=input_cost_usd + output_cost_usd,
    )


@dataclass(frozen=True)
class RulerRunCostEstimate:
    """Estimated cost for the full RULER matched-protocol run: all
    ``tasks`` x ``lengths`` x ``examples_per_cell`` x ``methods``."""

    tasks: Sequence[str]
    length_labels: Sequence[str]
    examples_per_cell: int
    methods: Sequence[str]
    max_output_tokens: int
    total_cases: int
    total_completions: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    cells: List[TaskLengthCostEstimate] = field(default_factory=list)


def estimate_ruler_run_cost(
    *,
    tasks: Sequence[str] = DEFAULT_RULER_TASKS,
    length_labels: Sequence[str] = RULER_LENGTH_LABELS,
    data_roots: Mapping[str, Path] = DEFAULT_DATA_ROOTS,
    examples_per_cell: int = DEFAULT_EXAMPLES_PER_CELL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    methods: Sequence[str] = RULER_METHODS,
) -> RulerRunCostEstimate:
    """Estimate the total gemini-3.1-pro-preview completion cost for the real
    RULER matrix, using the real, already-generated JSONL data -- not
    synthetic/hypothetical token counts."""
    cells: List[TaskLengthCostEstimate] = []
    for length_label in length_labels:
        for task in tasks:
            cells.append(
                estimate_task_length_cost(
                    length_label,
                    task,
                    data_roots=data_roots,
                    limit=examples_per_cell,
                    max_output_tokens=max_output_tokens,
                    methods=methods,
                )
            )

    total_cases = sum(cell.n_examples for cell in cells)
    total_completions = sum(cell.total_completions for cell in cells)
    total_input_tokens = sum(cell.total_input_tokens for cell in cells)
    total_output_tokens = sum(cell.total_output_tokens for cell in cells)
    total_cost_usd = sum(cell.total_cost_usd for cell in cells)

    return RulerRunCostEstimate(
        tasks=tasks,
        length_labels=length_labels,
        examples_per_cell=examples_per_cell,
        methods=methods,
        max_output_tokens=max_output_tokens,
        total_cases=total_cases,
        total_completions=total_completions,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_cost_usd=total_cost_usd,
        cells=cells,
    )


def format_ruler_cost_report(estimate: RulerRunCostEstimate) -> str:
    lines = [
        "=" * 78,
        f"RULER MATCHED-PROTOCOL RUN -- COST ESTIMATE ({RULER_MODEL_NAME})",
        "=" * 78,
        f"Tasks ({len(estimate.tasks)}): {', '.join(estimate.tasks)}",
        f"Context lengths: {', '.join(estimate.length_labels)}",
        f"Examples per (task, length) cell: {estimate.examples_per_cell}",
        f"Methods per case ({len(estimate.methods)}): {', '.join(estimate.methods)}",
        f"Total cases (tasks x lengths x examples): {estimate.total_cases}",
        f"Total real Gemini completions (cases x {len(estimate.methods)} methods): "
        f"{estimate.total_completions}",
        f"Assumed max output tokens/completion: {estimate.max_output_tokens} "
        f"(matches the run's --max-output-tokens default)",
        "",
        f"Total estimated input tokens (UPPER BOUND -- see caveat below): "
        f"{estimate.total_input_tokens:,}",
        f"Total estimated output tokens: {estimate.total_output_tokens:,}",
        "",
        f"Estimated input cost:  ${sum(c.input_cost_usd for c in estimate.cells):,.2f}",
        f"Estimated output cost: ${sum(c.output_cost_usd for c in estimate.cells):,.2f}",
        f"Estimated TOTAL cost:  ${estimate.total_cost_usd:,.2f}",
        "",
        PRICING_SOURCE_NOTE,
        "=" * 78,
        "CONFIRMATION REQUIRED: the run must not spend against this estimate until a "
        "human has reviewed the figures above and explicitly confirmed (e.g. the run "
        "driver's `--confirm-spend` flag, after reviewing `--cost-estimate-only` "
        "output).",
        "=" * 78,
    ]
    return "\n".join(lines)


def print_ruler_cost_estimate(
    *,
    tasks: Sequence[str] = DEFAULT_RULER_TASKS,
    length_labels: Sequence[str] = RULER_LENGTH_LABELS,
    data_roots: Mapping[str, Path] = DEFAULT_DATA_ROOTS,
    examples_per_cell: int = DEFAULT_EXAMPLES_PER_CELL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    methods: Sequence[str] = RULER_METHODS,
) -> RulerRunCostEstimate:
    """Compute and print the RULER run cost estimate. Returns the estimate so
    callers (e.g. the run driver's ``--cost-estimate-only`` / spend-confirmation
    gate) can gate further execution on it without recomputing."""
    estimate = estimate_ruler_run_cost(
        tasks=tasks,
        length_labels=length_labels,
        data_roots=data_roots,
        examples_per_cell=examples_per_cell,
        max_output_tokens=max_output_tokens,
        methods=methods,
    )
    print(format_ruler_cost_report(estimate))
    return estimate


CONFIRM_SPEND_ENV_VAR = "CSBENCH_RULER_CONFIRM_SPEND"


def spend_is_confirmed(*, confirm_spend_flag: bool, env: Mapping[str, str]) -> bool:
    """A single, testable predicate for whether the explicit spend-confirmation
    gate has been satisfied: either the CLI flag was passed, or the env var
    override is set to a truthy value (for non-interactive/CI invocation).
    Mirrors the "explicit confirmation flag" requirement without hardcoding CLI
    parsing into the predicate itself."""
    if confirm_spend_flag:
        return True
    value = str(env.get(CONFIRM_SPEND_ENV_VAR, "")).strip().lower()
    return value in {"1", "true", "yes"}


if __name__ == "__main__":
    print_ruler_cost_estimate()
