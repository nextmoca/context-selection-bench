"""Pre-run OpenAI API cost estimators for the GSM boundary suite's two sweeps.

BOUNDARY-SUITE FRAMING (read first). The GSM suite is a DIAGNOSTIC, NOT a
flagship result. It exists to demonstrate two disclosed BOUNDARY cases for
context selection, NOT to headline a win:

  1. GSM8K -- clean, single-document math word problems. There is nothing to
     select away, so record-level selection is a NO-OP BY DESIGN: the arm
     correctly preserves the whole prompt.
  2. GSM-IC -- in-context distractors injected WITHIN a single record. These
     are OUTSIDE the selection regime, because record-level selection cannot
     remove a distractor sentence that lives inside the one record it must
     keep.

Flagship claims live with the RULER and the tool/document suites, not here.
No headline or marketing numbers belong in this module.

---------------------------------------------------------------------------
What this module does
---------------------------------------------------------------------------

This module estimates the token/dollar cost of the completion calls each GSM
sweep will make -- ONE completion per (method x budget x condition x item)
cell. It never calls a model/completion API itself; it is pure arithmetic over
real, already-computed token counts (real GSM8K / GSM-IC data via the suite's
data loaders, real per-item budget resolution via the same budget math the
sweep runners use). It sizes the aggregate spend so a human can review the
figure -- and it returns the estimate so a caller can gate execution on it
before spending anything (the confirm-spend / ``BENCH_BUDGET_USD`` budget
gate). Each estimator prints both a "requested run" figure and a SEPARATE,
explicitly rough full-run projection, and never executes either run.

Two sweeps are covered here, side by side:

  * GSM8K sweep -- methods x 4 budget labels (25/50/75%/native-default) x
    {clean, distractor} x N items.
  * GSM-IC sweep -- methods x 3 budget labels (25/75%/native-default; NO 50%)
    x {clean, distractor} x N base items.

---------------------------------------------------------------------------
PRICING CAVEAT -- read before trusting these numbers for a real budget decision
---------------------------------------------------------------------------

As of the date below, gpt-4o-mini did **not** appear on OpenAI's CURRENT
public pricing page (that page listed only the then-current gpt-5.x-era
models). Several 2026-dated third-party trackers still quoted gpt-4o-mini at
its long-standing launch rate of $0.15 / 1M input tokens and $0.60 / 1M output
tokens -- but none of those are OpenAI's own authoritative pricing page.

**This is a placeholder, not a verified-current figure.** Before either sweep
actually spends real API budget:

  1. Confirm gpt-4o-mini is still callable on the account that will run this.
  2. Re-check the live OpenAI pricing page (or the account's billing page) for
     its current per-token price.
  3. If gpt-4o-mini is gone, update ``BASE_MODEL_NAME`` and the two price
     constants below (and re-run this estimator) before proceeding -- do not
     silently keep spending against a guessed number.

The completion-length assumption (``EXPECTED_COMPLETION_TOKENS``) is likewise a
documented ASSUMPTION, not a measurement: GSM completions are short
chain-of-thought + a final numeric answer; 100-300 tokens is a typical range
for gpt-4o-mini-scale GSM harness runs, and the midpoint is used as a single
point estimate. It will be wrong in either direction for any individual item;
it is only meant to size the *aggregate* cost.

---------------------------------------------------------------------------
Few-shot-on-top-of-question-budget accounting
---------------------------------------------------------------------------

Percentage budgets (25/50/75%) are resolved as a fraction of the QUESTION
content's token count only, NOT the full prompt's -- the fixed, always-present
few-shot block is scaffolding that neither the Needlepath nor the LLMLingua-2
selection step reduces. That few-shot block is therefore added back into the
per-cell prompt-token estimate for those two methods, or this estimator would
systematically UNDERCOUNT real prompt size / real API cost. ``full_context``
sends the item's real, full prompt unchanged, so its estimate is just the
item's full-prompt token count.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from csbench.suites.gsm.gsm8k_data import (
    DEFAULT_PILOT_N,
    DEFAULT_SEED,
    GSM8KPromptExample,
    build_fewshot_prompt,
    build_prompt_examples,
    load_gsm8k_test_items,
)
from csbench.suites.gsm.gsmic_data import DEFAULT_SEED as GSMIC_DEFAULT_SEED
from csbench.suites.gsm.gsmic_data import load_gsmic_2step_items
from csbench.suites.gsm.request import GSMICAdapterItem, build_gsmic_adapter_items
from csbench.tokenizing import estimate_tokens


# ===========================================================================
# Shared pricing + completion-length assumptions (single source of truth for
# both sweeps -- avoids two independently-maintained, possibly drifting figures)
# ===========================================================================

BASE_MODEL_NAME = "gpt-4o-mini"

PRICING_CHECKED_DATE = "2026-07-04"
GPT4O_MINI_INPUT_USD_PER_1M_TOKENS = 0.15
GPT4O_MINI_OUTPUT_USD_PER_1M_TOKENS = 0.60
PRICING_SOURCE_NOTE = (
    f"PRICING CAVEAT: gpt-4o-mini pricing was NOT found on OpenAI's current "
    f"public pricing page as of {PRICING_CHECKED_DATE} (that page listed only "
    f"then-current gpt-5.x-era models). The figures used here "
    f"(${GPT4O_MINI_INPUT_USD_PER_1M_TOKENS}/1M input, "
    f"${GPT4O_MINI_OUTPUT_USD_PER_1M_TOKENS}/1M output) are the long-standing, "
    f"widely-republished rate quoted since gpt-4o-mini's launch and still cited "
    f"by third-party trackers in 2026 -- but this is a placeholder, NOT a "
    f"verified-current figure. VERIFY current gpt-4o-mini pricing/availability "
    f"directly with OpenAI before relying on this for a real budget decision."
)

# GSM completion length assumption (documented, not fabricated as precise --
# see module docstring): short chain-of-thought + final numeric answer.
EXPECTED_COMPLETION_TOKENS_LOW = 100
EXPECTED_COMPLETION_TOKENS_HIGH = 300
EXPECTED_COMPLETION_TOKENS = 200  # midpoint assumption used as the point estimate

# The Needlepath arm's default selection budget (max selected tokens) at the
# np-2026-07-r1 operating point -- what the "native_default" budget resolves to
# for the needlepath method with no explicit override.
NEEDLEPATH_NATIVE_DEFAULT_TOKENS = 8000
LLMLINGUA_NATIVE_DEFAULT_RATE = 0.5  # LLMLingua-2's own default compression rate


def budget_tokens_for_reduction(question_tokens: int, target_reduction: float) -> int:
    """Sweep target X% reduction -> budget_tokens = (1 - X) * question_tokens.

    ``question_tokens`` is the reducible QUESTION content's token count (the
    bare question, plus any in-record distractor block), NOT the full prompt's
    token count. The few-shot block is fixed scaffolding and is excluded from
    this basis entirely -- see module docstring."""
    return max(1, round((1.0 - target_reduction) * question_tokens))


@dataclass(frozen=True)
class CellCostEstimate:
    """Estimated cost for one (method, budget_label) cell, aggregated over the
    items passed in (a single condition's worth, matching each sweep runner's
    per-condition-call convention)."""

    method: str
    budget_label: str
    n_items: int
    mean_prompt_tokens: float
    total_prompt_tokens: int
    total_completion_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float


def _cell_cost_from_prompt_tokens(
    method: str, budget_label: str, per_item_prompt_tokens: Sequence[int]
) -> CellCostEstimate:
    """Assemble a ``CellCostEstimate`` from a list of per-item prompt-token
    counts -- the one place the pricing arithmetic lives, so both sweeps price
    a cell identically."""
    n_items = len(per_item_prompt_tokens)
    total_prompt_tokens = sum(per_item_prompt_tokens)
    total_completion_tokens = n_items * EXPECTED_COMPLETION_TOKENS

    input_cost_usd = total_prompt_tokens / 1_000_000 * GPT4O_MINI_INPUT_USD_PER_1M_TOKENS
    output_cost_usd = total_completion_tokens / 1_000_000 * GPT4O_MINI_OUTPUT_USD_PER_1M_TOKENS

    return CellCostEstimate(
        method=method,
        budget_label=budget_label,
        n_items=n_items,
        mean_prompt_tokens=(total_prompt_tokens / n_items) if n_items else 0.0,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        input_cost_usd=input_cost_usd,
        output_cost_usd=output_cost_usd,
        total_cost_usd=input_cost_usd + output_cost_usd,
    )


# ===========================================================================
# GSM8K sweep
# ===========================================================================

# Budget labels for the GSM8K sweep (re-declared locally: the sweep-runner
# module is not a dependency of this estimator).
GSM8K_BUDGET_LABELS: Sequence[str] = ("25pct", "50pct", "75pct", "native_default")
_GSM8K_TARGET_REDUCTION_BY_BUDGET_LABEL = {"25pct": 0.25, "50pct": 0.5, "75pct": 0.75}

PILOT_METHODS: Sequence[str] = ("full_context", "needlepath", "llmlingua2")
PILOT_CONDITIONS: Sequence[str] = ("clean", "distractor")
DEFAULT_ESTIMATE_N = DEFAULT_PILOT_N  # the real GSM8K sweep size


def _decompose_prompt(item: GSM8KPromptExample) -> Tuple[str, str, Optional[str]]:
    """Recover (fewshot_prefix, bare_question, distractor_block) from a
    ``GSM8KPromptExample``, inverting the fixed assembly convention the data
    loader uses::

        prompt = f"{fewshot_prompt}\\n\\nQuestion: {question_text}\\nAnswer:"
        question_text (distractor only) = f"{distractor_text}\\n\\n{bare_question}"

    Raises ``ValueError`` if the item does not match that documented shape --
    a real contract violation between the data model and this estimator, not
    something to silently paper over. Only ``fewshot_prefix`` is consumed by
    this module (to add the fixed few-shot block back into the reduced-method
    prompt-token estimate); the other two are returned for completeness."""
    if item.condition == "distractor":
        if not item.distractor_text:
            raise ValueError(f"distractor item {item.item_id!r} is missing distractor_text")
        prefix = f"{item.distractor_text}\n\n"
        if not item.question.startswith(prefix):
            raise ValueError(
                f"distractor item {item.item_id!r} question does not start with its own distractor_text block"
            )
        bare_question = item.question[len(prefix):]
    else:
        bare_question = item.question

    suffix = f"\n\nQuestion: {item.question}\nAnswer:"
    if not item.prompt.endswith(suffix):
        raise ValueError(f"item {item.item_id!r} prompt does not end with the expected question suffix")
    fewshot_prefix = item.prompt[: -len(suffix)]
    return fewshot_prefix, bare_question, item.distractor_text


def _gsm8k_llmlingua_native_default_budget_tokens(
    item: GSM8KPromptExample, native_rate: float = LLMLINGUA_NATIVE_DEFAULT_RATE
) -> int:
    """Back-solve LLMLingua-2's own default rate against the QUESTION-content
    token count only (the few-shot block is not compressible)."""
    question_tokens = estimate_tokens(item.question)
    return max(1, round(native_rate * question_tokens))


def resolve_gsm8k_budget_tokens(method: str, budget_label: str, item: GSM8KPromptExample) -> int:
    """Resolve a (method, budget_label) pair to a concrete ``budget_tokens``
    value for this specific GSM8K item. Percentage budgets are matched as a %
    of the item's QUESTION-content tokens (not the full prompt's), so the
    requested percentage is meaningful. ``full_context``'s ``native_default``
    resolves to the full prompt tokens -- a documented, ignored placeholder
    (that method never uses its ``budget_tokens`` argument)."""
    if budget_label == "native_default":
        if method == "needlepath":
            return NEEDLEPATH_NATIVE_DEFAULT_TOKENS
        if method == "llmlingua2":
            return _gsm8k_llmlingua_native_default_budget_tokens(item)
        if method == "full_context":
            return estimate_tokens(item.prompt)
        raise ValueError(f"unknown method {method!r}")

    if method not in PILOT_METHODS:
        raise ValueError(f"unknown method {method!r}")
    if budget_label not in _GSM8K_TARGET_REDUCTION_BY_BUDGET_LABEL:
        raise ValueError(f"unknown budget_label {budget_label!r}")

    target_reduction = _GSM8K_TARGET_REDUCTION_BY_BUDGET_LABEL[budget_label]
    question_tokens = estimate_tokens(item.question)
    return budget_tokens_for_reduction(question_tokens, target_reduction)


def resolve_gsm8k_prompt_tokens_for_method(method: str, budget_label: str, item: GSM8KPromptExample) -> int:
    """Estimate the actual prompt tokens the base model will see for one
    (method, budget_label, item) cell -- reusing the real budget-resolution
    logic, not a second invented estimate.

    - ``full_context``: the item's real, full prompt token count.
    - ``needlepath`` / ``llmlingua2``: the resolved budget targets the QUESTION
      content only; the fixed, always-present few-shot block is added back here
      (or this would systematically UNDERCOUNT real prompt size / cost)."""
    if method == "full_context":
        return estimate_tokens(item.prompt)
    fewshot_prefix, _bare_question, _distractor_block = _decompose_prompt(item)
    fewshot_tokens = estimate_tokens(fewshot_prefix)
    return fewshot_tokens + resolve_gsm8k_budget_tokens(method, budget_label, item)


def estimate_gsm8k_cell_cost(method: str, budget_label: str, items: Sequence[GSM8KPromptExample]) -> CellCostEstimate:
    per_item_prompt_tokens = [
        resolve_gsm8k_prompt_tokens_for_method(method, budget_label, item) for item in items
    ]
    return _cell_cost_from_prompt_tokens(method, budget_label, per_item_prompt_tokens)


@dataclass(frozen=True)
class PilotCostEstimate:
    """Estimated cost for the full GSM8K matrix: N items x methods x budgets x
    conditions."""

    n_items: int
    methods: Sequence[str]
    budget_labels: Sequence[str]
    conditions: Sequence[str]
    total_completions: int
    cells: List[CellCostEstimate]
    total_cost_usd: float


def estimate_pilot_cost(
    n: int = DEFAULT_ESTIMATE_N,
    seed: int = DEFAULT_SEED,
    methods: Sequence[str] = PILOT_METHODS,
    budget_labels: Sequence[str] = GSM8K_BUDGET_LABELS,
    conditions: Sequence[str] = PILOT_CONDITIONS,
) -> PilotCostEstimate:
    """Estimate the total gpt-4o-mini completion cost for the real GSM8K sweep
    matrix, using real GSM8K data (first ``n`` test-split items via the same
    seeded selection the suite uses) and real per-item budget resolution."""
    items = load_gsm8k_test_items()[:n]
    fewshot = build_fewshot_prompt()
    examples = build_prompt_examples(items, fewshot, seed=seed)
    examples_by_condition = {
        condition: [e for e in examples if e.condition == condition] for condition in conditions
    }

    cells: List[CellCostEstimate] = []
    for method in methods:
        for budget_label in budget_labels:
            for condition in conditions:
                cells.append(estimate_gsm8k_cell_cost(method, budget_label, examples_by_condition[condition]))

    total_completions = len(items) * len(methods) * len(budget_labels) * len(conditions)
    total_cost_usd = sum(cell.total_cost_usd for cell in cells)

    return PilotCostEstimate(
        n_items=len(items),
        methods=methods,
        budget_labels=budget_labels,
        conditions=conditions,
        total_completions=total_completions,
        cells=cells,
        total_cost_usd=total_cost_usd,
    )


def format_pilot_cost_report(estimate: PilotCostEstimate) -> str:
    lines = [
        "=" * 72,
        f"GSM8K SWEEP COST ESTIMATE -- {BASE_MODEL_NAME} (N={estimate.n_items} matrix)",
        "=" * 72,
        f"Methods: {', '.join(estimate.methods)}",
        f"Budgets: {', '.join(estimate.budget_labels)}",
        f"Conditions: {', '.join(estimate.conditions)}",
        f"Total completions (1 per method x budget x condition x item): {estimate.total_completions}",
        f"Assumed completion tokens/item: {EXPECTED_COMPLETION_TOKENS} "
        f"(documented assumption, range {EXPECTED_COMPLETION_TOKENS_LOW}-{EXPECTED_COMPLETION_TOKENS_HIGH})",
        "",
        f"Estimated total cost: ${estimate.total_cost_usd:.4f}",
        "",
        PRICING_SOURCE_NOTE,
        "=" * 72,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full multi-regime run -- rough, approximate, NOT executed
# ---------------------------------------------------------------------------
#
# The full matched protocol covers multiple regimes (tool-calling, long-horizon
# agents, long context, reasoning, doc-QA), more methods, and publishable scale
# (hundreds per regime). Much of that infrastructure is built elsewhere in the
# benchmark; this GSM-only estimator cannot price it precisely, so this is a
# deliberately rough sizing: it scales the GSM8K sweep's real, empirically-
# computed per-completion cost by the ratio of matrix sizes. It is an
# order-of-magnitude tool, NOT a precise budget.

FULL_RUN_REGIMES = 5  # tool-calling, long-horizon agents, long context, reasoning, doc-QA
FULL_RUN_METHODS = 6  # full_context, needlepath, llmlingua2, plus three additional methods
FULL_RUN_BUDGETS = 4  # 25% / 50% / 75% / native-default, per the matched-budget sweep
FULL_RUN_CONDITIONS = 2  # assumes the clean/distractor split generalizes; undecided per-regime
FULL_RUN_N_PER_REGIME = 500  # publishable scale (hundreds), per the matched protocol


@dataclass(frozen=True)
class FullRunCostEstimate:
    """A deliberately rough, order-of-magnitude estimate for the eventual full
    multi-regime run. NOT to be executed by this estimator."""

    regimes: int
    methods: int
    budgets: int
    conditions: int
    n_per_regime: int
    total_completions: int
    total_cost_usd: float
    caveat: str


def estimate_full_multiregime_cost(pilot_estimate: PilotCostEstimate) -> FullRunCostEstimate:
    """Scale the GSM8K sweep's real per-completion cost up to the full protocol's
    matrix size. Deliberately rough -- see section docstring above."""
    total_completions = (
        FULL_RUN_REGIMES * FULL_RUN_METHODS * FULL_RUN_BUDGETS * FULL_RUN_CONDITIONS * FULL_RUN_N_PER_REGIME
    )
    per_completion_cost_usd = pilot_estimate.total_cost_usd / pilot_estimate.total_completions
    total_cost_usd = per_completion_cost_usd * total_completions

    caveat = (
        "ROUGH, APPROXIMATE ORDER-OF-MAGNITUDE ESTIMATE ONLY. Derived by scaling the "
        "GSM8K sweep's real per-completion cost by the ratio of matrix sizes (5 regimes x "
        "6 methods x 4 budgets x 2 conditions x 500 items/regime vs this sweep's actual "
        "matrix) -- NOT a fresh per-regime token estimate. Real prompt sizes for those "
        "regimes (e.g. long-context tasks routinely run 10k-100k+ token contexts) could be "
        "dramatically larger or smaller than GSM8K's short word-problem prompts, so this "
        "number could be off by an order of magnitude in either direction. It exists only "
        "to give a directional sense of scale -- it is NOT a budget commitment."
    )

    return FullRunCostEstimate(
        regimes=FULL_RUN_REGIMES,
        methods=FULL_RUN_METHODS,
        budgets=FULL_RUN_BUDGETS,
        conditions=FULL_RUN_CONDITIONS,
        n_per_regime=FULL_RUN_N_PER_REGIME,
        total_completions=total_completions,
        total_cost_usd=total_cost_usd,
        caveat=caveat,
    )


def format_full_run_cost_estimate(estimate: FullRunCostEstimate) -> str:
    lines = [
        "=" * 72,
        "FULL MULTI-REGIME RUN -- COST ESTIMATE ONLY (NOT being run -- full-run estimate only.)",
        "=" * 72,
        f"Regimes: {estimate.regimes} | Methods: {estimate.methods} | Budgets: {estimate.budgets} | "
        f"Conditions: {estimate.conditions} | N/regime: {estimate.n_per_regime}",
        f"Total completions (hypothetical): {estimate.total_completions}",
        f"Estimated total cost (hypothetical): ${estimate.total_cost_usd:,.2f}",
        "",
        estimate.caveat,
        "=" * 72,
    ]
    return "\n".join(lines)


def print_cost_estimates(
    n: int = DEFAULT_ESTIMATE_N,
    seed: int = DEFAULT_SEED,
    conditions: Sequence[str] = PILOT_CONDITIONS,
) -> PilotCostEstimate:
    """Compute and print BOTH GSM8K estimates (sweep + rough full-run), per the
    "estimate cost BEFORE running; print the estimate; print a SEPARATE cost
    estimate for the eventual full run" rule. Returns the sweep estimate so a
    caller can gate further execution (confirm-spend / ``BENCH_BUDGET_USD``)
    without recomputing.

    ``conditions`` must match whatever the caller will actually run -- the
    printed estimate matching the real run is a hard invariant."""
    pilot = estimate_pilot_cost(n=n, seed=seed, conditions=conditions)
    print(format_pilot_cost_report(pilot))
    full_run = estimate_full_multiregime_cost(pilot)
    print(format_full_run_cost_estimate(full_run))
    return pilot


# ===========================================================================
# GSM-IC sweep
# ===========================================================================

# Budget labels for the GSM-IC sweep: 25% / 75% / native-default only -- NO 50%
# (the GSM-IC sweep's established scope). Re-declared locally.
GSMIC_BUDGET_LABELS: Sequence[str] = ("25pct", "75pct", "native_default")
_GSMIC_TARGET_REDUCTION_BY_BUDGET_LABEL = {"25pct": 0.25, "75pct": 0.75}

GSMIC_METHODS: Sequence[str] = ("full_context", "needlepath", "llmlingua2")
GSMIC_CONDITIONS: Sequence[str] = ("clean", "distractor")
DEFAULT_GSMIC_ESTIMATE_N = 2400  # the real GSM-IC stratified sample size


def _gsmic_llmlingua_native_default_budget_tokens(
    item: GSMICAdapterItem, native_rate: float = LLMLINGUA_NATIVE_DEFAULT_RATE
) -> int:
    """Back-solve LLMLingua-2's own default rate against the QUESTION-content
    token count only (the few-shot block is not compressible)."""
    question_tokens = estimate_tokens(item.question)
    return max(1, round(native_rate * question_tokens))


def resolve_gsmic_budget_tokens(method: str, budget_label: str, item: GSMICAdapterItem) -> int:
    """Resolve a (method, budget_label) pair to a concrete ``budget_tokens``
    value for this specific GSM-IC item. Percentage budgets are matched as a %
    of the item's QUESTION-content tokens (not the full prompt's).
    ``full_context``'s ``native_default`` resolves to the full prompt tokens --
    a documented, ignored placeholder."""
    if budget_label == "native_default":
        if method == "needlepath":
            return NEEDLEPATH_NATIVE_DEFAULT_TOKENS
        if method == "llmlingua2":
            return _gsmic_llmlingua_native_default_budget_tokens(item)
        if method == "full_context":
            return estimate_tokens(item.prompt)
        raise ValueError(f"unknown method {method!r}")

    if method not in GSMIC_METHODS:
        raise ValueError(f"unknown method {method!r}")
    if budget_label not in _GSMIC_TARGET_REDUCTION_BY_BUDGET_LABEL:
        raise ValueError(f"unknown budget_label {budget_label!r}")

    target_reduction = _GSMIC_TARGET_REDUCTION_BY_BUDGET_LABEL[budget_label]
    question_tokens = estimate_tokens(item.question)
    return budget_tokens_for_reduction(question_tokens, target_reduction)


def resolve_gsmic_prompt_tokens_for_method(method: str, budget_label: str, item: GSMICAdapterItem) -> int:
    """Estimate the actual prompt tokens the base model will see for one
    (method, budget_label, item) cell -- mirroring the GSM8K estimator.

    - ``full_context``: the item's real, full prompt token count.
    - ``needlepath`` / ``llmlingua2``: the resolved budget targets the QUESTION
      content only; the item's fixed few-shot prefix is added back here."""
    if method == "full_context":
        return estimate_tokens(item.prompt)
    fewshot_tokens = estimate_tokens(item.fewshot_prefix)
    return fewshot_tokens + resolve_gsmic_budget_tokens(method, budget_label, item)


def estimate_gsmic_cell_cost(
    method: str, budget_label: str, items: Sequence[GSMICAdapterItem]
) -> CellCostEstimate:
    per_item_prompt_tokens = [
        resolve_gsmic_prompt_tokens_for_method(method, budget_label, item) for item in items
    ]
    return _cell_cost_from_prompt_tokens(method, budget_label, per_item_prompt_tokens)


@dataclass(frozen=True)
class GsmicPilotCostEstimate:
    """Estimated cost for the full GSM-IC matrix: N base items x methods x
    budgets x conditions."""

    n_items: int
    methods: Sequence[str]
    budget_labels: Sequence[str]
    conditions: Sequence[str]
    total_completions: int
    cells: List[CellCostEstimate]
    total_cost_usd: float


def estimate_gsmic_pilot_cost(
    n: int = DEFAULT_GSMIC_ESTIMATE_N,
    seed: int = GSMIC_DEFAULT_SEED,
    methods: Sequence[str] = GSMIC_METHODS,
    budget_labels: Sequence[str] = GSMIC_BUDGET_LABELS,
    conditions: Sequence[str] = GSMIC_CONDITIONS,
) -> GsmicPilotCostEstimate:
    """Estimate the total gpt-4o-mini completion cost for the real GSM-IC sweep
    matrix, using real GSM-IC data (first ``n`` base items) and real per-item
    budget resolution.

    ``n`` refers to the number of BASE GSM-IC items (NOT doubled for
    clean+distractor -- each base item yields one example per requested
    condition, exactly mirroring how the sweep actually runs)."""
    items = load_gsmic_2step_items()[:n]
    fewshot = build_fewshot_prompt()
    adapter_items = build_gsmic_adapter_items(items, fewshot)
    adapter_items_by_condition = {
        condition: [ai for ai in adapter_items if ai.condition == condition] for condition in conditions
    }

    cells: List[CellCostEstimate] = []
    for method in methods:
        for budget_label in budget_labels:
            for condition in conditions:
                cells.append(
                    estimate_gsmic_cell_cost(method, budget_label, adapter_items_by_condition[condition])
                )

    total_completions = len(items) * len(methods) * len(budget_labels) * len(conditions)
    total_cost_usd = sum(cell.total_cost_usd for cell in cells)

    return GsmicPilotCostEstimate(
        n_items=len(items),
        methods=methods,
        budget_labels=budget_labels,
        conditions=conditions,
        total_completions=total_completions,
        cells=cells,
        total_cost_usd=total_cost_usd,
    )


def format_gsmic_pilot_cost_report(estimate: GsmicPilotCostEstimate) -> str:
    lines = [
        "=" * 72,
        f"GSM-IC SWEEP COST ESTIMATE -- {BASE_MODEL_NAME} (N={estimate.n_items} base items)",
        "=" * 72,
        f"Methods: {', '.join(estimate.methods)}",
        f"Budgets: {', '.join(estimate.budget_labels)}",
        f"Conditions: {', '.join(estimate.conditions)}",
        f"Total completions (1 per method x budget x condition x item): {estimate.total_completions}",
        f"Assumed completion tokens/item: {EXPECTED_COMPLETION_TOKENS} "
        f"(documented assumption, range {EXPECTED_COMPLETION_TOKENS_LOW}-{EXPECTED_COMPLETION_TOKENS_HIGH})",
        "",
        f"Estimated total cost: ${estimate.total_cost_usd:.4f}",
        "",
        PRICING_SOURCE_NOTE,
        "=" * 72,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full-scale GSM-IC run -- rough linear scale-up, NOT executed here
# ---------------------------------------------------------------------------
#
# The GSM-IC sweep's own real target scale is already what
# ``estimate_gsmic_pilot_cost`` computes at its default ``n``. This section lets
# a caller who ran a SMALLER estimate (e.g. a smoke-scale sanity check) still
# see a rough projection to the real full scale without re-deriving the ratio
# by hand -- a simple linear scale-up of the smaller estimate's real,
# empirically-computed per-completion cost, not a second, independently
# fabricated number.

FULL_GSMIC_RUN_N = DEFAULT_GSMIC_ESTIMATE_N


@dataclass(frozen=True)
class FullGsmicRunCostEstimate:
    """A rough, linearly-scaled projection to the real full GSM-IC scale. NOT
    to be executed by this estimator -- informational only."""

    n_items: int
    total_completions: int
    total_cost_usd: float
    caveat: str


def estimate_full_gsmic_run_cost(pilot_estimate: GsmicPilotCostEstimate) -> FullGsmicRunCostEstimate:
    """Scale ``pilot_estimate``'s real per-completion cost up to the full
    ``FULL_GSMIC_RUN_N``-item scale (a no-op scale-up, i.e. returns the same
    numbers, if ``pilot_estimate.n_items`` already equals ``FULL_GSMIC_RUN_N``)."""
    n_completions_per_item = pilot_estimate.total_completions / pilot_estimate.n_items if pilot_estimate.n_items else 0
    total_completions = round(n_completions_per_item * FULL_GSMIC_RUN_N)
    per_completion_cost_usd = (
        pilot_estimate.total_cost_usd / pilot_estimate.total_completions if pilot_estimate.total_completions else 0.0
    )
    total_cost_usd = per_completion_cost_usd * total_completions

    caveat = (
        "ROUGH, LINEARLY-SCALED PROJECTION to the real full GSM-IC scale -- derived "
        "from the given estimate's real, empirically-computed per-completion cost "
        "(NOT a second, independently fabricated figure). If the given estimate's n "
        "already equals the full scale, this is simply that same estimate restated. "
        "This is NOT being executed -- see the pricing caveat above before treating "
        "either number as a final budget."
    )

    return FullGsmicRunCostEstimate(
        n_items=FULL_GSMIC_RUN_N,
        total_completions=total_completions,
        total_cost_usd=total_cost_usd,
        caveat=caveat,
    )


def format_full_gsmic_run_cost_estimate(estimate: FullGsmicRunCostEstimate) -> str:
    lines = [
        "=" * 72,
        f"GSM-IC FULL-SCALE (N={estimate.n_items:,}) COST ESTIMATE -- NOT being run (estimate only).",
        "=" * 72,
        f"N items: {estimate.n_items}",
        f"Total completions (hypothetical): {estimate.total_completions}",
        f"Estimated total cost (hypothetical): ${estimate.total_cost_usd:,.2f}",
        "",
        estimate.caveat,
        "=" * 72,
    ]
    return "\n".join(lines)


def print_gsmic_cost_estimates(
    n: int = DEFAULT_GSMIC_ESTIMATE_N,
    seed: int = GSMIC_DEFAULT_SEED,
    conditions: Sequence[str] = GSMIC_CONDITIONS,
) -> GsmicPilotCostEstimate:
    """Compute and print BOTH GSM-IC estimates (the requested-run estimate + a
    rough full-scale projection), per the "estimate cost BEFORE running; print
    the estimate; print a SEPARATE cost estimate for the eventual full run"
    rule. Returns the requested-run estimate so a caller can gate further
    execution (confirm-spend / ``BENCH_BUDGET_USD``) without recomputing.

    ``conditions`` must match whatever the caller will actually run -- the
    printed estimate matching the real run is a hard invariant."""
    pilot = estimate_gsmic_pilot_cost(n=n, seed=seed, conditions=conditions)
    print(format_gsmic_pilot_cost_report(pilot))
    full_run = estimate_full_gsmic_run_cost(pilot)
    print(format_full_gsmic_run_cost_estimate(full_run))
    return pilot


if __name__ == "__main__":
    print_cost_estimates()
    print_gsmic_cost_estimates()
