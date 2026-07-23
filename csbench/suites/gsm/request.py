"""GSM boundary-suite request construction (engine-free).

This suite is a DIAGNOSTIC, NOT a flagship result. It exists to demonstrate two
disclosed BOUNDARY cases for context selection, NOT to headline a win:

  1. GSM8K -- clean, single-document math word problems. There is nothing to
     select away, so record-level selection is a NO-OP BY DESIGN: the arm
     correctly preserves the whole prompt.
  2. GSM-IC -- an in-context distractor injected WITHIN a single record (one
     sentence woven into the narrative). This is OUTSIDE the selection regime,
     because record-level selection cannot remove a distractor sentence that
     lives inside the one record it must keep.

Flagship claims live with the RULER and the tool/document suites, not here. No
headline/marketing numbers belong in this module.

Scope of THIS module: only the engine-free, request-construction half of the
reference method adapters. The reference adapters combined two responsibilities
-- (a) turning one (example, condition, budget) into a selector/compressor
input, and (b) calling the engine/compressor and post-processing its output.
Half (b) is now the arm's job, behind the uniform ``arm.select(request) ->
ContextResponse`` contract; this module keeps only half (a): a request builder
that turns a GSM8K prompt example or a GSM-IC adapter item into a
``ContextRequest`` whose records/task/budget reproduce the selection input the
reference adapter built, so the generic csbench arms (full-context passthrough,
the hosted Needlepath selector, and whole-content compression arms) produce an
equivalent selection.

Fixed few-shot scaffolding (a matched-protocol rule preserved from the
reference design): the few-shot block is FIXED, always-present scaffolding --
it is never a selection candidate and is never compressed. It is therefore NOT
placed in the selection records for the ``needlepath`` and compression arms;
only the QUESTION content is. The runner re-attaches the identical few-shot
block verbatim at prompt-assembly time for those arms. The ``full_context``
baseline instead carries the whole assembled prompt as one record, so its
rendered context is byte-identical to the raw prompt (the runner does not
re-prepend the few-shot for that arm). This three-way split mirrors the
reference control flow exactly and keeps every method on identical few-shot
scaffolding, so no method gets a setup edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence

from csbench.contracts import BudgetSpec, ContextRecord, ContextRequest, TaskSpec

from .gsm8k_data import build_fewshot_prompt

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import
    from .gsm8k_data import GSM8KPromptExample
    from .gsmic_data import GSMICItem

# --------------------------------------------------------------------------- #
# Constants (ported from the reference adapters -- kind/source/importance/tags
# and the plain-format per-record excerpt ceiling that preserves matched-
# protocol parity: generous enough that no shared record is ever truncated by a
# small default per-record ceiling, so the only truncation that applies is the
# overall per-item token budget under test).
# --------------------------------------------------------------------------- #

_PLAIN_EXCERPT_TOKEN_CEILING = 20_000

# Record kinds mirror the reference state kinds, as plain wire strings.
_KIND_USER_INPUT = "user_input"
_KIND_EXTERNAL_DATA = "external_data"

# GSM8K selection-input fields (reference: the GSM8K method adapter).
_GSM8K_TOOL_NAME = "answer_gsm8k_question"
_GSM8K_QUESTION_SOURCE = "gsm8k_question"
_GSM8K_DISTRACTOR_SOURCE = "distractor_filler"
_GSM8K_QUESTION_IMPORTANCE = 10.0
_GSM8K_TASK_TAGS = ("gsm8k", "math_word_problem")

# GSM-IC selection-input fields (reference: the GSM-IC method adapter).
_GSMIC_TOOL_NAME = "answer_gsmic_question"
_GSMIC_SENTENCE_SOURCE = "gsmic_question_sentence"
_GSMIC_SENTENCE_IMPORTANCE = 8.0
_GSMIC_TASK_TAGS = ("gsmic", "math_word_problem")

# Distractor content (block for GSM8K, one sentence for GSM-IC) is deliberately
# low-scoring so a working selector filters it out.
_DISTRACTOR_IMPORTANCE = 0.0
_DISTRACTOR_TAGS = ("distractor", "irrelevant_context")

# Arm-name groups. The baseline carries the whole prompt (byte-identical
# passthrough); compression arms carry only the question (they compress it, and
# the runner re-attaches the fixed few-shot afterwards).
_BASELINE_ARM = "full_context"
_NEEDLEPATH_ARM = "needlepath"


# --------------------------------------------------------------------------- #
# GSM8K prompt decomposition
# --------------------------------------------------------------------------- #


def _decompose_prompt(example: "GSM8KPromptExample") -> tuple:
    """Recover ``(fewshot_prefix, bare_question, distractor_block)`` from a
    GSM8K prompt example, inverting the dataset builder's fixed assembly
    convention:

        prompt = f"{fewshot_prompt}\\n\\nQuestion: {question_text}\\nAnswer:"
        question_text (distractor only) = f"{distractor_text}\\n\\n{bare_question}"

    Raises ``ValueError`` if the example does not match that documented shape --
    a real contract violation between the data model and this builder, not
    something to silently paper over. ``distractor_block`` is ``None`` for the
    CLEAN condition (there is nothing to select away -- boundary case 1).
    """
    if example.condition == "distractor":
        if not example.distractor_text:
            raise ValueError(
                f"distractor example {example.item_id!r} is missing distractor_text"
            )
        prefix = f"{example.distractor_text}\n\n"
        if not example.question.startswith(prefix):
            raise ValueError(
                f"distractor example {example.item_id!r} question does not start with "
                "its own distractor_text block"
            )
        bare_question = example.question[len(prefix):]
    else:
        bare_question = example.question

    suffix = f"\n\nQuestion: {example.question}\nAnswer:"
    if not example.prompt.endswith(suffix):
        raise ValueError(
            f"example {example.item_id!r} prompt does not end with the expected "
            "question suffix"
        )
    fewshot_prefix = example.prompt[: -len(suffix)]
    return fewshot_prefix, bare_question, example.distractor_text


# --------------------------------------------------------------------------- #
# GSM-IC adapter items (engine-free question-with-segmentation unit)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GSMICAdapterItem:
    """One (GSM-IC item, condition) unit, ready for request construction.

    Carries the fields the request builder and a runner need: the assembled
    ``prompt`` and its fixed ``fewshot_prefix``, the ``question`` text, the
    parsed ``gold_numeric``, and the per-condition sentence-unit segmentation
    the GSM-IC ``needlepath`` request consumes. ``distractor_index`` is ``None``
    for the CLEAN condition (nothing to flag) and the flagged sentence index for
    the DISTRACTOR condition.
    """

    item_id: int
    base_problem_id: int
    condition: str
    question: str
    gold_numeric: float
    prompt: str
    fewshot_prefix: str
    sentences: List[str]
    distractor_index: Optional[int]


def build_gsmic_adapter_items(
    items: "Sequence[GSMICItem]", fewshot_prompt: Optional[str] = None
) -> List[GSMICAdapterItem]:
    """Build CLEAN + DISTRACTOR ``GSMICAdapterItem``s for each GSM-IC item.

    Reuses the GSM-IC data layer's prompt assembly (single source of truth for
    "how a GSM-IC prompt is assembled") and pairs each resulting example with
    its sentence-unit segmentation. The GSM-IC data layer (dataset loader,
    prompt assembly, and sentence segmentation) is imported lazily so this
    module imports without it.
    """
    # Lazy import: the GSM-IC data layer is an optional sibling; importing it
    # here keeps this module importable on its own.
    from .gsmic_data import (
        build_prompt_examples,
        segment_clean_question,
        segment_distractor_question,
    )

    fewshot = fewshot_prompt if fewshot_prompt is not None else build_fewshot_prompt()
    prompt_examples = {
        (pe.item_id, pe.condition): pe for pe in build_prompt_examples(items, fewshot)
    }

    out: List[GSMICAdapterItem] = []
    for item in items:
        clean_pe = prompt_examples[(item.item_id, "clean")]
        out.append(
            GSMICAdapterItem(
                item_id=item.item_id,
                base_problem_id=item.base_problem_id,
                condition="clean",
                question=clean_pe.question,
                gold_numeric=clean_pe.gold_numeric,
                prompt=clean_pe.prompt,
                fewshot_prefix=fewshot,
                sentences=segment_clean_question(item),
                distractor_index=None,
            )
        )

        distractor_pe = prompt_examples[(item.item_id, "distractor")]
        seg = segment_distractor_question(item)
        out.append(
            GSMICAdapterItem(
                item_id=item.item_id,
                base_problem_id=item.base_problem_id,
                condition="distractor",
                question=distractor_pe.question,
                gold_numeric=distractor_pe.gold_numeric,
                prompt=distractor_pe.prompt,
                fewshot_prefix=fewshot,
                sentences=seg.sentences,
                distractor_index=seg.distractor_index,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Request construction
# --------------------------------------------------------------------------- #


def _budget_spec(
    budget: int,
    *,
    operating_point: Optional[str],
    max_records: Optional[int],
) -> BudgetSpec:
    """The per-item operating point.

    ``budget`` is the per-item token budget the reference sweep computed as a
    percentage of the QUESTION content's token count (never the full prompt's),
    so the budget label constrains exactly the selectable content. The
    plain-format per-record excerpt ceiling is passed as an arm-neutral
    override; selecting methods honor it, compression / full-context arms ignore
    it harmlessly.
    """
    return BudgetSpec(
        max_context_tokens=max(1, int(budget)),
        operating_point=operating_point,
        max_records=max_records,
        max_excerpt_tokens_per_record=_PLAIN_EXCERPT_TOKEN_CEILING,
        mode="fixed",
    )


def _gsm8k_records_and_task(
    example: "GSM8KPromptExample",
    *,
    arm_name: str,
    record_prefix: str,
) -> tuple:
    """Records + task for one GSM8K (example, arm).

    - ``needlepath``: the QUESTION content decomposed into up to two records --
      the bare question (``user_input``, high importance, and the one required
      record so it is never silently dropped) and, for the DISTRACTOR condition
      only, the marked filler block (``external_data``, importance 0, tagged as
      a distractor). The few-shot block is NOT a record (fixed scaffolding).
    - ``full_context``: ONE record holding the whole assembled prompt, so the
      passthrough arm's rendered context is byte-identical to the raw prompt.
    - compression arms: ONE record holding the question text (the compressible
      content); the few-shot block is re-attached by the runner afterwards.
    """
    fewshot_prefix, bare_question, distractor_block = _decompose_prompt(example)

    if arm_name == _NEEDLEPATH_ARM:
        question_id = f"{record_prefix}-question"
        records = [
            ContextRecord(
                text=bare_question,
                kind=_KIND_USER_INPUT,
                id=question_id,
                source=_GSM8K_QUESTION_SOURCE,
                title="GSM8K question",
                importance=_GSM8K_QUESTION_IMPORTANCE,
            )
        ]
        required_record_ids = [question_id]
        if distractor_block:
            records.append(
                ContextRecord(
                    text=distractor_block,
                    kind=_KIND_EXTERNAL_DATA,
                    id=f"{record_prefix}-distractor",
                    source=_GSM8K_DISTRACTOR_SOURCE,
                    title="Irrelevant filler context",
                    importance=_DISTRACTOR_IMPORTANCE,
                    tags=list(_DISTRACTOR_TAGS),
                )
            )
    elif arm_name == _BASELINE_ARM:
        records = [ContextRecord(text=example.prompt, kind=_KIND_EXTERNAL_DATA)]
        required_record_ids = []
    else:  # compression arms (e.g. llmlingua2 / cpc): compress the question only
        records = [ContextRecord(text=example.question, kind=_KIND_EXTERNAL_DATA)]
        required_record_ids = []

    task = TaskSpec(
        prompt=bare_question,
        tool_name=_GSM8K_TOOL_NAME,
        required_record_ids=required_record_ids,
        tags=list(_GSM8K_TASK_TAGS),
    )
    return records, task


def _gsmic_records_and_task(
    item: "GSMICAdapterItem",
    *,
    arm_name: str,
    record_prefix: str,
) -> tuple:
    """Records + task for one GSM-IC (item, arm).

    - ``needlepath``: one record per question sentence-unit
      (``user_input``). NO sentence-unit is a required record -- the selector's
      own scoring decides what survives, matching a compression arm having no
      protected content either, so neither method gets a setup edge (boundary
      case 2: the distractor sentence lives inside the question, not in a
      separable block). The distractor sentence-unit, when present, is tagged as
      a distractor with importance 0.
    - ``full_context``: ONE record holding the whole assembled prompt.
    - compression arms: ONE record holding the full question text (distractor
      sentence included -- the question is compressible content for every method
      here).
    """
    if arm_name == _NEEDLEPATH_ARM:
        records = []
        for i, sentence in enumerate(item.sentences):
            is_distractor = (
                item.distractor_index is not None and i == item.distractor_index
            )
            records.append(
                ContextRecord(
                    text=sentence,
                    kind=_KIND_USER_INPUT,
                    id=f"{record_prefix}-sent-{i}",
                    source=_GSMIC_SENTENCE_SOURCE,
                    title=f"GSM-IC question sentence {i}",
                    importance=_DISTRACTOR_IMPORTANCE if is_distractor else _GSMIC_SENTENCE_IMPORTANCE,
                    tags=list(_DISTRACTOR_TAGS) if is_distractor else [],
                )
            )
    elif arm_name == _BASELINE_ARM:
        records = [ContextRecord(text=item.prompt, kind=_KIND_EXTERNAL_DATA)]
    else:  # compression arms: compress the full question (distractor included)
        records = [ContextRecord(text=item.question, kind=_KIND_EXTERNAL_DATA)]

    # required_record_ids intentionally EMPTY for GSM-IC (unlike GSM8K): no
    # sentence-unit is force-included.
    task = TaskSpec(
        prompt=item.question,
        tool_name=_GSMIC_TOOL_NAME,
        tags=list(_GSMIC_TASK_TAGS),
    )
    return records, task


def build_gsm_request(
    example_or_item,
    *,
    arm_name: str,
    budget: int,
    condition: Optional[str] = None,
    operating_point: Optional[str] = None,
    max_records: Optional[int] = None,
    request_id: Optional[str] = None,
) -> ContextRequest:
    """Build the ``ContextRequest`` for one arm and one GSM example/item.

    ``example_or_item`` is either a GSM8K prompt example (has ``distractor_text``)
    or a ``GSMICAdapterItem`` (has ``sentences``); the two are distinguished by
    presence of the ``sentences`` attribute. ``budget`` is the per-item token
    budget for the selectable content (see ``_budget_spec``). ``condition``, when
    given, must match the example's own condition; it is accepted for the
    caller's convenience and defaults to the example's ``condition``.

    The records/task/budget reproduce the selection input the reference adapter
    built for ``arm_name`` (see ``_gsm8k_records_and_task`` /
    ``_gsmic_records_and_task``), so the generic csbench arm produces an
    equivalent selection. ``render_format='plain'`` mirrors the reference
    plain-text format.
    """
    is_gsmic = hasattr(example_or_item, "sentences")
    cond = condition if condition is not None else getattr(example_or_item, "condition")
    item_id = getattr(example_or_item, "item_id")

    suite = "gsmic" if is_gsmic else "gsm8k"
    record_prefix = f"{suite}-{cond}-{item_id}"
    rid = request_id or f"{suite}:{cond}:{item_id}:{arm_name}"

    if is_gsmic:
        records, task = _gsmic_records_and_task(
            example_or_item, arm_name=arm_name, record_prefix=record_prefix
        )
    else:
        records, task = _gsm8k_records_and_task(
            example_or_item, arm_name=arm_name, record_prefix=record_prefix
        )

    budget_spec = _budget_spec(
        budget, operating_point=operating_point, max_records=max_records
    )

    return ContextRequest(
        request_id=rid,
        records=records,
        task=task,
        budget=budget_spec,
        render=True,
        render_format="plain",
        return_per_record=True,
    )
