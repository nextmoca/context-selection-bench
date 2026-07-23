"""GSM8K data surface + distractor augmentation (GSM boundary suite).

BOUNDARY-SUITE FRAMING (this suite is a DIAGNOSTIC, not a flagship result).
This module builds the two disclosed BOUNDARY cases for context selection; it
is here to characterize them plainly, NOT to headline a win:

  1. GSM8K clean -- single-document math word problems where there is nothing
     to select away, so record-level selection is a NO-OP BY DESIGN (the arm
     correctly preserves the whole prompt).
  2. GSM-IC-style distractors -- irrelevant filler injected WITHIN a single
     record. This is OUTSIDE the selection regime: record-level selection
     cannot remove a distractor sentence inside the one record it must keep.

Flagship claims live with the RULER and tool/document suites, not here.

Faithful-reproduction notes (kept so the numbers reproduce):

- **Data source**: real GSM8K via ``datasets.load_dataset("gsm8k", "main", ...)``
  -- the ``test`` split (1319 items) for evaluation items, the ``train`` split
  for the few-shot exemplars. No fabricated or hand-authored problem content
  anywhere. ``datasets`` is an optional dependency and is imported lazily inside
  the loader so this module imports without it.
- **Few-shot prompt**: rather than hand-transcribing a "canonical" 8-shot prompt
  from memory (risk of misquoting it), the prefix is built deterministically
  from the first ``NUM_FEWSHOT`` train-split examples, in dataset order (not
  randomly sampled), formatted with the standard ``Question: ... / Answer: ...``
  template and calculator annotations (``<<...>>``) stripped. This is the same
  convention used by standard eval harnesses: fixed, in-order train-split
  exemplars, chain-of-thought answer text ending in the ``#### <answer>`` marker.
  It is fully reproducible and never fabricated.
- **Item selection**: the evaluation selection is a plain
  ``load_gsm8k_test_items()[:n]`` -- the first ``n`` items in the ``test``
  split's own dataset order -- so the cost estimate and the actual run always
  agree on which items are used. ``select_pilot_indices`` below (a seeded
  shuffle-then-take helper) is NOT wired into that path; it is retained as a
  documented, unused helper for a future variant that wants a seeded/shuffled
  sample instead of dataset-order "first N". If it is ever wired in, the runner
  and the cost estimator must be updated together so they keep agreeing on the
  same item set.
- **Distractor augmentation**: irrelevant filler is real text -- sentences drawn
  from OTHER GSM8K test-split questions (never the target item's own question,
  never fabricated text), seeded-shuffled per item, and wrapped in an explicit
  ``[IRRELEVANT CONTEXT ...]`` marker block so it can never be silently mistaken
  for the real problem statement.
- **CLEAN vs DISTRACTOR**: always produced as two separate
  ``GSM8KPromptExample`` rows sharing ``item_id`` and ``gold_numeric`` but with
  distinct ``condition`` labels and distinct ``prompt`` text -- never blended.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence


# Pin the dataset fetch so numbers reproduce.
GSM8K_DATASET_ID = "gsm8k"
GSM8K_CONFIG = "main"

DEFAULT_SEED = 42
DEFAULT_PILOT_N = 200
NUM_FEWSHOT = 8
DISTRACTOR_SENTENCES = 5  # number of filler sentences drawn per distractor block

_GOLD_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_CALC_ANNOTATION_RE = re.compile(r"<<[^>]*>>")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

Condition = Literal["clean", "distractor"]


@dataclass(frozen=True)
class GSM8KItem:
    """A single GSM8K item with its gold answer parsed to a float."""

    index: int  # position in the source split (test: 0..1318, train: 0..N-1)
    question: str
    raw_answer: str  # full solution text including the trailing "#### <n>"
    gold_numeric: float


@dataclass(frozen=True)
class GSM8KPromptExample:
    """One (item, condition) prompt instance -- the unit the arms consume.

    ``condition`` is the only thing that may differ between the CLEAN and
    DISTRACTOR rows for the same ``item_id``; everything else (gold answer,
    few-shot prefix) is identical, per the matched-protocol requirement.
    """

    item_id: int  # the originating GSM8KItem.index
    condition: Condition
    question: str  # the (possibly distractor-augmented) question text
    gold_numeric: float
    distractor_text: Optional[str]  # None for "clean"; the filler block otherwise
    prompt: str  # fully assembled prompt: fixed few-shot prefix + question


def parse_gold_answer(raw_answer: str) -> float:
    """Parse the gold final numeric answer from a GSM8K `#### <number>` suffix."""
    match = _GOLD_ANSWER_RE.search(raw_answer)
    if not match:
        raise ValueError(f"No '#### <answer>' marker found in: {raw_answer!r}")
    return float(match.group(1).replace(",", ""))


def _items_from_split(split_name: str) -> List[GSM8KItem]:
    from datasets import load_dataset  # optional dependency, lazy-imported

    dataset = load_dataset(GSM8K_DATASET_ID, GSM8K_CONFIG, split=split_name)
    items: List[GSM8KItem] = []
    for idx, row in enumerate(dataset):
        items.append(
            GSM8KItem(
                index=idx,
                question=row["question"],
                raw_answer=row["answer"],
                gold_numeric=parse_gold_answer(row["answer"]),
            )
        )
    return items


def load_gsm8k_test_items() -> List[GSM8KItem]:
    """Load the real GSM8K `test` split (1319 items) with parsed gold answers."""
    return _items_from_split("test")


def load_gsm8k_train_items() -> List[GSM8KItem]:
    """Load the real GSM8K `train` split, used only for few-shot exemplars."""
    return _items_from_split("train")


def select_pilot_indices(
    total: int, n: int = DEFAULT_PILOT_N, seed: int = DEFAULT_SEED
) -> List[int]:
    """Seeded, reproducible selection of `n` indices out of `total`.

    Same (total, n, seed) always yields the same selection (sorted, unique,
    in range). Fixed default seed = DEFAULT_SEED.

    **NOT wired into the evaluation path** -- that path uses a plain
    ``load_gsm8k_test_items()[:n]`` (first ``n`` items in dataset order), and
    the runner and the cost estimator already agree with each other on that,
    which is the property that actually matters (the cost estimate must describe
    the same item set the real run uses). This function is kept only as a
    documented, unused helper for a variant that wants a seeded/shuffled sample
    instead of dataset-order "first N" -- do not assume it reflects what the
    evaluation run actually uses. See the module docstring's item-selection note
    for the full rationale.
    """
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    return sorted(indices[:n])


def _clean_answer_text(raw_answer: str) -> str:
    """Strip GSM8K calculator annotations (`<<...>>`), keep reasoning + `####`."""
    return _CALC_ANNOTATION_RE.sub("", raw_answer)


def build_fewshot_prompt(num_shots: int = NUM_FEWSHOT) -> str:
    """Build the fixed few-shot prefix used identically by every arm/condition.

    Uses the first `num_shots` GSM8K TRAIN-split examples, in dataset order
    (no randomness -- fully deterministic), formatted as
    `Question: ...\\nAnswer: ...` with calculator annotations stripped. See
    module docstring for why this convention was chosen over a hand-transcribed
    "canonical" prompt.
    """
    train_items = load_gsm8k_train_items()[:num_shots]
    blocks = [
        f"Question: {item.question}\nAnswer: {_clean_answer_text(item.raw_answer)}"
        for item in train_items
    ]
    return "\n\n".join(blocks)


def build_distractor_block(
    items: Sequence[GSM8KItem],
    exclude_index: int,
    seed: int = DEFAULT_SEED,
    n_sentences: int = DISTRACTOR_SENTENCES,
) -> str:
    """Build a deterministic, clearly-labeled block of irrelevant filler context.

    Filler is real text: sentences drawn from OTHER items' questions (never
    `exclude_index`'s own question), seeded-shuffled for reproducibility, and
    wrapped in an explicit marker so it can never be mistaken for the actual
    problem statement.
    """
    candidate_sentences: List[str] = []
    for item in items:
        if item.index == exclude_index:
            continue
        for sentence in _SENTENCE_SPLIT_RE.split(item.question.strip()):
            sentence = sentence.strip()
            if sentence:
                candidate_sentences.append(sentence)

    rng = random.Random(seed + exclude_index)
    rng.shuffle(candidate_sentences)
    filler = " ".join(candidate_sentences[:n_sentences])

    return (
        "[IRRELEVANT CONTEXT: DO NOT USE TO ANSWER; unrelated filler sentences "
        f"from other problems]\n{filler}\n[END IRRELEVANT CONTEXT]"
    )


def _assemble_prompt(fewshot_prompt: str, question_text: str) -> str:
    return f"{fewshot_prompt}\n\nQuestion: {question_text}\nAnswer:"


def build_prompt_examples(
    items: Sequence[GSM8KItem],
    fewshot_prompt: str,
    seed: int = DEFAULT_SEED,
    n_distractor_sentences: int = DISTRACTOR_SENTENCES,
) -> List[GSM8KPromptExample]:
    """Build CLEAN + DISTRACTOR prompt examples for each item.

    Every item yields exactly two `GSM8KPromptExample` rows sharing `item_id`
    and `gold_numeric`, distinguished only by `condition` (and, for
    "distractor", the appended filler block) -- CLEAN and DISTRACTOR content
    are never silently blended into a single row.
    """
    examples: List[GSM8KPromptExample] = []
    for item in items:
        examples.append(
            GSM8KPromptExample(
                item_id=item.index,
                condition="clean",
                question=item.question,
                gold_numeric=item.gold_numeric,
                distractor_text=None,
                prompt=_assemble_prompt(fewshot_prompt, item.question),
            )
        )

        distractor_text = build_distractor_block(
            items,
            exclude_index=item.index,
            seed=seed,
            n_sentences=n_distractor_sentences,
        )
        augmented_question = f"{distractor_text}\n\n{item.question}"
        examples.append(
            GSM8KPromptExample(
                item_id=item.index,
                condition="distractor",
                question=augmented_question,
                gold_numeric=item.gold_numeric,
                distractor_text=distractor_text,
                prompt=_assemble_prompt(fewshot_prompt, augmented_question),
            )
        )
    return examples
